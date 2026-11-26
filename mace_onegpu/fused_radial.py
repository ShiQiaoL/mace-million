#!/usr/bin/env python
"""
Fused Triton kernel for MACE's radial MLP (e3nn FullyConnectedNet [10, 64, 64, 64, D], normalize2mom(silu)).

e3nn semantics (verified from e3nn/nn/_fc.py): for hidden layers  h_{k+1} = cst * silu(h_k @ W_k / sqrt(fan_in)),
last layer  out = h_3 @ W_3 / sqrt(fan_in)  (no bias anywhere, var_in = var_out = 1 for MACE).
Folding 1/sqrt(fan_in) and cst into the NEXT layer's matrix gives
    out = silu(silu(silu(x A0) A1) A2) A3,
    A0 = W0/sqrt(10),  A1 = cst W1/sqrt(64),  A2 = cst W2/sqrt(64),  A3 = cst W3/sqrt(64).

Forward kernel : one program = BLOCK_E edges; all 4 matmuls (tensor cores via tl.dot) and 3 activations in
                 registers; writes only W [E, D].
Backward kernel: recomputes the activations in registers and back-propagates dL/dW -> dL/dx [E, 10];
                 nothing but x (10 floats/edge) is saved for backward.
Precision: 'ieee' (fp32 FMA, same arithmetic class as the original), 'tf32x3' (~fp32 accuracy on tensor cores),
           'tf32' (fastest). fp32 models only.

    from mace_onegpu.fused_radial import apply_fused_radial
    n = apply_fused_radial(model, precision="tf32x3")      # swaps block.conv_tp_weights in place
"""
import math
import os
import sys

import torch
import triton
import triton.language as tl


@triton.jit
def _silu(z):
    return z * tl.sigmoid(z)


@triton.jit
def _dsilu(z):
    s = tl.sigmoid(z)
    return s * (1.0 + z * (1.0 - s))


@triton.jit
def _radial_fwd(X, A0, A1, A2, A3, OUT, E,
                D: tl.constexpr, K0: tl.constexpr, H: tl.constexpr, BE: tl.constexpr, BD: tl.constexpr,
                PREC: tl.constexpr):
    pid = tl.program_id(0)
    offs_e = (pid * BE + tl.arange(0, BE)).to(tl.int64)  # int64: E*D exceeds 2^31 for large chunks
    me = offs_e < E
    ok = tl.arange(0, K0)
    oh = tl.arange(0, H)
    x = tl.load(X + offs_e[:, None] * 10 + ok[None, :], mask=me[:, None] & (ok[None, :] < 10), other=0.0)
    a0 = tl.load(A0 + ok[:, None] * H + oh[None, :])
    h = _silu(tl.dot(x, a0, input_precision=PREC))
    a1 = tl.load(A1 + oh[:, None] * H + oh[None, :])
    h = _silu(tl.dot(h, a1, input_precision=PREC))
    a2 = tl.load(A2 + oh[:, None] * H + oh[None, :])
    h = _silu(tl.dot(h, a2, input_precision=PREC))
    for n in tl.static_range(0, D, BD):
        od = n + tl.arange(0, BD)
        a3 = tl.load(A3 + oh[:, None] * D + od[None, :])
        o = tl.dot(h, a3, input_precision=PREC)
        tl.store(OUT + offs_e[:, None] * D + od[None, :], o, mask=me[:, None])


@triton.jit
def _radial_bwd(X, GOUT, A0, A1, A2, A3, GX, E,
                D: tl.constexpr, K0: tl.constexpr, H: tl.constexpr, BE: tl.constexpr, BD: tl.constexpr,
                PREC: tl.constexpr):
    pid = tl.program_id(0)
    offs_e = (pid * BE + tl.arange(0, BE)).to(tl.int64)  # int64: E*D exceeds 2^31 for large chunks
    me = offs_e < E
    ok = tl.arange(0, K0)
    oh = tl.arange(0, H)
    x = tl.load(X + offs_e[:, None] * 10 + ok[None, :], mask=me[:, None] & (ok[None, :] < 10), other=0.0)
    a0 = tl.load(A0 + ok[:, None] * H + oh[None, :])
    a1 = tl.load(A1 + oh[:, None] * H + oh[None, :])
    a2 = tl.load(A2 + oh[:, None] * H + oh[None, :])
    z0 = tl.dot(x, a0, input_precision=PREC)
    h0 = _silu(z0)
    z1 = tl.dot(h0, a1, input_precision=PREC)
    h1 = _silu(z1)
    z2 = tl.dot(h1, a2, input_precision=PREC)
    gh2 = tl.zeros((BE, H), dtype=tl.float32)
    for n in tl.static_range(0, D, BD):
        od = n + tl.arange(0, BD)
        g = tl.load(GOUT + offs_e[:, None] * D + od[None, :], mask=me[:, None], other=0.0)
        a3 = tl.load(A3 + oh[:, None] * D + od[None, :])
        gh2 += tl.dot(g, tl.trans(a3), input_precision=PREC)
    gz2 = gh2 * _dsilu(z2)
    gz1 = tl.dot(gz2, tl.trans(a2), input_precision=PREC) * _dsilu(z1)
    gz0 = tl.dot(gz1, tl.trans(a1), input_precision=PREC) * _dsilu(z0)
    gx = tl.dot(gz0, tl.trans(a0), input_precision=PREC)
    tl.store(GX + offs_e[:, None] * 10 + ok[None, :], gx, mask=me[:, None] & (ok[None, :] < 10))


BE_DEFAULT = int(os.environ.get("FUSED_RADIAL_BE", "64"))
BD_DEFAULT = 128


class _FusedRadialFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, A0, A1, A2, A3, prec, be, bd):
        x = x.contiguous()
        E = x.shape[0]
        D = A3.shape[1]
        out = torch.empty(E, D, dtype=x.dtype, device=x.device)
        grid = (triton.cdiv(E, be),)
        _radial_fwd[grid](x, A0, A1, A2, A3, out, E, D=D, K0=A0.shape[0], H=A1.shape[0], BE=be, BD=bd, PREC=prec)
        ctx.save_for_backward(x)
        ctx.mats = (A0, A1, A2, A3)
        ctx.cfg = (prec, be, bd)
        return out

    @staticmethod
    def backward(ctx, gout):
        (x,) = ctx.saved_tensors
        A0, A1, A2, A3 = ctx.mats
        prec, be, bd = ctx.cfg
        E = x.shape[0]
        D = A3.shape[1]
        gx = torch.empty_like(x)
        grid = (triton.cdiv(E, be),)
        _radial_bwd[grid](x, gout.contiguous(), A0, A1, A2, A3, gx, E, D=D, K0=A0.shape[0], H=A1.shape[0],
                          BE=be, BD=bd, PREC=prec)
        return gx, None, None, None, None, None, None, None


class FusedRadialMLP(torch.nn.Module):
    """Drop-in replacement for an e3nn FullyConnectedNet[10, H, H, H, D] with normalize2mom(silu)."""

    def __init__(self, net, precision="tf32x3", be=BE_DEFAULT, bd=BD_DEFAULT):
        super().__init__()
        layers = list(net)
        assert len(layers) == 4, "expects 3 hidden layers + output"
        hs = [layers[0].h_in] + [l.h_out for l in layers]
        assert hs[0] == 10, hs
        H = hs[1]
        assert hs[2] == H and hs[3] == H, hs
        csts = []
        for l in layers[:3]:
            act = l.act
            csts.append(1.0 if getattr(act, "_is_id", False) else float(act.cst))
            assert l.var_in == 1 and l.var_out == 1
        assert layers[3].act is None and layers[3].var_in == 1 and layers[3].var_out == 1
        dt = layers[0].weight.dtype
        assert dt == torch.float32, "fused radial kernel is fp32-only"
        W = [l.weight.detach() for l in layers]
        A0 = W[0] / math.sqrt(hs[0])
        A1 = csts[0] * W[1] / math.sqrt(H)
        A2 = csts[1] * W[2] / math.sqrt(H)
        A3 = csts[2] * W[3] / math.sqrt(H)
        K0 = 16
        A0p = torch.zeros(K0, H, dtype=dt, device=A0.device)
        A0p[:10] = A0
        D = hs[4]
        assert D % bd == 0, f"output width {D} must be a multiple of {bd}"
        self.register_buffer("A0", A0p.contiguous())
        self.register_buffer("A1", A1.contiguous())
        self.register_buffer("A2", A2.contiguous())
        self.register_buffer("A3", A3.contiguous())
        self.prec, self.be, self.bd = precision, be, bd
        self.hs = hs

    def forward(self, x):
        return _FusedRadialFn.apply(x, self.A0, self.A1, self.A2, self.A3, self.prec, self.be, self.bd)


def apply_fused_radial(model, precision="tf32x3"):
    n = 0
    for blk in model.interactions:
        if hasattr(blk, "conv_tp_weights") and not isinstance(blk.conv_tp_weights, FusedRadialMLP):
            blk.conv_tp_weights = FusedRadialMLP(blk.conv_tp_weights, precision=precision)
            n += 1
    return n


# ----------------------------------------------------------------------------- self-test
def _selftest(model_path, E=1_000_000, repeat=20):
    import copy
    import time

    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    torch.set_default_dtype(torch.float32)
    m = torch.load(model_path, map_location="cuda").float()
    for blk_i, blk in enumerate(m.interactions):
        net = blk.conv_tp_weights
        x = torch.randn(E, 10, device="cuda") * 0.5
        # reference: original e3nn module, strict fp32 matmuls
        torch.backends.cuda.matmul.allow_tf32 = False
        xr = x.clone().requires_grad_(True)
        yr = net(xr)
        g = torch.randn_like(yr)
        (gxr,) = torch.autograd.grad(yr, xr, g)

        def timeit(fn):
            for _ in range(3):
                fn()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(repeat):
                fn()
            torch.cuda.synchronize()
            return (time.perf_counter() - t0) / repeat * 1e3

        def ref_fb():
            xx = x.clone().requires_grad_(True)
            yy = net(xx)
            torch.autograd.grad(yy, xx, g)

        t_ref = timeit(ref_fb)
        print(f"block {blk_i}: D={yr.shape[1]}  E={E:,}  original e3nn fwd+bwd {t_ref:.2f} ms", flush=True)
        for prec in ("ieee", "tf32x3", "tf32"):
            fm = FusedRadialMLP(net, precision=prec).cuda()
            xf = x.clone().requires_grad_(True)
            yf = fm(xf)
            (gxf,) = torch.autograd.grad(yf, xf, g)
            ey = float((yf - yr).abs().max() / yr.abs().max())
            eg = float((gxf - gxr).abs().max() / gxr.abs().max())

            def f_fb():
                xx = x.clone().requires_grad_(True)
                yy = fm(xx)
                torch.autograd.grad(yy, xx, g)

            t = timeit(f_fb)
            print(f"   fused {prec:7s}: fwd+bwd {t:7.2f} ms  ({t_ref/t:5.1f}x)   rel.err W {ey:.2e}   rel.err dW/dx {eg:.2e}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python -m mace_onegpu.fused_radial <mace model file, e.g. mace-mpa-0-medium.model>")
    _selftest(sys.argv[1])
