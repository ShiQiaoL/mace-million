#!/usr/bin/env python
"""
fp32-accurate tensor-core GEMMs ("3xTF32") for MACE's radial MLPs (e3nn FullyConnectedNet, no bias).

Every fp32 operand is split exactly as  a = a_hi + a_lo  with a_hi = round_to_tf32(a) and a_lo = round_to_tf32(a - a_hi),
and  a @ b  ~=  a_hi@b_hi + a_hi@b_lo + a_lo@b_hi   (the dropped a_lo@b_lo term is ~2^-22 relative, i.e. fp32 level).
Since hi/lo parts are exactly representable in TF32, the TF32 tensor-core GEMM loses nothing on input conversion.
  forward : one GEMM with K concatenated:  [x_hi | x_hi | x_lo] @ [W_hi ; W_lo ; W_hi]
  backward: dx = g_hi @ [W_hi^T | W_lo^T] (halves summed) + g_lo @ W_hi^T
TF32 is switched on ONLY around these matmuls; everything else in the model stays strict fp32.

    from mace_onegpu import radial_3xtf32; radial_3xtf32.apply(model)      # swaps blk.conv_tp_weights in place
"""
import math
import os
import sys

import torch


_MASK = -8192  # ~0x1FFF as int32: keep sign, exponent and the top 10 mantissa bits


def to_tf32(x):
    """Round fp32 to the nearest TF32-representable value (ties away from zero), returned as fp32."""
    i = x.contiguous().view(torch.int32)
    return ((i + 0x1000) & _MASK).view(torch.float32)


def split(x):
    hi = to_tf32(x)
    lo = to_tf32(x - hi)
    return hi, lo


def _tf32_mm(a, b):
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        return a @ b
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev


class _MM3(torch.autograd.Function):
    """y = x @ W (W constant) with 3xTF32 forward and backward (grad wrt x only)."""

    @staticmethod
    def forward(ctx, x, Bcat, WT2, WhT):
        xh, xl = split(x)
        y = _tf32_mm(torch.cat([xh, xh, xl], 1), Bcat)
        ctx.save_for_backward(WT2, WhT)
        return y

    @staticmethod
    def backward(ctx, g):
        WT2, WhT = ctx.saved_tensors
        gh, gl = split(g)
        t = _tf32_mm(gh, WT2)                       # [E, 2K] = [gh@WhT | gh@WlT]
        K = WhT.shape[1]
        gx = t[:, :K] + t[:, K:] + _tf32_mm(gl, WhT)
        return gx, None, None, None


class Radial3xTF32(torch.nn.Module):
    """Drop-in for e3nn FullyConnectedNet (no bias, var_in=var_out=1) with 3xTF32 GEMMs."""

    def __init__(self, net):
        super().__init__()
        self.acts = []
        for j, l in enumerate(net):
            assert getattr(l, "bias", None) is None and l.var_in == 1 and l.var_out == 1
            W = (l.weight.detach() / math.sqrt(l.h_in)).float()
            Wh, Wl = split(W)
            self.register_buffer(f"B{j}", torch.cat([Wh, Wl, Wh], 0).contiguous())        # [3K, N]
            self.register_buffer(f"WT2_{j}", torch.cat([Wh.T, Wl.T], 1).contiguous())      # [N, 2K]
            self.register_buffer(f"WhT_{j}", Wh.T.contiguous())                            # [N, K]
            self.acts.append(l.act)
        self.n = len(self.acts)

    def forward(self, x):
        for j in range(self.n):
            x = _MM3.apply(x, getattr(self, f"B{j}"), getattr(self, f"WT2_{j}"), getattr(self, f"WhT_{j}"))
            if self.acts[j] is not None:
                x = self.acts[j](x)
        return x


def apply(model):
    n = 0
    for blk in model.interactions:
        if hasattr(blk, "conv_tp_weights") and not isinstance(blk.conv_tp_weights, Radial3xTF32):
            blk.conv_tp_weights = Radial3xTF32(blk.conv_tp_weights)
            n += 1
    return n


def _selftest(model_path, E=1_500_000, repeat=10):
    import time

    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    torch.set_default_dtype(torch.float32)
    torch.backends.cuda.matmul.allow_tf32 = False
    m = torch.load(model_path, map_location="cuda").float()
    for bi, blk in enumerate(m.interactions):
        net = blk.conv_tp_weights
        for p in net.parameters():
            p.requires_grad_(False)
        f = Radial3xTF32(net).cuda()
        x = torch.rand(E, 10, device="cuda") * 2 - 0.5
        xr = x.clone().requires_grad_(True)
        yr = net(xr)
        g = torch.randn_like(yr)
        (gr,) = torch.autograd.grad(yr, xr, g)
        xf = x.clone().requires_grad_(True)
        yf = f(xf)
        (gf,) = torch.autograd.grad(yf, xf, g)
        # fp64 reference to judge both
        with torch.no_grad():
            xd = x.double().requires_grad_(True)
        netd = __import__("copy").deepcopy(net).double()
        with torch.enable_grad():
            yd = netd(xd)
            (gd,) = torch.autograd.grad(yd, xd, g.double())
        def rel(a, b):
            return float((a.double() - b).abs().max() / b.abs().max())
        print(f"block {bi}: D={yr.shape[1]} E={E:,}   vs fp64:  fp32 W {rel(yr, yd):.2e} dx {rel(gr, gd):.2e}   "
              f"3xTF32 W {rel(yf, yd):.2e} dx {rel(gf, gd):.2e}", flush=True)

        def timeit(fn):
            for _ in range(3):
                fn()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(repeat):
                fn()
            torch.cuda.synchronize()
            return (time.perf_counter() - t0) / repeat * 1e3

        def fb(mod):
            def _():
                xx = x.clone().requires_grad_(True)
                torch.autograd.grad(mod(xx), xx, g)
            return _

        def fo(mod):
            def _():
                with torch.no_grad():
                    mod(x)
            return _

        print(f"   fwd: fp32 {timeit(fo(net)):7.2f} ms   3xTF32 {timeit(fo(f)):7.2f} ms      "
              f"fwd+bwd: fp32 {timeit(fb(net)):7.2f} ms   3xTF32 {timeit(fb(f)):7.2f} ms", flush=True)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python -m mace_onegpu.radial_3xtf32 <mace model file, e.g. mace-mpa-0-medium.model>")
    _selftest(sys.argv[1])
