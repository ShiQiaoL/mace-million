#!/usr/bin/env python
"""
Fully fused MACE edge convolution: radial MLP + channel-wise tensor product + receiver reduction in ONE Triton
kernel (forward) and ONE kernel (backward). Neither the per-edge TP weights W [E, 128*npaths] nor the per-edge
messages m_ji [E, 128*nout] are ever written to memory.

Semantics are taken from the cuEquivariance descriptor that MACE itself uses for the fused convolution
(cue.descriptors.channelwise_tensor_product(in1, sh, mid).flatten_coefficient_modes().squeeze_modes()):
    out[i, o, c] = sum_{e: recv(e)=i} sum_{paths (p,k,m,o,coef)} coef * W[e,p,c] * X[send(e),k,c] * Y[e,m]
with segments of 128 channels (layout ir_mul), and W = radial MLP(edge_feats) (see fused_radial.py).
The kernels are generated per interaction block from its path table and written to mace_onegpu/_generated/.

Requirements: fp32 model, edges sorted by receiver (CSR). Rows of `out` for nodes that are not receivers are 0.
Set FUSED_CTX.n_recv to limit the receivers (three-pass chunks: receivers are the first nC local nodes;
padding edges must point to a node index >= n_recv).

    from mace_onegpu.fused_edge_conv import apply_fused_edge_conv, FUSED_CTX
    apply_fused_edge_conv(model)            # patches every interaction block in place
"""
import hashlib
import importlib.util
import math
import os
import sys

import torch
import triton
import triton.language as tl  # noqa: F401  (used by generated code)

HERE = os.path.dirname(os.path.abspath(__file__))
# generated Triton sources are written next to the package (must be writable; see README)
GEN_DIR = os.path.join(HERE, "_generated")


class _Ctx:
    n_recv = None     # None -> all nodes are receivers


FUSED_CTX = _Ctx()


# ----------------------------------------------------------------------------- path table from cuEquivariance
def path_table(block):
    import cuequivariance as cue
    from e3nn import o3

    from mace.modules.irreps_tools import tp_out_irreps_with_instructions
    from mace.tools.cg import O3_e3nn

    mid, _ = tp_out_irreps_with_instructions(o3.Irreps(str(block.edge_irreps)), o3.Irreps(str(block.edge_attrs_irreps)),
                                             o3.Irreps(str(block.target_irreps)))
    d = cue.descriptors.channelwise_tensor_product(cue.Irreps(O3_e3nn, str(block.edge_irreps)),
                                                   cue.Irreps(O3_e3nn, str(block.edge_attrs_irreps)),
                                                   cue.Irreps(O3_e3nn, str(mid)))
    p = d.flatten_coefficient_modes().squeeze_modes().polynomial
    assert len(p.operations) == 1
    opd, stp = p.operations[0]
    assert tuple(opd.buffers) == (0, 1, 2, 3), opd
    assert stp.subscripts == "u,u,,u" if isinstance(stp.subscripts, str) else str(stp.subscripts) == "u,u,,u"
    C = stp.operands[0].segments[0][0]
    terms = []
    for pp in stp.paths:
        coef = float(pp.coefficients) if not hasattr(pp.coefficients, "shape") or pp.coefficients.size == 1 else None
        assert coef is not None, "non-scalar coefficient"
        w, k, m, o = pp.indices
        terms.append((int(w), int(k), int(m), int(o), coef))
    nW, nX, nY, nO = (len(op.segments) for op in stp.operands)
    return dict(C=int(C), nW=nW, nX=nX, nY=nY, nO=nO, terms=terms, mid=str(mid))


# ----------------------------------------------------------------------------- code generation
def _gen_source(tab, H, NB, name):
    C, nW, nX, nY, nO, terms = tab["C"], tab["nW"], tab["nX"], tab["nY"], tab["nO"], tab["terms"]
    groups = {}
    for (w, k, m, o, c) in terms:
        groups.setdefault(w, {}).setdefault(k, []).append((m, o, c))
    L = []
    A = L.append
    A("import triton\nimport triton.language as tl\n")
    A("@triton.jit\ndef _silu(z):\n    return z * tl.sigmoid(z)\n")
    A("@triton.jit\ndef _dsilu(z):\n    s = tl.sigmoid(z)\n    return s * (1.0 + z * (1.0 - s))\n")
    # ---------------- forward
    A("@triton.jit")
    A(f"def {name}_fwd(EF, Y, X, SEND, ROWPTR, A0, A1, A2, A3, OUT, ZSTEP, PREC: tl.constexpr):")
    A(f"    i = tl.program_id(0)")
    A(f"    start = tl.load(ROWPTR + i)")
    A(f"    end = tl.load(ROWPTR + i + 1)")
    A(f"    oc = tl.arange(0, {C})")
    A(f"    ok = tl.arange(0, 16)")
    A(f"    oh = tl.arange(0, {H})")
    A(f"    onb = tl.arange(0, {NB})")
    for o in range(nO):
        A(f"    acc{o} = tl.zeros([{C}], dtype=tl.float32)")
    A(f"    p3 = A3 + oh[:, None] * {nW * C} + oc[None, :]  # advanced by ZSTEP(=0) each iteration: blocks LICM of W slices")
    A(f"    for e0 in range(start, end, {NB}):")
    A(f"        e = e0 + onb")
    A(f"        me = e < end")
    A(f"        ef = tl.load(EF + e[:, None] * 10 + ok[None, :], mask=me[:, None] & (ok[None, :] < 10), other=0.0)")
    A(f"        h = _silu(tl.dot(ef, tl.load(A0 + ok[:, None] * {H} + oh[None, :]), input_precision=PREC))")
    A(f"        h = _silu(tl.dot(h, tl.load(A1 + oh[:, None] * {H} + oh[None, :]), input_precision=PREC))")
    A(f"        h = _silu(tl.dot(h, tl.load(A2 + oh[:, None] * {H} + oh[None, :]), input_precision=PREC))")
    A(f"        s = tl.load(SEND + e, mask=me, other=0)")
    for m in range(nY):
        A(f"        y{m} = tl.load(Y + e * {nY} + {m}, mask=me, other=0.0)")
    for k in range(nX):
        A(f"        x{k} = tl.load(X + s[:, None] * {nX * C} + {k * C} + oc[None, :], mask=me[:, None], other=0.0)")
    for w in sorted(groups):
        A(f"        w = tl.dot(h, tl.load(p3 + {w * C}), input_precision=PREC)")
        for k in sorted(groups[w]):
            A(f"        pr = w * x{k}")
            for (m, o, c) in groups[w][k]:
                A(f"        acc{o} += tl.sum(pr * y{m}[:, None], axis=0) * {c!r}")
    A(f"        p3 += ZSTEP")
    for o in range(nO):
        A(f"    tl.store(OUT + i * {nO * C} + {o * C} + oc, acc{o})")
    A("")
    # ---------------- backward
    A("@triton.jit")
    A(f"def {name}_bwd(EF, Y, X, SEND, ROWPTR, A0, A1, A2, A3, G, GEF, GY, GX, ZSTEP, PREC: tl.constexpr):")
    A(f"    i = tl.program_id(0)")
    A(f"    start = tl.load(ROWPTR + i)")
    A(f"    end = tl.load(ROWPTR + i + 1)")
    A(f"    oc = tl.arange(0, {C})")
    A(f"    ok = tl.arange(0, 16)")
    A(f"    oh = tl.arange(0, {H})")
    A(f"    onb = tl.arange(0, {NB})")
    for o in range(nO):
        A(f"    g{o} = tl.load(G + i * {nO * C} + {o * C} + oc)[None, :]")
    A(f"    a0 = tl.load(A0 + ok[:, None] * {H} + oh[None, :])")
    A(f"    a1 = tl.load(A1 + oh[:, None] * {H} + oh[None, :])")
    A(f"    a2 = tl.load(A2 + oh[:, None] * {H} + oh[None, :])")
    A(f"    p3 = A3 + oh[:, None] * {nW * C} + oc[None, :]  # advanced by ZSTEP(=0) each iteration: blocks LICM of W slices")
    A(f"    for e0 in range(start, end, {NB}):")
    A(f"        e = e0 + onb")
    A(f"        me = e < end")
    A(f"        ef = tl.load(EF + e[:, None] * 10 + ok[None, :], mask=me[:, None] & (ok[None, :] < 10), other=0.0)")
    A(f"        z0 = tl.dot(ef, a0, input_precision=PREC)")
    A(f"        h0 = _silu(z0)")
    A(f"        z1 = tl.dot(h0, a1, input_precision=PREC)")
    A(f"        h1 = _silu(z1)")
    A(f"        z2 = tl.dot(h1, a2, input_precision=PREC)")
    A(f"        h = _silu(z2)")
    A(f"        s = tl.load(SEND + e, mask=me, other=0)")
    for m in range(nY):
        A(f"        y{m} = tl.load(Y + e * {nY} + {m}, mask=me, other=0.0)")
        A(f"        gy{m} = tl.zeros([{NB}], dtype=tl.float32)")
    for k in range(nX):
        A(f"        x{k} = tl.load(X + s[:, None] * {nX * C} + {k * C} + oc[None, :], mask=me[:, None], other=0.0)")
        A(f"        gx{k} = tl.zeros([{NB}, {C}], dtype=tl.float32)")
    A(f"        gh = tl.zeros([{NB}, {H}], dtype=tl.float32)")
    for w in sorted(groups):
        A(f"        a3 = tl.load(p3 + {w * C})")
        A(f"        w = tl.dot(h, a3, input_precision=PREC)")
        A(f"        gw = tl.zeros([{NB}, {C}], dtype=tl.float32)")
        for k in sorted(groups[w]):
            A(f"        wx = w * x{k}")
            for (m, o, c) in groups[w][k]:
                A(f"        t = g{o} * {c!r}")
                A(f"        ty = t * y{m}[:, None]")
                A(f"        gw += ty * x{k}")
                A(f"        gx{k} += ty * w")
                A(f"        gy{m} += tl.sum(t * wx, axis=1)")
        A(f"        gh += tl.dot(gw, tl.trans(a3), input_precision=PREC)")
    A(f"        gz = gh * _dsilu(z2)")
    A(f"        gz = tl.dot(gz, tl.trans(a2), input_precision=PREC) * _dsilu(z1)")
    A(f"        gz = tl.dot(gz, tl.trans(a1), input_precision=PREC) * _dsilu(z0)")
    A(f"        gef = tl.dot(gz, tl.trans(a0), input_precision=PREC)")
    A(f"        tl.store(GEF + e[:, None] * 10 + ok[None, :], gef, mask=me[:, None] & (ok[None, :] < 10))")
    for m in range(nY):
        A(f"        tl.store(GY + e * {nY} + {m}, gy{m}, mask=me)")
    for k in range(nX):
        A(f"        tl.atomic_add(GX + s[:, None] * {nX * C} + {k * C} + oc[None, :], gx{k}, mask=me[:, None])")
    A(f"        p3 += ZSTEP")
    return "\n".join(L) + "\n"


_KERNELS = {}


def _load_kernels(tab, H, NB):
    key = hashlib.sha1(repr((tab["terms"], tab["C"], tab["nW"], tab["nX"], tab["nY"], tab["nO"], H, NB)).encode()).hexdigest()[:12]
    if key in _KERNELS:
        return _KERNELS[key]
    name = f"conv_{key}"
    os.makedirs(GEN_DIR, exist_ok=True)
    path = os.path.join(GEN_DIR, f"{name}.py")
    src = _gen_source(tab, H, NB, name)
    if not os.path.exists(path) or open(path).read() != src:
        with open(path, "w") as fh:
            fh.write(src)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    k = (getattr(mod, f"{name}_fwd"), getattr(mod, f"{name}_bwd"))
    _KERNELS[key] = k
    return k


# ----------------------------------------------------------------------------- autograd + module
class _FusedConvFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ef, Y, X, send, rowptr, n_rows, mod):
        ef = ef.contiguous()
        Y = Y.contiguous()
        X = X.contiguous()
        n_recv = rowptr.numel() - 1
        out = torch.zeros(n_rows, mod.nO * mod.C, dtype=X.dtype, device=X.device)
        fwd, _ = mod.kernels
        fwd[(n_recv,)](ef, Y, X, send, rowptr, mod.A0, mod.A1, mod.A2, mod.A3, out, 0, PREC=mod.prec,
                       num_warps=mod.num_warps, num_stages=1)
        ctx.save_for_backward(ef, Y, X, send, rowptr)
        ctx.mod = mod
        return out

    @staticmethod
    def backward(ctx, g):
        ef, Y, X, send, rowptr = ctx.saved_tensors
        mod = ctx.mod
        n_recv = rowptr.numel() - 1
        g = g.contiguous()
        gef = torch.zeros_like(ef)
        gY = torch.zeros_like(Y)
        gX = torch.zeros_like(X)
        _, bwd = mod.kernels
        bwd[(n_recv,)](ef, Y, X, send, rowptr, mod.A0, mod.A1, mod.A2, mod.A3, g, gef, gY, gX, 0, PREC=mod.prec,
                       num_warps=mod.num_warps, num_stages=1)
        return gef, gY, gX, None, None, None, None


class _PassThrough(torch.nn.Module):
    def forward(self, x):
        return x


class FusedEdgeConvTP(torch.nn.Module):
    """Replaces (conv_tp_weights, conv_tp) of a fused MACE interaction block.
    Called by the block as conv_tp(node_feats, edge_attrs, tp_weights, edge_index) where tp_weights is now the raw
    edge_feats (conv_tp_weights is replaced by a pass-through)."""

    def __init__(self, block, precision="ieee", NB=16, num_warps=4):
        super().__init__()
        from .fused_radial import FusedRadialMLP

        fr = FusedRadialMLP(block.conv_tp_weights, precision=precision)
        self.register_buffer("A0", fr.A0)
        self.register_buffer("A1", fr.A1)
        self.register_buffer("A2", fr.A2)
        self.register_buffer("A3", fr.A3)
        tab = path_table(block)
        assert fr.hs[-1] == tab["nW"] * tab["C"], (fr.hs, tab["nW"], tab["C"])
        self.C, self.nO, self.nX, self.nY, self.nW = tab["C"], tab["nO"], tab["nX"], tab["nY"], tab["nW"]
        self.tab = tab
        self.prec, self.num_warps = precision, num_warps
        self.kernels = _load_kernels(tab, fr.hs[1], NB)
        # keep the original weight numel so MACE code that inspects it still works
        self.weight_numel = self.nW * self.C

    def forward(self, node_feats, edge_attrs, edge_feats, edge_index):
        send = edge_index[0].to(torch.int32) if edge_index.dtype != torch.int32 else edge_index[0]
        recv = edge_index[1]
        n_rows = node_feats.shape[0]
        n_recv = FUSED_CTX.n_recv if FUSED_CTX.n_recv is not None else n_rows
        rowptr = torch.searchsorted(recv, torch.arange(n_recv + 1, device=recv.device, dtype=recv.dtype)).to(torch.int32)
        return _FusedConvFn.apply(edge_feats, edge_attrs, node_feats, send.contiguous(), rowptr, n_rows, self)


def apply_fused_edge_conv(model, precision="ieee"):
    n = 0
    for blk in model.interactions:
        if not hasattr(blk, "conv_fusion"):
            raise RuntimeError("fused edge conv needs a cuEq conv_fusion model (variant *_fused)")
        if isinstance(blk.conv_tp, FusedEdgeConvTP):
            continue
        blk.conv_tp = FusedEdgeConvTP(blk, precision=precision)
        blk.conv_tp_weights = _PassThrough()
        n += 1
    return n


# ----------------------------------------------------------------------------- self-test vs the original model
def _selftest():
    import argparse
    import copy
    import time

    import numpy as np

    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    from . import receiver_chunk_mace as rc
    from .ase_gradient_probe import make_batch, read_structure

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--structure", required=True)
    ap.add_argument("--precisions", nargs="+", default=["ieee", "tf32x3", "tf32"])
    ap.add_argument("--repeat", type=int, default=5)
    a = ap.parse_args()
    dev = "cuda"
    ref, dt = rc.build_model(a.model, "fp32_fused", dev)
    atoms = read_structure(a.structure)
    batch = make_batch(ref, atoms, dev)
    # sort edges by receiver (required by the fused kernel; does not change the model result)
    order = torch.argsort(batch.edge_index[1], stable=True)
    batch.edge_index = batch.edge_index[:, order]
    batch.shifts = batch.shifts[order]
    if hasattr(batch, "unit_shifts") and batch.unit_shifts is not None:
        batch.unit_shifts = batch.unit_shifts[order]

    def run(m):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        o = m(batch.to_dict(), training=False, compute_force=True)
        torch.cuda.synchronize()
        return float(o["energy"].detach().double().sum()), o["forces"].detach().double(), time.perf_counter() - t0

    run(ref)
    ts = []
    for _ in range(a.repeat):
        E0, F0, t = run(ref)
        ts.append(t)
    t0 = sorted(ts)[len(ts) // 2]
    torch.cuda.reset_peak_memory_stats()
    run(ref)
    pk0 = torch.cuda.max_memory_allocated() / 1e9
    print(f"N={len(atoms)} E={int(batch.edge_index.shape[1])}  original fp32_fused: {t0*1e3:.1f} ms  peak {pk0:.2f} GB  E={E0:.6f}", flush=True)
    for prec in a.precisions:
        m = copy.deepcopy(ref)
        apply_fused_edge_conv(m, precision=prec)
        run(m)
        ts = []
        for _ in range(a.repeat):
            E1, F1, t = run(m)
            ts.append(t)
        t1 = sorted(ts)[len(ts) // 2]
        torch.cuda.reset_peak_memory_stats()
        run(m)
        pk1 = torch.cuda.max_memory_allocated() / 1e9
        d = F1 - F0
        print(f"fused edge conv [{prec:6s}]: {t1*1e3:.1f} ms ({t0/t1:.2f}x)  peak {pk1:.2f} GB   dE={E1-E0:+.3e} eV  "
              f"max|dF|={float(d.abs().max()):.3e}  RMS={float(d.pow(2).mean().sqrt()):.3e} eV/A", flush=True)
        del m
        torch.cuda.empty_cache()


if __name__ == "__main__":
    _selftest()
