#!/usr/bin/env python
"""
Fused MACE edge convolution with TABULATED radial weights (element-general).

For MACE 0.3.15 density blocks the TP weights of an edge depend only on its length r and on the ORDERED species
pair (sender, receiver):  W(r; pair) = radial_MLP(Bessel(Agnesi(r; pair)) * f_cut(r))   [nW * C per edge].
We tabulate W and dW/dr per ordered pair on a uniform r-grid (sampled in fp64 from the model itself) and evaluate a
cubic Hermite spline inside the kernel, so neither the MLP nor its 64 x (nW*C) last-layer GEMM runs per edge:

    out[i, o, c] = sum_{e: recv(e)=i} sum_{terms (w,k,m,o,coef)} coef * W(r_e; p_e)[w,c] * X[send(e),k,c] * Y[e,m]

Tables only exist for the species pairs present in the system (any of the 89 elements).  Accuracy (fp64):
512 knots -> max|dF| 7.5e-7 eV/A, 1024 knots -> 1.8e-7 eV/A vs the original MLP (fp32 roundoff is ~2e-5).
Grid = (receivers, C // CB); per-edge-row accumulators reduced once at the end.  Forward only + benchmark here.
"""
import hashlib
import importlib.util
import os
import sys
import time

import torch
import triton  # noqa: F401
import triton.language as tl  # noqa: F401

HERE = os.path.dirname(os.path.abspath(__file__))
# generated Triton sources are written next to the package (must be writable; see README)
GEN_DIR = os.path.join(HERE, "_generated")


def _groups(tab):
    g = {}
    for (w, k, m, o, c) in tab["terms"]:
        g.setdefault(w, {}).setdefault(k, []).append((m, o, c))
    return g


def gen_fwd(tab, NB, CB, name):
    C, nW, nX, nY, nO = tab["C"], tab["nW"], tab["nX"], tab["nY"], tab["nO"]
    D = nW * C
    groups = _groups(tab)
    used_m = sorted({m for (w, k, m, o, c) in tab["terms"]})
    used_o = sorted({o for (w, k, m, o, c) in tab["terms"]})
    L = []
    A = L.append
    A("import triton\nimport triton.language as tl\n")
    A("@triton.jit")
    A(f"def {name}(R, PAIR, Y, X, SEND, ROWPTR, TV, TG, OUT, r_lo, inv_h, M):")
    A("    i = tl.program_id(0)")
    A("    cb = tl.program_id(1)")
    A("    start = tl.load(ROWPTR + i)")
    A("    end = tl.load(ROWPTR + i + 1)")
    A(f"    oc = cb * {CB} + tl.arange(0, {CB})")
    A(f"    onb = tl.arange(0, {NB})")
    for o in used_o:
        A(f"    acc{o} = tl.zeros([{NB}, {CB}], dtype=tl.float32)")
    A(f"    for e0 in range(start, end, {NB}):")
    A("        e = e0 + onb")
    A("        me = e < end")
    A("        r = tl.load(R + e, mask=me, other=0.0)")
    A("        p = tl.load(PAIR + e, mask=me, other=0)")
    A("        x = (r - r_lo) * inv_h")
    A("        x = tl.minimum(tl.maximum(x, 0.0), (M - 1).to(tl.float32) - 1e-4)")
    A("        kf = tl.floor(x)")
    A("        t = x - kf")
    A("        t2 = t * t")
    A("        t3 = t2 * t")
    A("        b00 = (2.0 * t3 - 3.0 * t2 + 1.0)[:, None]")
    A("        b10 = (t3 - 2.0 * t2 + t)[:, None]")
    A("        b01 = (-2.0 * t3 + 3.0 * t2)[:, None]")
    A("        b11 = (t3 - t2)[:, None]")
    A("        row = (p * M + kf.to(tl.int32)).to(tl.int64)")
    A(f"        base0 = row[:, None] * {D} + oc[None, :]")
    A(f"        base1 = base0 + {D}")
    A("        s = tl.load(SEND + e, mask=me, other=0)")
    for m in used_m:
        A(f"        y{m} = tl.load(Y + e * {nY} + {m}, mask=me, other=0.0)[:, None]")
    for k in range(nX):
        A(f"        x{k} = tl.load(X + s[:, None] * {nX * C} + {k * C} + oc[None, :], mask=me[:, None], other=0.0)")
    A("        mm = me[:, None]")
    for w in sorted(groups):
        off = w * C
        A(f"        ww = (b00 * tl.load(TV + base0 + {off}, mask=mm, other=0.0) + b10 * tl.load(TG + base0 + {off}, mask=mm, other=0.0)"
          f" + b01 * tl.load(TV + base1 + {off}, mask=mm, other=0.0) + b11 * tl.load(TG + base1 + {off}, mask=mm, other=0.0))")
        for k in sorted(groups[w]):
            A(f"        pr = ww * x{k}")
            for (m, o, c) in groups[w][k]:
                A(f"        acc{o} += pr * (y{m} * {c!r})")
    for o in range(nO):
        if o in used_o:
            A(f"    tl.store(OUT + i * {nO * C} + {o * C} + oc, tl.sum(acc{o}, 0))")
        else:
            A(f"    tl.store(OUT + i * {nO * C} + {o * C} + oc, tl.zeros([{CB}], dtype=tl.float32))")
    return "\n".join(L) + "\n"


def gen_bwd(tab, NB, CB, name, need_gx, gy_mode="tile"):
    """Backward of gen_fwd: given G = dL/dout [n_recv, nO*C] produce dL/dr [E], dL/dY [E, nY] (atomic over channel
    blocks) and, if need_gx, dL/dX [n_nodes, nX*C] (atomic over edges).  W and dW/dr come from the same table."""
    C, nW, nX, nY, nO = tab["C"], tab["nW"], tab["nX"], tab["nY"], tab["nO"]
    D = nW * C
    groups = _groups(tab)
    used_m = sorted({m for (w, k, m, o, c) in tab["terms"]})
    used_o = sorted({o for (w, k, m, o, c) in tab["terms"]})
    L = []
    A = L.append
    A("import triton\nimport triton.language as tl\n")
    A("@triton.jit")
    A(f"def {name}(R, PAIR, Y, X, SEND, ROWPTR, TV, TG, G, GR, GY, GX, r_lo, inv_h, M):")
    A("    i = tl.program_id(0)")
    A("    cb = tl.program_id(1)")
    A("    start = tl.load(ROWPTR + i)")
    A("    end = tl.load(ROWPTR + i + 1)")
    A(f"    oc = cb * {CB} + tl.arange(0, {CB})")
    A(f"    onb = tl.arange(0, {NB})")
    for o in used_o:
        A(f"    g{o} = tl.load(G + i * {nO * C} + {o * C} + oc)[None, :]")
    A(f"    for e0 in range(start, end, {NB}):")
    A("        e = e0 + onb")
    A("        me = e < end")
    A("        r = tl.load(R + e, mask=me, other=0.0)")
    A("        p = tl.load(PAIR + e, mask=me, other=0)")
    A("        x = (r - r_lo) * inv_h")
    A("        x = tl.minimum(tl.maximum(x, 0.0), (M - 1).to(tl.float32) - 1e-4)")
    A("        kf = tl.floor(x)")
    A("        t = x - kf")
    A("        t2 = t * t")
    A("        t3 = t2 * t")
    A("        b00 = (2.0 * t3 - 3.0 * t2 + 1.0)[:, None]")
    A("        b10 = (t3 - 2.0 * t2 + t)[:, None]")
    A("        b01 = (-2.0 * t3 + 3.0 * t2)[:, None]")
    A("        b11 = (t3 - t2)[:, None]")
    A("        d00 = (6.0 * t2 - 6.0 * t)[:, None]")
    A("        d10 = (3.0 * t2 - 4.0 * t + 1.0)[:, None]")
    A("        d01 = (-6.0 * t2 + 6.0 * t)[:, None]")
    A("        d11 = (3.0 * t2 - 2.0 * t)[:, None]")
    A("        row = (p * M + kf.to(tl.int32)).to(tl.int64)")
    A(f"        base0 = row[:, None] * {D} + oc[None, :]")
    A(f"        base1 = base0 + {D}")
    A("        s = tl.load(SEND + e, mask=me, other=0)")
    A("        mm = me[:, None]")
    for m in used_m:
        A(f"        y{m} = tl.load(Y + e * {nY} + {m}, mask=me, other=0.0)[:, None]")
        if gy_mode == "tile":
            A(f"        gy{m} = tl.zeros([{NB}, {CB}], dtype=tl.float32)")
        else:
            A(f"        gy{m} = tl.zeros([{NB}], dtype=tl.float32)")
    for k in range(nX):
        A(f"        x{k} = tl.load(X + s[:, None] * {nX * C} + {k * C} + oc[None, :], mask=mm, other=0.0)")
        if need_gx:
            A(f"        gx{k} = tl.zeros([{NB}, {CB}], dtype=tl.float32)")
    A(f"        gr = tl.zeros([{NB}, {CB}], dtype=tl.float32)")
    for w in sorted(groups):
        off = w * C
        A(f"        v0 = tl.load(TV + base0 + {off}, mask=mm, other=0.0)")
        A(f"        q0 = tl.load(TG + base0 + {off}, mask=mm, other=0.0)")
        A(f"        v1 = tl.load(TV + base1 + {off}, mask=mm, other=0.0)")
        A(f"        q1 = tl.load(TG + base1 + {off}, mask=mm, other=0.0)")
        A("        ww = b00 * v0 + b10 * q0 + b01 * v1 + b11 * q1")
        A("        dw = d00 * v0 + d10 * q0 + d01 * v1 + d11 * q1")
        A(f"        gw = tl.zeros([{NB}, {CB}], dtype=tl.float32)")
        for k in sorted(groups[w]):
            A(f"        wx = ww * x{k}")
            for (m, o, c) in groups[w][k]:
                A(f"        tc = g{o} * {c!r}")
                A(f"        ty = tc * y{m}")
                A(f"        gw += ty * x{k}")
                if need_gx:
                    A(f"        gx{k} += ty * ww")
                if gy_mode == "tile":
                    A(f"        gy{m} += tc * wx")
                else:
                    A(f"        gy{m} += tl.sum(tc * wx, 1)")
        A("        gr += gw * dw")
    A("        tl.atomic_add(GR + e, tl.sum(gr, 1) * inv_h, mask=me)")
    for m in used_m:
        A(f"        tl.atomic_add(GY + e * {nY} + {m}, {'tl.sum(gy%d, 1)' % m if gy_mode == 'tile' else 'gy%d' % m}, mask=me)")
    if need_gx:
        for k in range(nX):
            A(f"        tl.atomic_add(GX + s[:, None] * {nX * C} + {k * C} + oc[None, :], gx{k}, mask=mm)")
    return "\n".join(L) + "\n"


_K = {}


def _load(src_fn, key_extra, prefix, *args):
    tab = args[0]
    key = hashlib.sha1(repr((tab["terms"], tab["C"], tab["nW"], tab["nX"], tab["nY"], tab["nO"], args[1:], key_extra)).encode()).hexdigest()[:12]
    if key in _K:
        return _K[key]
    name = f"{prefix}_{key}"
    os.makedirs(GEN_DIR, exist_ok=True)
    path = os.path.join(GEN_DIR, f"{name}.py")
    src = src_fn(*args[:3], name, *args[3:])
    if not os.path.exists(path) or open(path).read() != src:
        with open(path, "w") as fh:
            fh.write(src)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    _K[key] = getattr(mod, name)
    return _K[key]


def load_bwd(tab, NB, CB, need_gx, gy_mode="tile"):
    return _load(gen_bwd, "splb", "cspb", tab, NB, CB, need_gx, gy_mode)


def load_fwd(tab, NB, CB):
    key = hashlib.sha1(repr((tab["terms"], tab["C"], tab["nW"], tab["nX"], tab["nY"], tab["nO"], NB, CB, "splf")).encode()).hexdigest()[:12]
    if key in _K:
        return _K[key]
    name = f"cspf_{key}"
    os.makedirs(GEN_DIR, exist_ok=True)
    path = os.path.join(GEN_DIR, f"{name}.py")
    src = gen_fwd(tab, NB, CB, name)
    if not os.path.exists(path) or open(path).read() != src:
        with open(path, "w") as fh:
            fh.write(src)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    _K[key] = getattr(mod, name)
    return _K[key]


def build_tables(model, species_z, n_grid=512, r_lo=0.5, device="cuda"):
    """Per interaction block: TV, TG [P*M, D] fp32 (TG = dW/dr * grid spacing), ordered pairs (sender, receiver).
    Sampled in fp64 from a float64 copy of the model's radial parts (same code path as spline_radial.py)."""
    import copy

    from .spline_radial import _sample

    m64 = copy.deepcopy(model)
    fns = [blk.conv_tp_weights for blk in m64.interactions]
    for fn in fns + [m64.radial_embedding]:
        fn.double()
    r_hi = float(model.r_max)
    tabs = _sample(m64, fns, species_z, n_grid, r_lo, r_hi, device, torch.float64)
    h = (r_hi - r_lo) / (n_grid - 1)
    out = []
    for (vals, ders) in tabs:                       # [P, M, D] each, fp64
        P, M, D = vals.shape
        out.append((vals.reshape(P * M, D).float().contiguous(), (ders * h).reshape(P * M, D).float().contiguous()))
    del m64
    return dict(tables=out, r_lo=r_lo, r_hi=r_hi, M=n_grid, inv_h=1.0 / h, species_z=list(species_z))


class _SplineConvFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, r, Y, X, mod, pair, send, rowptr, need_gx):
        out = mod.forward(r, pair, Y, X, send, rowptr)
        ctx.save_for_backward(r, Y, X, pair, send, rowptr)
        ctx.mod, ctx.need_gx = mod, need_gx
        return out

    @staticmethod
    def backward(ctx, g):
        r, Y, X, pair, send, rowptr = ctx.saved_tensors
        mod = ctx.mod
        GR = torch.zeros_like(r)
        GY = torch.zeros_like(Y)
        GX = torch.zeros_like(X) if ctx.need_gx else torch.empty(1, device=X.device)
        kern = mod.bwd_gx if ctx.need_gx else mod.bwd_nogx
        n_recv = rowptr.numel() - 1
        grid = (n_recv, mod.tab["C"] // mod.CB_bwd)
        kern[grid](r, pair, Y, X, send, rowptr, mod.TV, mod.TG, g.contiguous(), GR, GY, GX,
                   float(mod.meta["r_lo"]), float(mod.meta["inv_h"]), int(mod.meta["M"]), num_warps=mod.num_warps_bwd)
        return GR, GY, (GX if ctx.need_gx else None), None, None, None, None, None


class ConvSpline:
    def __init__(self, block, TV, TG, meta, NB=16, CB=32, num_warps=4, NB_bwd=16, CB_bwd=16, num_warps_bwd=4):
        from .fused_edge_conv import path_table

        self.tab = path_table(block)
        assert TV.shape[1] == self.tab["nW"] * self.tab["C"]
        self.TV, self.TG, self.meta = TV, TG, meta
        self.NB, self.CB, self.num_warps = NB, CB, num_warps
        self.CB_bwd, self.num_warps_bwd = CB_bwd, num_warps_bwd
        self.kern = load_fwd(self.tab, NB, CB)
        self.bwd_gx = load_bwd(self.tab, NB_bwd, CB_bwd, True)
        self.bwd_nogx = load_bwd(self.tab, NB_bwd, CB_bwd, False)

    @torch._dynamo.disable   # our Triton kernels: graph break for torch.compile, run eagerly
    def __call__(self, r, pair, Y, X, send, rowptr, need_gx=True):
        """Autograd-aware: r [E] fp32, Y [E, nY], X [n_nodes, nX*C]; returns messages [n_recv, nO*C]."""
        return _SplineConvFn.apply(r.contiguous(), Y.contiguous(), X.contiguous(), self, pair, send, rowptr, need_gx)

    @torch._dynamo.disable   # our Triton kernels: graph break for torch.compile, run eagerly
    def forward(self, r, pair, Y, X, send, rowptr):
        n_recv = rowptr.numel() - 1
        t = self.tab
        out = torch.empty(n_recv, t["nO"] * t["C"], dtype=torch.float32, device=r.device)
        grid = (n_recv, t["C"] // self.CB)
        self.kern[grid](r, pair, Y, X, send, rowptr, self.TV, self.TG, out,
                        float(self.meta["r_lo"]), float(self.meta["inv_h"]), int(self.meta["M"]), num_warps=self.num_warps)
        return out


def _bench():
    import argparse

    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    from . import receiver_chunk_mace as rc

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--rep", nargs=3, type=int, default=[8, 8, 8])
    ap.add_argument("--chunk", type=int, default=16384)
    ap.add_argument("--grids", nargs="+", type=int, default=[512, 1024])
    ap.add_argument("--configs", nargs="+", default=["16,32,4", "16,64,4"])
    ap.add_argument("--bwd-configs", nargs="+", default=["16,16,4", "16,32,4", "16,16,8", "16,32,8", "32,16,8"])
    a = ap.parse_args()
    dev = "cuda"
    model, dt = rc.build_model(a.model, "fp32_fused", dev)

    class A_:
        structure = None
        base = a.base
        rep = a.rep

    pos, box, sp, _ = rc.load_structure(A_, model, dt, dev)
    sysm = rc.System(pos, box, sp, float(model.r_max), dev)
    sysm.build_neighbors()
    zs = [int(z) for z in model.atomic_numbers]
    cols = sorted(int(c) for c in torch.unique(sysm.spidx.long()).tolist())
    species_z = [zs[c] for c in cols]
    col2s = torch.full((len(zs),), -1, dtype=torch.long, device=dev)
    col2s[torch.tensor(cols, device=dev)] = torch.arange(len(cols), device=dev)
    loc = torch.full((sysm.N,), -1, dtype=torch.int32, device=dev)
    g = rc.LocalGraph(sysm, 0, min(a.chunk, sysm.N), loc)
    na = rc.node_attrs_of(model, sysm.spidx[g.nodes], dt)
    v = g.vectors.to(dt)
    recv = g.edge_index[1]
    assert bool((recv[1:] >= recv[:-1]).all())
    rowptr = torch.searchsorted(recv, torch.arange(g.nC + 1, device=dev, dtype=recv.dtype)).to(torch.int32)
    send = g.edge_index[0].to(torch.int32).contiguous()
    snode = col2s[na.argmax(1)]
    pair = (snode[g.edge_index[0]] * len(cols) + snode[g.edge_index[1]]).to(torch.int32).contiguous()
    print(f"chunk: {g.nC} receivers, {g.n_nodes} nodes, {int(g.nE):,} edges; species Z={species_z}", flush=True)
    with torch.no_grad():
        lengths, Y, ef, cut = rc.edge_embed(model, g, na, v)
        assert cut is None
        r = lengths.reshape(-1).float().contiguous()
        Y = Y.contiguous()
        h1 = rc.layer1(model, g, na, v)

    def tm(fn, n=10):
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / n * 1e3

    for M in a.grids:
        t0 = time.perf_counter()
        meta = build_tables(model, species_z, n_grid=M, device=dev)
        mb = sum(tv.numel() + tg.numel() for tv, tg in meta["tables"]) * 4 / 2**20
        print(f"=== grid M={M}: tables {mb:.1f} MiB total (build {time.perf_counter()-t0:.1f}s)", flush=True)
        for bi, blk in enumerate(model.interactions):
            with torch.no_grad():
                if bi == 0:
                    X = blk.linear_up(model.node_embedding(na))
                else:
                    H1 = torch.randn(g.n_nodes, h1.shape[1], device=dev) * h1.std()
                    H1[: g.nC] = h1
                    X = blk.linear_up(H1)
                X = X.contiguous()

            def ref():
                with torch.no_grad():
                    W = blk.conv_tp_weights(ef)
                    return blk.conv_tp(X, Y, W, g.edge_index)[: g.nC]

            rr = ref()
            t_ref = tm(ref)
            TV, TG = meta["tables"][bi]
            print(f"--- block {bi}: reference (radial MLP + cuEq conv_tp) forward {t_ref:.2f} ms; "
                  f"table {TV.numel()*8/2**20:.1f} MiB", flush=True)
            for cfg in a.configs:
                NB, CB, nw = (int(x) for x in cfg.split(","))
                try:
                    cs = ConvSpline(blk, TV, TG, meta, NB=NB, CB=CB, num_warps=nw)
                    out = cs.forward(r, pair, Y, X, send, rowptr)
                    err = float((out - rr).abs().max() / rr.abs().max())
                    t_k = tm(lambda: cs.forward(r, pair, Y, X, send, rowptr))
                    print(f"   fwd NB={NB:2d} CB={CB:2d} warps={nw}: kernel {t_k:7.2f} ms  ({t_ref/t_k:5.2f}x)   "
                          f"rel.max err vs MLP {err:.2e}", flush=True)
                except Exception as exc:
                    print(f"   fwd NB={NB} CB={CB} warps={nw}: FAILED {type(exc).__name__}: {str(exc)[:200]}", flush=True)
            # ---- backward kernel alone: which part is expensive?
            Gk = torch.randn(g.nC, rr.shape[1], device=dev)
            for gy_mode in ("tile", "row"):
                for ngx in ((False, True) if bi == 1 else (False,)):
                    for cfg in a.bwd_configs:
                        NBb, CBb, nwb = (int(x) for x in cfg.split(","))
                        try:
                            kb = load_bwd(cs.tab, NBb, CBb, ngx, gy_mode)
                            GR = torch.zeros_like(r)
                            GYt = torch.zeros_like(Y)
                            GX = torch.zeros_like(X)

                            def run_b():
                                kb[(g.nC, cs.tab["C"] // CBb)](r, pair, Y, X, send, rowptr, TV, TG, Gk, GR, GYt, GX,
                                                              float(meta["r_lo"]), float(meta["inv_h"]), int(meta["M"]),
                                                              num_warps=nwb)
                            t_b = tm(run_b)
                            print(f"   bwd kernel only gy={gy_mode:4s} gx={str(ngx):5s} NB={NBb:2d} CB={CBb:2d} warps={nwb}: "
                                  f"{t_b:7.2f} ms", flush=True)
                        except Exception as exc:
                            print(f"   bwd kernel gy={gy_mode} gx={ngx} {cfg}: FAILED {type(exc).__name__}: {str(exc)[:160]}", flush=True)
            # ---- forward + backward through the whole edge pipeline (vectors -> r, Y -> messages), vs autograd reference
            need_gx = bi == 1
            Gr = torch.randn_like(rr)

            def ref_fb():
                vv = v.detach().requires_grad_(True)
                XX = X.detach().requires_grad_(need_gx)
                _, Yr, efr, _ = rc.edge_embed(model, g, na, vv)
                W = blk.conv_tp_weights(efr)
                o = blk.conv_tp(XX, Yr, W, g.edge_index)[: g.nC]
                gs = torch.autograd.grad((o * Gr).sum(), [vv, XX] if need_gx else [vv])
                return gs

            g_ref = ref_fb()
            t_rfb = tm(ref_fb)
            print(f"   reference fwd+bwd (edge embed + MLP + cuEq TP): {t_rfb:.2f} ms", flush=True)
            for cfg in a.bwd_configs:
                NBb, CBb, nwb = (int(x) for x in cfg.split(","))
                try:
                    cs = ConvSpline(blk, TV, TG, meta, NB=16, CB=32, num_warps=4, NB_bwd=NBb, CB_bwd=CBb, num_warps_bwd=nwb)

                    def new_fb():
                        vv = v.detach().requires_grad_(True)
                        XX = X.detach().requires_grad_(need_gx)
                        ln, Yn, _, _ = rc.edge_embed(model, g, na, vv)
                        o = cs(ln.reshape(-1), pair, Yn, XX, send, rowptr, need_gx=need_gx)
                        return torch.autograd.grad((o * Gr).sum(), [vv, XX] if need_gx else [vv])

                    g_new = new_fb()
                    ev = float((g_new[0] - g_ref[0]).abs().max() / g_ref[0].abs().max())
                    ex = float((g_new[1] - g_ref[1]).abs().max() / g_ref[1].abs().max()) if need_gx else float("nan")
                    t_nfb = tm(new_fb)
                    print(f"   fwd+bwd NB_bwd={NBb:2d} CB_bwd={CBb:2d} warps={nwb}: {t_nfb:7.2f} ms ({t_rfb/t_nfb:5.2f}x)   "
                          f"rel.max err d/dvec {ev:.2e}  d/dX {ex:.2e}", flush=True)
                except Exception as exc:
                    print(f"   fwd+bwd NB_bwd={NBb} CB_bwd={CBb} warps={nwb}: FAILED {type(exc).__name__}: {str(exc)[:200]}", flush=True)


if __name__ == "__main__":
    _bench()
