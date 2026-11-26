#!/usr/bin/env python
"""
Exact node-level speed-ups for the three-pass MACE layers (receiver_chunk_mace.layer1 / layer2_energy):

1. skip_tp = FullyConnectedTensorProduct(x, one_hot(89 elements)) is evaluated by cuEq "naive" as a contraction over
   all 89 elements (88 of them multiply zeros).  For each element s present, the map x -> skip_tp(x, e_s) is linear;
   we extract it numerically (identity probe), split it into its dense sub-blocks (G groups of 128x128, found from the
   sparsity pattern, and verified to reproduce the full map exactly) and apply it as a batched GEMM + species select.
2. linear / skip / density are evaluated only for the nC receivers of the chunk (block.forward does all nodes).

Everything else calls the original MACE modules unchanged.  Differences vs the original are fp32 re-association only.

    from mace_onegpu import fast_layers; fast_layers.install(model, species_cols)    # replaces rc.layer1 and rc.layer2_energy
"""
import os

import torch


class _SelectCols(torch.autograd.Function):
    """y = x[:, p] for UNIQUE column indices p (a permutation or a subset).  Backward is a plain index_copy into zeros
    instead of PyTorch's slow sort-based advanced-indexing backward."""

    @staticmethod
    def forward(ctx, x, p):
        ctx.save_for_backward(p)
        ctx.d = x.shape[1]
        return x.index_select(1, p)

    @staticmethod
    def backward(ctx, g):
        (p,) = ctx.saved_tensors
        gx = torch.zeros(g.shape[0], ctx.d, dtype=g.dtype, device=g.device)
        return gx.index_copy_(1, p, g), None


def select_cols(x, p, is_identity):
    return x if is_identity else _SelectCols.apply(x, p)


class SpeciesSkip(torch.nn.Module):
    """Exact replacement of skip_tp(x, node_attrs) for one-hot node_attrs restricted to `cols`."""

    def __init__(self, skip_tp, cols, n_elem, d_in, dtype, device):
        super().__init__()
        S = len(cols)
        with torch.no_grad():
            eye = torch.eye(d_in, dtype=dtype, device=device)
            Ms = []
            for c in cols:
                oh = torch.zeros(d_in, n_elem, dtype=dtype, device=device)
                oh[:, c] = 1.0
                Ms.append(skip_tp(eye, oh))                                     # [d_in, d_out]
            M = torch.stack(Ms)                                                   # [S, d_in, d_out]
        d_out = M.shape[2]
        nz = (M != 0).any(0)                                                      # [d_in, d_out] union pattern
        groups = {}
        for j in range(d_out):
            rows = tuple(torch.nonzero(nz[:, j]).flatten().tolist())
            groups.setdefault(rows, []).append(j)
        gi = [list(r) for r in groups]
        go = [groups[r] for r in groups]
        K = {len(r) for r in gi}
        Wd = {len(c) for c in go}
        assert len(K) == 1 and len(Wd) == 1, f"irregular block structure {K} {Wd}"
        k, w = K.pop(), Wd.pop()
        G = len(gi)
        rin = torch.tensor(gi, device=device)                                     # [G, k]
        cout = torch.tensor(go, device=device)                                    # [G, w]
        Wb = torch.stack([M[:, rin[g]][:, :, cout[g]] for g in range(G)], 1)      # [S, G, k, w]
        # exactness of the decomposition: rebuild the full maps
        R = torch.zeros_like(M)
        for g in range(G):
            R[:, rin[g][:, None], cout[g][None, :]] = Wb[:, g]
        assert torch.equal(R, M), "block decomposition does not reproduce skip_tp"
        covered = torch.zeros(d_out, dtype=torch.bool, device=device)
        covered[cout.flatten()] = True
        rf = rin.flatten()
        assert torch.unique(rf).numel() == rf.numel(), "skip_tp input groups overlap"
        self.register_buffer("rin", rf)
        self.rin_id = bool(rf.numel() == d_in and torch.equal(rf, torch.arange(d_in, device=device)))
        self.register_buffer("Wb", Wb.contiguous())
        inv = torch.empty(G * w, dtype=torch.long, device=device)
        inv[cout.flatten()] = torch.arange(G * w, device=device)
        self.register_buffer("inv", inv)
        self.inv_id = bool(torch.equal(inv, torch.arange(G * w, device=device)))
        self.all_covered = bool(covered.all())
        assert self.all_covered, "some skip_tp outputs are identically zero; not handled"
        col2s = torch.zeros(n_elem, dtype=torch.long, device=device)
        col2s[torch.as_tensor(cols, device=device)] = torch.arange(S, device=device)
        self.register_buffer("col2s", col2s)
        self.S, self.G, self.k, self.w, self.d_out = S, G, k, w, d_out
        self.info = (f"S={S} blocks G={G} of {k}x{w} (d_in={d_in}, d_out={d_out}; dense would be {d_in}x{d_out}x{n_elem})"
                     f" in-perm identity={self.rin_id} out-perm identity={self.inv_id}")

    def forward(self, x, node_attrs):
        n = x.shape[0]
        xg = select_cols(x, self.rin, self.rin_id).view(n, self.G, self.k).transpose(0, 1)   # [G, n, k]
        s = self.col2s[node_attrs.argmax(1)]                                      # [n]
        out = None
        for si in range(self.S):
            y = torch.bmm(xg, self.Wb[si])                                        # [G, n, w]
            out = y if out is None else torch.where((s == si).view(1, n, 1), y, out)
        out = out.transpose(0, 1).reshape(n, self.G * self.w)
        return select_cols(out, self.inv, self.inv_id)


class FastLayers:
    def __init__(self, model, species_cols, spline_grid=0):
        from . import receiver_chunk_mace as rc

        self.rc = rc
        self.m = model
        dev = next(model.parameters()).device
        dt = next(model.parameters()).dtype
        n_elem = len(model.atomic_numbers)
        b0, b1 = model.interactions
        assert type(b0).__name__ == "RealAgnosticDensityInteractionBlock", type(b0).__name__
        assert type(b1).__name__ == "RealAgnosticDensityResidualInteractionBlock", type(b1).__name__
        from e3nn import o3

        # input widths of skip_tp: layer-1 = output of linear (target irreps), layer-2 = node_feats (hidden irreps)
        d_in0 = o3.Irreps(str(b0.irreps_out)).dim
        d_in1 = o3.Irreps(str(b1.node_feats_irreps)).dim
        self.d_in1 = d_in1
        self.skip0 = SpeciesSkip(b0.skip_tp, species_cols, n_elem, d_in0, dt, dev)
        self.skip1 = SpeciesSkip(b1.skip_tp, species_cols, n_elem, d_in1, dt, dev)
        # optional: tabulated radial TP weights (512 knots: <= 1.6e-5 eV/A in forces, ~fp32 roundoff; see spline_w.py)
        self.sw = None
        if spline_grid:
            from . import spline_w

            zs = [int(z) for z in model.atomic_numbers]
            self.sw, self.sw_meta = spline_w.make(model, [zs[c] for c in species_cols], n_grid=spline_grid, device=dev)
            col2s = torch.full((n_elem,), 0, dtype=torch.long, device=dev)
            col2s[torch.as_tensor(species_cols, device=dev)] = torch.arange(len(species_cols), device=dev)
            self.col2s = col2s
            self.S = len(species_cols)
            # fused tabulated-radial convolution for FORWARD-ONLY layer-1 calls (pass A): conv_spline.py
            from .conv_spline import ConvSpline

            TV0, TG0 = self.sw_meta["tables"][0]
            from .fused_edge_conv import path_table

            C0 = path_table(b0)["C"]
            cb = next(b for b in (32, 16, 8) if C0 % b == 0)
            self.cs0 = ConvSpline(b0, TV0, TG0, self.sw_meta, NB=16, CB=cb, num_warps=4,
                                  CB_bwd=min(16, cb))

    def _tp_weights(self, bi, blk, lengths, edge_feats, cutoff, node_attrs, edge_index):
        if self.sw is None:
            tpw = blk.conv_tp_weights(edge_feats)
            return tpw * cutoff if cutoff is not None else tpw
        s = self.col2s[node_attrs.argmax(1)]
        pair = (s[edge_index[0].long()] * self.S + s[edge_index[1].long()]).to(torch.int32)
        return self.sw[bi](lengths.reshape(-1), pair)

    @staticmethod
    def _rowptr(g):
        rp = getattr(g, "rowptr", None)
        if rp is None:                                  # eager LocalGraph: real edges first, grouped by receiver
            recv = g.edge_index[1][: g.nE].contiguous()
            ar = torch.arange(g.nC + 1, device=recv.device, dtype=recv.dtype)
            rp = torch.searchsorted(recv, ar).to(torch.int32)
            g.rowptr = rp
        return rp

    def layer1(self, model, g, node_attrs, vectors):
        rc, blk, nC = self.rc, model.interactions[0], g.nC
        lengths, edge_attrs, edge_feats, cutoff = rc.edge_embed(model, g, node_attrs, vectors)
        node_feats = blk.linear_up(model.node_embedding(node_attrs))
        if self.sw is not None and not torch.is_grad_enabled():
            # pass A: fused lookup + TP + receiver sum, W never materialised
            s = self.col2s[node_attrs.argmax(1)]
            send = g.edge_index[0]
            pair = (s[send.long()] * self.S + s[g.edge_index[1].long()]).to(torch.int32)
            msg = self.cs0.forward(lengths.reshape(-1).contiguous(), pair, edge_attrs.contiguous(),
                                   node_feats.contiguous(), send.to(torch.int32).contiguous(), self._rowptr(g))
            ed = torch.tanh(blk.density_fn(edge_feats) ** 2)
            if cutoff is not None:
                ed = ed * cutoff
            recv = g.edge_index[1].long()
            density = torch.zeros(node_feats.shape[0], 1, dtype=ed.dtype, device=ed.device).index_add(0, recv, ed)[:nC].clone()
            na = node_attrs[:nC]
            message = blk.linear(msg) / (density + 1)
            message = self.skip0(message, na)
            nf = blk.reshape(message)
            return model.products[0](node_feats=nf, sc=None, node_attrs=na)
        tpw = self._tp_weights(0, blk, lengths, edge_feats, cutoff, node_attrs, g.edge_index)
        ed = torch.tanh(blk.density_fn(edge_feats) ** 2)
        if cutoff is not None:
            ed = ed * cutoff
        recv = g.edge_index[1].long()
        density = torch.zeros(node_feats.shape[0], 1, dtype=ed.dtype, device=ed.device).index_add(0, recv, ed)[:nC].clone()
        msg = blk.conv_tp(node_feats, edge_attrs, tpw, g.edge_index)[:nC]
        na = node_attrs[:nC]
        message = blk.linear(msg) / (density + 1)
        message = self.skip0(message, na)
        nf = blk.reshape(message)
        return model.products[0](node_feats=nf, sc=None, node_attrs=na)

    def layer2_energy(self, model, g, node_attrs, vectors, h1_local, heads):
        rc, blk, nC = self.rc, model.interactions[1], g.nC
        lengths, edge_attrs, edge_feats, cutoff = rc.edge_embed(model, g, node_attrs, vectors)
        na = node_attrs[:nC]
        sc = self.skip1(h1_local[:nC], na)
        node_feats = blk.linear_up(h1_local)
        tpw = self._tp_weights(1, blk, lengths, edge_feats, cutoff, node_attrs, g.edge_index)
        ed = torch.tanh(blk.density_fn(edge_feats) ** 2)
        if cutoff is not None:
            ed = ed * cutoff
        recv = g.edge_index[1].long()
        density = torch.zeros(node_feats.shape[0], 1, dtype=ed.dtype, device=ed.device).index_add(0, recv, ed)[:nC].clone()
        msg = blk.conv_tp(node_feats, edge_attrs, tpw, g.edge_index)[:nC]
        message = blk.linear(msg) / (density + 1)
        nf = blk.reshape(message)
        h2 = model.products[1](node_feats=nf, sc=sc, node_attrs=na)
        ar = torch.arange(nC, device=h2.device)
        e = model.readouts[1](h2, heads)[ar, heads]
        if hasattr(model, "pair_repulsion"):
            e = e + model.pair_repulsion_fn(lengths, node_attrs, g.edge_index, model.atomic_numbers)[:nC]
        return e


def install(model, species_cols, spline_grid=0):
    from . import receiver_chunk_mace as rc

    fl = FastLayers(model, species_cols, spline_grid=spline_grid)
    rc._layer1_orig = getattr(rc, "_layer1_orig", rc.layer1)
    rc._layer2_orig = getattr(rc, "_layer2_orig", rc.layer2_energy)
    rc.layer1 = fl.layer1
    rc.layer2_energy = fl.layer2_energy
    return fl


# ----------------------------------------------------------------------------- self-test on one real chunk
def _selftest():
    import argparse
    import time

    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    from . import receiver_chunk_mace as rc

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--rep", nargs=3, type=int, default=[8, 8, 8])
    ap.add_argument("--chunk", type=int, default=16384)
    ap.add_argument("--spline-grid", type=int, default=0)
    ap.add_argument("--compile", nargs="*", default=None,
                    help="also test torch.compile of the layer functions with these modes (e.g. default max-autotune-no-cudagraphs)")
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
    cols = sorted(int(c) for c in torch.unique(sysm.spidx.long()).tolist())
    L1o, L2o = rc.layer1, rc.layer2_energy
    fl = FastLayers(model, cols, spline_grid=a.spline_grid)
    print(f"spline grid: {a.spline_grid}")
    print(f"skip0: {fl.skip0.info}\nskip1: {fl.skip1.info}", flush=True)
    loc = torch.full((sysm.N,), -1, dtype=torch.int32, device=dev)
    g = rc.LocalGraph(sysm, 0, min(a.chunk, sysm.N), loc)
    na = rc.node_attrs_of(model, sysm.spidx[g.nodes], dt)
    v = g.vectors.to(dt)
    heads = torch.zeros(g.nC, dtype=torch.long, device=dev)
    print(f"chunk: {g.nC} receivers, {g.n_nodes} nodes, {int(g.nE):,} edges", flush=True)
    # skip_tp unit check on real features
    with torch.no_grad():
        x = torch.randn(g.nC, fl.d_in1, device=dev)
        r = model.interactions[1].skip_tp(x, na[: g.nC])
        f = fl.skip1(x, na[: g.nC])
        print(f"  skip1 on random x: rel.max err {float((f - r).abs().max() / r.abs().max()):.3e}", flush=True)
        h1 = L1o(model, g, na, v)
    H1 = torch.zeros(g.n_nodes, h1.shape[1], device=dev, dtype=dt)
    H1[: g.nC] = h1
    H1[g.nC:] = torch.randn(g.n_nodes - g.nC, h1.shape[1], device=dev) * h1.std()
    Gr = torch.randn_like(h1)

    def l1(fn):
        vv = v.detach().requires_grad_(True)
        h = fn(model, g, na, vv)
        (gv,) = torch.autograd.grad((h * Gr).sum(), [vv])
        return h.detach(), gv

    def l2(fn):
        vv = v.detach().requires_grad_(True)
        hh = H1.detach().requires_grad_(True)
        e = fn(model, g, na, vv, hh, heads)
        gv, gh = torch.autograd.grad(e.sum(), [vv, hh])
        return e.detach(), gv, gh

    def rel(a_, b_):
        return float((a_ - b_).abs().max() / b_.abs().max())

    (h_o, gv_o), (h_f, gv_f) = l1(L1o), l1(fl.layer1)
    print(f"  layer-1: h1 rel.max err {rel(h_f, h_o):.3e}   d/dvec {rel(gv_f, gv_o):.3e}", flush=True)
    with torch.no_grad():
        h_a = fl.layer1(model, g, na, v)
    print(f"  layer-1 forward-only path (pass A{', fused spline conv' if fl.sw is not None else ''}): "
          f"h1 rel.max err {rel(h_a, h_o):.3e}", flush=True)
    (e_o, gv_o2, gh_o), (e_f, gv_f, gh_f) = l2(L2o), l2(fl.layer2_energy)
    print(f"  layer-2: E_i rel.max err {rel(e_f, e_o):.3e}  sum dE {float((e_f - e_o).sum()):+.3e} eV   "
          f"d/dvec {rel(gv_f, gv_o2):.3e}   d/dh1 {rel(gh_f, gh_o):.3e}", flush=True)

    def timeit(fn, n=10):
        for _ in range(2):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / n * 1e3

    variants = [("original", L1o, L2o), ("fast", fl.layer1, fl.layer2_energy)]
    if a.compile is not None:
        import torch._dynamo as _dynamo

        _dynamo.config.cache_size_limit = 64
        for mode in (a.compile or ["default"]):
            c1 = torch.compile(fl.layer1, dynamic=False, mode=None if mode == "default" else mode)
            c2 = torch.compile(fl.layer2_energy, dynamic=False, mode=None if mode == "default" else mode)
            t0 = time.perf_counter()
            (hc, gvc), (ec, gvc2, ghc) = l1(c1), l2(c2)
            with torch.no_grad():
                hca = c1(model, g, na, v)
            torch.cuda.synchronize()
            print(f"  compiled[{mode}] first calls (compile time) {time.perf_counter()-t0:.1f}s", flush=True)
            print(f"  compiled[{mode}] vs original: L1 h1 {rel(hc, h_o):.3e}  d/dvec {rel(gvc, gv_o):.3e}  L1 fwd-only {rel(hca, h_o):.3e}"
                  f"   L2 E_i {rel(ec, e_o):.3e}  d/dvec {rel(gvc2, gv_o2):.3e}  d/dh1 {rel(ghc, gh_o):.3e}", flush=True)
            variants.append((f"cmp:{mode[:10]}", c1, c2))

    def gpu_ms(fn, n=5):
        from torch.profiler import ProfilerActivity, profile

        for _ in range(2):
            fn()
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as pr:
            for _ in range(n):
                fn()
            torch.cuda.synchronize()
        tot, cnt = 0.0, 0
        for ev in pr.events():
            if ev.device_type == torch.autograd.DeviceType.CUDA:
                tot += getattr(ev, "device_time_total", 0) or ev.cuda_time_total
                cnt += 1
        return tot / n / 1e3, cnt // n

    for name, f1, f2 in variants:
        def pa():
            with torch.no_grad():
                f1(model, g, na, v)
        ta = timeit(pa)
        tb = timeit(lambda: l2(f2))
        tc = timeit(lambda: l1(f1))
        print(f"  {name:8s}: pass A (L1 fwd) {ta:7.2f} ms   pass B (L2 fwd+bwd) {tb:7.2f} ms   "
              f"pass C (L1 fwd+bwd) {tc:7.2f} ms   total {ta+tb+tc:7.2f} ms", flush=True)
        ga, ka = gpu_ms(pa)
        gb, kb = gpu_ms(lambda: l2(f2))
        gc, kc = gpu_ms(lambda: l1(f1))
        print(f"  {name:8s}: GPU kernel time A {ga:6.2f} ms ({ka} kernels)  B {gb:6.2f} ms ({kb})  C {gc:6.2f} ms ({kc})"
              f"  total {ga+gb+gc:6.2f} ms", flush=True)


if __name__ == "__main__":
    _selftest()


class GenericLayers:
    """Same interface as FastLayers, built on the model's own modules (receiver_chunk_mace.layer1 / layer2_energy).
    Used for interaction blocks that FastLayers does not specialise (e.g. the non-linear blocks of MACE-MH-1):
    exact, but without the species-block skip connection and the tabulated radial weights."""

    def layer1(self, model, g, node_attrs, vectors):
        from . import receiver_chunk_mace as rc

        return rc.layer1(model, g, node_attrs, vectors)

    def layer2_energy(self, model, g, node_attrs, vectors, h1_local, heads):
        from . import receiver_chunk_mace as rc

        return rc.layer2_energy(model, g, node_attrs, vectors, h1_local, heads)
