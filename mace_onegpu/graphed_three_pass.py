#!/usr/bin/env python
"""
CUDA-Graph version of the three-pass receiver-chunked MACE (receiver_chunk_mace.py).

Removes the kernel-launch / Python overhead that left the GPU idle ~50% of the time:
  * per-chunk local graphs (receiver+sender node lists, padded local edge_index) are built ONCE per neighbour
    list and cached as int32 (~0.8 kB/atom);
  * all full chunks share fixed shapes (n_cap nodes, e_cap edges), so each pass is captured ONCE as a CUDA graph
    and replayed per chunk after copying that chunk's indices into static buffers;
  * edge vectors are computed INSIDE the graph from the (static-address, fp64) position array, so an MD
    integrator can update positions in place and replay;
  * forces, G1 and the energy are accumulated inside the graphs into global tensors;
  * the three graphs share one memory pool (replayed sequentially, never concurrently).
The last, partial chunk runs eagerly. Results are identical to the eager three-pass (same kernels).

usage (GPU):
  python -m mace_onegpu.graphed_three_pass --model M.model --base water64_1x1x1.extxyz --rep 20 20 20 --variant fp32_fused \
         --chunk 16384 [--tf32] [--compare-eager] [--check-replica]
"""
import argparse
import json
import math
import os
import time

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
import torch  # noqa: E402

from . import receiver_chunk_mace as rc  # noqa: E402

GB = 1e9


def make_capture_safe():
    """MACE 0.3.15 EquivariantProductBasisBlock (cuEq path) computes the species index with
    torch.nonzero(node_attrs)[:, 1], a host-synchronising op that cannot be captured in a CUDA graph.
    For one-hot node_attrs, argmax(dim=1) returns the identical index without synchronisation.
    Everything else in the forward is copied verbatim."""
    from mace.modules import blocks as B

    def forward(self, node_feats, sc, node_attrs):
        use_cueq = False
        use_cueq_mul_ir = False
        if hasattr(self, "use_agnostic_product"):
            if self.use_agnostic_product:
                node_attrs = torch.ones((node_feats.shape[0], 1), dtype=node_feats.dtype, device=node_feats.device)
        if hasattr(self, "cueq_config"):
            if self.cueq_config is not None:
                if self.cueq_config.enabled and (self.cueq_config.optimize_all or self.cueq_config.optimize_symmetric):
                    use_cueq = True
                if self.cueq_config.layout_str == "mul_ir":
                    use_cueq_mul_ir = True
        if use_cueq:
            if use_cueq_mul_ir:
                node_feats = torch.transpose(node_feats, 1, 2)
            index_attrs = node_attrs.argmax(dim=1).int()          # was: torch.nonzero(node_attrs)[:, 1].int()
            node_feats = self.symmetric_contractions(node_feats.flatten(1), index_attrs)
        else:
            node_feats = self.symmetric_contractions(node_feats, node_attrs)
        if self.use_sc and sc is not None:
            return self.linear(node_feats) + sc
        return self.linear(node_feats)

    B.EquivariantProductBasisBlock.forward = forward

    # Multi-head readouts (e.g. MACE-MH-1): mace.modules.irreps_tools.mask_head writes `mask[idx, :, head] = 1`, an
    # index_put with a host scalar that cannot be captured. The same 0/1 mask (same shape, channel order and dtype)
    # built from a device-side comparison gives bit-identical results.
    from mace.modules import irreps_tools as IT

    def mask_head(x, head, num_heads: int):
        sel = head.unsqueeze(-1) == torch.arange(num_heads, device=x.device)                 # [N, H]
        mask = sel.unsqueeze(-1).expand(x.shape[0], num_heads, x.shape[1] // num_heads)      # [N, H, C/H]
        return x * mask.reshape(x.shape).to(torch.get_default_dtype())

    IT.mask_head = mask_head
    B.mask_head = mask_head


make_capture_safe()


class ChunkCache:
    """Padded local graphs of all FULL chunks, built once per neighbour list."""

    def __init__(self, sysm, chunk, e_cap_align=65536):
        dev = sysm.device
        N = sysm.N
        self.chunk = chunk
        self.nfull = N // chunk
        loc = torch.full((N,), -1, dtype=torch.int32, device=dev)
        raw = []
        n_max = 0
        e_max = 0
        for k in range(self.nfull):
            a, b = k * chunk, (k + 1) * chunk
            g = rc.LocalGraph(sysm, a, b, loc)            # unpadded
            raw.append((g.nodes.to(torch.int32), g.edge_index.to(torch.int32), g.nE))
            n_max = max(n_max, g.n_nodes)
            e_max = max(e_max, g.nE)
            del g
        self.n_cap = n_max + 1                              # >= 1 spare slot
        self.e_cap = ((e_max + e_cap_align - 1) // e_cap_align) * e_cap_align
        self.nodes = torch.empty(self.nfull, self.n_cap, dtype=torch.int32, device=dev)
        self.ei = torch.empty(self.nfull, 2, self.e_cap, dtype=torch.int32, device=dev)
        self.nE = torch.empty(self.nfull, dtype=torch.int64, device=dev)
        self.rowptr = torch.empty(self.nfull, chunk + 1, dtype=torch.int32, device=dev)
        ar = torch.arange(chunk + 1, device=dev, dtype=torch.int32)
        for k, (nodes, ei, nE) in enumerate(raw):
            self.rowptr[k] = torch.searchsorted(ei[1, :nE].contiguous(), ar).to(torch.int32)   # receivers 0..chunk-1
            nn = nodes.numel()
            self.nodes[k, :nn] = nodes
            self.nodes[k, nn:] = nodes[0]                   # padding nodes alias a real atom; they only ever get +0
            self.ei[k, :, :nE] = ei
            pad = torch.arange(self.e_cap - nE, device=dev, dtype=torch.int32) % self.n_cap
            self.ei[k, 0, nE:] = pad                        # zero-weight self loops spread over all nodes
            self.ei[k, 1, nE:] = pad
            self.nE[k] = nE
        del raw
        self.bytes = self.nodes.numel() * 4 + self.ei.numel() * 4 + self.nE.numel() * 8 + self.rowptr.numel() * 4


class GraphedThreePass:
    def __init__(self, model, dtype, chunk, sysm):
        self.m, self.dt, self.chunk = model, dtype, chunk
        self.sysm = sysm
        dev = sysm.device
        self.dev = dev
        heads0 = torch.zeros(1, dtype=torch.long, device=dev)
        z = torch.zeros(1, dtype=dtype, device=dev)
        self.shift = float(model.scale_shift(z, heads0))
        self.scale = float(model.scale_shift(z + 1, heads0)) - self.shift
        t0 = time.perf_counter()
        self.cache = ChunkCache(sysm, chunk)
        torch.cuda.synchronize()
        self.t_cache = time.perf_counter() - t0
        N = sysm.N
        self.N = N
        # global state (static addresses)
        self.F = torch.zeros(N, 3, dtype=torch.float64, device=dev)
        self.H1 = None
        self.G1 = None
        self.EB = torch.zeros((), dtype=torch.float64, device=dev)
        self.EC = torch.zeros((), dtype=torch.float64, device=dev)
        # static per-chunk inputs
        c = self.cache
        self.s_nodes = torch.zeros(c.n_cap, dtype=torch.long, device=dev)
        self.s_ei = torch.zeros(2, c.e_cap, dtype=torch.long, device=dev)
        self.s_nE = torch.zeros((), dtype=torch.long, device=dev)
        self.s_rowptr = torch.zeros(chunk + 1, dtype=torch.int32, device=dev)
        self.e_ar = torch.arange(c.e_cap, device=dev)
        self.padvec = torch.tensor([1.5 * sysm.r_max, 0.0, 0.0], dtype=torch.float64, device=dev)
        self.heads = torch.zeros(chunk, dtype=torch.long, device=dev)
        self.graphs = None
        self.e0 = None

    # ---------------------------------------------------------------- per-chunk bodies (capturable)
    def _local_inputs(self):
        s = self.sysm
        send, recv = self.s_ei[0], self.s_ei[1]
        gs, gr = self.s_nodes[send], self.s_nodes[recv]
        v = s.pos[gr] - s.pos[gs]
        v = v - s.box * torch.round(v / s.box)
        pad = (self.e_ar >= self.s_nE).unsqueeze(-1)
        v = torch.where(pad, self.padvec, v).to(self.dt)
        na = torch.nn.functional.one_hot(s.spidx[self.s_nodes].long(), num_classes=len(self.m.atomic_numbers)).to(self.dt)
        return na, v

    class _G:  # duck-typed LocalGraph for rc.layer1 / rc.layer2_energy
        pass

    def _g(self):
        g = self._G()
        g.edge_index = self.s_ei
        g.nC = self.chunk
        g.rowptr = self.s_rowptr
        return g

    def _accum_forces(self, gv):
        Floc = torch.zeros(self.cache.n_cap, 3, dtype=torch.float64, device=self.dev)
        gv = gv.to(torch.float64)
        Floc.index_add_(0, self.s_ei[1], -gv)
        Floc.index_add_(0, self.s_ei[0], gv)
        self.F.index_add_(0, self.s_nodes, Floc)

    def body_A(self):
        with torch.no_grad():
            na, v = self._local_inputs()
            h1 = rc.layer1(self.m, self._g(), na, v)
            self.H1.index_copy_(0, self.s_nodes[: self.chunk], h1)

    def body_B(self):
        na, v = self._local_inputs()
        vec = v.detach().requires_grad_(True)
        h1l = self.H1[self.s_nodes].detach().requires_grad_(True)
        with torch.enable_grad():
            e = rc.layer2_energy(self.m, self._g(), na, vec, h1l, self.heads)
            eb = self.scale * e.sum()
            gv, gh = torch.autograd.grad(eb, [vec, h1l])
        self.EB.add_(eb.detach().to(torch.float64))
        self.G1.index_add_(0, self.s_nodes, gh)
        self._accum_forces(gv)

    def body_C(self):
        na, v = self._local_inputs()
        vec = v.detach().requires_grad_(True)
        rec = self.s_nodes[: self.chunk]
        with torch.enable_grad():
            h1 = rc.layer1(self.m, self._g(), na, vec)
            r1 = self.m.readouts[0](h1, self.heads)[torch.arange(self.chunk, device=self.dev), self.heads]
            ec = self.scale * r1.sum()
            loss = ec + (self.G1[rec] * h1).sum()
            (gv,) = torch.autograd.grad(loss, [vec])
        self.EC.add_(ec.detach().to(torch.float64))
        self._accum_forces(gv)

    def _load(self, k):
        c = self.cache
        self.s_nodes.copy_(c.nodes[k])
        self.s_ei.copy_(c.ei[k])
        self.s_nE.copy_(c.nE[k])
        self.s_rowptr.copy_(c.rowptr[k])

    # ---------------------------------------------------------------- capture
    def _capture(self):
        pool = torch.cuda.graph_pool_handle()
        self._load(0)
        graphs = {}
        side = torch.cuda.Stream()
        for name, fn in (("A", self.body_A), ("B", self.body_B), ("C", self.body_C)):
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(3):
                    fn()
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=pool):
                fn()
            graphs[name] = g
        torch.cuda.synchronize()
        self.graphs = graphs

    # ---------------------------------------------------------------- one energy+force evaluation
    def compute(self):
        m, dt, dev, s = self.m, self.dt, self.dev, self.sysm
        N, chunk = self.N, self.chunk
        t = {}
        if self.e0 is None:
            e0 = 0.0
            with torch.no_grad():
                for a in range(0, N, 1 << 20):
                    b = min(a + (1 << 20), N)
                    na = rc.node_attrs_of(m, s.spidx[a:b], dt)
                    hh = torch.zeros(b - a, dtype=torch.long, device=dev)
                    e0 += float(m.atomic_energies_fn(na)[torch.arange(b - a, device=dev), hh].double().sum())
            self.e0 = e0 + self.shift * N
        if self.H1 is None:
            # infer h1 width with one eager chunk
            loc = torch.full((N,), -1, dtype=torch.int32, device=dev)
            g = rc.LocalGraph(s, 0, min(chunk, N), loc)
            with torch.no_grad():
                h1 = rc.layer1(m, g, rc.node_attrs_of(m, s.spidx[g.nodes], dt), g.vectors.to(dt))
            self.H1 = torch.zeros(N, h1.shape[1], dtype=dt, device=dev)
            self.G1 = torch.zeros_like(self.H1)
            del g, h1, loc
        if self.graphs is None:
            t0 = time.perf_counter()
            self._capture()
            t["capture"] = time.perf_counter() - t0
        self.F.zero_()
        self.G1.zero_()
        self.EB.zero_()
        self.EC.zero_()
        eager = rc.ReceiverChunkMACE(m, dt, chunk)       # for the partial last chunk only
        a_last = self.cache.nfull * chunk
        loc = torch.full((N,), -1, dtype=torch.int32, device=dev) if a_last < N else None
        if loc is not None:
            eager.e_cap = ((chunk * s.MN + 65535) // 65536) * 65536

        torch.cuda.synchronize(); t0 = time.perf_counter()
        gA = self.graphs["A"]
        for k in range(self.cache.nfull):
            self._load(k)
            gA.replay()
        if loc is not None:
            g = rc.LocalGraph(s, a_last, N, loc, eager.e_cap, eager.node_bucket)
            with torch.no_grad():
                self.H1[a_last:N] = rc.layer1(m, g, rc.node_attrs_of(m, s.spidx[g.nodes_padded], dt), g.vectors.to(dt))
        torch.cuda.synchronize(); t["A"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        gB = self.graphs["B"]
        for k in range(self.cache.nfull):
            self._load(k)
            gB.replay()
        EB_tail = 0.0
        if loc is not None:
            g = rc.LocalGraph(s, a_last, N, loc, eager.e_cap, eager.node_bucket)
            na = rc.node_attrs_of(m, s.spidx[g.nodes_padded], dt)
            vec = g.vectors.to(dt).requires_grad_(True)
            h1l = self.H1[g.nodes_padded].detach().requires_grad_(True)
            heads = torch.zeros(g.nC, dtype=torch.long, device=dev)
            with torch.enable_grad():
                e = rc.layer2_energy(m, g, na, vec, h1l, heads)
                eb = self.scale * e.sum()
                gv, gh = torch.autograd.grad(eb, [vec, h1l])
            EB_tail = float(eb.detach().double())
            self.G1.index_add_(0, g.nodes, gh[: g.n_nodes])
            eager._accum_forces(self.F, g, gv)
        torch.cuda.synchronize(); t["B"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        gC = self.graphs["C"]
        for k in range(self.cache.nfull):
            self._load(k)
            gC.replay()
        EC_tail = 0.0
        if loc is not None:
            g = rc.LocalGraph(s, a_last, N, loc, eager.e_cap, eager.node_bucket)
            na = rc.node_attrs_of(m, s.spidx[g.nodes_padded], dt)
            vec = g.vectors.to(dt).requires_grad_(True)
            heads = torch.zeros(g.nC, dtype=torch.long, device=dev)
            with torch.enable_grad():
                h1 = rc.layer1(m, g, na, vec)
                r1 = m.readouts[0](h1, heads)[torch.arange(g.nC, device=dev), heads]
                ec = self.scale * r1.sum()
                loss = ec + (self.G1[a_last:N] * h1).sum()
                (gv,) = torch.autograd.grad(loss, [vec])
            EC_tail = float(ec.detach().double())
            eager._accum_forces(self.F, g, gv)
        torch.cuda.synchronize(); t["C"] = time.perf_counter() - t0
        E = self.e0 + float(self.EB) + EB_tail + float(self.EC) + EC_tail
        return E, self.F, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--rep", nargs=3, type=int, default=[20, 20, 20])
    ap.add_argument("--variant", default="fp32_fused")
    ap.add_argument("--chunk", type=int, default=16384)
    ap.add_argument("--tf32", action="store_true")
    ap.add_argument("--fuse-radial", default="none", choices=["none", "ieee", "tf32x3", "tf32"],
                    help="replace the radial MLPs by the fused Triton kernel (fp32 only)")
    ap.add_argument("--radial3x", action="store_true", help="3xTF32 tensor-core GEMMs in the radial MLPs")
    ap.add_argument("--spline-grid", type=int, default=0, help="tabulated radial weights (knots), needs --fast-layers")
    ap.add_argument("--fast-layers", action="store_true", help="exact species-block skip_tp + receiver-only node ops")
    ap.add_argument("--ace-layer1", action="store_true", help="exact ACE-style rewrite of layer 1 (ace_layer1.py)")
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--compare-eager", action="store_true", help="also run eager three-pass and compare forces")
    ap.add_argument("--check-replica", action="store_true")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    if a.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    dev = "cuda"
    model, dt = rc.build_model(a.model, a.variant, dev)
    ref_model = model
    if a.radial3x:
        import copy

        from . import radial_3xtf32

        ref_model = copy.deepcopy(model)          # untouched original for the replica check
        print(f"3xTF32 radial MLP installed in {radial_3xtf32.apply(model)} interaction blocks", flush=True)
    if a.fuse_radial != "none":
        import copy

        from .fused_radial import apply_fused_radial

        ref_model = copy.deepcopy(model)          # untouched original for the replica check
        nfr = apply_fused_radial(model, precision=a.fuse_radial)
        print(f"fused radial MLP ({a.fuse_radial}) installed in {nfr} interaction blocks", flush=True)

    class A:
        structure = None
        base = a.base
        rep = a.rep

    pos, box, sp, _ = rc.load_structure(A, model, dt, dev)
    sysm = rc.System(pos, box, sp, float(model.r_max), dev)
    del pos, sp
    sysm.build_neighbors()
    N = sysm.N
    if a.fast_layers:
        from . import fast_layers

        cols = sorted(int(c) for c in torch.unique(sysm.spidx.long()).tolist())
        fl = fast_layers.install(model, cols, spline_grid=a.spline_grid)
        print(f"fast layers installed: skip0 {fl.skip0.info}; skip1 {fl.skip1.info}", flush=True)
    if a.ace_layer1:
        from . import ace_layer1

        cols = sorted(int(c) for c in torch.unique(sysm.spidx.long()).tolist())
        ace = ace_layer1.install(model, cols)
        ace.layout = os.environ.get("ACE_L1_LAYOUT", "seg")
        print(f"ACE layer-1 installed: species columns {cols}, layout {ace.layout}", flush=True)
    ev = GraphedThreePass(model, dt, a.chunk, sysm)
    c = ev.cache
    print(f"N={N:,} chunk={a.chunk} full chunks={c.nfull} n_cap={c.n_cap} e_cap={c.e_cap} "
          f"cache={c.bytes/N:.0f} B/atom build={ev.t_cache:.1f}s tf32={a.tf32}", flush=True)
    res = {"N": N, "chunk": a.chunk, "variant": a.variant, "tf32": a.tf32, "fuse_radial": a.fuse_radial,
           "ace_layer1": a.ace_layer1, "fast_layers": a.fast_layers,
           "radial3x": a.radial3x,
           "spline_grid": a.spline_grid, "runs": []}
    E = F = None
    for r in range(a.repeat):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        E, F, t = ev.compute()
        torch.cuda.synchronize()
        tt = time.perf_counter() - t0
        pk = torch.cuda.max_memory_allocated() / GB
        tt_run = tt - t.get("capture", 0.0)
        print(f"[run {r}] total {tt:.2f}s (capture {t.get('capture',0):.1f}s)  A {t['A']:.3f}  B {t['B']:.3f}  C {t['C']:.3f}  "
              f"-> {N/tt_run:.3e} atom-step/s   peak {pk:.1f} GB   E={E:.6f}", flush=True)
        res["runs"].append({"total": tt, **t, "atom_step_per_s": N / tt_run, "peak_gb": pk, "E": E})
    Fg = F.clone()
    if a.compare_eager:
        ee = rc.ReceiverChunkMACE(model, dt, a.chunk)
        Ee, Fe, te, _ = ee.compute(sysm, log=lambda *x, **k: None)
        Ee, Fe, te, _ = ee.compute(sysm, log=lambda *x, **k: None)
        d = Fg - Fe
        print(f"[graphed vs eager] max|dF|={float(d.abs().max()):.3e} eV/A  dE={E-Ee:+.3e} eV   "
              f"eager steady {te['A']+te['B']+te['C']:.2f}s", flush=True)
        res["vs_eager"] = {"max_dF": float(d.abs().max()), "dE": E - Ee, "eager_t": te}
    if a.check_replica:
        from ase import io as aseio

        from .ase_gradient_probe import make_batch

        tf = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        base = aseio.read(a.base)
        nb = len(base)
        bb = make_batch(ref_model, base, dev)
        o = ref_model(bb.to_dict(), training=False, compute_force=True)
        Fb = o["forces"].detach().double()
        Eb = float(o["energy"].detach().double().sum())
        torch.backends.cuda.matmul.allow_tf32 = tf
        k = (sysm.perm % nb).to(Fb.device)
        d = Fg - Fb[k]
        print(f"REPLICA CHECK vs original MACE (fp32 base cell, no tf32): every one of {N:,} atoms  "
              f"max|dF|={float(d.abs().max()):.3e}  RMS={float(d.pow(2).mean().sqrt()):.3e} eV/A  "
              f"E/N-E_b/N_b={E/N-Eb/nb:+.3e} eV/atom", flush=True)
        res["replica"] = {"max_dF": float(d.abs().max()), "rms_dF": float(d.pow(2).mean().sqrt()), "dE_per_atom": E / N - Eb / nb}
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
