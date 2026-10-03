"""
LAMMPS ML-IAP ("pair_style mliap unified <file.pt> 0") wrapper for the memory-lean, exact three-pass MACE.

Per force call LAMMPS hands us the local atoms, their ghosts, and the full pair list (receiver pair_i is local,
sender pair_j local or ghost, rij = the pair vector used by MACE's own ML-IAP wrapper).  We return per-atom energies
and per-pair forces dE/drij (same convention as mace.calculators.lammps_mliap_mace).  Algorithm (exact):
    pass A  receiver chunks of local atoms:  h1 = layer1  (no grad)            -> H1[local]
            forward_exchange(H1)                                                -> H1 on ghosts
    pass B  chunks:  E2 = layer-2 energy;  dE2/drij -> pair forces,  dE2/dh1 -> G1[local + ghost]
            reverse_exchange(G1)                                                -> ghost contributions summed on owners
    pass C  chunks:  recompute layer 1 with grad;  loss = E1 + G1 . h1;  d/drij -> pair forces
Only H1, G1 (2 x 512 floats per atom) and per-pair arrays persist; every wide intermediate lives in one chunk.
Uses the exact speed-ups of fast_layers.py (species-block skip_tp, receiver-only node ops) and, if spline_grid > 0,
the fp64-sampled tabulated radial weights (spline_w.py, conv_spline.py).  Works with any subset of the model's
elements and with multi-GPU domain decomposition (ghost exchange goes through LAMMPS).
"""
import json
import os
import time

import torch

try:
    from lammps.mliap.mliap_unified_abc import MLIAPUnified
except ImportError:  # pragma: no cover
    class MLIAPUnified:
        def __init__(self):
            pass

_RUNTIME = ("_fl", "_dev", "_cols", "_e0", "_loc", "_log", "_D1", "_pad", "_prof", "_t_graph", "_cache_graphs", "_topo", "_gx", "_use_graphs", "_fmax_abort", "_flag", "_fl_l1", "_fl_l2", "_t_end", "_topo_mode", "_peak", "_hi")


class _G:
    pass


_INT_MAX = 2**31 - 1


def _exchange(fn, src, dst, D):
    """LAMMPS' ML-IAP comm buffers are sized with 32-bit ints (ntotal * D); split the feature columns so that every
    call stays below 2^31 elements (otherwise Kokkos aborts at ~4.2M atoms for D = 512)."""
    nt = src.shape[0]
    if nt * D < _INT_MAX:
        fn(src, dst, D)
        return
    w = D
    while nt * w >= _INT_MAX and w > 1:
        w //= 2
    for c in range(0, D, w):
        cw = min(w, D - c)
        s = src[:, c:c + cw].contiguous()
        d = torch.empty_like(s)
        fn(s, d, cw)
        dst[:, c:c + cw] = d


class _GraphExec:
    """CUDA graphs of the three per-chunk bodies for FULL chunks (nC == chunk), with static buffers of fixed capacity
    (n_cap nodes and e_cap edges per chunk, nt_cap atoms incl. ghosts).  Inputs are copied into the static buffers
    before each replay and outputs read back from static buffers; the arithmetic is identical to the eager path."""

    def __init__(self, owner, C, n_cap, e_cap, nt_cap, D1):
        dev, dt = owner._dev, owner.dtype
        self.o, self.C, self.n_cap, self.e_cap, self.nt_cap, self.D1 = owner, C, n_cap, e_cap, nt_cap, D1
        self.ncol = owner.num_species
        self.s_nodes = torch.zeros(n_cap, dtype=torch.long, device=dev)
        self.s_spec = torch.zeros(n_cap, dtype=torch.long, device=dev)
        self.s_ei = torch.zeros(2, e_cap, dtype=torch.long, device=dev)
        self.s_rowptr = torch.zeros(C + 1, dtype=torch.int32, device=dev)
        self.s_vec = torch.zeros(e_cap, 3, dtype=dt, device=dev)
        self.s_h1 = torch.zeros(C, D1, dtype=dt, device=dev)
        self.s_gv = torch.zeros(e_cap, 3, dtype=dt, device=dev)
        self.s_eat = torch.zeros(C, dtype=torch.float64, device=dev)
        self.s_g1c = torch.zeros(C, D1, dtype=dt, device=dev)
        self.H1x = torch.zeros(nt_cap, D1, dtype=dt, device=dev)
        self.G1 = torch.zeros(nt_cap, D1, dtype=dt, device=dev)
        self.heads = torch.full((C,), owner._hi, dtype=torch.long, device=dev)
        self.ar = torch.arange(C, device=dev)
        self.padpat = torch.arange(e_cap, device=dev) % n_cap
        self.padvec = owner._pad.to(dt)
        self.graphs = None

    def fits(self, n_nodes, nE, nt):
        return n_nodes <= self.n_cap - 1 and nE <= self.e_cap and nt <= self.nt_cap

    def _g(self):
        g = _G()
        g.edge_index = self.s_ei
        g.nC = self.C
        g.rowptr = self.s_rowptr
        return g

    def _na(self):
        return torch.nn.functional.one_hot(self.s_spec, num_classes=self.ncol).to(self.o.dtype)

    def body_A(self):
        with torch.no_grad():
            self.s_h1.copy_(self.o._fl_l1(self.o.model, self._g(), self._na(), self.s_vec))

    def body_B(self):
        o = self.o
        vec = self.s_vec.detach().requires_grad_(True)
        h1l = self.H1x[self.s_nodes].detach().requires_grad_(True)
        with torch.enable_grad():
            e = o._fl_l2(o.model, self._g(), self._na(), vec, h1l, self.heads)
            gv, gh = torch.autograd.grad(o.scale * e.sum(), [vec, h1l])
        self.s_gv.copy_(gv)
        self.G1.index_add_(0, self.s_nodes, gh)       # padding rows alias a real node and carry exactly 0
        self.s_eat.copy_(o.scale * e.detach().double())

    def body_C(self):
        o = self.o
        vec = self.s_vec.detach().requires_grad_(True)
        with torch.enable_grad():
            h1 = o._fl_l1(o.model, self._g(), self._na(), vec)
            r0 = o.model.readouts[0](h1, self.heads)[self.ar, self.heads]
            loss = o.scale * r0.sum() + (self.s_g1c * h1).sum()
            (gv,) = torch.autograd.grad(loss, [vec])
        self.s_gv.copy_(gv)
        self.s_eat.copy_(o.scale * r0.detach().double())

    def load(self, nodes, ls, e0, e1, a, b, rij, elems, rp_cpu, pi):
        n = nodes.numel()
        nE = e1 - e0
        self.s_nodes[:n].copy_(nodes)
        self.s_nodes[n:].fill_(a)                       # padding nodes alias the chunk's first receiver (= atom a)
        self.s_spec.copy_(elems[self.s_nodes])
        self.s_ei[0, :nE].copy_(ls)
        self.s_ei[1, :nE].copy_(pi[e0:e1] - a)          # receivers are the chunk's own atoms: local index = global - a
        self.s_ei[:, nE:].copy_(self.padpat[nE:].expand(2, -1))
        self.s_vec[:nE].copy_(rij[e0:e1])
        self.s_vec[nE:].copy_(self.padvec.expand(self.e_cap - nE, 3))
        self.s_rowptr.copy_(torch.as_tensor(rp_cpu[a:b + 1] - e0, device=self.s_rowptr.device))

    def capture(self):
        pool = torch.cuda.graph_pool_handle()
        side = torch.cuda.Stream()
        graphs = {}
        for name, fn in (("A", self.body_A), ("B", self.body_B), ("C", self.body_C)):
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(3):
                    fn()
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()
            gr = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gr, pool=pool):
                fn()
            graphs[name] = gr
        torch.cuda.synchronize()
        self.graphs = graphs


class ThreePassMLIAP(MLIAPUnified):
    def __init__(self, model, chunk=16384, spline_grid=512, node_bucket=8192, e_align=65536, rebuild_config=None):
        super().__init__()
        from ase.data import chemical_symbols

        self.model = model
        self.element_types = [chemical_symbols[int(z)] for z in model.atomic_numbers]
        self.num_species = len(self.element_types)
        self.rcutfac = 0.5 * float(model.r_max)
        self.ndescriptors = 1
        self.nparams = 1
        self.chunk = int(chunk)
        self.spline_grid = int(spline_grid)
        self.node_bucket = int(node_bucket)
        self.e_align = int(e_align)
        self.dtype = next(model.parameters()).dtype
        self.step = 0
        # rebuild recipe (class, config, state_dict) for models that keep e3nn TorchScript modules after conversion
        self._rebuild = None
        if rebuild_config is not None:
            self._rebuild = (type(model), rebuild_config, {k: v.detach().cpu() for k, v in model.state_dict().items()})

    # the runtime objects (Triton kernels, tables, cuEq JIT state) are rebuilt after unpickling
    def __getstate__(self):
        d = self.__dict__.copy()
        for k in _RUNTIME:
            d.pop(k, None)
        if d.get("_rebuild") is not None:       # MLIAPUnified is a plain class: the model is an ordinary attribute
            d.pop("model", None)
        return d

    def _rebuild_model(self):
        """Build the model from its recipe with e3nn's fx code generation only (no TorchScript in the LAMMPS process)."""
        import e3nn

        from .fused_conv import repair_fused_blocks

        cls, cfg, sd = self._rebuild
        e3nn.set_optimization_defaults(jit_script_fx=False)
        prev = torch.get_default_dtype()
        torch.set_default_dtype(self.dtype)
        try:
            m = cls(**cfg)
        finally:
            torch.set_default_dtype(prev)
        repair_fused_blocks(m)
        m.load_state_dict(sd, strict=True)
        return m.to(self.dtype)

    # ------------------------------------------------------------------ setup
    def _setup(self, elems):
        from . import fast_layers

        dev = elems.device
        self._dev = dev
        if os.environ.get("MACE3P_CHUNK"):
            self.chunk = int(os.environ["MACE3P_CHUNK"])
        if getattr(self, "_rebuild", None) is not None and getattr(self, "model", None) is None:
            self.model = self._rebuild_model()
        self.model = self.model.to(dev).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        cols = sorted(int(c) for c in torch.unique(elems).tolist())
        self._cols = cols
        from .receiver_chunk_mace import head_index
        self._hi, hname = head_index(self.model)
        try:
            self._fl = fast_layers.FastLayers(self.model, cols, spline_grid=self.spline_grid)
            kind = "fast"
        except AssertionError as err:      # interaction blocks other than MPA-0's: exact generic layers, no spline/skip
            self._fl = fast_layers.GenericLayers()
            kind = f"generic ({err})"
        print(f"ThreePassMLIAP: head {hname!r} (index {self._hi}), layers: {kind}", flush=True)
        self._flag = None
        # MACE3P_COMPILE=1: torch.compile (inductor) of the layer functions used inside the CUDA graphs of full chunks
        # (fixed shapes -> compiled once); the partial tail chunk keeps the eager functions (its shapes change every step)
        self._fl_l1, self._fl_l2 = self._fl.layer1, self._fl.layer2_energy
        if os.environ.get("MACE3P_COMPILE", "0") == "1":
            import torch._dynamo as _dynamo

            _dynamo.config.cache_size_limit = 64
            # MACE3P_COMPILE_MODE: inductor mode, e.g. "max-autotune-no-cudagraphs" (we capture CUDA graphs ourselves)
            mode = os.environ.get("MACE3P_COMPILE_MODE") or None
            self._fl_l1 = torch.compile(self._fl.layer1, dynamic=False, mode=mode)
            self._fl_l2 = torch.compile(self._fl.layer2_energy, dynamic=False, mode=mode)
        m = self.model
        ncol = len(m.atomic_numbers)
        with torch.no_grad():
            eye = torch.eye(ncol, dtype=self.dtype, device=dev)
            self._e0 = m.atomic_energies_fn(eye)[:, self._hi].double()      # [ncol]
            h0 = torch.full((1,), self._hi, dtype=torch.long, device=dev)
            z = torch.zeros(1, dtype=self.dtype, device=dev)
            self.shift = float(m.scale_shift(z, h0))
            self.scale = float(m.scale_shift(z + 1, h0)) - self.shift
        self._loc = None
        self._D1 = None
        self._pad = torch.tensor([1.5 * float(m.r_max), 0.0, 0.0], dtype=torch.float64, device=dev)
        self._prof = os.environ.get("MACE3P_PROFILE", "0") == "1"
        self._fmax_abort = float(os.environ.get("MACE3P_FMAX_ABORT", "100.0"))   # eV/A per pair
        self._cache_graphs = os.environ.get("MACE3P_CACHE_GRAPHS", "0") == "1"
        self._use_graphs = os.environ.get("MACE3P_GRAPHS", "0") == "1"
        if os.environ.get("MACE3P_GC", "1") == "0":      # diagnostic: no cyclic garbage collection during the run
            import gc

            gc.collect()
            gc.freeze()
            gc.disable()
        if self._use_graphs:
            # torch.nonzero in MACE's product block cannot be captured; argmax is identical for one-hot node_attrs
            from .graphed_three_pass import make_capture_safe

            make_capture_safe()
        self._gx = None           # CUDA-graph executor for full chunks (built from the 2nd force call on)
        self._topo = {}
        # MACE3P_TOPO=batched (default): the topologies of all receiver chunks are built in a few batched passes per
        # force call (_topology_all); chunk: one build per chunk (the pre-October-2026 path, kept as the reference)
        self._topo_mode = os.environ.get("MACE3P_TOPO", "batched")
        self._peak = 0.0
        self._t_graph = 0.0
        path = os.environ.get("MACE3P_LOG")
        self._log = open(path, "a", buffering=1) if path else None

    # ------------------------------------------------------------------ one receiver chunk as a padded local graph
    def _graph(self, a, b, e0, e1, pi, pj, rij, elems, rp_cpu, e_cap):
        if self._prof:
            torch.cuda.synchronize()
            _t0 = time.perf_counter()
        out = self._graph_impl(a, b, e0, e1, pi, pj, rij, elems, rp_cpu, e_cap)
        if self._prof:
            torch.cuda.synchronize()
            self._t_graph += time.perf_counter() - _t0
        return out

    def _topology(self, a, b, e0, e1, pi, pj):
        """Receiver chunk [a, b): local node list (receivers first, then outside senders) and the local index of the
        sender of each edge (int32); the local index of the receiver is its global index minus a."""
        dev = self._dev
        nC = b - a
        recv_g = pi[e0:e1].long()
        send_g = pj[e0:e1].long()
        loc = self._loc
        loc[a:b] = torch.arange(nC, device=dev, dtype=loc.dtype)
        # outside senders, ascending (identical to torch.unique(send_g) minus [a, b)); a mark + compaction replaces the
        # radix sort of torch.unique (Nsight: ~4 ms per 2.5M-edge chunk)
        flag = self._flag
        if flag is None or flag.numel() < loc.numel():
            flag = self._flag = torch.zeros(loc.numel(), dtype=torch.bool, device=dev)
        flag[send_g] = True
        flag[a:b] = False
        new = torch.nonzero(flag).squeeze(1)
        flag[new] = False
        loc[new] = torch.arange(nC, nC + new.numel(), device=dev, dtype=loc.dtype)
        nodes = torch.cat([torch.arange(a, b, device=dev, dtype=torch.int32), new.to(torch.int32)])
        ls = loc[send_g]                                                # int32 [nE]
        loc[nodes.long()] = -1
        return nodes, ls

    def _assemble(self, a, b, e0, e1, nodes, ls, rij, elems, rp_cpu, e_cap, pi):
        dev = self._dev
        nC = b - a
        n_nodes = nodes.numel()
        nodes = nodes.long()
        nb = self.node_bucket
        n_pad = ((n_nodes + 1 + nb - 1) // nb) * nb
        nodes_padded = torch.cat([nodes, nodes[:1].expand(n_pad - n_nodes)])
        nE = e1 - e0
        k = e_cap - nE
        tgt = torch.arange(k, device=dev) % n_pad
        g = _G()
        ei = torch.stack([ls.long(), (pi[e0:e1] - a).long()])
        g.edge_index = torch.cat([ei, torch.stack([tgt, tgt])], 1)
        vec = torch.cat([rij[e0:e1], self._pad.expand(k, 3)], 0).to(self.dtype)
        g.nC, g.nE, g.n_nodes = nC, nE, n_nodes
        g.nodes, g.nodes_padded = nodes, nodes_padded
        g.rowptr = (torch.as_tensor(rp_cpu[a:b + 1], device=dev) - e0).to(torch.int32)
        na = torch.nn.functional.one_hot(elems[nodes_padded], num_classes=self.num_species).to(self.dtype)
        return g, na, vec

    def _graph_impl(self, a, b, e0, e1, pi, pj, rij, elems, rp_cpu, e_cap):
        # self._topo holds the topologies of this force call (every chunk with MACE3P_TOPO=batched, the full chunks
        # with CUDA graphs otherwise); MACE3P_CACHE_GRAPHS=1 also keeps per-chunk builds for passes B and C
        key = (a, b)
        topo = self._topo.get(key)
        if topo is None:
            topo = self._topology(a, b, e0, e1, pi, pj)
            if self._cache_graphs:
                self._topo[key] = topo
        return self._assemble(a, b, e0, e1, topo[0], topo[1], rij, elems, rp_cpu, e_cap, pi)

    def _topology_all(self, bounds, rp_cpu, pi, pj, nl, nt):
        """Topologies of all receiver chunks, {(a, b): (nodes int32, local sender index int32 [nE])}, identical to
        _topology() chunk by chunk but built for groups of consecutive chunks at a time: one torch.unique over the
        (chunk, sender) keys of the group's outside edges instead of a dozen launches and a device sync per chunk.
        The per-chunk build is host-bound (1.0 s per call at 983k atoms with 30 chunks, 57 s per step at 7.9M atoms
        with 960 chunks); a group is bounded by MACE3P_TOPO_EDGES edges (~50 B per edge of transient memory)."""
        dev = self._dev
        C = self.chunk
        gmax = int(os.environ.get("MACE3P_TOPO_EDGES", "30000000"))
        out = {}
        nb = len(bounds)
        k0 = 0
        while k0 < nb:
            k1 = k0 + 1
            while k1 < nb and int(rp_cpu[bounds[k1][1]] - rp_cpu[bounds[k0][0]]) <= gmax:
                k1 += 1
            a0 = bounds[k0][0]
            E0, E1 = int(rp_cpu[a0]), int(rp_cpu[bounds[k1 - 1][1]])
            ng = k1 - k0
            recv = pi[E0:E1]                                             # int32, non-decreasing
            send = pj[E0:E1]
            cid = torch.div(recv - a0, C, rounding_mode="floor")         # chunk within the group
            lo = cid * C + a0                                            # chunk start a
            hi = torch.clamp(lo + C, max=nl)                             # chunk end b
            ls = send - lo                                               # local index of inside senders
            idx = torch.nonzero((send < lo) | (send >= hi)).squeeze(1)   # outside edges
            key = cid[idx].long() * nt + send[idx].long()
            uniq, inv = torch.unique(key, sorted=True, return_inverse=True)
            uc = torch.div(uniq, nt, rounding_mode="floor")              # chunk of each unique outside sender
            us = (uniq - uc * nt).to(torch.int32)                        # the sender, ascending within a chunk
            cnt = torch.bincount(uc, minlength=ng)
            start = torch.cumsum(cnt, 0) - cnt
            ca = torch.arange(ng, device=dev, dtype=torch.long) * C + a0
            nC = torch.clamp(ca + C, max=nl) - ca                         # receivers per chunk
            lid = nC[uc] + torch.arange(uniq.numel(), device=dev) - start[uc]
            ls[idx] = lid.to(torch.int32)[inv]
            cnt_h, start_h = cnt.tolist(), start.tolist()
            for k in range(k0, k1):
                a, b = bounds[k]
                e0, e1 = int(rp_cpu[a]), int(rp_cpu[b])
                s, c = start_h[k - k0], cnt_h[k - k0]
                nodes = torch.cat([torch.arange(a, b, device=dev, dtype=torch.int32), us[s:s + c]])
                out[(a, b)] = (nodes, ls[e0 - E0:e1 - E0])
            k0 = k1
        return out

    def _prepare_graphs(self, bounds, rp_cpu, pi, pj, rij, elems, nt, C):
        """A captured executor that fits the topologies of all full chunks (rebuilt if its capacities are exceeded);
        None on the first force call (eager warm-up; D1 unknown).  Uses self._topo, building the full chunks' entries
        one by one if the batched build did not run (MACE3P_TOPO=chunk)."""
        full = [(a, b) for a, b in bounds if b - a == C]
        if not full:
            return None
        if self._D1 is None:          # width of h1: one eager layer-1 call on the first full chunk (also JIT warm-up)
            a, b = full[0]
            e0, e1 = int(rp_cpu[a]), int(rp_cpu[b])
            e_cap0 = ((e1 - e0 + self.e_align - 1) // self.e_align) * self.e_align
            with torch.no_grad():
                g, na, vec = self._graph(a, b, e0, e1, pi, pj, rij, elems, rp_cpu, e_cap0)
                self._D1 = int(self._fl.layer1(self.model, g, na, vec).shape[1])
        missing = [k for k in full if k not in self._topo]
        if missing:
            _t_topo = time.perf_counter()
            for a, b in missing:
                self._topo[(a, b)] = self._topology(a, b, int(rp_cpu[a]), int(rp_cpu[b]), pi, pj)
            torch.cuda.synchronize()
            self._t_topo += time.perf_counter() - _t_topo
        n_max = max(self._topo[k][0].numel() for k in full)
        e_max = max(int(rp_cpu[b] - rp_cpu[a]) for a, b in full)
        gx = self._gx
        if gx is None or not gx.fits(n_max, e_max, nt):
            if gx is not None and self._log:
                self._log.write(json.dumps(dict(recapture=dict(n_max=n_max, n_cap=gx.n_cap, e_max=e_max, e_cap=gx.e_cap,
                                                               nt=nt, nt_cap=gx.nt_cap))) + "\n")
            self._gx = gx = None
            torch.cuda.empty_cache()
            # a recapture costs seconds (tens with MACE3P_COMPILE=1: new shapes -> recompilation), so leave room
            mg = float(os.environ.get("MACE3P_CAP_MARGIN", "1.25"))
            n_cap = ((int(n_max * mg) + 2 + 1023) // 1024) * 1024
            e_cap = ((int(e_max * mg) + self.e_align - 1) // self.e_align) * self.e_align
            nt_cap = int(nt * mg) + 1024
            gx = _GraphExec(self, C, n_cap, e_cap, nt_cap, self._D1)
            a, b = full[0]
            gx.load(*self._topo[(a, b)], int(rp_cpu[a]), int(rp_cpu[b]), a, b, rij, elems, rp_cpu, pi)
            gx.capture()
            self._gx = gx
            self.n_captures = getattr(self, "n_captures", 0) + 1
        return gx

    def _peak_gb(self):
        """Peak torch allocation (GB) since the previous call of this function; the running maximum is self._peak.
        Only when the per-call log is on (the reset changes what torch.cuda.max_memory_allocated reports)."""
        if self._log is None:
            return None
        p = torch.cuda.max_memory_allocated() / 1e9
        torch.cuda.reset_peak_memory_stats()
        self._peak = max(self._peak, p)
        return round(p, 3)

    # ------------------------------------------------------------------ LAMMPS entry point
    def compute_forces(self, data):
        nl, nt, P = int(data.nlocal), int(data.ntotal), int(data.npairs)
        if nl == 0 or P <= 1:
            return
        t_start = time.perf_counter()
        t_gap = (t_start - self._t_end) if getattr(self, "_t_end", None) else None
        # Safety guard (incident on an H200 node, 2026-09-24): LAMMPS/Kokkos indexes the full neighbour list with
        # 32-bit ints; nlocal x (neighbour slots incl. skin) beyond 2^31 gives an illegal GPU memory access that can take
        # nodes down.  Pair list inside r_max ~ P; slots incl. skin ~ P * ((r_max + skin)/r_max)^3 (average).
        # Calibrated on an H200 node: 8.96M atoms (0.89 of 2^31) ran, 9.88M (0.98) crashed -> abort above 0.95.
        # Abort cleanly in Python instead.  Override only deliberately: MACE3P_ALLOW_HUGE=1.
        skin = float(os.environ.get("MACE3P_SKIN", "2.0"))
        rmax = 2.0 * self.rcutfac
        est_slots = P * ((rmax + skin) / rmax) ** 3
        if est_slots > 0.95 * 2**31 and os.environ.get("MACE3P_ALLOW_HUGE", "0") != "1":
            raise RuntimeError(f"ThreePassMLIAP guard: nlocal={nl:,}, pairs={P:,} -> estimated LAMMPS neighbour slots "
                               f"{est_slots:.3e} is close to the 2^31 int32 limit of the Kokkos neighbour list "
                               f"(skin {skin} A). Use fewer atoms per GPU, a smaller skin, or more GPUs.")
        elems = torch.as_tensor(data.elems).long()
        if not hasattr(self, "_fl") or not set(torch.unique(elems).tolist()) <= set(self._cols):
            self._setup(elems)
        self.step += 1
        dev = self._dev
        elems = elems.to(dev)
        pi = torch.as_tensor(data.pair_i).to(dev).int()
        pj = torch.as_tensor(data.pair_j).to(dev).int()
        rij = torch.as_tensor(data.rij).to(dev).double()
        order = None
        if not bool((pi[1:] >= pi[:-1]).all()):
            order = torch.argsort(pi, stable=True)
            pi, pj, rij = pi[order], pj[order], rij[order]
        rp = torch.searchsorted(pi, torch.arange(nl + 1, device=dev, dtype=pi.dtype))
        rp_cpu = rp.cpu().numpy()
        C = self.chunk
        bounds = [(a, min(a + C, nl)) for a in range(0, nl, C)]
        e_max = max(int(rp_cpu[b] - rp_cpu[a]) for a, b in bounds)
        e_cap = ((e_max + self.e_align - 1) // self.e_align) * self.e_align
        if self._loc is None or self._loc.numel() < nt:
            self._loc = torch.full((max(nt, 1),), -1, dtype=torch.int32, device=dev)
        fl, m, scale = self._fl, self.model, self.scale
        eat = torch.zeros(nl, dtype=torch.float64, device=dev)
        PF = torch.zeros(P, 3, dtype=torch.float32, device=dev)     # fp32 while H1/G1 are alive; -> fp64 at the end
        heads = torch.full((C,), self._hi, dtype=torch.long, device=dev)
        # ---------------- pass A
        t = {}
        self._t_graph = 0.0
        self._topo = {}
        if self._prof:
            torch.cuda.synchronize()
            t["prep"] = time.perf_counter() - t_start
        t0 = time.perf_counter()
        H1 = None
        self._t_topo = self._t_load = 0.0
        _t_prep = time.perf_counter()
        if self._topo_mode == "batched":
            self._topo = self._topology_all(bounds, rp_cpu, pi, pj, nl, nt)
            torch.cuda.synchronize()
            self._t_topo = time.perf_counter() - _t_prep
        gx = self._prepare_graphs(bounds, rp_cpu, pi, pj, rij, elems, nt, C) if self._use_graphs else None
        t["A_prepare"] = time.perf_counter() - _t_prep
        with torch.no_grad():
            for a, b in bounds:
                e0, e1 = int(rp_cpu[a]), int(rp_cpu[b])
                if gx is not None and b - a == C:
                    _tl = time.perf_counter()
                    gx.load(*self._topo[(a, b)], e0, e1, a, b, rij, elems, rp_cpu, pi)
                    self._t_load += time.perf_counter() - _tl
                    gx.graphs["A"].replay()
                    h1 = gx.s_h1
                else:
                    g, na, vec = self._graph(a, b, e0, e1, pi, pj, rij, elems, rp_cpu, e_cap)
                    h1 = fl.layer1(m, g, na, vec)
                if H1 is None:
                    self._D1 = h1.shape[1]
                    H1 = torch.zeros(nt, self._D1, dtype=self.dtype, device=dev)
                H1[a:b] = h1
        D1 = self._D1
        H1x = gx.H1x[:nt] if gx is not None else torch.empty_like(H1)
        _exchange(data.forward_exchange, H1, H1x, D1)
        del H1
        torch.cuda.synchronize()
        t["A"] = time.perf_counter() - t0
        t["peak_A"] = self._peak_gb()
        t["graph_A"] = self._t_graph
        t["A_topology"] = self._t_topo
        t["A_load"] = self._t_load
        # ---------------- pass B
        t0 = time.perf_counter()
        if gx is not None:
            G1 = gx.G1[:nt]
            G1.zero_()
        else:
            G1 = torch.zeros_like(H1x)
        for a, b in bounds:
            e0, e1 = int(rp_cpu[a]), int(rp_cpu[b])
            if gx is not None and b - a == C:
                gx.load(*self._topo[(a, b)], e0, e1, a, b, rij, elems, rp_cpu, pi)
                gx.graphs["B"].replay()
                PF[e0:e1] += gx.s_gv[: e1 - e0].float()
                eat[a:b] += gx.s_eat
                continue
            g, na, vec = self._graph(a, b, e0, e1, pi, pj, rij, elems, rp_cpu, e_cap)
            vec = vec.requires_grad_(True)
            h1l = H1x[g.nodes_padded].detach().requires_grad_(True)
            with torch.enable_grad():
                e = fl.layer2_energy(m, g, na, vec, h1l, heads[: g.nC])
                gv, gh = torch.autograd.grad(scale * e.sum(), [vec, h1l])
            PF[e0:e1] += gv[: g.nE].float()
            G1.index_add_(0, g.nodes, gh[: g.n_nodes])
            eat[a:b] += scale * e.detach().double()
        del H1x
        G1r = torch.empty_like(G1)
        _exchange(data.reverse_exchange, G1, G1r, D1)
        del G1
        torch.cuda.synchronize()
        t["B"] = time.perf_counter() - t0
        t["peak_B"] = self._peak_gb()
        t["graph_AB"] = self._t_graph
        # ---------------- pass C
        t0 = time.perf_counter()
        for a, b in bounds:
            e0, e1 = int(rp_cpu[a]), int(rp_cpu[b])
            if gx is not None and b - a == C:
                gx.load(*self._topo[(a, b)], e0, e1, a, b, rij, elems, rp_cpu, pi)
                gx.s_g1c.copy_(G1r[a:b])
                gx.graphs["C"].replay()
                PF[e0:e1] += gx.s_gv[: e1 - e0].float()
                eat[a:b] += gx.s_eat
                continue
            g, na, vec = self._graph(a, b, e0, e1, pi, pj, rij, elems, rp_cpu, e_cap)
            vec = vec.requires_grad_(True)
            with torch.enable_grad():
                h1 = fl.layer1(m, g, na, vec)
                ar = torch.arange(g.nC, device=dev)
                r0 = m.readouts[0](h1, heads[: g.nC])[ar, heads[: g.nC]]
                loss = scale * r0.sum() + (G1r[a:b] * h1).sum()
                (gv,) = torch.autograd.grad(loss, [vec])
            PF[e0:e1] += gv[: g.nE].float()
            eat[a:b] += scale * r0.detach().double()
        del G1r
        torch.cuda.synchronize()
        t["C"] = time.perf_counter() - t0
        t["peak_C"] = self._peak_gb()
        t["graph_ABC"] = self._t_graph
        self._topo = {}
        eat += self._e0[elems[:nl]] + self.shift
        PF = PF.double()
        if order is not None:
            out = torch.empty_like(PF)
            out[order] = PF
            PF = out
        # Safety guard (incident on an H200 node, 2026-09-26): non-finite or absurd forces lead to NaN positions and an
        # illegal GPU memory access inside LAMMPS/Kokkos neighbour binning.  Abort cleanly in Python instead.
        fmax = float(PF.abs().max()) if PF.numel() else 0.0
        if not (bool(torch.isfinite(eat).all()) and fmax == fmax and fmax < self._fmax_abort):
            raise RuntimeError(f"ThreePassMLIAP guard: non-finite or absurd output at call {self.step} "
                               f"(max |pair force| = {fmax:.3e} eV/A, limit {self._fmax_abort:g}); aborting before LAMMPS "
                               f"integrates it. The model is probably extrapolating (check the trajectory / training data).")
        t_pre_out = time.perf_counter()
        torch.as_tensor(data.eatoms).copy_(eat)
        data.energy = float(eat.sum())
        data.update_pair_forces_gpu(PF)
        if self._log is not None:
            torch.cuda.synchronize()
            t_out = time.perf_counter() - t_pre_out
            # t_gap_lammps: wall time LAMMPS spent between the end of our previous call and the start of this one
            # (pair-list build inside ML-IAP, force tally, integration, output)
            self._log.write(json.dumps(dict(step=self.step, nlocal=nl, ntotal=nt, npairs=P, chunks=len(bounds),
                                            e_cap=e_cap, sorted_pairs=order is None, graphs=gx is not None,
                                            captures=getattr(self, "n_captures", 0), t_total=time.perf_counter() - t_start, **t,
                                            t_output=t_out, t_gap_lammps=t_gap,
                                            peak_out=self._peak_gb(), peak_gb=self._peak)) + "\n")
            self._t_end = time.perf_counter()

    def compute_descriptors(self, data):
        pass

    def compute_gradients(self, data):
        pass
