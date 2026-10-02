"""
Exact receiver-chunked evaluation of an L-layer NequIP-type network (SevenNet), standalone and inside LAMMPS.

A network with L message-passing layers needs, to compute the layer-k features of a receiver, the layer-(k-1)
features of the receiver and of its neighbours, and nothing else. Hence (exact, 2L chunk passes):

  forward,  k = 1..L : X_k[C] = layer_k(X_{k-1}[C + S(C)], edges into C)     chunk by chunk, no autograd
                       (inside LAMMPS: forward_exchange(X_k) fills the ghost rows)
  read-out           : E = sum_i e(X_L[i]),  G_L = dE/dX_L                     per atom
  backward, k = L..1 : recompute layer_k on each chunk with autograd, loss = <G_k[C], X_k[C]>
                       -> pair forces dE/dr_ij, and G_{k-1}[C + S(C)] += dloss/dX_{k-1}
                       (inside LAMMPS: reverse_exchange(G_{k-1}) sums ghost contributions on their owners)

Resident: X_0..X_L and one G (per atom); edge intermediates and the autograd graph exist for one chunk only.
For L = 2 this is the three-pass scheme of mliap_threepass.py (second forward and first backward fused).

SevenNet's conventions: edge_index[0] = receiver, edge_index[1] = sender, edge_vec = r_sender - r_receiver. LAMMPS'
ML-IAP pair vector rij = x_j - x_i (j = neighbour) is the same vector, so it is passed unchanged.

Needs sevenn (>= 0.10) and its e3nn (>= 0.5). With the LAMMPS build of this repository, which embeds the Python of
the MACE environment (e3nn 0.4.4), run LAMMPS with the SevenNet environment's site-packages FIRST on PYTHONPATH
(see examples/sevennet_lammps.sh); both environments must use the same Python minor version.
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


# ----------------------------------------------------------------------------- model
def load_sevennet(name, dtype, device, script=True):
    """SevenNet model in `dtype` on `device`, built from a checkpoint path or a pretrained name (e.g. "7net-0").
    script=False builds e3nn's generated code as fx modules only (no TorchScript), as needed inside LAMMPS."""
    import e3nn
    import sevenn.util as su

    path = name if os.path.isfile(name) else su.pretrained_name_to_path(name)
    if not script:
        e3nn.set_optimization_defaults(jit_script_fx=False)
    # e3nn's generated tensor products fix their dtype at construction: build under the target default dtype.
    # sevenn's patch for old checkpoints compares float32 tables with the new ones; let that comparison cast.
    prev, allclose = torch.get_default_dtype(), torch.allclose
    torch.set_default_dtype(dtype)
    torch.allclose = lambda x, y, *aa, **kw: allclose(x.to(y.dtype), y, *aa, **kw)
    try:
        model, cfg = su.model_from_checkpoint(path)
    finally:
        torch.allclose = allclose
        torch.set_default_dtype(prev)
    model = model.to(device=device, dtype=dtype).eval()
    # some sevenn/e3nn modules keep float tensors as plain attributes (not parameters/buffers): cast them as well
    for mod in model.modules():
        for nm, val in list(vars(mod).items()):
            if isinstance(val, torch.Tensor) and torch.is_floating_point(val) and val.dtype != dtype:
                setattr(mod, nm, val.to(device=device, dtype=dtype))
    for p in model.parameters():
        p.requires_grad_(False)
    return model, cfg


class SevenNetLayers:
    """The modules of an AtomGraphSequential SevenNet model, split into embedding, L layers and read-out."""

    def __init__(self, model, cfg, dtype):
        mods = model._modules
        self.m, self.dt = model, dtype
        self.L = int(cfg["num_convolution_layer"])
        self.r_max = float(cfg["cutoff"])
        self.edge_embedding = mods["edge_embedding"]
        self.embed_mod = mods["onehot_to_feature_x"]
        self.layers = [[mods[n] for n in mods if n.startswith(f"{k}_")] for k in range(self.L)]
        self.readout = [mods["reduce_input_to_hidden"], mods["reduce_hidden_to_energy"], mods["rescale_atomic_energy"]]
        self.n_types = int(cfg.get("_number_of_species", len(model.type_map)))

    def onehot(self, t):
        return torch.nn.functional.one_hot(t.long(), self.n_types).to(self.dt)

    def embed(self, types):
        oh = self.onehot(types)
        return self.embed_mod({"x": oh, "node_attr": oh})["x"]

    def layer(self, k, x_nodes, types_nodes, edge_index_sevenn, edge_vec):
        d = {"x": x_nodes, "node_attr": self.onehot(types_nodes), "atom_type": types_nodes.long(),
             "edge_index": edge_index_sevenn, "edge_vec": edge_vec}
        d = self.edge_embedding(d)
        for mod in self.layers[k]:
            d = mod(d)
        return d["x"]

    def atomic_energy(self, xL, types):
        d = {"x": xL, "atom_type": types.long()}
        for mod in self.readout:
            d = mod(d)
        return d["atomic_energy"].reshape(-1)


# ----------------------------------------------------------------------------- standalone (periodic box)
class LPassSevenNet(SevenNetLayers):
    def __init__(self, model, cfg, dtype, chunk):
        super().__init__(model, cfg, dtype)
        self.chunk = int(chunk)

    def compute(self, sysm, types, log=print):
        """Energy and forces (sorted order of sysm) for a receiver_chunk_mace.System with built neighbour list."""
        from .receiver_chunk_mace import LocalGraph

        N, dev, dt = sysm.N, sysm.device, self.dt
        loc = torch.full((N,), -1, dtype=torch.int32, device=dev)
        F = torch.zeros(N, 3, dtype=torch.float64, device=dev)
        t0 = time.perf_counter()

        def graph(a, b):
            g = LocalGraph(sysm, a, b, loc)
            # LocalGraph: edge_index = [sender, receiver], vectors = r_receiver - r_sender
            return g, torch.stack([g.edge_index[1], g.edge_index[0]]), -g.vectors.to(dt)

        with torch.no_grad():
            X = [self.embed(types)]
            for k in range(self.L):                                         # forward passes
                Xk = None
                for a in range(0, N, self.chunk):
                    b = min(a + self.chunk, N)
                    g, ei, ev = graph(a, b)
                    xk = self.layer(k, X[k][g.nodes], types[g.nodes], ei, ev)[: g.nC]
                    if Xk is None:
                        Xk = torch.empty(N, xk.shape[1], dtype=dt, device=dev)
                    Xk[a:b] = xk
                    del g, xk
                X.append(Xk)
        t_fwd = time.perf_counter() - t0
        XL = X[-1].clone().requires_grad_(True)
        e_at = self.atomic_energy(XL, types)
        E = float(e_at.double().sum())
        (G,) = torch.autograd.grad(e_at.sum(), [XL])
        del XL, e_at
        t0 = time.perf_counter()
        for k in range(self.L - 1, -1, -1):                                  # backward passes (recompute)
            Gprev = torch.zeros_like(X[k]) if k > 0 else None
            for a in range(0, N, self.chunk):
                b = min(a + self.chunk, N)
                g, ei, ev = graph(a, b)
                ev = ev.requires_grad_(True)                                 # ev = r_sender - r_receiver
                xin = X[k][g.nodes].detach().clone().requires_grad_(k > 0)
                with torch.enable_grad():
                    xk = self.layer(k, xin, types[g.nodes], ei, ev)[: g.nC]
                    loss = (G[a:b] * xk).sum()
                    if k > 0:
                        gv, gx = torch.autograd.grad(loss, [ev, xin])
                        Gprev.index_add_(0, g.nodes, gx)
                    else:
                        (gv,) = torch.autograd.grad(loss, [ev])
                # ev = r_send - r_recv:  F_send -= dE/dev,  F_recv += dE/dev
                F.index_add_(0, g.send_g, -gv.double())
                F.index_add_(0, g.recv_g, gv.double())
                del g, ev, xin, xk, loss, gv
            G = Gprev
        t_bwd = time.perf_counter() - t0
        resident = sum(x.numel() * x.element_size() for x in X)
        log(f"  L={self.L} forward {t_fwd:.1f}s backward {t_bwd:.1f}s  resident X_0..X_L {resident / N:.0f} B/atom")
        return E, F, resident


# ----------------------------------------------------------------------------- LAMMPS (ML-IAP unified, Kokkos)
_RUNTIME = ("_lay", "_dev", "_loc", "_flag", "_log", "_t_end", "_fmax_abort")


class LPassSevenNetMLIAP(MLIAPUnified):
    """pair_style mliap unified <file.pt> 0  for a SevenNet model, evaluated exactly in receiver chunks.
    The file stores only the checkpoint path and the settings; the model is built inside LAMMPS on the first call
    (fx code generation only: TorchScript cannot be deserialised in the LAMMPS process, see mliap_threepass.py)."""

    def __init__(self, checkpoint, chunk=8192, dtype="float32"):
        super().__init__()
        from ase.data import chemical_symbols

        import sevenn.util as su

        path = checkpoint if os.path.isfile(checkpoint) else su.pretrained_name_to_path(checkpoint)
        self.checkpoint = os.path.abspath(path)
        model, cfg = load_sevennet(self.checkpoint, torch.float32, "cpu")
        order = sorted(model.type_map.items(), key=lambda kv: kv[1])          # (Z, type index), by type index
        assert [i for _, i in order] == list(range(len(order)))
        self.element_types = [chemical_symbols[int(z)] for z, _ in order]       # LAMMPS elems = SevenNet type index
        self.num_species = len(self.element_types)
        self.rcutfac = 0.5 * float(cfg["cutoff"])
        self.ndescriptors = 1
        self.nparams = 1
        self.chunk = int(chunk)
        self.dtype_name = str(dtype)
        self.step = 0

    def __getstate__(self):
        d = self.__dict__.copy()
        for k in _RUNTIME:
            d.pop(k, None)
        return d

    def _setup(self):
        self._dev = torch.device("cuda")
        if os.environ.get("MACE3P_CHUNK"):
            self.chunk = int(os.environ["MACE3P_CHUNK"])
        dt = getattr(torch, self.dtype_name)
        model, cfg = load_sevennet(self.checkpoint, dt, self._dev, script=False)
        self._lay = SevenNetLayers(model, cfg, dt)
        self._loc = None
        self._flag = None
        self._fmax_abort = float(os.environ.get("MACE3P_FMAX_ABORT", "100.0"))
        path = os.environ.get("MACE3P_LOG")
        self._log = open(path, "a", buffering=1) if path else None
        import e3nn
        import sevenn

        print(f"LPassSevenNetMLIAP: {os.path.basename(self.checkpoint)} L={self._lay.L} r_max={self._lay.r_max} "
              f"dtype={self.dtype_name} chunk={self.chunk} | torch {torch.__version__} ({os.path.dirname(torch.__file__)}) "
              f"e3nn {e3nn.__version__} sevenn {sevenn.__version__}", flush=True)

    def _topology(self, a, b, e0, e1, pi, pj):
        """Receiver chunk [a, b): nodes (receivers first, then outside senders incl. ghosts) and the SevenNet edge
        index [receiver, sender] in local numbering."""
        dev = self._dev
        nC = b - a
        recv_g = pi[e0:e1].long()
        send_g = pj[e0:e1].long()
        loc = self._loc
        loc[a:b] = torch.arange(nC, device=dev, dtype=loc.dtype)
        flag = self._flag
        flag[send_g] = True
        flag[a:b] = False
        new = torch.nonzero(flag).squeeze(1)
        flag[new] = False
        loc[new] = torch.arange(nC, nC + new.numel(), device=dev, dtype=loc.dtype)
        nodes = torch.cat([torch.arange(a, b, device=dev), new])
        ei = torch.stack([loc[recv_g], loc[send_g]]).long()
        loc[nodes] = -1
        return nodes, ei

    def compute_forces(self, data):
        from .mliap_threepass import _exchange

        nl, nt, P = int(data.nlocal), int(data.ntotal), int(data.npairs)
        if nl == 0 or P <= 1:
            return
        t_start = time.perf_counter()
        if not hasattr(self, "_lay"):
            self._setup()
        self.step += 1
        lay, dev, dt, C = self._lay, self._dev, self._lay.dt, self.chunk
        elems = torch.as_tensor(data.elems).to(dev).long()
        pi = torch.as_tensor(data.pair_i).to(dev).long()
        pj = torch.as_tensor(data.pair_j).to(dev).long()
        rij = torch.as_tensor(data.rij).to(dev).double()
        order = None
        if not bool((pi[1:] >= pi[:-1]).all()):
            order = torch.argsort(pi, stable=True)
            pi, pj, rij = pi[order], pj[order], rij[order]
        rp_cpu = torch.searchsorted(pi, torch.arange(nl + 1, device=dev, dtype=pi.dtype)).cpu().numpy()
        bounds = [(a, min(a + C, nl)) for a in range(0, nl, C)]
        if self._loc is None or self._loc.numel() < nt:
            self._loc = torch.full((nt,), -1, dtype=torch.long, device=dev)
            self._flag = torch.zeros(nt, dtype=torch.bool, device=dev)
        topo = {}
        t = {}
        # ---------------- forward: X_0 (all atoms incl. ghosts, from the species), X_1..X_L by chunks + exchange
        t0 = time.perf_counter()
        with torch.no_grad():
            X = [lay.embed(elems)]
            for k in range(lay.L):
                Xk = None
                for a, b in bounds:
                    e0, e1 = int(rp_cpu[a]), int(rp_cpu[b])
                    if (a, b) not in topo:
                        topo[(a, b)] = self._topology(a, b, e0, e1, pi, pj)
                    nodes, ei = topo[(a, b)]
                    xk = lay.layer(k, X[k][nodes], elems[nodes], ei, rij[e0:e1].to(dt))[: b - a]
                    if Xk is None:
                        Xk = torch.zeros(nt, xk.shape[1], dtype=dt, device=dev)
                    Xk[a:b] = xk
                if k < lay.L - 1:                                            # X_L is needed on local atoms only
                    Xx = torch.empty_like(Xk)
                    _exchange(data.forward_exchange, Xk, Xx, Xk.shape[1])
                    Xk = Xx
                X.append(Xk)
        torch.cuda.synchronize()
        t["forward"] = time.perf_counter() - t0
        # ---------------- read-out
        XL = X[-1][:nl].clone().requires_grad_(True)
        with torch.enable_grad():
            e_at = lay.atomic_energy(XL, elems[:nl])
            (gL,) = torch.autograd.grad(e_at.sum(), [XL])
        eat = e_at.detach().double()
        G = torch.zeros(nt, gL.shape[1], dtype=dt, device=dev)
        G[:nl] = gL
        del XL, e_at, gL
        X.pop()
        # ---------------- backward: recompute layer k, pair forces, G_{k-1} (+ reverse exchange)
        t0 = time.perf_counter()
        PF = torch.zeros(P, 3, dtype=torch.float64, device=dev)
        for k in range(lay.L - 1, -1, -1):
            Gprev = torch.zeros_like(X[k]) if k > 0 else None
            for a, b in bounds:
                e0, e1 = int(rp_cpu[a]), int(rp_cpu[b])
                nodes, ei = topo[(a, b)]
                ev = rij[e0:e1].to(dt).requires_grad_(True)
                xin = X[k][nodes].detach().requires_grad_(k > 0)
                with torch.enable_grad():
                    xk = lay.layer(k, xin, elems[nodes], ei, ev)[: b - a]
                    loss = (G[a:b] * xk).sum()
                    if k > 0:
                        gv, gx = torch.autograd.grad(loss, [ev, xin])
                        Gprev.index_add_(0, nodes, gx)
                    else:
                        (gv,) = torch.autograd.grad(loss, [ev])
                PF[e0:e1] += gv.double()                                    # dE/drij, as mliap_threepass.py
                del ev, xin, xk, loss, gv
            X.pop()
            if k > 0:
                Gr = torch.empty_like(Gprev)
                _exchange(data.reverse_exchange, Gprev, Gr, Gprev.shape[1])
                G = Gr
                del Gprev
        torch.cuda.synchronize()
        t["backward"] = time.perf_counter() - t0
        if order is not None:
            out = torch.empty_like(PF)
            out[order] = PF
            PF = out
        fmax = float(PF.abs().max()) if PF.numel() else 0.0
        if not (bool(torch.isfinite(eat).all()) and fmax == fmax and fmax < self._fmax_abort):
            raise RuntimeError(f"LPassSevenNetMLIAP guard: non-finite or absurd output at call {self.step} "
                               f"(max |pair force| = {fmax:.3e} eV/A, limit {self._fmax_abort:g}); aborting.")
        torch.as_tensor(data.eatoms).copy_(eat)
        data.energy = float(eat.sum())
        data.update_pair_forces_gpu(PF)
        if self._log is not None:
            self._log.write(json.dumps(dict(step=self.step, nlocal=nl, ntotal=nt, npairs=P, chunks=len(bounds),
                                            t_total=time.perf_counter() - t_start, **t,
                                            peak_gb=torch.cuda.max_memory_allocated() / 1e9)) + "\n")

    def compute_descriptors(self, data):
        pass

    def compute_gradients(self, data):
        pass
