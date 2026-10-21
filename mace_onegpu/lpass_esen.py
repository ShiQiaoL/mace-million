"""
Exact receiver-chunked evaluation of eSEN (fairchem-core 1.10, equivariant transformer with SO(2) convolutions),
standalone and inside LAMMPS. Third architecture family after MACE (mliap_threepass.py) and NequIP-type SevenNet
(lpass_sevennet.py); the schedule is the 2L-sweep one of lpass_sevennet.py:

  eSEN couples atoms only through node states: layer 0 (edge-degree embedding) and every eSEN block build their
  per-edge quantities (Wigner rotations, radial embedding, source/target species embedding, SO(2) messages) from the
  node features of the previous layer and the edge geometry, aggregate them on the receiver and discard them.
  Hence, with L blocks, L+1 message layers (k = 0 the edge-degree embedding, k = 1..L the blocks):

  forward,  k = 0..L : X_{k+1}[C] = layer_k(X_k[C + S(C)], edges into C)       chunk by chunk, no autograd
  read-out           : E = sum_i e(norm(X_{L+1})[i]),  G = dE/dX_{L+1}        per atom
  backward, k = L..0 : recompute layer_k on each chunk with autograd, loss = <G[C], X_{k+1}[C]>
                       -> pair forces dE/dr_ij and G_k[C + S(C)] += dloss/dX_k

Resident: X_0..X_{L+1} and one G per atom; edge tensors and the autograd graph exist for one chunk only.

eSEN's conventions (fairchem.core.models.esen.esen): edge_index[0] = source (sender j), edge_index[1] = target
(receiver i), edge_distance_vec = pos[source] - pos[target] = r_j - r_i. LAMMPS' ML-IAP pair vector rij = x_j - x_i
(j = neighbour) is the same vector and is passed unchanged; receiver_chunk_mace.LocalGraph.vectors = x_recv - x_send
is negated.

Energies: fairchem trainers predict a normalised energy per structure, E = rmsd * sum_i y_i + mean (+ sum_i ref[Z_i]
when linear element references are used). Per atom we use e_i = rmsd * y_i + ref[Z_i]; the per-structure constant
`mean` is added once in the standalone evaluation and dropped inside LAMMPS (it does not affect forces or virials).
The forces are exactly those of the original model.
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


# ----------------------------------------------------------------------------- deterministic edge frame
# eSEN (like eSCN and EquiformerV2) rotates every edge into a local frame whose axis perpendicular to the edge is
# chosen with torch.rand_like (fairchem.core.models.esen.common.rotation.init_edge_rot_mat). Whether the output
# depends on that choice is a property of the activation: with the gate activation and the spectral atom-wise MLP of
# eSEN-30M-OAM the network is invariant to rotations about the edge axis, and two calls agree to rounding (6e-15 eV/A
# on 192 water atoms, float64; experimental/esen_exactness_cpu.py --frame random); with the S2-grid activation
# (class default, mmax < lmax) it is not, and two calls on the same configuration differ visibly (0.1-0.3 eV/A for a
# random-weight test model). We replace the random vector by a constant (the alignment fall-backs of the original code
# remain), which makes every evaluation a deterministic function of the geometry; for eSEN-30M-OAM this changes
# nothing beyond rounding.
_FRAME_VEC = (0.3, 0.5, 0.8)


def init_edge_rot_mat_fixed(edge_distance_vec, rot_clip=False):
    """fairchem 1.10 init_edge_rot_mat (MIT, Meta) with the random vector replaced by a constant."""
    YTOL = 0.999999
    edge_vec_0 = edge_distance_vec
    edge_vec_0_distance = torch.sqrt(torch.sum(edge_vec_0 ** 2, dim=1))
    norm_x = edge_vec_0 / (edge_vec_0_distance.view(-1, 1))
    if rot_clip:
        yprod = norm_x @ norm_x.new_tensor([0.0, 1.0, 0.0])
        norm_x[yprod > YTOL] = norm_x.new_tensor([0.0, 1.0, 0.0])
        norm_x[yprod < -YTOL] = norm_x.new_tensor([0.0, -1.0, 0.0])
    edge_vec_2 = edge_vec_0.new_tensor(_FRAME_VEC).expand_as(edge_vec_0).clone()
    edge_vec_2 = edge_vec_2 / (torch.sqrt(torch.sum(edge_vec_2 ** 2, dim=1)).view(-1, 1))
    edge_vec_2b = edge_vec_2.clone()
    edge_vec_2b[:, 0] = -edge_vec_2[:, 1]
    edge_vec_2b[:, 1] = edge_vec_2[:, 0]
    edge_vec_2c = edge_vec_2.clone()
    edge_vec_2c[:, 1] = -edge_vec_2[:, 2]
    edge_vec_2c[:, 2] = edge_vec_2[:, 1]
    vec_dot_b = torch.abs(torch.sum(edge_vec_2b * norm_x, dim=1)).view(-1, 1)
    vec_dot_c = torch.abs(torch.sum(edge_vec_2c * norm_x, dim=1)).view(-1, 1)
    vec_dot = torch.abs(torch.sum(edge_vec_2 * norm_x, dim=1)).view(-1, 1)
    edge_vec_2 = torch.where(torch.gt(vec_dot, vec_dot_b), edge_vec_2b, edge_vec_2)
    vec_dot = torch.abs(torch.sum(edge_vec_2 * norm_x, dim=1)).view(-1, 1)
    edge_vec_2 = torch.where(torch.gt(vec_dot, vec_dot_c), edge_vec_2c, edge_vec_2)
    vec_dot = torch.abs(torch.sum(edge_vec_2 * norm_x, dim=1))
    if len(vec_dot) > 0:
        assert torch.max(vec_dot) < 0.99
    norm_z = torch.cross(norm_x, edge_vec_2, dim=1)
    norm_z = norm_z / (torch.sqrt(torch.sum(norm_z ** 2, dim=1, keepdim=True)))
    norm_z = norm_z / (torch.sqrt(torch.sum(norm_z ** 2, dim=1)).view(-1, 1))
    norm_y = torch.cross(norm_x, norm_z, dim=1)
    norm_y = norm_y / (torch.sqrt(torch.sum(norm_y ** 2, dim=1, keepdim=True)))
    norm_x = norm_x.view(-1, 3, 1)
    norm_y = -norm_y.view(-1, 3, 1)
    norm_z = norm_z.view(-1, 3, 1)
    edge_rot_mat_inv = torch.cat([norm_z, norm_x, norm_y], dim=2)
    edge_rot_mat = torch.transpose(edge_rot_mat_inv, 1, 2)
    return edge_rot_mat if rot_clip else edge_rot_mat.detach()


_FRAME_INSTALLED = {"mode": None}


def install_edge_frame(mode=None):
    """mode 'fixed' (default; env MACE3P_ESEN_FRAME) patches the frame construction used by the eSEN backbone,
    'random' restores fairchem's original. Affects every eSEN evaluation in the process, including the reference."""
    import fairchem.core.models.esen.esen as esen_mod
    from fairchem.core.models.esen.common import rotation as rot

    mode = mode or os.environ.get("MACE3P_ESEN_FRAME", "fixed")
    if not hasattr(rot, "_init_edge_rot_mat_original"):
        rot._init_edge_rot_mat_original = rot.init_edge_rot_mat
    fn = init_edge_rot_mat_fixed if mode == "fixed" else rot._init_edge_rot_mat_original
    rot.init_edge_rot_mat = fn
    esen_mod.init_edge_rot_mat = fn
    try:
        import fairchem.core.models.esen.esen_dens as dens_mod
        dens_mod.init_edge_rot_mat = fn
    except Exception:  # pragma: no cover
        pass
    _FRAME_INSTALLED["mode"] = mode
    return mode


# ----------------------------------------------------------------------------- model
def load_esen(checkpoint, dtype, device, log=print):
    """Backbone, energy MLP and energy scaling of an eSEN checkpoint (fairchem-core 1.10 format), in `dtype` on
    `device`, built through fairchem's own loader so that the weights, normalizer and element references are exactly
    those the OCPCalculator would use."""
    from fairchem.core.common.relaxation.ase_utils import OCPCalculator

    calc = OCPCalculator(checkpoint_path=checkpoint, cpu=True, seed=0)
    tr = calc.trainer
    model = getattr(tr.model, "module", tr.model)
    backbone = model.backbone
    heads = model.output_heads
    head = heads["energy"] if "energy" in heads else next(iter(heads.values()))
    energy_block = head.energy_block
    backbone.direct_forces = False                      # conservative path: rot_clip in the Wigner construction
    backbone.activation_checkpointing = False
    rmsd, mean, refs = 1.0, 0.0, None
    nz = getattr(tr, "normalizers", {}) or {}
    if "energy" in nz:
        n = nz["energy"]
        rmsd = float(getattr(n, "rmsd", getattr(n, "std", 1.0)))
        mean = float(getattr(n, "mean", 0.0))
    er = getattr(tr, "elementrefs", {}) or {}
    if "energy" in er:
        refs = er["energy"].element_references.detach().clone()
    log(f"load_esen: {os.path.basename(checkpoint)} blocks={backbone.num_layers} cutoff={backbone.cutoff} "
        f"max_neighbors={backbone.max_neighbors} lmax={backbone.lmax} mmax={backbone.mmax} "
        f"sphere={backbone.sphere_channels} hidden={backbone.hidden_channels} | energy rmsd={rmsd:g} mean={mean:g} "
        f"element refs={'yes' if refs is not None else 'no'}")
    backbone = backbone.to(device=device, dtype=dtype).eval()
    energy_block = energy_block.to(device=device, dtype=dtype).eval()
    for p in list(backbone.parameters()) + list(energy_block.parameters()):
        p.requires_grad_(False)
    if refs is not None:
        refs = refs.to(device=device, dtype=torch.float64)
    return backbone, energy_block, dict(rmsd=rmsd, mean=mean, refs=refs, calc=calc)


class ESENLayers:
    """An eSEN backbone split into node embedding, L+1 message layers and the per-atom read-out."""

    def __init__(self, backbone, energy_block, dtype, rmsd=1.0, mean=0.0, refs=None):
        if _FRAME_INSTALLED["mode"] is None:
            install_edge_frame()
        self.bb, self.eb, self.dt = backbone, energy_block, dtype
        self.L = int(backbone.num_layers) + 1           # layer 0: edge-degree embedding; 1..L: eSEN blocks
        self.r_max = float(backbone.cutoff)
        self.rmsd, self.mean, self.refs = float(rmsd), float(mean), refs

    def embed(self, Z):
        """X_0: species embedding in the l = 0 channel (node-only)."""
        x = torch.zeros(Z.shape[0], self.bb.sph_feature_size, self.bb.sphere_channels, dtype=self.dt, device=Z.device)
        x[:, 0, :] = self.bb.sphere_embedding(Z.long())
        return x

    def _edges(self, Z_nodes, ei, ev):
        d = torch.linalg.norm(ev, dim=-1)
        _, wigner, wigner_inv = self.bb.get_rotmat_and_wigner(ev)
        Z = Z_nodes.long()
        x_edge = torch.cat((self.bb.distance_expansion(d), self.bb.source_embedding(Z[ei[0]]),
                            self.bb.target_embedding(Z[ei[1]])), dim=1)
        return d, wigner, wigner_inv, x_edge

    def layer(self, k, x_nodes, Z_nodes, ei, ev):
        """Layer k on a local graph: nodes = receivers first, then senders; ei = [sender, receiver] (local indices);
        ev = r_sender - r_receiver. Returns the new node features of all local nodes (rows beyond the receivers are
        residual-only and are discarded by the caller)."""
        d, w, winv, x_edge = self._edges(Z_nodes, ei, ev)
        if k == 0:
            return self.bb.edge_degree_embedding(x_nodes, x_edge, d, ei, winv)
        return self.bb.blocks[k - 1](x_nodes, x_edge, d, ei, w, winv, node_offset=0)

    def atomic_energy(self, xL, Z):
        """Per-atom energies of the original model (eV): rmsd * MLP(norm(x)_{l=0}) + element reference."""
        x = self.bb.norm(xL)
        e = self.eb(x.narrow(1, 0, 1).squeeze(1)).reshape(-1)
        e = e * self.rmsd
        if self.refs is not None:
            e = e + self.refs[Z.long()].to(e.dtype)
        return e


# ----------------------------------------------------------------------------- standalone (periodic box)
class LPassESEN(ESENLayers):
    def __init__(self, backbone, energy_block, dtype, chunk, **kw):
        super().__init__(backbone, energy_block, dtype, **kw)
        self.chunk = int(chunk)

    def compute(self, sysm, Z, log=print):
        """Energy (eV, incl. the per-structure mean) and forces (sorted order of sysm) for a receiver_chunk_mace.System
        with built neighbour list; Z = atomic numbers in the sorted order."""
        from .receiver_chunk_mace import LocalGraph

        N, dev, dt = sysm.N, sysm.device, self.dt
        loc = torch.full((N,), -1, dtype=torch.int32, device=dev)
        F = torch.zeros(N, 3, dtype=torch.float64, device=dev)
        t0 = time.perf_counter()

        def graph(a, b):
            g = LocalGraph(sysm, a, b, loc)
            return g, g.edge_index, (-g.vectors).to(dt)          # [sender, receiver], r_sender - r_receiver

        with torch.no_grad():
            X = [self.embed(Z)]
            for k in range(self.L):                                          # forward sweeps
                Xk = None
                for a in range(0, N, self.chunk):
                    b = min(a + self.chunk, N)
                    g, ei, ev = graph(a, b)
                    xk = self.layer(k, X[k][g.nodes], Z[g.nodes], ei, ev)[: g.nC]
                    if Xk is None:
                        Xk = torch.empty((N,) + tuple(xk.shape[1:]), dtype=dt, device=dev)
                    Xk[a:b] = xk
                    del g, xk
                X.append(Xk)
        t_fwd = time.perf_counter() - t0
        XL = X[-1].clone().requires_grad_(True)
        with torch.enable_grad():
            e_at = self.atomic_energy(XL, Z)
            (G,) = torch.autograd.grad(e_at.sum(), [XL])
        E = float(e_at.detach().double().sum()) + self.mean
        del XL, e_at
        t0 = time.perf_counter()
        for k in range(self.L - 1, -1, -1):                                   # backward sweeps (recompute)
            Gprev = torch.zeros_like(X[k]) if k > 0 else None
            for a in range(0, N, self.chunk):
                b = min(a + self.chunk, N)
                g, ei, ev = graph(a, b)
                ev = ev.requires_grad_(True)
                xin = X[k][g.nodes].detach().clone().requires_grad_(k > 0)
                with torch.enable_grad():
                    xk = self.layer(k, xin, Z[g.nodes], ei, ev)[: g.nC]
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
        log(f"  eSEN L={self.L} forward {t_fwd:.1f}s backward {t_bwd:.1f}s  resident X_0..X_L {resident / N:.0f} B/atom")
        return E, F, resident


# ----------------------------------------------------------------------------- LAMMPS (ML-IAP unified, Kokkos)
_RUNTIME = ("_lay", "_dev", "_loc", "_flag", "_log", "_fmax_abort", "_Z_of_type", "_full")


class LPassESENMLIAP(MLIAPUnified):
    """pair_style mliap unified <file.pt> 0  for an eSEN checkpoint, evaluated exactly in receiver chunks.
    The file stores the checkpoint path, the element list (LAMMPS type order) and the settings; the model is built
    inside LAMMPS on the first call. MACE3P_ESEN_FULL=1 switches to the stock full-graph evaluation (one forward over
    all local and ghost atoms with the complete autograd graph resident), the reference for speed and capacity."""

    def __init__(self, checkpoint, elements, chunk=8192, dtype="float32"):
        super().__init__()
        from ase.data import atomic_numbers

        self.checkpoint = os.path.abspath(checkpoint)
        self.element_types = list(elements)
        self.Z_list = [int(atomic_numbers[s]) for s in self.element_types]
        self.num_species = len(self.element_types)
        self.ndescriptors = 1
        self.nparams = 1
        self.chunk = int(chunk)
        self.dtype_name = str(dtype)
        self.step = 0
        self.rcutfac = None                                 # set in _setup (needs the checkpoint); LAMMPS reads it
        bb, eb, meta = load_esen(self.checkpoint, torch.float32, "cpu")
        self.rcutfac = 0.5 * float(bb.cutoff)
        del bb, eb, meta

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
        bb, eb, meta = load_esen(self.checkpoint, dt, self._dev)
        self._lay = ESENLayers(bb, eb, dt, rmsd=meta["rmsd"], mean=meta["mean"], refs=meta["refs"])
        self._Z_of_type = torch.tensor(self.Z_list, device=self._dev, dtype=torch.long)
        self._loc = None
        self._flag = None
        self._full = os.environ.get("MACE3P_ESEN_FULL", "0") == "1"
        self._fmax_abort = float(os.environ.get("MACE3P_FMAX_ABORT", "100.0"))
        path = os.environ.get("MACE3P_LOG")
        self._log = open(path, "a", buffering=1) if path else None
        import fairchem.core

        print(f"LPassESENMLIAP: {os.path.basename(self.checkpoint)} L={self._lay.L} r_max={self._lay.r_max} "
              f"dtype={self.dtype_name} chunk={self.chunk} full_graph={self._full} elements={self.element_types} | "
              f"torch {torch.__version__} ({os.path.dirname(torch.__file__)}) fairchem-core {fairchem.core.__version__}",
              flush=True)

    def _topology(self, a, b, e0, e1, pi, pj):
        """Receiver chunk [a, b): nodes (receivers first, then outside senders incl. ghosts) and the eSEN edge index
        [sender, receiver] in local numbering."""
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
        ei = torch.stack([loc[send_g], loc[recv_g]]).long()
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
        Z = self._Z_of_type[torch.as_tensor(data.elems).to(dev).long()]
        pi = torch.as_tensor(data.pair_i).to(dev).long()
        pj = torch.as_tensor(data.pair_j).to(dev).long()
        rij = torch.as_tensor(data.rij).to(dev).double()
        order = None
        if not bool((pi[1:] >= pi[:-1]).all()):
            order = torch.argsort(pi, stable=True)
            pi, pj, rij = pi[order], pj[order], rij[order]
        t = {}
        if self._full:
            PF, eat, t = self._full_graph(lay, Z, pi, pj, rij, nl, nt, P)
        else:
            rp_cpu = torch.searchsorted(pi, torch.arange(nl + 1, device=dev, dtype=pi.dtype)).cpu().numpy()
            bounds = [(a, min(a + C, nl)) for a in range(0, nl, C)]
            if self._loc is None or self._loc.numel() < nt:
                self._loc = torch.full((nt,), -1, dtype=torch.long, device=dev)
                self._flag = torch.zeros(nt, dtype=torch.bool, device=dev)
            topo = {}
            # ---------------- forward: X_0 (all atoms incl. ghosts), X_1..X_L by chunks + exchange
            t0 = time.perf_counter()
            with torch.no_grad():
                X = [lay.embed(Z)]
                for k in range(lay.L):
                    Xk = None
                    for a, b in bounds:
                        e0, e1 = int(rp_cpu[a]), int(rp_cpu[b])
                        if (a, b) not in topo:
                            topo[(a, b)] = self._topology(a, b, e0, e1, pi, pj)
                        nodes, ei = topo[(a, b)]
                        xk = lay.layer(k, X[k][nodes], Z[nodes], ei, rij[e0:e1].to(dt))[: b - a]
                        if Xk is None:
                            Xk = torch.zeros((nt,) + tuple(xk.shape[1:]), dtype=dt, device=dev)
                        Xk[a:b] = xk
                    if k < lay.L - 1:                                        # X_L is needed on local atoms only
                        Xx = torch.empty_like(Xk)
                        _exchange(data.forward_exchange, Xk.reshape(nt, -1), Xx.reshape(nt, -1), Xk[0].numel())
                        Xk = Xx
                    X.append(Xk)
            torch.cuda.synchronize()
            t["forward"] = time.perf_counter() - t0
            # ---------------- read-out
            XL = X[-1][:nl].clone().requires_grad_(True)
            with torch.enable_grad():
                e_at = lay.atomic_energy(XL, Z[:nl])
                (gL,) = torch.autograd.grad(e_at.sum(), [XL])
            eat = e_at.detach().double()
            G = torch.zeros((nt,) + tuple(gL.shape[1:]), dtype=dt, device=dev)
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
                        xk = lay.layer(k, xin, Z[nodes], ei, ev)[: b - a]
                        loss = (G[a:b] * xk).sum()
                        if k > 0:
                            gv, gx = torch.autograd.grad(loss, [ev, xin])
                            Gprev.index_add_(0, nodes, gx)
                        else:
                            (gv,) = torch.autograd.grad(loss, [ev])
                    PF[e0:e1] += gv.double()                                # dE/drij, as mliap_threepass.py
                    del ev, xin, xk, loss, gv
                X.pop()
                if k > 0:
                    Gr = torch.empty_like(Gprev)
                    _exchange(data.reverse_exchange, Gprev.reshape(nt, -1), Gr.reshape(nt, -1), Gprev[0].numel())
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
            raise RuntimeError(f"LPassESENMLIAP guard: non-finite or absurd output at call {self.step} "
                               f"(max |pair force| = {fmax:.3e} eV/A, limit {self._fmax_abort:g}); aborting.")
        torch.as_tensor(data.eatoms).copy_(eat)
        data.energy = float(eat.sum())
        data.update_pair_forces_gpu(PF)
        if self._log is not None:
            self._log.write(json.dumps(dict(step=self.step, nlocal=nl, ntotal=nt, npairs=P, full_graph=self._full,
                                            t_total=time.perf_counter() - t_start, **t,
                                            peak_gb=torch.cuda.max_memory_allocated() / 1e9)) + "\n")

    def _full_graph(self, lay, Z, pi, pj, rij, nl, nt, P):
        """Stock evaluation: the whole local + ghost graph in one forward with the complete autograd graph resident."""
        dt = lay.dt
        t0 = time.perf_counter()
        ei = torch.stack([pj, pi]).long()                                   # [sender, receiver], global local ids
        ev = rij.to(dt).requires_grad_(True)
        with torch.enable_grad():
            x = lay.embed(Z)
            for k in range(lay.L):
                x = lay.layer(k, x, Z, ei, ev)
            e_at = lay.atomic_energy(x[:nl], Z[:nl])
            (gv,) = torch.autograd.grad(e_at.sum(), [ev])
        PF = gv.double()
        torch.cuda.synchronize()
        return PF, e_at.detach().double(), {"full": time.perf_counter() - t0}

    def compute_descriptors(self, data):
        pass

    def compute_gradients(self, data):
        pass
