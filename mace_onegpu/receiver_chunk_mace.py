#!/usr/bin/env python
"""
Receiver-chunked three-pass MACE energy + forces with O(N) persistent memory of a few kB per atom.

Design targets: accuracy identical to the original MACE, atom count >= 50% of NEP89 on one H200
(>= 9.3e6 atoms, i.e. <= 14.6 kB persistent per atom), speed >= 1/10 of NEP89 (>= 2e6 atom-step/s).
This file proves the MEMORY part exactly, in plain PyTorch (speed comes later with custom kernels).

Math. For a 2-layer MACE (ScaleShiftMACE, e.g. MACE-MPA-0) the energy is a sum of receiver-local terms
    E = sum_i [ e0(Z_i) + shift ] + scale * sum_i [ pair_i + R1(h1_i) + R2(h2_i) ]
    h1_i = P1( sum_{j in N(i)} phi1(r_ij, Z_i, Z_j) )                      (species-only input -> ACE density)
    h2_i = P2( sum_{j in N(i)} phi2(r_ij, h1_j, Z_i, Z_j), h1_i )
Only h1 (512 numbers per atom for MPA-0 medium) couples different receivers. Hence:
    pass A  (no grad, chunks of receivers)   : H1[i] = h1_i for all atoms                  -> global [N, 512]
    pass B  (grad, chunks of receivers C)    : E_B = scale * sum_{i in C} (pair_i + R2(h2_i)),
                                               d/d r_ij -> forces,  d/d h1_j -> G1[j] += ...  -> global [N, 512]
    pass C  (grad, chunks of receivers C)    : L_C = scale * sum_{i in C} R1(h1_i) + sum_{i in C} G1[i] . h1_i,
                                               d/d r_ij -> forces
Every wide intermediate (edge messages, 5120-wide node messages, symmetric-contraction tensors, autograd graph)
lives only for one chunk. Persistent state: positions, species, neighbour list (int32), H1, G1, forces.

Neighbour search: GPU cell list with Morton-ordered atoms (chunks are spatially compact) and orthorhombic
minimum image (requires every box length > 2 r_max). Small boxes fall back to brute force.

usage (GPU):
  validate : python -m mace_onegpu.receiver_chunk_mace --model M.model --structure water64_2x2x2_N1536.data --variant fp64 --validate
  scale    : python -m mace_onegpu.receiver_chunk_mace --model M.model --base water64_1x1x1.extxyz --rep 44 44 44 --variant fp32_fused --chunk 16384
"""
import argparse
import copy
import json
import math
import os
import time

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
import numpy as np  # noqa: E402
import torch  # noqa: E402

GB = 1e9
VERBOSE = os.environ.get("MACE_RC_VERBOSE", "0") == "1"


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _alloc():
    return torch.cuda.memory_allocated() if torch.cuda.is_available() else 0


def _peak():
    return torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0


# ----------------------------------------------------------------------------- model
def build_model(path, variant, device):
    from mace.cli.convert_e3nn_cueq import run as run_e3nn_to_cueq

    dt = torch.float64 if variant.startswith("fp64") else torch.float32
    torch.set_default_dtype(dt)
    m = torch.load(path, map_location=device)
    m = m.double() if dt == torch.float64 else m.float()
    if "e3nn" in variant:
        pass
    elif "fused" in variant:
        from .fused_conv import repair_fused_blocks

        m = run_e3nn_to_cueq(copy.deepcopy(m).to(device), device=device)
        repair_fused_blocks(m)
    else:
        m = run_e3nn_to_cueq(copy.deepcopy(m).to("cpu"), device="cpu")
    m = m.to(device).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    assert len(m.interactions) == 2, "prototype supports 2-layer MACE"
    return m, dt


# ----------------------------------------------------------------------------- structure
def morton3(cx, cy, cz, bits=10):
    code = torch.zeros_like(cx, dtype=torch.int64)
    for b in range(bits):
        code |= ((cx >> b) & 1) << (3 * b + 2)
        code |= ((cy >> b) & 1) << (3 * b + 1)
        code |= ((cz >> b) & 1) << (3 * b)
    return code


class System:
    """Positions sorted in Morton order of cells; perm maps sorted -> original index."""

    def __init__(self, pos, box, spidx, r_max, device):
        self.device = device
        self.box = torch.as_tensor(box, dtype=pos.dtype, device=device)
        assert bool((self.box > 2 * r_max).all()), "minimum image needs box > 2 r_max in every direction"
        pos = torch.remainder(pos, self.box)
        self.ncell = torch.clamp((self.box / r_max).floor().long(), min=1)
        c = torch.minimum((pos / (self.box / self.ncell)).floor().long(), self.ncell - 1)
        code = morton3(c[:, 0], c[:, 1], c[:, 2])
        order = torch.argsort(code, stable=True)
        self.perm = order
        self.pos = pos[order].contiguous()
        self.spidx = spidx[order].contiguous()
        self.cell = c[order].to(torch.int16).contiguous()
        self.N = self.pos.shape[0]
        self.r_max = r_max

    def min_image(self, v):
        return v - self.box * torch.round(v / self.box)

    # --- neighbour list: NL[i, :NN[i]] = senders j with 0 < |x_i - x_j|_mic < r_max
    def build_neighbors(self, chunk=8192, max_mn=None):
        N, dev = self.N, self.device
        nc = self.ncell
        brute = bool((nc < 3).any())
        if not brute:
            cl64 = self.cell.long()
            lin = (cl64[:, 0] * nc[1] + cl64[:, 1]) * nc[2] + cl64[:, 2]
            del cl64
            ncells = int(nc.prod())
            counts = torch.bincount(lin, minlength=ncells)
            start = torch.full((ncells,), N, dtype=torch.long, device=dev)
            start.scatter_reduce_(0, lin, torch.arange(N, device=dev), reduce="amin")
            mcc = int(counts.max())
            offs = torch.stack(torch.meshgrid(*[torch.arange(-1, 2, device=dev)] * 3, indexing="ij"), -1).reshape(27, 3)
        r2 = self.r_max ** 2

        def candidates(a, b):
            ri = torch.arange(a, b, device=dev)
            if brute:
                cand = torch.arange(N, device=dev).unsqueeze(0).expand(b - a, N)
                valid = torch.ones_like(cand, dtype=torch.bool)
            else:
                cc = (self.cell[a:b].long().unsqueeze(1) + offs.unsqueeze(0)) % nc          # [n,27,3]
                cl = (cc[..., 0] * nc[1] + cc[..., 1]) * nc[2] + cc[..., 2]           # [n,27]
                st, ct = start[cl], counts[cl]
                k = torch.arange(mcc, device=dev)
                cand = (st.unsqueeze(-1) + k).reshape(b - a, -1)                      # [n,27*mcc]
                valid = (k < ct.unsqueeze(-1)).reshape(b - a, -1)
                cand = torch.where(valid, cand, torch.zeros_like(cand))
            d = self.min_image(self.pos[ri].unsqueeze(1) - self.pos[cand])
            dd = (d * d).sum(-1)
            ok = valid & (dd < r2) & (cand != ri.unsqueeze(1))
            return cand, ok

        # count pass
        NN = torch.empty(N, dtype=torch.int32, device=dev)
        for a in range(0, N, chunk):
            b = min(a + chunk, N)
            _, ok = candidates(a, b)
            NN[a:b] = ok.sum(1).to(torch.int32)
        MN = int(NN.max()) if max_mn is None else max_mn
        NL = torch.full((N, MN), -1, dtype=torch.int32, device=dev)
        for a in range(0, N, chunk):
            b = min(a + chunk, N)
            cand, ok = candidates(a, b)
            key = (~ok).to(torch.int8)
            idx = torch.argsort(key, dim=1, stable=True)[:, :MN]
            sel = torch.gather(cand, 1, idx)
            okk = torch.gather(ok, 1, idx)
            NL[a:b] = torch.where(okk, sel, torch.full_like(sel, -1)).to(torch.int32)
        self.NL, self.NN, self.MN = NL, NN, MN
        return MN


# ----------------------------------------------------------------------------- local graph for a receiver chunk
class LocalGraph:
    """Receiver chunk [a, b) with its senders. Optionally padded to fixed shapes (e_cap edges, node_bucket
    multiples) so JIT-compiled cuEquivariance kernels see only a handful of distinct shapes. Padding edges point
    from/to a dummy node, have length 1.5 r_max (cutoff = 0, radial features = 0, bias-free radial MLPs -> zero
    message, zero density, zero ZBL) and are excluded from force / G1 accumulation."""

    def __init__(self, sysm, a, b, loc, e_cap=None, node_bucket=None):
        dev = sysm.device
        C = torch.arange(a, b, device=dev)
        nC = b - a
        nl = sysm.NL[a:b].long()
        mask = nl >= 0
        recv_g = C.unsqueeze(1).expand_as(nl)[mask]
        send_g = nl[mask]
        loc[C] = torch.arange(nC, device=dev, dtype=loc.dtype)
        su = torch.unique(send_g)
        new = su[loc[su] < 0]
        loc[new] = torch.arange(nC, nC + new.numel(), device=dev, dtype=loc.dtype)
        self.nodes = torch.cat([C, new])
        edge_index = torch.stack([loc[send_g].long(), loc[recv_g].long()])
        loc[self.nodes] = -1
        self.send_g, self.recv_g = send_g, recv_g
        self.nC = nC
        self.n_nodes = self.nodes.numel()
        self.nE = send_g.numel()
        vectors = sysm.min_image(sysm.pos[recv_g] - sysm.pos[send_g])
        if node_bucket:
            n_pad = ((self.n_nodes + 1 + node_bucket - 1) // node_bucket) * node_bucket
            fill = torch.full((n_pad - self.n_nodes,), int(self.nodes[0]), dtype=self.nodes.dtype, device=dev)
            self.nodes_padded = torch.cat([self.nodes, fill])
            dummy = n_pad - 1
        else:
            self.nodes_padded = self.nodes
            dummy = None
        if e_cap and dummy is not None and e_cap > self.nE:
            k = e_cap - self.nE
            # spread the (exactly zero) padding edges over distinct nodes as self-loops: pointing them all at one
            # dummy row caused ~1e5-way atomic-add contention in the fused scatter
            tgt = torch.arange(k, device=dev) % n_pad
            pad_ei = torch.stack([tgt, tgt])
            edge_index = torch.cat([edge_index, pad_ei], 1)
            pv = torch.zeros(k, 3, dtype=vectors.dtype, device=dev)
            pv[:, 0] = 1.5 * sysm.r_max
            vectors = torch.cat([vectors, pv], 0)
        self.edge_index = edge_index
        # vectors = x_recv - x_send (+ minimum-image shift), as in mace.modules.utils.get_edge_vectors_and_lengths
        self.vectors = vectors


def node_attrs_of(model, spidx_nodes, dtype):
    return torch.nn.functional.one_hot(spidx_nodes.long(), num_classes=len(model.atomic_numbers)).to(dtype)


def edge_embed(model, g, node_attrs, vectors):
    lengths = torch.linalg.vector_norm(vectors, dim=-1, keepdim=True)
    edge_attrs = model.spherical_harmonics(vectors)
    edge_feats, cutoff = model.radial_embedding(lengths, node_attrs, g.edge_index, model.atomic_numbers)
    return lengths, edge_attrs, edge_feats, cutoff


def layer1(model, g, node_attrs, vectors):
    _, edge_attrs, edge_feats, cutoff = edge_embed(model, g, node_attrs, vectors)
    node_feats = model.node_embedding(node_attrs)
    nf, sc = model.interactions[0](node_attrs=node_attrs, node_feats=node_feats, edge_attrs=edge_attrs,
                                   edge_feats=edge_feats, edge_index=g.edge_index, cutoff=cutoff, first_layer=True)
    nC = g.nC
    return model.products[0](node_feats=nf[:nC], sc=None if sc is None else sc[:nC], node_attrs=node_attrs[:nC])


def layer2_energy(model, g, node_attrs, vectors, h1_local, heads):
    lengths, edge_attrs, edge_feats, cutoff = edge_embed(model, g, node_attrs, vectors)
    nf, sc = model.interactions[1](node_attrs=node_attrs, node_feats=h1_local, edge_attrs=edge_attrs,
                                   edge_feats=edge_feats, edge_index=g.edge_index, cutoff=cutoff, first_layer=False)
    nC = g.nC
    h2 = model.products[1](node_feats=nf[:nC], sc=None if sc is None else sc[:nC], node_attrs=node_attrs[:nC])
    ar = torch.arange(nC, device=h2.device)
    e = model.readouts[1](h2, heads)[ar, heads]
    if hasattr(model, "pair_repulsion"):
        e = e + model.pair_repulsion_fn(lengths, node_attrs, g.edge_index, model.atomic_numbers)[:nC]
    return e


# ----------------------------------------------------------------------------- three-pass evaluator
class ReceiverChunkMACE:
    def __init__(self, model, dtype, chunk, pad=True, node_bucket=8192):
        self.m, self.dt, self.chunk = model, dtype, chunk
        self.pad, self.node_bucket = pad, node_bucket
        self.e_cap = None
        heads0 = torch.zeros(1, dtype=torch.long, device=next(model.parameters()).device)
        z = torch.zeros(1, dtype=dtype, device=heads0.device)
        self.shift = float(model.scale_shift(z, heads0))
        self.scale = float(model.scale_shift(z + 1, heads0)) - self.shift
        self.h1_dim = None

    def _accum_forces(self, F, g, gv):
        # v = x_recv - x_send  ->  dE/dx_recv += gv, dE/dx_send -= gv ;  F = -dE/dx
        gv = gv[: g.nE]
        F.index_add_(0, g.recv_g, -gv.to(F.dtype))
        F.index_add_(0, g.send_g, gv.to(F.dtype))

    def compute(self, sysm, log=print):
        m, dt, dev = self.m, self.dt, sysm.device
        N = sysm.N
        loc = torch.full((N,), -1, dtype=torch.int32, device=dev)
        if self.pad:
            self.e_cap = ((self.chunk * sysm.MN + 65535) // 65536) * 65536
        F = torch.zeros(N, 3, dtype=torch.float64, device=dev)
        E = 0.0
        t = {"A": 0.0, "B": 0.0, "C": 0.0}
        # e0 + shift (no forces)
        with torch.no_grad():
            for a in range(0, N, 1 << 20):
                b = min(a + (1 << 20), N)
                na = node_attrs_of(m, sysm.spidx[a:b], dt)
                heads = torch.zeros(b - a, dtype=torch.long, device=dev)
                E += float(m.atomic_energies_fn(na)[torch.arange(b - a, device=dev), heads].double().sum())
        E += self.shift * N

        # pass A: H1 for all atoms
        _sync(); t0 = time.perf_counter()
        H1 = None
        with torch.no_grad():
            for a in range(0, N, self.chunk):
                b = min(a + self.chunk, N)
                _sync(); _tc = time.perf_counter()
                g = LocalGraph(sysm, a, b, loc, self.e_cap, self.node_bucket if self.pad else None)
                if VERBOSE:
                    print(f'[chunk A] a={a} nC={g.nC} nodes={g.n_nodes} pad_nodes={g.nodes_padded.numel()} edges={g.nE} padded_edges={g.edge_index.shape[1]}', flush=True)
                    _g_pass = 'A'
                na = node_attrs_of(m, sysm.spidx[g.nodes_padded], dt)
                h1 = layer1(m, g, na, g.vectors.to(dt))
                if H1 is None:
                    self.h1_dim = h1.shape[1]
                    H1 = torch.empty(N, self.h1_dim, dtype=dt, device=dev)
                H1[a:b] = h1
                del g, na, h1
                if VERBOSE:
                    _sync(); print(f'[chunk A] a={a} done {time.perf_counter()-_tc:.3f}s', flush=True)
        _sync(); t["A"] = time.perf_counter() - t0
        log(f"  pass A done: H1 {tuple(H1.shape)}  alloc={_alloc()/GB:.2f} GB  t={t['A']:.1f}s")

        # pass B: layer 2 energy, forces from layer-2 edges, G1 = dE_B/dH1
        G1 = torch.zeros_like(H1)
        t0 = time.perf_counter()
        EB = 0.0
        ctimes = []
        for a in range(0, N, self.chunk):
            _sync(); tc = time.perf_counter()
            b = min(a + self.chunk, N)
            _sync(); _tc = time.perf_counter()
            g = LocalGraph(sysm, a, b, loc, self.e_cap, self.node_bucket if self.pad else None)
            if VERBOSE:
                print(f'[chunk B] a={a} nC={g.nC} nodes={g.n_nodes} pad_nodes={g.nodes_padded.numel()} edges={g.nE} padded_edges={g.edge_index.shape[1]}', flush=True)
                _g_pass = 'B'
            na = node_attrs_of(m, sysm.spidx[g.nodes_padded], dt)
            vec = g.vectors.to(dt).requires_grad_(True)
            h1l = H1[g.nodes_padded].detach().requires_grad_(True)
            heads = torch.zeros(g.nC, dtype=torch.long, device=dev)
            with torch.enable_grad():
                e = layer2_energy(m, g, na, vec, h1l, heads)
                eb = self.scale * e.sum()
                gv, gh = torch.autograd.grad(eb, [vec, h1l])
            EB += float(eb.detach().double())
            G1.index_add_(0, g.nodes, gh[: g.n_nodes])
            self._accum_forces(F, g, gv)
            del g, na, vec, h1l, e, eb, gv, gh
            if VERBOSE:
                _sync(); print(f'[chunk B] a={a} done {time.perf_counter()-_tc:.3f}s', flush=True)
            _sync(); ctimes.append(time.perf_counter() - tc)
        _sync(); t["B"] = time.perf_counter() - t0
        if ctimes:
            log(f"  pass B chunk times: first {[round(x,2) for x in ctimes[:4]]} median {sorted(ctimes)[len(ctimes)//2]:.3f}s over {len(ctimes)} chunks")
        log(f"  pass B done: alloc={_alloc()/GB:.2f} GB  peak={_peak()/GB:.2f} GB  t={t['B']:.1f}s")

        # pass C: layer 1 with grad, loss = scale*R1(h1) + G1 . h1
        t0 = time.perf_counter()
        EC = 0.0
        for a in range(0, N, self.chunk):
            b = min(a + self.chunk, N)
            _sync(); _tc = time.perf_counter()
            g = LocalGraph(sysm, a, b, loc, self.e_cap, self.node_bucket if self.pad else None)
            if VERBOSE:
                print(f'[chunk C] a={a} nC={g.nC} nodes={g.n_nodes} pad_nodes={g.nodes_padded.numel()} edges={g.nE} padded_edges={g.edge_index.shape[1]}', flush=True)
                _g_pass = 'C'
            na = node_attrs_of(m, sysm.spidx[g.nodes_padded], dt)
            vec = g.vectors.to(dt).requires_grad_(True)
            heads = torch.zeros(g.nC, dtype=torch.long, device=dev)
            with torch.enable_grad():
                h1 = layer1(m, g, na, vec)
                r1 = m.readouts[0](h1, heads)[torch.arange(g.nC, device=dev), heads]
                ec = self.scale * r1.sum()
                loss = ec + (G1[a:b] * h1).sum()
                (gv,) = torch.autograd.grad(loss, [vec])
            EC += float(ec.detach().double())
            self._accum_forces(F, g, gv)
            del g, na, vec, h1, r1, ec, loss, gv
            if VERBOSE:
                _sync(); print(f'[chunk C] a={a} done {time.perf_counter()-_tc:.3f}s', flush=True)
        _sync(); t["C"] = time.perf_counter() - t0
        log(f"  pass C done: peak={_peak()/GB:.2f} GB  t={t['C']:.1f}s")
        E += EB + EC
        persistent = {
            "pos": sysm.pos.numel() * sysm.pos.element_size(), "spidx": sysm.spidx.numel() * sysm.spidx.element_size(),
            "NL": sysm.NL.numel() * 4, "NN": N * 4, "H1": H1.numel() * H1.element_size(),
            "G1": G1.numel() * G1.element_size(), "F": F.numel() * 8, "loc": N * 4, "perm": N * 8,
            "cell": sysm.cell.numel() * sysm.cell.element_size(),
        }
        del H1, G1
        return E, F, t, persistent


# ----------------------------------------------------------------------------- main
def load_structure(args, model, dt, dev):
    from ase import io as aseio

    zs = [int(z) for z in model.atomic_numbers]
    if args.structure:
        p = args.structure
        if p.endswith(".data"):
            try:
                at = aseio.read(p, format="lammps-data", atom_style="atomic", Z_of_type={1: 8, 2: 1}, units="metal")
            except TypeError:
                at = aseio.read(p, format="lammps-data", style="atomic", Z_of_type={1: 8, 2: 1}, units="metal")
        else:
            at = aseio.read(p)
        cellm = np.array(at.get_cell())
        assert np.allclose(cellm, np.diag(np.diag(cellm))), "orthorhombic cells only"
        pos = torch.tensor(at.get_positions(), dtype=torch.float64, device=dev)  # fp64 coordinates, as LAMMPS
        box = np.diag(cellm)
        sp = torch.tensor([zs.index(int(z)) for z in at.get_atomic_numbers()], dtype=torch.int16, device=dev)
        return pos, box, sp, at
    base = aseio.read(args.base)
    cellm = np.array(base.get_cell())
    L0 = np.diag(cellm)
    nx, ny, nz = args.rep
    p0 = torch.tensor(base.get_positions(), dtype=torch.float64, device=dev)
    s0 = torch.tensor([zs.index(int(z)) for z in base.get_atomic_numbers()], dtype=torch.int16, device=dev)
    g = torch.stack(torch.meshgrid(torch.arange(nx, device=dev), torch.arange(ny, device=dev),
                                   torch.arange(nz, device=dev), indexing="ij"), -1).reshape(-1, 3).to(torch.float64)
    shift = g * torch.tensor(L0, dtype=torch.float64, device=dev)
    pos = (shift.unsqueeze(1) + p0.unsqueeze(0)).reshape(-1, 3)
    sp = s0.repeat(g.shape[0])
    return pos, L0 * np.array([nx, ny, nz]), sp, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--structure", default=None)
    ap.add_argument("--base", default=None)
    ap.add_argument("--rep", nargs=3, type=int, default=[1, 1, 1])
    ap.add_argument("--variant", default="fp32_fused")
    ap.add_argument("--chunk", type=int, default=8192)
    ap.add_argument("--validate", action="store_true", help="compare with the standard full-graph MACE evaluation")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--no-pad", action="store_true")
    ap.add_argument("--check-replica", action="store_true", help="with --base: compare every atom's force with the base cell")
    ap.add_argument("--json", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    dev = a.device
    model, dt = build_model(a.model, a.variant, dev)
    base_mem = _alloc()
    pos, box, sp, atoms = load_structure(a, model, dt, dev)
    N = pos.shape[0]
    print(f"variant={a.variant} N={N:,} box={np.round(box,3)} chunk={a.chunk}  model resident {base_mem/GB:.3f} GB")
    t0 = time.perf_counter()
    sysm = System(pos, box, sp, float(model.r_max), dev)
    del pos, sp
    MN = sysm.build_neighbors()
    _sync()
    tnl = time.perf_counter() - t0
    nedges = int(sysm.NN.sum())
    print(f"neighbour list: MN={MN} edges={nedges:,} ({nedges/N:.2f}/atom) t={tnl:.1f}s  alloc={_alloc()/GB:.2f} GB")
    ev = ReceiverChunkMACE(model, dt, a.chunk, pad=not a.no_pad)
    res = None
    for rep in range(a.repeat):
        torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
        t0 = time.perf_counter()
        E, F, t, pers = ev.compute(sysm)
        _sync()
        tt = time.perf_counter() - t0
        peak = _peak()
        res = dict(variant=a.variant, N=N, edges=nedges, MN=MN, chunk=a.chunk, energy=E, t_total=tt, t_pass=t,
                   peak_gb=peak / GB, persistent_bytes=pers, persistent_per_atom=sum(pers.values()) / N,
                   atom_step_per_s=N / tt)
        print(f"[rep {rep}] E={E:.6f} eV  t={tt:.2f}s ({N/tt:.3e} atom-step/s)  peak={peak/GB:.2f} GB  "
              f"persistent={sum(pers.values())/N:.0f} B/atom  ({ {k: round(v/N,1) for k,v in pers.items()} })")
    if a.validate:
        from .ase_gradient_probe import make_batch
        from ase import Atoms

        if atoms is None:
            zs = [int(z) for z in model.atomic_numbers]
            atoms = Atoms(numbers=[zs[int(s)] for s in sysm.spidx.cpu()], positions=sysm.pos.double().cpu().numpy(),
                          cell=np.diag(box), pbc=True)
            perm = None
        else:
            perm = sysm.perm.cpu().numpy()
        batch = make_batch(model, atoms, dev)
        out = model(batch.to_dict(), training=False, compute_force=True)
        Eref = float(out["energy"].detach().double().sum())
        Fref = out["forces"].detach().double().cpu().numpy()
        Fp = F.cpu().numpy()
        if perm is not None:
            Fref = Fref[perm]
        d = Fp - Fref
        print(f"VALIDATE vs full-graph MACE: dE={E-Eref:+.3e} eV ({(E-Eref)/N:+.2e}/atom)  max|dF|={np.abs(d).max():.3e}  "
              f"RMS(dF)={np.sqrt((d**2).mean()):.3e} eV/A   (E_ref={Eref:.6f})")
        res["validate"] = dict(dE=E - Eref, max_dF=float(np.abs(d).max()), rms_dF=float(np.sqrt((d ** 2).mean())))
    if a.check_replica and a.base:
        # Exactness at any N: a periodic replica of the base cell must reproduce, atom by atom, the forces the
        # ORIGINAL full-graph MACE gives on the base cell, and E/N must equal E_base/N_base.
        from ase import io as aseio

        from .ase_gradient_probe import make_batch

        base = aseio.read(a.base)
        nb = len(base)
        bb = make_batch(model, base, dev)
        out = model(bb.to_dict(), training=False, compute_force=True)
        Eb = float(out["energy"].detach().double().sum())
        Fb = out["forces"].detach().double()
        k = (sysm.perm % nb).to(Fb.device)          # sorted index -> base-cell atom
        maxd, sq = 0.0, 0.0
        for s in range(0, N, 1 << 22):
            e = min(s + (1 << 22), N)
            d = F[s:e].to(Fb.device) - Fb[k[s:e]]
            maxd = max(maxd, float(d.abs().max()))
            sq += float((d * d).sum())
        rms = math.sqrt(sq / (3 * N))
        print(f"REPLICA CHECK vs full-graph MACE on the {nb}-atom base cell: every one of {N:,} atoms compared  "
              f"max|dF|={maxd:.3e}  RMS(dF)={rms:.3e} eV/A   E/N - E_base/N_base = {E/N - Eb/nb:+.3e} eV/atom")
        res["replica_check"] = dict(max_dF=maxd, rms_dF=rms, dE_per_atom=E / N - Eb / nb, E_base=Eb)
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
