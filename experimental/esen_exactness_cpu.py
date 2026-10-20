"""CPU exactness test of the receiver-chunked eSEN evaluation (lpass_esen.py) against fairchem's own full-graph
pipeline, with a randomly initialised small eSEN (no checkpoint needed) or a real checkpoint (--checkpoint).

Checks, all in float64:
  1. full-graph forces/energy of the same cell: fairchem graph + backbone forward + autograd  vs  chunked evaluator
  2. replica check: chunked evaluation of an n x n x n replica vs the base-cell reference mapped onto every image
usage: python experimental/esen_exactness_cpu.py [--checkpoint esen_30m_oam.pt] [--n 60] [--chunk 16] [--rep 2]
"""
import argparse
import sys
import os
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def random_esen(seed, cutoff, dtype):
    from fairchem.core.models.esen.esen import eSEN_Backbone

    torch.manual_seed(seed)
    bb = eSEN_Backbone(num_layers=2, sphere_channels=32, hidden_channels=48, lmax=2, mmax=2, cutoff=cutoff,
                       max_neighbors=300, otf_graph=True, use_pbc=True, direct_forces=False, regress_forces=True,
                       num_distance_basis=64, edge_channels=32)
    eb = torch.nn.Sequential(torch.nn.Linear(32, 48), torch.nn.SiLU(), torch.nn.Linear(48, 1))
    with torch.no_grad():                       # non-trivial random weights everywhere
        for p in list(bb.parameters()) + list(eb.parameters()):
            p.add_(0.3 * torch.randn_like(p))
    bb.direct_forces = False
    return bb.to(dtype).eval(), eb.to(dtype).eval(), dict(rmsd=1.0, mean=0.0, refs=None)


def reference_full_graph(bb, eb, atoms, dtype, meta):
    """Independent reference: ASE's periodic neighbour list (all images within the cutoff) fed to the backbone's own
    forward (otf_graph=False path: edge vectors from pos, cell and cell offsets inside fairchem); forces by autograd
    of the total energy with respect to the positions."""
    from ase.neighborlist import neighbor_list

    i, j, S = neighbor_list("ijS", atoms, float(bb.cutoff))       # vector = pos[j] + S cell - pos[i]
    N, E = len(atoms), len(i)
    pos = torch.tensor(atoms.get_positions(), dtype=dtype).requires_grad_(True)
    Z = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long)
    batch = torch.zeros(N, dtype=torch.long)
    d = {"pos": pos, "cell": torch.tensor(np.array(atoms.get_cell()), dtype=dtype).view(1, 3, 3),
         "atomic_numbers": Z, "atomic_numbers_full": Z, "batch": batch, "batch_full": batch,
         "natoms": torch.tensor([N]), "nedges": torch.tensor([E]),
         "edge_index": torch.tensor(np.stack([j, i]), dtype=torch.long),   # [source = sender j, target = receiver i]
         "cell_offsets": torch.tensor(S, dtype=dtype)}
    otf = bb.otf_graph
    bb.otf_graph = False
    try:
        out = bb(d)
    finally:
        bb.otf_graph = otf
    x = out["node_embedding"]                     # the backbone's forward already applied the final norm
    e_at = eb(x.narrow(1, 0, 1).squeeze(1)).reshape(-1) * meta["rmsd"]
    if meta["refs"] is not None:
        e_at = e_at + meta["refs"].to(e_at.dtype)[Z]
    Etot = e_at.sum()
    (gpos,) = torch.autograd.grad(Etot, [pos])
    return float(Etot) + meta["mean"], (-gpos).detach().double(), E


def chunked(bb, eb, atoms, dtype, meta, chunk):
    from mace_onegpu.lpass_esen import LPassESEN
    from mace_onegpu.receiver_chunk_mace import System

    pos = torch.tensor(atoms.get_positions(), dtype=torch.float64)
    Z = torch.tensor(atoms.get_atomic_numbers())
    sysm = System(pos, np.diag(np.array(atoms.get_cell())), Z.to(torch.int16), float(bb.cutoff), "cpu")
    sysm.build_neighbors()
    Zs = Z[sysm.perm].long()
    ev = LPassESEN(bb, eb, dtype, chunk, **meta)
    E, Fs, resident = ev.compute(sysm, Zs, log=print)
    F = torch.empty_like(Fs)
    F[sysm.perm] = Fs                                   # back to the original atom order
    return E, F, int(sysm.NL.ge(0).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--n", type=int, default=60, help="atoms in the random test cell")
    ap.add_argument("--box", type=float, default=10.0)
    ap.add_argument("--cutoff", type=float, default=4.0, help="cutoff of the random model (ignored with a checkpoint)")
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--rep", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--frame", default="fixed", choices=["fixed", "random"],
                    help="edge frame: fixed (deterministic, default) or fairchem's random vector")
    a = ap.parse_args()
    from ase import Atoms

    from mace_onegpu.lpass_esen import install_edge_frame

    print("edge frame:", install_edge_frame(a.frame))
    dtype = torch.float64
    torch.set_default_dtype(dtype)
    if a.checkpoint:
        from mace_onegpu.lpass_esen import load_esen

        bb, eb, meta = load_esen(a.checkpoint, dtype, "cpu")
        meta = {k: meta[k] for k in ("rmsd", "mean", "refs")}
    else:
        bb, eb, meta = random_esen(a.seed, a.cutoff, dtype)
    rng = np.random.default_rng(a.seed)
    # random H/O mixture with a minimum separation of 1.0 A (random positions, rejection)
    pos = []
    while len(pos) < a.n:
        p = rng.uniform(0, a.box, 3)
        if all(np.linalg.norm((p - q + a.box / 2) % a.box - a.box / 2) > 1.0 for q in pos):
            pos.append(p)
    Z = rng.choice([1, 8], size=a.n, p=[2 / 3, 1 / 3])
    atoms = Atoms(numbers=Z, positions=np.array(pos), cell=np.eye(3) * a.box, pbc=True)
    print(f"model: blocks={bb.num_layers} cutoff={bb.cutoff} lmax={bb.lmax} sphere={bb.sphere_channels} | cell {a.n} atoms, "
          f"box {a.box} A, chunk {a.chunk}")

    t0 = time.perf_counter()
    E0, F0, nE0 = reference_full_graph(bb, eb, atoms, dtype, meta)
    t_ref = time.perf_counter() - t0
    t0 = time.perf_counter()
    E1, F1, nE1 = chunked(bb, eb, atoms, dtype, meta, a.chunk)
    t_chk = time.perf_counter() - t0
    dF = float((F1 - F0).abs().max())
    print(f"[1] full graph vs chunked: edges {nE0} vs {nE1} | E {E0:.10f} vs {E1:.10f} (|dE| {abs(E1 - E0):.2e} eV) | "
          f"max|dF| {dF:.2e} eV/A | |F|max {float(F0.abs().max()):.3f} | {t_ref:.1f}s vs {t_chk:.1f}s")
    ok1 = dF < 1e-9 and abs(E1 - E0) < 1e-8

    big = atoms.repeat(a.rep)
    E2, F2, nE2 = chunked(bb, eb, big, dtype, meta, a.chunk)
    nrep = a.rep ** 3
    Fmap = F0.repeat(nrep, 1)                           # ase repeat: image-major order, atom = index % n
    dF2 = float((F2 - Fmap).abs().max())
    print(f"[2] replica {a.rep}x{a.rep}x{a.rep} ({len(big)} atoms, {nE2} edges): E/atom {E2 / len(big):.10f} vs "
          f"{E0 / a.n:.10f} (|d| {abs(E2 / len(big) - E0 / a.n):.2e}) | max|dF| {dF2:.2e} eV/A")
    ok2 = dF2 < 1e-9 and abs(E2 / len(big) - E0 / a.n) < 1e-8
    print("RESULT:", "PASS" if (ok1 and ok2) else "FAIL")
    return 0 if (ok1 and ok2) else 1


if __name__ == "__main__":
    sys.exit(main())
