"""Layer-by-layer comparison of the chunked eSEN forward sweeps against the original forward (hooks), CPU, float64.
usage: python experimental/esen_debug_layers.py [--n 60] [--chunk 16]"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from esen_exactness_cpu import random_esen  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--box", type=float, default=10.0)
    ap.add_argument("--cutoff", type=float, default=4.0)
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    from ase import Atoms
    from ase.neighborlist import neighbor_list

    from mace_onegpu.lpass_esen import ESENLayers
    from mace_onegpu.receiver_chunk_mace import LocalGraph, System

    dtype = torch.float64
    torch.set_default_dtype(dtype)
    bb, eb, meta = random_esen(a.seed, a.cutoff, dtype)
    rng = np.random.default_rng(a.seed)
    pos = []
    while len(pos) < a.n:
        p = rng.uniform(0, a.box, 3)
        if all(np.linalg.norm((p - q + a.box / 2) % a.box - a.box / 2) > 1.0 for q in pos):
            pos.append(p)
    Z = rng.choice([1, 8], size=a.n, p=[2 / 3, 1 / 3])
    atoms = Atoms(numbers=Z, positions=np.array(pos), cell=np.eye(3) * a.box, pbc=True)

    # ---- original forward with hooks
    caps = []
    hooks = [bb.edge_degree_embedding.register_forward_hook(lambda m, i, o: caps.append(("edgedeg", o.detach().clone())))]
    for k, blk in enumerate(bb.blocks):
        hooks.append(blk.register_forward_hook(lambda m, i, o, k=k: caps.append((f"block{k + 1}", o.detach().clone()))))
    i, j, S = neighbor_list("ijS", atoms, float(bb.cutoff))
    N, E = len(atoms), len(i)
    P = torch.tensor(atoms.get_positions(), dtype=dtype)
    Zt = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long)
    batch = torch.zeros(N, dtype=torch.long)
    d = {"pos": P, "cell": torch.tensor(np.array(atoms.get_cell()), dtype=dtype).view(1, 3, 3), "atomic_numbers": Zt,
         "atomic_numbers_full": Zt, "batch": batch, "batch_full": batch, "natoms": torch.tensor([N]),
         "nedges": torch.tensor([E]), "edge_index": torch.tensor(np.stack([j, i]), dtype=torch.long),
         "cell_offsets": torch.tensor(S, dtype=dtype)}
    bb.otf_graph = False
    with torch.no_grad():
        out = bb(d)
    for h in hooks:
        h.remove()
    ref = {name: x for name, x in caps}
    ref["final_normed"] = out["node_embedding"]
    print("reference captured:", [(k, tuple(v.shape)) for k, v in ref.items()])

    # ---- chunked forward sweeps
    lay = ESENLayers(bb, eb, dtype, **meta)
    sysm = System(P.clone(), np.diag(np.array(atoms.get_cell())), Zt.to(torch.int16), float(bb.cutoff), "cpu")
    sysm.build_neighbors()
    Zs = Zt[sysm.perm]
    inv = torch.empty_like(sysm.perm)
    inv[sysm.perm] = torch.arange(N)
    loc = torch.full((N,), -1, dtype=torch.int32)
    names = ["edgedeg"] + [f"block{k}" for k in range(1, bb.num_layers + 1)]
    with torch.no_grad():
        X = [lay.embed(Zs)]
        for k in range(lay.L):
            Xk = None
            for s in range(0, N, a.chunk):
                e = min(s + a.chunk, N)
                g = LocalGraph(sysm, s, e, loc)
                xk = lay.layer(k, X[k][g.nodes], Zs[g.nodes], g.edge_index, (-g.vectors).to(dtype))[: g.nC]
                if Xk is None:
                    Xk = torch.empty((N,) + tuple(xk.shape[1:]), dtype=dtype)
                Xk[s:e] = xk
            X.append(Xk)
            r = ref[names[k]]
            mine = Xk[inv]                                     # back to the original atom order
            dmax = float((mine - r).abs().max())
            print(f"layer {k} ({names[k]}): max|dX| {dmax:.3e}  (|X|max {float(r.abs().max()):.3e})"
                  + ("" if dmax < 1e-10 else "   <-- differs"))
        fin = lay.bb.norm(X[-1])[inv]
        print(f"final norm: max|d| {float((fin - ref['final_normed']).abs().max()):.3e}")
    # per-edge check of layer 0 inputs: same edge set?
    print("edges: reference", E, "| System NL", int(sysm.NL.ge(0).sum()))


if __name__ == "__main__":
    main()
