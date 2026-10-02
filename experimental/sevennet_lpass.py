#!/usr/bin/env python
"""SPEC F2 (feasibility): exact receiver-chunked evaluation of an L-layer NequIP-type network (SevenNet-0, 5 layers).

The MACE three-pass scheme generalises to any message-passing network in which the node features of layer k depend
on the layer-(k-1) features of the receiver and its neighbours only:

  forward,  k = 1..L : X_k[C] = layer_k(X_{k-1}[C + N(C)], edges into C)     chunk by chunk, no autograd
  readout            : E = sum_i e(X_L[i]),  G_L = dE/dX_L                     per atom
  backward, k = L..1 : recompute layer_k on each chunk with autograd, loss = sum_{i in C} G_k[i] . X_k[i]
                       -> pair forces from d/d(edge vectors), and G_{k-1}[C + N(C)] += d/dX_{k-1}

Resident memory: the per-atom features X_0..X_L and one G at a time; edge-level intermediates and the autograd graph
exist for one chunk only. 2L chunk passes (MACE, L = 2, fuses the last forward with its backward: 3 passes).
In LAMMPS each layer would need one forward and one reverse ghost exchange, as for H1/G1 in mliap_threepass.py.

    python sevennet_lpass.py --model 7net-0 --base water64_1x1x1.extxyz --rep 4 4 4 --chunk 2048 --dtype float64 --check
Runs in an environment with sevenn (>= 0.10) and torch; mace is not needed.
"""
import argparse
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from mace_onegpu.receiver_chunk_mace import System, _sync, _peak  # noqa: E402
from mace_onegpu.lpass_sevennet import LPassSevenNet, load_sevennet  # noqa: E402

GB = 1e9


def reference(model, atoms, cutoff, device):
    """Original SevenNet evaluation (full graph, autograd through everything) on the same model and dtype."""
    import sevenn.nn.node_embedding as ne
    from sevenn.atom_graph_data import AtomGraphData
    from sevenn.train.dataload import unlabeled_atoms_to_graph

    dt = next(model.parameters()).dtype
    if not getattr(ne, "_f2_patched", False):        # the one-hot module casts to float32; follow the model dtype
        orig = ne.OnehotEmbedding.forward

        def fwd(self, data):
            out = orig(self, data)
            out[self.key_output] = out[self.key_output].to(dt)
            if self.key_additional_output is not None:
                out[self.key_additional_output] = out[self.key_additional_output].to(dt)
            return out
        ne.OnehotEmbedding.forward, ne._f2_patched = fwd, True
    gnp = unlabeled_atoms_to_graph(atoms, cutoff)
    data = AtomGraphData.from_numpy_dict(gnp).to(device)
    # edge vectors at full precision: r_sender - r_receiver by minimum image (orthorhombic, box > 2 r_cut)
    pos = torch.tensor(atoms.get_positions(), dtype=torch.float64, device=device)
    L = torch.tensor(np.diag(np.array(atoms.get_cell())), dtype=torch.float64, device=device)
    ei = data["edge_index"]
    v = pos[ei[1]] - pos[ei[0]]
    v = v - L * torch.round(v / L)
    print(f"reference: |edge_vec(input) - edge_vec(fp64)|max = {float((data['edge_vec'].double() - v).abs().max()):.2e} A", flush=True)
    data["edge_vec"] = v.to(dt)
    for k in list(data.keys()):
        v = data[k]
        if isinstance(v, torch.Tensor) and torch.is_floating_point(v) and v.dtype != dt:
            data[k] = v.to(dt)
    # from_numpy_dict stores positions/cell in float32: take them from the Atoms object at full precision
    if "pos" in data:
        data["pos"] = torch.tensor(atoms.get_positions(), dtype=dt, device=device)
    if "cell_lattice_vectors" in data:
        data["cell_lattice_vectors"] = torch.tensor(np.array(atoms.get_cell()), dtype=dt, device=device).reshape(data["cell_lattice_vectors"].shape)
    model.train(False)
    model.set_is_batch_data(False)          # as sevenn's calculator: single graph, no 'batch' key
    out = model(data)
    return float(out["inferred_total_energy"].double().sum()), out["inferred_force"].detach().double()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="7net-0")
    ap.add_argument("--base", required=True)
    ap.add_argument("--rep", nargs=3, type=int, default=[1, 1, 1])
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--dtype", default="float64", choices=["float32", "float64"])
    ap.add_argument("--check", action="store_true", help="compare every atom with the original model on the base cell")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--stock", action="store_true", help="also run the original model on the full replica (memory)")
    ap.add_argument("--write-ref", default=None, help="write the fp64 reference of the original model for --data")
    ap.add_argument("--data", default=None, help="LAMMPS data file (types 1 = O, 2 = H) for --write-ref")
    a = ap.parse_args()
    from ase.io import read

    dev = a.device
    if a.write_ref:                       # reference forces for a LAMMPS data file (compare_forces.py format)
        dt = getattr(torch, a.dtype)
        model, cfg = load_sevennet(a.model, dt, dev)
        torch.set_default_dtype(dt)
        at = read(a.data, format="lammps-data", atom_style="atomic", Z_of_type={1: 8, 2: 1}, units="metal")
        E, F = reference(model, at, float(cfg["cutoff"]), dev)
        np.savez(a.write_ref, energy=E, forces=F.cpu().numpy(), positions=at.get_positions(), numbers=at.get_atomic_numbers())
        print(f"reference ({a.dtype}): N={len(at)} E={E:.6f} eV  Fmax={float(F.abs().max()):.4f} eV/A -> {a.write_ref}")
        return
    dt = getattr(torch, a.dtype)
    model, cfg = load_sevennet(a.model, dt, dev)     # load under the float32 default (sevenn's checkpoint patching)
    torch.set_default_dtype(dt)
    ev = LPassSevenNet(model, cfg, dt, a.chunk)
    base = read(a.base)
    big = base.repeat(a.rep)
    tm = model.type_map
    types = torch.tensor([tm[int(z)] for z in big.get_atomic_numbers()], device=dev)
    cell = np.diag(np.array(big.get_cell()))
    pos = torch.tensor(big.get_positions(), dtype=torch.float64, device=dev)
    sysm = System(pos, cell, types.to(torch.int16), ev.r_max, dev)
    sysm.build_neighbors()
    types_s = types[sysm.perm]
    N = sysm.N
    print(f"SevenNet {a.model}: L={ev.L} r_max={ev.r_max} dtype={a.dtype} N={N:,} edges/atom={float(sysm.NN.float().mean()):.1f} "
          f"chunk={a.chunk}", flush=True)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    _sync(); t0 = time.perf_counter()
    E, F, resident = ev.compute(sysm, types_s)
    _sync(); tt = time.perf_counter() - t0
    print(f"L-pass: E={E:.6f} eV ({E / N:.6f}/atom)  t={tt:.2f}s ({N / tt:.3e} atom-step/s)  peak={_peak() / GB:.2f} GB "
          f"({_peak() / N / 1e3:.1f} kB/atom)", flush=True)
    Fo = torch.empty_like(F)
    Fo[sysm.perm] = F
    if a.check:
        nb = len(base)
        Eb, Fb = reference(model, base, ev.r_max, dev)
        k = torch.arange(N, device=dev) % nb                    # ase.repeat: copies are contiguous blocks of nb atoms
        d = Fo - Fb[k]
        print(f"CHECK vs original SevenNet on the {nb}-atom base cell: dE/atom={E / N - Eb / nb:+.3e} eV  "
              f"max|dF|={float(d.abs().max()):.3e}  RMS(dF)={math.sqrt(float((d * d).mean())):.3e} eV/A  (all {N:,} atoms)",
              flush=True)
    if a.stock:
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        try:
            _sync(); t0 = time.perf_counter()
            Es, Fs = reference(model, big, ev.r_max, dev)
            _sync(); ts = time.perf_counter() - t0
            print(f"stock SevenNet on N={N:,}: E={Es:.6f}  t={ts:.2f}s  peak={_peak() / GB:.2f} GB ({_peak() / N / 1e3:.1f} kB/atom)  "
                  f"max|F_stock - F_Lpass|={float((Fs - Fo).abs().max()):.3e}", flush=True)
        except torch.cuda.OutOfMemoryError:
            print(f"stock SevenNet on N={N:,}: OUT OF MEMORY", flush=True)


if __name__ == "__main__":
    main()
