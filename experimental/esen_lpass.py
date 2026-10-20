#!/usr/bin/env python
"""Standalone measurements of the exact receiver-chunked eSEN evaluation (mace_onegpu/lpass_esen.py), GPU.

  --check      chunked evaluation of a replica (--rep) vs the original model on the base cell, float64, every atom
  --official   also the official OCPCalculator (fp32, fairchem's own graph) on the base cell vs our fp64 reference
  --noise      original forward twice with fairchem's random edge frame: gauge noise of the released model
  --scan N...  chunked fp32 evaluation of replicas with about N atoms: time, peak memory (capacity)
  --stock N... original full-graph evaluation (fp32) of replicas with about N atoms, until out of memory
  --write-ref  fp64 reference forces of the original model for a LAMMPS data file (compare_forces.py format)
Examples:
  python experimental/esen_lpass.py --checkpoint esen_30m_oam.pt --base water64_1x1x1.extxyz --rep 4 4 4 --chunk 2048 --check --official --noise
  python experimental/esen_lpass.py --checkpoint esen_30m_oam.pt --base water64_1x1x1.extxyz --scan 12288 98304 192000 983040 1500000 --chunk 8192
"""
import argparse
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from mace_onegpu.lpass_esen import ESENLayers, LPassESEN, install_edge_frame, load_esen  # noqa: E402
from mace_onegpu.receiver_chunk_mace import System  # noqa: E402

GB = 1e9


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _peak():
    return torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0


def reference(bb, eb, meta, atoms, dtype, device):
    """Original model (same modules and weights) on the whole cell: ASE periodic neighbour list, fairchem's forward
    (otf_graph=False path), autograd forces. Returns E (eV, incl. the per-structure mean) and F (N, 3) float64."""
    from ase.neighborlist import neighbor_list

    lay = ESENLayers(bb, eb, dtype, rmsd=meta["rmsd"], mean=meta["mean"], refs=meta["refs"])
    i, j, S = neighbor_list("ijS", atoms, float(bb.cutoff))
    N, E = len(atoms), len(i)
    pos = torch.tensor(atoms.get_positions(), dtype=dtype, device=device).requires_grad_(True)
    Z = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long, device=device)
    batch = torch.zeros(N, dtype=torch.long, device=device)
    d = {"pos": pos, "cell": torch.tensor(np.array(atoms.get_cell()), dtype=dtype, device=device).view(1, 3, 3),
         "atomic_numbers": Z, "atomic_numbers_full": Z, "batch": batch, "batch_full": batch,
         "natoms": torch.tensor([N], device=device), "nedges": torch.tensor([E], device=device),
         "edge_index": torch.tensor(np.stack([j, i]), dtype=torch.long, device=device),
         "cell_offsets": torch.tensor(S, dtype=dtype, device=device)}
    otf = bb.otf_graph
    bb.otf_graph = False
    try:
        out = bb(d)
    finally:
        bb.otf_graph = otf
    x = out["node_embedding"]
    e_at = lay.eb(x.narrow(1, 0, 1).squeeze(1)).reshape(-1) * lay.rmsd
    if lay.refs is not None:
        e_at = e_at + lay.refs[Z].to(e_at.dtype)
    Etot = e_at.sum()
    (gpos,) = torch.autograd.grad(Etot, [pos])
    return float(Etot) + lay.mean, (-gpos).detach().double(), E


def chunked(bb, eb, meta, atoms, dtype, device, chunk, log=print):
    pos = torch.tensor(atoms.get_positions(), dtype=torch.float64, device=device)
    Z = torch.tensor(atoms.get_atomic_numbers(), device=device)
    sysm = System(pos, np.diag(np.array(atoms.get_cell())), Z.to(torch.int16), float(bb.cutoff), device)
    sysm.build_neighbors()
    Zs = Z[sysm.perm].long()
    ev = LPassESEN(bb, eb, dtype, chunk, rmsd=meta["rmsd"], mean=meta["mean"], refs=meta["refs"])
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    _sync(); t0 = time.perf_counter()
    E, Fs, resident = ev.compute(sysm, Zs, log=log)
    _sync(); tt = time.perf_counter() - t0
    F = torch.empty_like(Fs)
    F[sysm.perm] = Fs
    return E, F, dict(N=sysm.N, t=tt, peak=_peak(), resident=resident, edges=int(sysm.NL.ge(0).sum()))


def replica_for(base, n_target):
    """Smallest cubic-ish replica of base with at least n_target atoms."""
    nb = len(base)
    k = max(1, round((n_target / nb) ** (1 / 3)))
    reps = [k, k, k]
    i = 0
    while nb * reps[0] * reps[1] * reps[2] < n_target:
        reps[i % 3] += 1
        i += 1
    return base.repeat(tuple(reps)), reps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--base", required=True, help="base cell (extxyz), orthorhombic, box > 2 cutoff")
    ap.add_argument("--rep", nargs=3, type=int, default=[4, 4, 4])
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--dtype", default="float64", choices=["float32", "float64"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--official", action="store_true")
    ap.add_argument("--noise", action="store_true")
    ap.add_argument("--scan", nargs="*", type=int, default=None)
    ap.add_argument("--stock", nargs="*", type=int, default=None)
    ap.add_argument("--write-ref", default=None)
    ap.add_argument("--data", default=None, help="LAMMPS data file (types 1 = O, 2 = H) for --write-ref")
    ap.add_argument("--frame", default="fixed", choices=["fixed", "random"])
    a = ap.parse_args()
    from ase.io import read

    print("edge frame:", install_edge_frame(a.frame))
    dev = a.device
    dt = getattr(torch, a.dtype)
    bb, eb, meta = load_esen(a.checkpoint, dt, dev)
    _official = {}

    def official_calc():
        """fairchem's own OCPCalculator on the GPU, a separate model instance (the modules behind `bb` were moved to
        `dev` and would otherwise be fed CPU tensors by the calculator's own pipeline)."""
        if "calc" not in _official:
            from fairchem.core.common.relaxation.ase_utils import OCPCalculator

            _official["calc"] = OCPCalculator(checkpoint_path=a.checkpoint, cpu=(dev == "cpu"), seed=0)
        return _official["calc"]

    def official_forces(atoms):
        """Forces, energy and wall time of the official evaluation (second call timed; the first warms up)."""
        at = atoms.copy()
        at.calc = official_calc()
        f = at.get_forces()
        at.calc.reset()
        _sync(); t0 = time.perf_counter()
        f = at.get_forces()
        _sync(); tt = time.perf_counter() - t0
        return f, float(at.get_potential_energy()), tt
    print(f"eSEN: blocks={bb.num_layers} cutoff={bb.cutoff} lmax={bb.lmax} mmax={bb.mmax} sphere={bb.sphere_channels} "
          f"hidden={bb.hidden_channels} | dtype={a.dtype} device={dev}", flush=True)

    if a.write_ref:
        at = read(a.data, format="lammps-data", atom_style="atomic", Z_of_type={1: 8, 2: 1}, units="metal")
        at.set_pbc(True)
        E, F, nE = reference(bb, eb, meta, at, dt, dev)
        np.savez(a.write_ref, energy=E, forces=F.cpu().numpy(), positions=at.get_positions(), numbers=at.get_atomic_numbers())
        print(f"reference ({a.dtype}): N={len(at)} E={E:.6f} eV  Fmax={float(F.abs().max()):.4f} eV/A  edges={nE} -> {a.write_ref}")
        return

    base = read(a.base)
    base.set_pbc(True)
    nb = len(base)

    if a.noise:
        install_edge_frame("random")
        E1, F1, _ = reference(bb, eb, meta, base, dt, dev)
        E2, F2, _ = reference(bb, eb, meta, base, dt, dev)
        d = (F1 - F2).abs()
        print(f"NOISE (random edge frame, {a.dtype}, {nb} atoms): two calls differ by max|dF|={float(d.max()):.3e} "
              f"RMS(dF)={math.sqrt(float((d * d).mean())):.3e} eV/A  |dE|={abs(E1 - E2):.3e} eV  (|F|max {float(F1.abs().max()):.3f})",
              flush=True)
        install_edge_frame("fixed")

    if a.check or a.official:
        Eb, Fb, nEb = reference(bb, eb, meta, base, dt, dev)
        print(f"reference on the base cell ({a.dtype}): E={Eb:.6f} eV ({Eb / nb:.6f}/atom)  Fmax={float(Fb.abs().max()):.4f}  edges={nEb}",
              flush=True)
        if a.official:
            try:
                f_off, e_off, t_off = official_forces(base)
                f_off = torch.tensor(f_off, dtype=torch.float64, device=dev)
                d = f_off - Fb
                print(f"OFFICIAL OCPCalculator (fp32, fairchem graph, same fixed frame) vs our fp64 reference: "
                      f"dE={e_off - Eb:+.3e} eV  max|dF|={float(d.abs().max()):.3e}  RMS(dF)={math.sqrt(float((d * d).mean())):.3e} eV/A  "
                      f"t={t_off:.3f}s ({nb / t_off:.3e} atom-step/s)", flush=True)
            except Exception as e:                                  # keep the chunked check even if the official path fails
                print(f"OFFICIAL OCPCalculator failed: {str(e).splitlines()[0][:160]}", flush=True)
        if a.check:
            big = base.repeat(tuple(a.rep))
            E, F, info = chunked(bb, eb, meta, big, dt, dev, a.chunk)
            N = info["N"]
            k = torch.arange(N, device=dev) % nb               # ase.repeat: contiguous copies of the base cell
            d = F - Fb[k]
            print(f"CHUNKED {a.rep} N={N:,} edges={info['edges']:,}: t={info['t']:.2f}s ({N / info['t']:.3e} atom-step/s) "
                  f"peak={info['peak'] / GB:.2f} GB ({info['peak'] / N / 1e3:.1f} kB/atom) resident={info['resident'] / N / 1e3:.2f} kB/atom",
                  flush=True)
            print(f"CHECK vs the original model on the {nb}-atom base cell: dE/atom={E / N - Eb / nb:+.3e} eV  "
                  f"max|dF|={float(d.abs().max()):.3e}  RMS(dF)={math.sqrt(float((d * d).mean())):.3e} eV/A  (all {N:,} atoms)",
                  flush=True)

    if a.scan:
        for n in a.scan:
            big, reps = replica_for(base, n)
            try:
                E, F, info = chunked(bb, eb, meta, big, dt, dev, a.chunk, log=lambda *x: None)
                N = info["N"]
                print(f"SCAN chunked {a.dtype} reps={reps} N={N:,}: t={info['t']:.2f}s ({N / info['t']:.3e} atom-step/s) "
                      f"peak={info['peak'] / GB:.2f} GB ({info['peak'] / N / 1e3:.2f} kB/atom) resident={info['resident'] / N / 1e3:.2f} kB/atom "
                      f"E/atom={E / N:.6f}", flush=True)
            except torch.cuda.OutOfMemoryError:
                print(f"SCAN chunked reps={reps} N={len(big):,}: OUT OF MEMORY", flush=True)
                torch.cuda.empty_cache()
                break
            del big
            torch.cuda.empty_cache()

    if a.stock:
        for n in a.stock:
            big, reps = replica_for(base, n)
            torch.cuda.empty_cache()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            try:
                f, e, ts = official_forces(big)                  # the official evaluation: fairchem graph + full forward
                print(f"STOCK official reps={reps} N={len(big):,}: t={ts:.2f}s ({len(big) / ts:.3e} atom-step/s) "
                      f"peak={_peak() / GB:.2f} GB ({_peak() / len(big) / 1e3:.1f} kB/atom)  Fmax={float(np.abs(f).max()):.3f}", flush=True)
            except torch.cuda.OutOfMemoryError:
                print(f"STOCK official reps={reps} N={len(big):,}: OUT OF MEMORY", flush=True)
                torch.cuda.empty_cache()
                break
            except RuntimeError as e:                            # cuda errors surface as RuntimeError in some paths
                print(f"STOCK official reps={reps} N={len(big):,}: {str(e).splitlines()[0][:120]}", flush=True)
                torch.cuda.empty_cache()
                break
            del big, at
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
