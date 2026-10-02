#!/usr/bin/env python
"""Are teacher labels of carved vacuum clusters faithful to the periodic system they were cut from?

For a periodic frame, centres are chosen at random (and, with --student, at the atoms where the student deviates most
from the teacher). Around each centre, clusters are carved exactly as in the on-the-fly supervision
(scripts/al_driver.py: carve_clusters, sphere of radius R, bonds completed with covalent radii, 8 A of vacuum) for
several R. The teacher's forces on the cluster atoms (fp64, no periodicity) are compared with the teacher's forces on
the same atoms in the periodic frame, as a function of the depth below the cut surface (R - distance from the centre).
With --student, the student's own deviation from the teacher at the same centres is reported for scale.
For oxides, under-coordinated cations/anions in the clusters are counted (--coord Si:O:2.0:4:2).

    python validate_clusters.py --model mace-mpa-0-medium.model --xyz test_ref.extxyz --radii 5 7 9 11 13 --out w.json
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
from al_driver import carve_clusters  # noqa: E402  (the production carving code, unchanged)

DEPTH_BINS = [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 4.0), (4.0, 6.0), (6.0, 8.0), (8.0, 99.0)]


def selection(at, c, radius, bond_scale=1.2, closure=3):
    """Indices selected by carve_clusters for centre c (same rules; used to map cluster atoms back to the frame)."""
    from ase.data import covalent_radii
    from ase.geometry import find_mic

    rcov = covalent_radii[at.numbers] * bond_scale
    d, dist = find_mic(at.positions - at.positions[c], at.cell, True)
    sel = dist < radius
    shell = np.where((dist >= radius) & (dist < radius + 2 * rcov.max() * closure))[0]
    for _ in range(closure):
        ins = np.where(sel)[0]
        cand = shell[~sel[shell]]
        if not len(cand):
            break
        rij = np.linalg.norm(d[cand][:, None, :] - d[ins][None, :, :], axis=2)
        bonded = (rij < rcov[cand][:, None] + rcov[ins][None, :]).any(1)
        if not bonded.any():
            break
        sel[cand[bonded]] = True
    return np.where(sel)[0], d, dist


def coordination(cl, spec):
    """spec 'A:B:rc:nA:nB' -> fraction of A with < nA B-neighbours and of B with < nB A-neighbours within rc."""
    from ase.data import atomic_numbers

    A, B, rc, nA, nB = spec.split(":")
    zA, zB, rc, nA, nB = atomic_numbers[A], atomic_numbers[B], float(rc), int(nA), int(nB)
    p, z = cl.positions, cl.numbers
    iA, iB = np.where(z == zA)[0], np.where(z == zB)[0]
    if not len(iA) or not len(iB):
        return None
    dmat = np.linalg.norm(p[iA][:, None, :] - p[iB][None, :, :], axis=2) < rc
    return {"n_" + A: int(len(iA)), "n_" + B: int(len(iB)), "ratio_" + B + "_" + A: len(iB) / len(iA),
            "under_" + A: float((dmat.sum(1) < nA).mean()), "under_" + B: float((dmat.sum(0) < nB).mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="teacher")
    ap.add_argument("--student", default=None)
    ap.add_argument("--xyz", required=True)
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--radii", type=float, nargs="+", default=[5.0, 7.0, 9.0, 11.0, 13.0])
    ap.add_argument("--n-random", type=int, default=8)
    ap.add_argument("--n-worst", type=int, default=4)
    ap.add_argument("--coord", default=None, help="e.g. Si:O:2.0:4:2")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    from ase.data import covalent_radii
    from ase.io import read
    from mace.calculators import MACECalculator

    torch.set_default_dtype(torch.float64)
    teacher = MACECalculator(model_paths=a.model, device="cuda", default_dtype="float64")
    at = read(a.xyz, index=a.index)
    at.pbc = True
    L = np.diag(np.array(at.get_cell()))
    ext = max(a.radii) + 3 * 2 * 1.2 * float(covalent_radii[at.numbers].max())
    rep = [max(1, math.ceil((2 * ext + 1.0) / x)) for x in L]
    at = at.repeat(rep)
    at.wrap()
    N = len(at)
    at.calc = teacher
    Fp = at.get_forces().copy()
    at.calc = None
    print(f"{a.xyz}[{a.index}] x{rep}: N={N}, periodic teacher F_rms={np.sqrt((Fp ** 2).mean()):.4f} eV/A")

    rng = np.random.default_rng(11)
    centres = [("random", int(i)) for i in rng.choice(N, size=a.n_random, replace=False)]
    dstu = None
    if a.student:
        stu = MACECalculator(model_paths=a.student, device="cuda", default_dtype="float64")
        at.calc = stu
        Fs = at.get_forces().copy()
        at.calc = None
        del stu
        dstu = np.linalg.norm(Fs - Fp, axis=1)
        for i in np.argsort(-dstu)[: a.n_worst]:
            centres.append(("worst", int(i)))
        print(f"  student vs teacher (periodic): rms {np.sqrt(((Fs - Fp) ** 2).mean()):.4f}, max |dF| {dstu.max():.3f} eV/A")

    rows = []
    for kind, c in centres:
        for R in a.radii:
            dF = np.zeros(N)
            dF[c] = 1.0
            cls = carve_clusters(at, dF, 1, R, 10 ** 9)
            idx, d, dist = selection(at, c, R)
            cl = cls[0]
            assert len(cl) == len(idx) and np.array_equal(cl.numbers, at.numbers[idx])
            shift = cl.positions[0] - d[idx[0]]
            assert np.allclose(cl.positions - shift, d[idx], atol=1e-8)       # same atoms, same order, translated
            cl.calc = teacher
            Fc = cl.get_forces()
            cl.calc = None
            err = np.linalg.norm(Fc - Fp[idx], axis=1)
            depth = R - dist[idx]                                               # < 0 for atoms added by bond completion
            k0 = int(np.where(idx == c)[0][0])
            row = {"kind": kind, "centre": c, "Z": int(at.numbers[c]), "R": R, "n_atoms": len(cl),
                   "centre_dF": float(err[k0]), "centre_F": float(np.linalg.norm(Fp[c])),
                   "student_dF_centre": None if dstu is None else float(dstu[c]),
                   "added_by_bonds": int((depth <= 0).sum()), "max_dF_added": float(err[depth <= 0].max()) if (depth <= 0).any() else None,
                   "depth_bins": []}
            for lo, hi in DEPTH_BINS:
                m = (depth > lo) & (depth <= hi)
                row["depth_bins"].append({"lo": lo, "hi": hi, "n": int(m.sum()),
                                          "max_dF": float(err[m].max()) if m.any() else None,
                                          "rms_dF": float(np.sqrt((err[m] ** 2).mean())) if m.any() else None})
            if a.coord:
                row["coordination"] = coordination(cl, a.coord)
            rows.append(row)
        last = [r for r in rows if r["centre"] == c]
        print(f"  {kind:6s} centre {c:6d} (Z={at.numbers[c]:2d}): |dF_centre| by R " +
              " ".join(f"{r['R']:.0f}:{r['centre_dF'] * 1e3:.1f}" for r in last) + " meV/A" +
              ("" if dstu is None else f"   student |dF| {dstu[c] * 1e3:.0f} meV/A"))

    summ = []
    for R in a.radii:
        rr = [r for r in rows if r["R"] == R]
        ce = np.array([r["centre_dF"] for r in rr])
        core = [b for r in rr for b in r["depth_bins"] if b["lo"] >= 4.0 and b["max_dF"] is not None]
        summ.append({"R": R, "centre_dF_median": float(np.median(ce)), "centre_dF_max": float(ce.max()),
                     "depth_ge4_max_dF": max((b["max_dF"] for b in core), default=None),
                     "n_atoms_median": float(np.median([r["n_atoms"] for r in rr]))})
        print(f"  R={R:4.1f}: centre |dF| median {summ[-1]['centre_dF_median'] * 1e3:.1f}, max {ce.max() * 1e3:.1f} meV/A; "
              f"depth>=4 A max {summ[-1]['depth_ge4_max_dF']}; median {summ[-1]['n_atoms_median']:.0f} atoms")
    res = {"model": os.path.basename(a.model), "student": a.student and os.path.basename(a.student),
           "xyz": os.path.basename(a.xyz), "index": a.index, "rep": rep, "N": N,
           "periodic_F_rms": float(np.sqrt((Fp ** 2).mean())), "rows": rows, "summary": summ, "wall_s": time.time() - t0}
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
