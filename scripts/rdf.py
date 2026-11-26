#!/usr/bin/env python
"""
Partial radial distribution functions from LAMMPS custom dumps (id type x y z, orthorhombic box, NPT allowed).
    python rdf.py traj_a.lammpstrj [traj_b.lammpstrj] --types O H --rmax 6 --dr 0.02 --skip 0 --out rdf.npz
With two trajectories: prints the first-peak position/height of each pair and max|g_a - g_b| (r > 1.5 A for O-O).
"""
import argparse

import numpy as np


def frames(path):
    with open(path) as fh:
        while True:
            line = fh.readline()
            if not line:
                return
            if not line.startswith("ITEM: TIMESTEP"):
                continue
            fh.readline()
            fh.readline()
            n = int(fh.readline())
            fh.readline()
            box = np.array([[float(x) for x in fh.readline().split()[:2]] for _ in range(3)])
            cols = fh.readline().split()[2:]
            data = np.loadtxt([fh.readline() for _ in range(n)])
            yield box, cols, data


def rdf(path, names, rmax, dr, skip, stop=10**9):
    nb = int(rmax / dr)
    pairs = [(a, b) for i, a in enumerate(names) for b in names[i:]]
    hist = {p: np.zeros(nb) for p in pairs}
    norm = {p: 0.0 for p in pairs}
    nf = 0
    for k, (box, cols, d) in enumerate(frames(path)):
        if k < skip:
            continue
        if k >= stop:
            break
        L = box[:, 1] - box[:, 0]
        V = float(np.prod(L))
        ti = cols.index("type")
        xi = [cols.index(c) for c in ("x", "y", "z")]
        pos = d[:, xi]
        typ = d[:, ti].astype(int)
        for (a, b) in pairs:
            A = pos[typ == names.index(a) + 1]
            B = pos[typ == names.index(b) + 1]
            for s in range(0, len(A), 256):
                dv = A[s:s + 256, None, :] - B[None, :, :]
                dv -= L * np.round(dv / L)
                r = np.sqrt((dv ** 2).sum(-1)).ravel()
                r = r[(r > 1e-6) & (r < rmax)]
                hist[(a, b)] += np.histogram(r, bins=nb, range=(0, rmax))[0]
            na, nbb = len(A), len(B)
            npair = na * nbb - (na if a == b else 0)
            norm[(a, b)] += npair / V
        nf += 1
    edges = np.linspace(0, rmax, nb + 1)
    shell = 4.0 / 3.0 * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    r = 0.5 * (edges[1:] + edges[:-1])
    g = {f"{a}-{b}": hist[(a, b)] / (norm[(a, b)] * shell) for (a, b) in pairs}
    return r, g, nf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trajs", nargs="+")
    ap.add_argument("--types", nargs="+", default=["O", "H"])
    ap.add_argument("--rmax", type=float, default=6.0)
    ap.add_argument("--dr", type=float, default=0.02)
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--stop", type=int, default=10**9)
    ap.add_argument("--out", default="rdf.npz")
    ap.add_argument("--peak-min", nargs="*", default=[],
                    help="per pair lower r bound for the reported peak, e.g. O-O:2.2 O-H:1.3 (skip intramolecular peaks)")
    a = ap.parse_args()
    pmin = {s.split(":")[0]: float(s.split(":")[1]) for s in a.peak_min}
    res = []
    for t in a.trajs:
        r, g, nf = rdf(t, a.types, a.rmax, a.dr, a.skip, a.stop)
        res.append(g)
        print(f"{t}: {nf} frames")
        for k, v in g.items():
            lo = pmin.get(k, 0.5)
            m = r > lo
            i = np.argmax(np.where(m, v, -1))
            print(f"   g_{k}: main peak (r > {lo}) at r = {r[i]:.3f} A, g = {v[i]:.3f}")
    if len(res) == 2:
        for k in res[0]:
            m = r > 1.5
            d = np.abs(res[0][k] - res[1][k])[m]
            print(f"   max|g_a - g_b| for {k} (r > 1.5 A): {d.max():.3f}   mean {d.mean():.4f}")
    np.savez(a.out, r=r, **{f"{i}_{k}": v for i, g in enumerate(res) for k, v in g.items()})


if __name__ == "__main__":
    main()
