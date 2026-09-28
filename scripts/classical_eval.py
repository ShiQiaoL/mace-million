#!/usr/bin/env python
"""Classical force fields on the DFT benchmark frames: same metrics as eval_vs_teacher.py
(force RMSE/MAE, energy RMSE after removing the mean offset, virial-pressure RMSE and bias vs the REF_* = DFT labels).
One LAMMPS 'run 0' per frame (serial CPU, no Kokkos).

    python classical_eval.py --frames dft/sio2/dft.extxyz --ff vashishta --pot $LAMMPS/potentials/SiO.1990.vashishta --elems Si O
    python classical_eval.py --frames dft/cuni/dft.extxyz --ff eam/alloy --pot $LAMMPS/potentials/CuNi.eam.alloy --elems Cu Ni
    python classical_eval.py --frames dft/water/dft.extxyz --ff spcfw
spcfw: flexible SPC/Fw water (Wu, Tepper & Voth, J. Chem. Phys. 124, 024503 (2006)), Ewald 1e-6, molecules assigned by
nearest O for every H.
"""
import argparse
import json
import os
import subprocess
import tempfile

import numpy as np
from ase.io import read

EV_A3_TO_BAR = 1.602176634e6
KCAL_TO_EV = 0.0433641043
ATM_TO_BAR = 1.01325


def write_atomic(path, at, elems):
    c = at.cell.lengths()
    with open(path, "w") as f:
        f.write(f"frame\n\n{len(at)} atoms\n{len(elems)} atom types\n\n0 {c[0]:.10f} xlo xhi\n0 {c[1]:.10f} ylo yhi\n"
                f"0 {c[2]:.10f} zlo zhi\n\nMasses\n\n")
        from ase.data import atomic_masses, atomic_numbers
        for i, e in enumerate(elems):
            f.write(f"{i + 1} {atomic_masses[atomic_numbers[e]]:.6f}\n")
        f.write("\nAtoms # atomic\n\n")
        sym = at.get_chemical_symbols()
        for i, (s, p) in enumerate(zip(sym, at.get_positions(wrap=True))):
            f.write(f"{i + 1} {elems.index(s) + 1} {p[0]:.10f} {p[1]:.10f} {p[2]:.10f}\n")


def write_water_full(path, at):
    """SPC/Fw topology: every H bonded to its nearest O (minimum image)."""
    c = at.cell.lengths()
    sym = np.array(at.get_chemical_symbols())
    iO, iH = np.where(sym == "O")[0], np.where(sym == "H")[0]
    D = at.get_distances
    owner = {}
    for h in iH:
        d = D(h, iO, mic=True)
        owner[h] = iO[int(np.argmin(d))]
    mol = {o: k + 1 for k, o in enumerate(iO)}
    bonds = [(owner[h], h) for h in iH]
    angles = []
    for o in iO:
        hs = [h for h in iH if owner[h] == o]
        if len(hs) == 2:
            angles.append((hs[0], o, hs[1]))
    pos = at.get_positions(wrap=True)
    with open(path, "w") as f:
        f.write(f"water\n\n{len(at)} atoms\n{len(bonds)} bonds\n{len(angles)} angles\n2 atom types\n1 bond types\n"
                f"1 angle types\n\n0 {c[0]:.10f} xlo xhi\n0 {c[1]:.10f} ylo yhi\n0 {c[2]:.10f} zlo zhi\n\n"
                "Masses\n\n1 15.9994\n2 1.008\n\nAtoms # full\n\n")
        for i in range(len(at)):
            t, q, m = (1, -0.82, mol[i]) if sym[i] == "O" else (2, 0.41, mol[owner[i]])
            f.write(f"{i + 1} {m} {t} {q} {pos[i, 0]:.10f} {pos[i, 1]:.10f} {pos[i, 2]:.10f}\n")
        f.write("\nBonds\n\n" + "".join(f"{k + 1} 1 {a + 1} {b + 1}\n" for k, (a, b) in enumerate(bonds)))
        f.write("\nAngles\n\n" + "".join(f"{k + 1} 1 {a + 1} {b + 1} {c_ + 1}\n" for k, (a, b, c_) in enumerate(angles)))
    return len(angles) == len(iO)


def run_frame(at, a, wd):
    data = os.path.join(wd, "frame.data")
    if a.ff == "spcfw":
        ok = write_water_full(data, at)
        head = ("units real\natom_style full\nboundary p p p\nread_data frame.data\n"
                "pair_style lj/cut/coul/long 9.0\npair_coeff 1 1 0.1554253 3.165492\npair_coeff * 2 0.0 0.0\n"
                "bond_style harmonic\nbond_coeff 1 529.581 1.012\nangle_style harmonic\nangle_coeff 1 37.95 113.24\n"
                "kspace_style ewald 1.0e-6\nspecial_bonds lj/coul 0.0 0.0 0.0\n")
        fconv, econv, pconv = KCAL_TO_EV, KCAL_TO_EV, ATM_TO_BAR
    else:
        ok = True
        write_atomic(data, at, a.elems)
        head = (f"units metal\natom_style atomic\nboundary p p p\nread_data frame.data\npair_style {a.ff}\n"
                f"pair_coeff * * {a.pot} {' '.join(a.elems)}\n")
        fconv, econv, pconv = 1.0, 1.0, 1.0
    inp = head + ("compute pv all pressure NULL virial\nthermo_style custom step pe c_pv\nthermo_modify format float %20.12g\n"
                  "dump f all custom 1 forces.dump id fx fy fz\ndump_modify f sort id format float %20.12g\nrun 0\n")
    open(os.path.join(wd, "in.lmp"), "w").write(inp)
    r = subprocess.run([a.lmp, "-in", "in.lmp", "-log", "log.lammps", "-screen", "none"], cwd=wd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"lammps failed: {r.stdout[-800:]} {r.stderr[-800:]}")
    lines = open(os.path.join(wd, "log.lammps")).read().split("\n")
    k = next(i for i, l in enumerate(lines) if l.split()[:3] == ["Step", "PotEng", "c_pv"])
    _, pe, pv = lines[k + 1].split()[:3]
    dl = open(os.path.join(wd, "forces.dump")).read().split("\n")
    n = int(dl[3])
    F = np.array(" ".join(dl[9:9 + n]).split(), dtype=float).reshape(n, 4)[:, 1:] * fconv
    return float(pe) * econv, F, float(pv) * pconv, ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--ff", required=True, help="spcfw, or a LAMMPS pair style (eam/alloy, vashishta, tersoff, ...)")
    ap.add_argument("--pot", default=None)
    ap.add_argument("--elems", nargs="*", default=[])
    ap.add_argument("--lmp", default="lmp")
    ap.add_argument("--label", default=None)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    frames = read(a.frames, index=":")
    dF, dE, dP, n_bad_topo = [], [], [], 0
    with tempfile.TemporaryDirectory() as wd:
        for at in frames:
            E, F, P, ok = run_frame(at, a, wd)
            n_bad_topo += (not ok)
            dF.append((F - at.arrays["REF_forces"]).ravel())
            dE.append((E - at.info["REF_energy"]) / len(at))
            Pt = -np.trace(np.asarray(at.info["REF_stress"]).reshape(3, 3)) / 3 * EV_A3_TO_BAR
            dP.append(P - Pt)
    dF, dE, dP = np.concatenate(dF), np.asarray(dE), np.asarray(dP)
    res = dict(f_rmse=float(np.sqrt((dF ** 2).mean())), f_mae=float(np.abs(dF).mean()),
               e_rmse_meV_atom=float(1e3 * np.sqrt(((dE - dE.mean()) ** 2).mean())),
               p_rmse_bar=float(np.sqrt((dP ** 2).mean())), p_bias_bar=float(dP.mean()), n_frames=len(frames))
    lab = a.label or (a.ff + (f" ({os.path.basename(a.pot)})" if a.pot else ""))
    print(f"{lab:40s} {1e3*res['f_rmse']:9.1f} {1e3*res['f_mae']:9.1f} {res['e_rmse_meV_atom']:9.2f} "
          f"{res['p_rmse_bar']:9.0f} {res['p_bias_bar']:9.0f}   ({len(frames)} frames" +
          (f", {n_bad_topo} with broken water topology)" if a.ff == "spcfw" else ")"))
    if a.json:
        json.dump({lab: res}, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
