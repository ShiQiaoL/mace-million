#!/usr/bin/env python
"""DFT benchmark: single points with VASP on a few frames, then everyone vs DFT.

  prepare : pick --n frames spread over config_type from a REF_* extxyz, write one VASP directory per frame
            (Materials-Project/MPtrj-compatible static settings: PBE, PAW PBE_54 with MP POTCAR choices, ENCUT 520,
            Gaussian smearing 0.05 eV, LASPH; spin-polarised only if a magnetic element is present).
  collect : read the finished runs (vasprun.xml) into dft.extxyz with REF_energy / REF_forces / REF_stress (= DFT).
            Then:  python eval_vs_teacher.py --frames dft.extxyz --models teacher.model student.model --nep nep89.txt

    python dft_bench.py prepare --frames test_ref.extxyz --n 6 --out dft/sio2 --kpts 1 1 1
    python dft_bench.py collect --out dft/sio2
Element-general: POTCAR choice and magnetic moments come from tables covering the periodic table, not the system.
"""
import argparse
import glob
import os

import numpy as np
from ase.io import read, write

# VASP PAW PBE_54 POTCAR library, one sub-directory per POTCAR ($VASP_PP_PATH/O/POTCAR, $VASP_PP_PATH/Li_sv/POTCAR, ...).
# Set it before "prepare", e.g.  export VASP_PP_PATH=/path/to/pseudopotentials/POT_GGA_PAW_PBE_54
POT_ROOT = os.environ.get("VASP_PP_PATH")
# Materials Project (pymatgen MPRelaxSet) POTCAR choices; elements not listed use the plain symbol
MP_POTCAR = {"Li": "Li_sv", "Na": "Na_pv", "K": "K_sv", "Rb": "Rb_sv", "Cs": "Cs_sv", "Be": "Be_sv", "Mg": "Mg_pv",
             "Ca": "Ca_sv", "Sr": "Sr_sv", "Ba": "Ba_sv", "Sc": "Sc_sv", "Y": "Y_sv", "Ti": "Ti_pv", "Zr": "Zr_sv",
             "Hf": "Hf_pv", "V": "V_pv", "Nb": "Nb_pv", "Ta": "Ta_pv", "Cr": "Cr_pv", "Mo": "Mo_pv", "W": "W_pv",
             "Mn": "Mn_pv", "Tc": "Tc_pv", "Re": "Re_pv", "Fe": "Fe_pv", "Ru": "Ru_pv", "Os": "Os_pv", "Co": "Co",
             "Rh": "Rh_pv", "Ir": "Ir", "Ni": "Ni_pv", "Pd": "Pd", "Pt": "Pt", "Cu": "Cu_pv", "Ag": "Ag", "Au": "Au",
             "Zn": "Zn", "Cd": "Cd", "Hg": "Hg", "Ga": "Ga_d", "In": "In_d", "Tl": "Tl_d", "Ge": "Ge_d", "Sn": "Sn_d",
             "Pb": "Pb_d", "Bi": "Bi", "Sb": "Sb", "Te": "Te", "Po": "Po", "La": "La", "Ce": "Ce", "Nd": "Nd_3",
             "Sm": "Sm_3", "Eu": "Eu", "Gd": "Gd", "Er": "Er_3", "Yb": "Yb", "Lu": "Lu_3", "U": "U", "Th": "Th"}
MAGNETIC = {"V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Mo", "Ru", "Rh", "Nd", "Sm", "Eu", "Gd", "Er", "U"}

INCAR = """SYSTEM = dft_bench single point (MPtrj-compatible)
PREC = Accurate
ENCUT = 520
EDIFF = 1E-5
ISMEAR = 0
SIGMA = 0.05
LASPH = .TRUE.
LREAL = .FALSE.
ALGO = Fast
NELM = 200
ISYM = 0
IBRION = -1
NSW = 0
LWAVE = .FALSE.
LCHARG = .FALSE.
KPAR = 1
"""


def prepare(a):
    if not POT_ROOT or not os.path.isdir(POT_ROOT):
        raise SystemExit("dft_bench.py prepare: set VASP_PP_PATH to the VASP PAW PBE_54 POTCAR directory "
                         f"(one sub-directory per POTCAR); got VASP_PP_PATH={POT_ROOT!r}")
    frames = read(a.frames, index=":")
    by_type = {}
    for i, f in enumerate(frames):
        by_type.setdefault(f.info.get("config_type", "all"), []).append(i)
    types = sorted(by_type)
    pick, k = [], 0
    while len(pick) < min(a.n, len(frames)):                 # round-robin over config types
        lst = by_type[types[k % len(types)]]
        j = (k // len(types))
        if j < len(lst) and lst[j] not in pick:
            pick.append(lst[j])
        k += 1
        if k > 10 * len(frames):
            break
    for n, i in enumerate(pick):
        f = frames[i]
        d = os.path.join(a.out, f"f{n:02d}")
        os.makedirs(d, exist_ok=True)
        syms = []
        for s in f.get_chemical_symbols():
            if s not in syms:
                syms.append(s)
        at = f.copy()
        at.calc = None
        write(os.path.join(d, "POSCAR"), at, format="vasp", direct=True, sort=True)
        order = sorted(syms)            # ase's vasp writer with sort=True orders species alphabetically
        with open(os.path.join(d, "POTCAR"), "w") as out:
            for s in order:
                out.write(open(os.path.join(POT_ROOT, MP_POTCAR.get(s, s), "POTCAR")).read())
        inc = INCAR
        if any(s in MAGNETIC for s in syms):
            counts = [at.get_chemical_symbols().count(s) for s in order]
            inc += "ISPIN = 2\nMAGMOM = " + " ".join(f"{c}*{0.6 if s in MAGNETIC else 0.0}" for s, c in zip(order, counts)) + "\n"
        open(os.path.join(d, "INCAR"), "w").write(inc)
        open(os.path.join(d, "KPOINTS"), "w").write("Gamma\n0\nGamma\n" + " ".join(map(str, a.kpts)) + "\n0 0 0\n")
        write(os.path.join(d, "frame_ref.extxyz"), f)
        print(f"{d}: frame {i} ({f.info.get('config_type')}), {len(f)} atoms, species {order}")


def collect(a):
    out = []
    for d in sorted(glob.glob(os.path.join(a.out, "f*"))):
        vr = os.path.join(d, "vasprun.xml")
        if not os.path.exists(vr):
            print(f"{d}: not finished")
            continue
        try:
            at = read(vr, index=-1)
        except Exception as e:          # noqa: BLE001
            print(f"{d}: unreadable ({e})")
            continue
        if at.calc is None or "free_energy" not in getattr(at.calc, "results", {}) and "energy" not in getattr(at.calc, "results", {}):
            print(f"{d}: no converged result")
            continue
        E, F, S = at.get_potential_energy(), at.get_forces(), at.get_stress(voigt=False)
        at.calc = None
        at.info.update(REF_energy=E, REF_stress=S.reshape(-1), config_type=os.path.basename(d))
        at.arrays["REF_forces"] = F
        out.append(at)
        t = [ln for ln in open(os.path.join(d, "OUTCAR"), errors="replace") if "Elapsed time" in ln]
        print(f"{d}: E = {E:.4f} eV, |F|_RMS = {np.sqrt((F ** 2).mean()):.3f} eV/A, {t[-1].strip() if t else ''}")
    if out:
        write(os.path.join(a.out, "dft.extxyz"), out)
        print(f"wrote {len(out)} frames -> {os.path.join(a.out, 'dft.extxyz')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["prepare", "collect"])
    ap.add_argument("--frames")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--out", required=True)
    ap.add_argument("--kpts", nargs=3, type=int, default=[1, 1, 1])
    a = ap.parse_args()
    prepare(a) if a.mode == "prepare" else collect(a)


if __name__ == "__main__":
    main()
