#!/usr/bin/env python
"""Accuracy table: deviation of students (MACE .model) and, as the reference point, NEP89 (calorine CPUNEP)
from the teacher labels on held-out frames (extxyz with REF_energy / REF_forces / REF_stress, e.g. test_ref.extxyz).
Element-general.  Prints, per model: force RMSE and MAE (eV/A), energy RMSE (meV/atom, after removing the mean offset,
since students and NEP89 have different energy zeros), pressure RMSE and mean bias (bar, virial part).

    python eval_vs_teacher.py --frames test_ref.extxyz --models S1.model S2.model --nep nep89.txt --json out.json
"""
import argparse
import json
import os

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
import numpy as np

KEYS = {"energy": "REF_energy", "forces": "REF_forces", "stress": "REF_stress"}  # noqa: E402
from ase.io import read  # noqa: E402

EV_A3_TO_BAR = 1.602176634e6


def evaluate(frames, calc):
    dF, dE, dP = [], [], []
    for at in frames:
        a = at.copy()
        a.calc = calc
        F = a.get_forces()
        E = a.get_potential_energy()
        dF.append((F - at.arrays[KEYS["forces"]]).ravel())
        dE.append((E - at.info[KEYS["energy"]]) / len(at))
        if "REF_stress" in at.info and all(at.pbc):
            S = a.get_stress(voigt=False)
            St = np.asarray(at.info[KEYS["stress"]]).reshape(3, 3)
            dP.append((-np.trace(S) + np.trace(St)) / 3 * EV_A3_TO_BAR)
    dF = np.concatenate(dF)
    dE = np.asarray(dE)
    dP = np.asarray(dP)
    return dict(f_rmse=float(np.sqrt((dF ** 2).mean())), f_mae=float(np.abs(dF).mean()),
                e_rmse_meV_atom=float(1e3 * np.sqrt(((dE - dE.mean()) ** 2).mean())),
                p_rmse_bar=float(np.sqrt((dP ** 2).mean())) if len(dP) else None,
                p_bias_bar=float(dP.mean()) if len(dP) else None, n_frames=len(frames))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--models", nargs="*", default=[])
    ap.add_argument("--nep", default=None)
    ap.add_argument("--max-frames", type=int, default=200)
    ap.add_argument("--json", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--energy-key", default="REF_energy")
    ap.add_argument("--forces-key", default="REF_forces")
    ap.add_argument("--stress-key", default="REF_stress")
    a = ap.parse_args()
    global KEYS
    KEYS = {"energy": a.energy_key, "forces": a.forces_key, "stress": a.stress_key}
    frames = read(a.frames, index=":")[: a.max_frames]
    res = {}
    for m in a.models:
        from mace.calculators import MACECalculator

        res[os.path.basename(m)] = evaluate(frames, MACECalculator(model_paths=m, device=a.device, default_dtype="float64"))
    if a.nep:
        from calorine.calculators import CPUNEP

        res["NEP89"] = evaluate(frames, CPUNEP(a.nep))
    print(f"{'model':40s} {'F_RMSE':>9s} {'F_MAE':>9s} {'E_RMSE':>9s} {'P_RMSE':>9s} {'P_bias':>9s}   ({len(frames)} frames)")
    print(f"{'':40s} {'meV/A':>9s} {'meV/A':>9s} {'meV/at':>9s} {'bar':>9s} {'bar':>9s}")
    for k, r in res.items():
        p = (f"{r['p_rmse_bar']:9.0f} {r['p_bias_bar']:9.0f}" if r["p_rmse_bar"] is not None else f"{'-':>9s} {'-':>9s}")
        print(f"{k:40s} {1e3*r['f_rmse']:9.1f} {1e3*r['f_mae']:9.1f} {r['e_rmse_meV_atom']:9.2f} {p}")
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
