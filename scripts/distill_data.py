#!/usr/bin/env python
"""
Step 1 of the MACE -> NEP distillation / on-the-fly workflow (VASP-MLFF-like, with MACE-MPA-0 as the "DFT").
Element-general: nothing here depends on the chemistry; give any periodic start structure and (T, density) ranges.

Runs short Langevin NVT MD with the ORIGINAL MACE model (fp64, ASE) over a grid of temperatures and isotropic cell
scalings, samples frames, and writes them with MACE energies / forces / virials in GPUMD NEP format
(train.xyz, test.xyz) via calorine.

    python distill_data.py --model M.model --structure water64_1x1x1.extxyz --temps 260 300 350 420 \
        --scales 1.027 1.0 0.975 --equil-ps 1 --sample-ps 5 --every-ps 0.1 --out-dir distill_water
"""
import argparse
import os
import time

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
import numpy as np  # noqa: E402
from ase import units  # noqa: E402
from ase.calculators.singlepoint import SinglePointCalculator  # noqa: E402
from ase.io import read, write  # noqa: E402
from ase.md.langevin import Langevin  # noqa: E402
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--structure", required=True)
    ap.add_argument("--temps", nargs="+", type=float, default=[260, 300, 350, 420])
    ap.add_argument("--scales", nargs="+", type=float, default=[1.027, 1.0, 0.975],
                    help="isotropic length scale factors (density ~ 1/scale^3)")
    ap.add_argument("--dt-fs", type=float, default=1.0)
    ap.add_argument("--equil-ps", type=float, default=1.0)
    ap.add_argument("--sample-ps", type=float, default=5.0)
    ap.add_argument("--every-ps", type=float, default=0.1)
    ap.add_argument("--test-fraction", type=float, default=0.1)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--dtype", default="float64")
    a = ap.parse_args()
    from calorine.nep import write_structures
    from mace.calculators import MACECalculator

    os.makedirs(a.out_dir, exist_ok=True)
    calc = MACECalculator(model_paths=a.model, device="cuda", default_dtype=a.dtype)
    base = read(a.structure)
    frames = []
    rng = np.random.default_rng(20260925)
    t_all = time.time()
    for T in a.temps:
        for s in a.scales:
            at = base.copy()
            at.set_cell(at.cell * s, scale_atoms=True)
            at.calc = calc
            MaxwellBoltzmannDistribution(at, temperature_K=T, rng=rng)
            Stationary(at)
            dyn = Langevin(at, a.dt_fs * units.fs, temperature_K=T, friction=0.01 / units.fs, rng=rng)
            n_eq = int(a.equil_ps * 1000 / a.dt_fs)
            n_every = max(1, int(a.every_ps * 1000 / a.dt_fs))
            n_samp = int(a.sample_ps * 1000 / a.dt_fs) // n_every
            t0 = time.time()
            dyn.run(n_eq)
            for k in range(n_samp):
                dyn.run(n_every)
                f = at.copy()
                e = at.get_potential_energy()
                F = at.get_forces()
                S = at.get_stress(voigt=False)
                f.calc = SinglePointCalculator(f, energy=e, forces=F, stress=S)
                f.info["config_type"] = f"T{int(T)}_s{s:.3f}"
                frames.append(f)
            rho = sum(base.get_masses()) / 0.602214076 / at.get_volume()
            print(f"T={T:.0f} K scale={s:.3f} (rho={rho:.3f} g/cm3): {n_samp} frames, T_now={at.get_temperature():.0f} K, "
                  f"{time.time()-t0:.0f}s", flush=True)
    idx = rng.permutation(len(frames))
    n_test = max(1, int(round(a.test_fraction * len(frames))))
    test = [frames[i] for i in idx[:n_test]]
    train = [frames[i] for i in idx[n_test:]]
    write_structures(os.path.join(a.out_dir, "train.xyz"), train)
    write_structures(os.path.join(a.out_dir, "test.xyz"), test)
    write(os.path.join(a.out_dir, "all_frames_mace.extxyz"), frames)
    # REF_* keys for mace_run_train (avoids the reserved energy/forces/stress keys); same train/test split
    def ref(fs):
        out = []
        for f in fs:
            c = f.copy()
            c.calc = None
            c.info.update(REF_energy=f.get_potential_energy(), REF_stress=f.get_stress(voigt=False).reshape(-1),
                          config_type=f.info["config_type"])
            c.arrays["REF_forces"] = f.get_forces()
            out.append(c)
        return out
    write(os.path.join(a.out_dir, "train_ref.extxyz"), ref(train))
    write(os.path.join(a.out_dir, "test_ref.extxyz"), ref(test))
    F = np.concatenate([f.get_forces() for f in frames])
    print(f"wrote {len(train)} train + {len(test)} test frames ({len(frames)*len(base)} atoms); "
          f"|F| RMS {np.sqrt((F**2).mean()):.3f} eV/A; total {time.time()-t_all:.0f}s", flush=True)


if __name__ == "__main__":
    main()
