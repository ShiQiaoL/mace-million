#!/usr/bin/env python
"""Capacity and throughput of a MACE model through its own ASE calculator (the official standalone evaluation, full
graph, cuEquivariance if available), on replicas of a base cell until out of memory. For the "original" column of the
extended Table 2 where the stock LAMMPS ML-IAP file cannot be used (MACE-MH-1: TorchScript modules cannot be
deserialised inside LAMMPS).
    python experimental/mace_official_capacity.py --model mace-mh-1.model --head omat_pbe --base water64_1x1x1.extxyz --sizes 1536 5184 12288 24000 41472
"""
import argparse
import time

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--head", default=None)
    ap.add_argument("--base", required=True)
    ap.add_argument("--sizes", nargs="+", type=int, required=True)
    ap.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    ap.add_argument("--no-cueq", action="store_true")
    a = ap.parse_args()
    from ase.io import read
    from mace.calculators import MACECalculator

    kw = dict(model_paths=a.model, device="cuda", default_dtype=a.dtype, enable_cueq=not a.no_cueq)
    if a.head:
        kw["head"] = a.head
    calc = MACECalculator(**kw)
    base = read(a.base)
    base.set_pbc(True)
    nb = len(base)
    print(f"MACE official ASE calculator: {a.model} head={a.head} dtype={a.dtype} cueq={not a.no_cueq} | base {nb} atoms", flush=True)
    for n in a.sizes:
        k = max(1, round((n / nb) ** (1 / 3)))
        reps = [k, k, k]
        i = 0
        while nb * reps[0] * reps[1] * reps[2] < n:
            reps[i % 3] += 1
            i += 1
        big = base.repeat(tuple(reps))
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            at = big.copy()
            at.calc = calc
            at.get_forces()                                     # warm-up (kernel compilation)
            at.calc.reset()
            torch.cuda.synchronize(); t0 = time.perf_counter()
            f = at.get_forces()
            torch.cuda.synchronize(); tt = time.perf_counter() - t0
            pk = torch.cuda.max_memory_allocated()
            print(f"OFFICIAL reps={reps} N={len(big):,}: t={tt:.3f}s ({len(big) / tt:.3e} atom-step/s) peak={pk / 1e9:.2f} GB "
                  f"({pk / len(big) / 1e3:.1f} kB/atom) Fmax={float(np.abs(f).max()):.3f}", flush=True)
        except torch.cuda.OutOfMemoryError:
            print(f"OFFICIAL reps={reps} N={len(big):,}: OUT OF MEMORY", flush=True)
            break
        except RuntimeError as e:
            print(f"OFFICIAL reps={reps} N={len(big):,}: {str(e).splitlines()[0][:140]}", flush=True)
            break
        finally:
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
