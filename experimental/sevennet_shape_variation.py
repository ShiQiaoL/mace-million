#!/usr/bin/env python
"""Does the cost of the L-layer chunked SevenNet evaluation depend on whether the tensor shapes change between calls?
Inside LAMMPS every force call sees a different number of pairs and of ghost atoms; the standalone benchmark repeats one
system. Here one force evaluation is timed for a sequence of systems with different atom counts (random subsets of a
replicated water box), first with the e3nn backend, then with cuEquivariance, with and without a repeat of the same
system in between.
    python sevennet_shape_variation.py --base water64_1x1x1.extxyz --rep 4 4 4 --chunk 8192 [--cueq]
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from mace_onegpu.lpass_sevennet import LPassSevenNet, load_sevennet  # noqa: E402
from mace_onegpu.receiver_chunk_mace import System  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="7net-0")
    ap.add_argument("--base", required=True)
    ap.add_argument("--rep", nargs=3, type=int, default=[4, 4, 4])
    ap.add_argument("--chunk", type=int, default=8192)
    ap.add_argument("--cueq", action="store_true")
    ap.add_argument("--n-systems", type=int, default=6)
    a = ap.parse_args()
    from ase.io import read

    dev, dt = "cuda", torch.float32
    model, cfg = load_sevennet(a.model, dt, dev, cueq=a.cueq)
    ev = LPassSevenNet(model, cfg, dt, a.chunk)
    big = read(a.base).repeat(a.rep)
    tm = model.type_map
    rng = np.random.default_rng(3)
    systems = []
    for k in range(a.n_systems):
        at = big.copy()
        if k:                                  # remove 0.5-3% of the molecules: different N, edges and shapes
            n_mol = len(at) // 3
            drop = rng.choice(n_mol, size=int(n_mol * (0.005 + 0.005 * k)), replace=False)
            idx = np.concatenate([3 * drop, 3 * drop + 1, 3 * drop + 2])
            del at[idx]
        types = torch.tensor([tm[int(z)] for z in at.get_atomic_numbers()], device=dev)
        sysm = System(torch.tensor(at.get_positions(), dtype=torch.float64, device=dev), np.diag(np.array(at.get_cell())),
                      types.to(torch.int16), ev.r_max, dev)
        sysm.build_neighbors()
        systems.append((sysm, types[sysm.perm]))

    def timed(sysm, types):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        ev.compute(sysm, types, log=lambda *x: None)
        torch.cuda.synchronize()
        return time.perf_counter() - t0

    tag = "cueq" if a.cueq else "e3nn"
    s0, t0 = systems[0]
    print(f"[{tag}] warm-up: {timed(s0, t0):.3f} s, repeat: {timed(s0, t0):.3f} s, repeat: {timed(s0, t0):.3f} s  (N={s0.N:,})")
    for k, (sysm, types) in enumerate(systems[1:], 1):
        t_first = timed(sysm, types)
        t_second = timed(sysm, types)
        print(f"[{tag}] system {k}: N={sysm.N:,}  first call {t_first:.3f} s  second call {t_second:.3f} s")
    print(f"[{tag}] back to system 0: {timed(s0, t0):.3f} s")


if __name__ == "__main__":
    main()
