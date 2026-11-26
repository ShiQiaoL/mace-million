#!/usr/bin/env python
"""Speed of the chunked SevenNet evaluation, eager against torch.compile (MACE3P_COMPILE=1), same structure; reports
the time per force evaluation and the largest force difference between the two (fp32).
    python bench_lpass_sevennet.py --base water64_1x1x1.extxyz --rep 8 8 4 --chunk 8192
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


def run(model, cfg, sysm, types, chunk, compile_, n=3):
    os.environ["MACE3P_COMPILE"] = "1" if compile_ else "0"
    ev = LPassSevenNet(model, cfg, torch.float32, chunk)
    t0 = time.perf_counter()
    E, F, _ = ev.compute(sysm, types, log=lambda *x: None)                 # warm-up (compilation)
    torch.cuda.synchronize()
    t_first = time.perf_counter() - t0
    ts = []
    for _ in range(n):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        E, F, _ = ev.compute(sysm, types, log=lambda *x: None)
        torch.cuda.synchronize(); ts.append(time.perf_counter() - t0)
    return E, F, t_first, min(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="7net-0")
    ap.add_argument("--base", required=True)
    ap.add_argument("--rep", nargs=3, type=int, default=[8, 8, 4])
    ap.add_argument("--chunk", type=int, default=8192)
    a = ap.parse_args()
    from ase.io import read

    dev = "cuda"
    model, cfg = load_sevennet(a.model, torch.float32, dev)
    big = read(a.base).repeat(a.rep)
    types = torch.tensor([model.type_map[int(z)] for z in big.get_atomic_numbers()], device=dev)
    sysm = System(torch.tensor(big.get_positions(), dtype=torch.float64, device=dev), np.diag(np.array(big.get_cell())),
                  types.to(torch.int16), float(cfg["cutoff"]), dev)
    sysm.build_neighbors()
    ts = types[sysm.perm]
    E0, F0, f0, t0 = run(model, cfg, sysm, ts, a.chunk, False)
    print(f"N={sysm.N:,} eager:   {t0:.3f} s ({sysm.N / t0:.3e} atom-step/s)  first call {f0:.1f} s", flush=True)
    E1, F1, f1, t1 = run(model, cfg, sysm, ts, a.chunk, True)
    print(f"N={sysm.N:,} compile: {t1:.3f} s ({sysm.N / t1:.3e} atom-step/s)  first call {f1:.1f} s  speed-up {t0 / t1:.2f}x  "
          f"|dE|/N={abs(E1 - E0) / sysm.N:.2e} eV  max|dF|={float((F1 - F0).abs().max()):.2e} eV/A", flush=True)


if __name__ == "__main__":
    main()
