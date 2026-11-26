#!/usr/bin/env python
"""Where the time of the L-layer chunked SevenNet evaluation goes: torch.profiler over one force evaluation, with a
record_function range around every module call (edge embedding, linear layers, convolution, gate, ...).
    python profile_lpass_sevennet.py --base water64_1x1x1.extxyz --rep 8 8 4 --chunk 8192
"""
import argparse
import os
import sys
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from mace_onegpu.lpass_sevennet import LPassSevenNet, SevenNetLayers, load_sevennet  # noqa: E402
from mace_onegpu.receiver_chunk_mace import System  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="7net-0")
    ap.add_argument("--base", required=True)
    ap.add_argument("--rep", nargs=3, type=int, default=[8, 8, 4])
    ap.add_argument("--chunk", type=int, default=8192)
    ap.add_argument("--cueq", action="store_true", help="SevenNet's cuEquivariance backend")
    ap.add_argument("--rows", type=int, default=22)
    a = ap.parse_args()
    from ase.io import read

    dev, dt = "cuda", torch.float32
    model, cfg = load_sevennet(a.model, dt, dev, cueq=a.cueq)
    ev = LPassSevenNet(model, cfg, dt, a.chunk)
    big = read(a.base).repeat(a.rep)
    tm = model.type_map
    types = torch.tensor([tm[int(z)] for z in big.get_atomic_numbers()], device=dev)
    sysm = System(torch.tensor(big.get_positions(), dtype=torch.float64, device=dev), np.diag(np.array(big.get_cell())),
                  types.to(torch.int16), ev.r_max, dev)
    sysm.build_neighbors()
    types_s = types[sysm.perm]

    names = {id(m): f"{n.split('_', 1)[1] if n[0].isdigit() else n}" for n, m in model._modules.items()}
    orig_layer = SevenNetLayers.layer

    def layer(self, k, x_nodes, types_nodes, ei, ev_):
        d = {"x": x_nodes, "node_attr": self.onehot(types_nodes), "atom_type": types_nodes.long(),
             "edge_index": ei, "edge_vec": ev_}
        with torch.profiler.record_function("mod::edge_embedding"):
            d = self.edge_embedding(d)
        for mod in self.layers[k]:
            with torch.profiler.record_function("mod::" + names.get(id(mod), type(mod).__name__)):
                d = mod(d)
        return d["x"]

    SevenNetLayers.layer = layer
    import time

    for _ in range(2):                                                   # warm-up (and kernel planning)
        ev.compute(sysm, types_s, log=lambda *x: None)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    ev.compute(sysm, types_s, log=lambda *x: None)
    torch.cuda.synchronize()
    print(f"cueq={a.cueq}  N={sysm.N:,}  one force evaluation (warm): {time.perf_counter() - t0:.3f} s")
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
        ev.compute(sysm, types_s, log=print)
        torch.cuda.synchronize()
    SevenNetLayers.layer = orig_layer
    rows = [e for e in prof.key_averages() if e.key.startswith("mod::")]
    tot = defaultdict(float)
    for e in rows:
        tot[e.key] += e.device_time_total if hasattr(e, "device_time_total") else e.cuda_time_total
    s = sum(tot.values())
    print(f"N={sysm.N:,}  chunk={a.chunk}  module ranges (GPU time incl. backward via autograd attribution is NOT included):")
    for k, v in sorted(tot.items(), key=lambda kv: -kv[1]):
        print(f"  {k:32s} {v / 1e6:8.3f} s  {100 * v / s:5.1f}%")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=a.rows))
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=12))


if __name__ == "__main__":
    main()
