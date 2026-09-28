#!/usr/bin/env python
"""
Separate the autograd (force-VJP) memory from the forward edge-tensor memory on the SAME graph, outside LAMMPS.

For each structure and each backend variant it measures torch peak allocated memory for
  E_nograd : energy only under torch.no_grad()            (pure inference, nothing saved)
  E_graph  : energy with autograd graph, no backward       (saved tensors, no VJP)
  E+F      : energy + forces (one VJP wrt positions)
  E+F+S    : energy + forces + stress
and reports n_edges, so that  M_{E+F} - M_E  (why inference needs autograd memory)
and  M_E_graph - M_E_nograd  are obtained directly.

usage (GPU node, mace env):
  python ase_gradient_probe.py --model mace-mpa-0-medium.model --structures a.extxyz b.data ... \
         --variants fp64 fp64_fused fp32 --out probe.json
"""
import argparse
import copy
import json
import os
import time

os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

import numpy as np  # noqa: E402
import torch  # noqa: E402
from ase import io  # noqa: E402

from mace import data as mace_data  # noqa: E402
from mace.cli.convert_e3nn_cueq import run as run_e3nn_to_cueq  # noqa: E402
from mace.tools import torch_geometric, utils  # noqa: E402

VARIANTS = {
    "fp64": dict(dtype=torch.float64, backend="cueq", fusion=False),
    "fp64_fused": dict(dtype=torch.float64, backend="cueq", fusion=True),
    "fp32": dict(dtype=torch.float32, backend="cueq", fusion=False),
    "fp32_fused": dict(dtype=torch.float32, backend="cueq", fusion=True),
    "fp64_e3nn": dict(dtype=torch.float64, backend="e3nn", fusion=False),
}


def read_structure(path):
    if path.endswith(".data"):
        try:
            return io.read(path, format="lammps-data", atom_style="atomic", Z_of_type={1: 8, 2: 1}, units="metal")
        except TypeError:
            return io.read(path, format="lammps-data", style="atomic", Z_of_type={1: 8, 2: 1}, units="metal")
    return io.read(path)


def make_model(path, spec, device):
    torch.set_default_dtype(spec["dtype"])
    model = torch.load(path, map_location=device)
    model = model.double() if spec["dtype"] == torch.float64 else model.float()
    if spec["backend"] == "cueq":
        if spec["fusion"]:
            model = run_e3nn_to_cueq(copy.deepcopy(model).to(device), device=device)
        else:
            model = run_e3nn_to_cueq(copy.deepcopy(model).to("cpu"), device="cpu")
    model = model.to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def make_batch(model, atoms, device):
    z_table = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])
    try:
        config = mace_data.config_from_atoms(atoms)
    except TypeError:
        config = mace_data.config_from_atoms(atoms, charges_key="charges")
    heads = getattr(model, "heads", None)
    try:
        ad = mace_data.AtomicData.from_config(config, z_table=z_table, cutoff=float(model.r_max), heads=heads)
    except TypeError:
        ad = mace_data.AtomicData.from_config(config, z_table=z_table, cutoff=float(model.r_max))
    loader = torch_geometric.dataloader.DataLoader(dataset=[ad], batch_size=1, shuffle=False, drop_last=False)
    batch = next(iter(loader)).to(device)
    return batch


def measure(model, batch, mode, device):
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    d = batch.to_dict()
    if mode == "E_nograd":
        with torch.no_grad():
            out = model(d, training=False, compute_force=False, compute_stress=False)
    elif mode == "E_graph":
        out = model(d, training=False, compute_force=False, compute_stress=False)
    elif mode == "E+F":
        out = model(d, training=False, compute_force=True, compute_stress=False)
    elif mode == "E+F+S":
        out = model(d, training=False, compute_force=True, compute_stress=True)
    else:
        raise ValueError(mode)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    peak = torch.cuda.max_memory_allocated()
    res = {
        "mode": mode, "peak_alloc_gb": peak / 1e9, "peak_over_base_gb": (peak - base) / 1e9,
        "peak_reserved_gb": torch.cuda.max_memory_reserved() / 1e9, "t_ms": (t1 - t0) * 1e3,
        "energy_eV": float(out["energy"].detach().double().sum().item()) if out.get("energy") is not None else None,
    }
    if out.get("forces") is not None:
        f = out["forces"].detach().double()
        res["fmax_eV_A"] = float(f.abs().max().item())
        res["frms_eV_A"] = float(f.pow(2).mean().sqrt().item())
    del out, d
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--structures", nargs="+", required=True)
    ap.add_argument("--variants", nargs="+", default=["fp64", "fp64_fused", "fp32"])
    ap.add_argument("--modes", nargs="+", default=["E_nograd", "E_graph", "E+F", "E+F+S"])
    ap.add_argument("--repeat", type=int, default=2, help="repeat each mode; the 2nd run excludes JIT/compile effects")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    device = "cuda"
    results = []
    props = torch.cuda.get_device_properties(0)
    header = {"gpu": props.name, "gpu_total_gb": props.total_memory / 1e9, "torch": torch.__version__}
    print(header)
    for vname in a.variants:
        spec = VARIANTS[vname]
        try:
            model = make_model(a.model, spec, device)
        except Exception as exc:
            results.append({"variant": vname, "error": f"model build failed: {exc!r}"})
            print(results[-1])
            continue
        model_bytes = torch.cuda.memory_allocated()
        for s in a.structures:
            atoms = read_structure(s)
            torch.set_default_dtype(spec["dtype"])
            batch = make_batch(model, atoms, device)
            n_edges = int(batch.edge_index.shape[1])
            for mode in a.modes:
                rec = {"variant": vname, "structure": os.path.basename(s), "natoms": len(atoms), "n_edges": n_edges,
                       "edges_per_atom": n_edges / len(atoms), "model_resident_gb": model_bytes / 1e9}
                try:
                    r = None
                    for _ in range(a.repeat):
                        r = measure(model, batch, mode, device)
                    rec.update(r)
                except torch.cuda.OutOfMemoryError as exc:
                    rec.update({"mode": mode, "error": "OOM", "detail": str(exc)[:200]})
                    torch.cuda.empty_cache()
                except Exception as exc:
                    rec.update({"mode": mode, "error": repr(exc)[:300]})
                results.append(rec)
                print(json.dumps(rec))
            del batch
            torch.cuda.empty_cache()
        del model
        torch.cuda.empty_cache()
    with open(a.out, "w") as fh:
        json.dump({"header": header, "results": results}, fh, indent=1)
    print("->", a.out)


if __name__ == "__main__":
    main()
