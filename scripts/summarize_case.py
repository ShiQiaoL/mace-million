#!/usr/bin/env python
"""Merge memprobe JSONL + LAMMPS log + nvidia-smi samples of one case into one TSV row (append)."""
import argparse
import csv
import glob
import json
import os
import re

COLS = [
    "case", "status", "rc", "variant", "data", "natoms", "steps", "skin", "ngpu",
    "gpu_name", "gpu_total_gb", "dtype", "conv_fusion",
    "nlocal_max", "nghost_max", "npairs_sum", "npairs_per_local", "fullnghs_log", "nghost_log",
    "peak_alloc_gb", "peak_reserved_gb", "peak_alloc_per_rank_gb", "nvsmi_peak_gb",
    "t_force_ms_median", "timesteps_per_s", "pair_pct", "comm_pct", "neigh_builds", "wall_s", "model", "case_dir",
]


def parse_log(path):
    out = {}
    if not os.path.exists(path):
        return out
    with open(path, errors="replace") as fh:
        txt = fh.read()
    m = re.search(r"Performance:.*?([0-9.]+) timesteps/s", txt)
    if m:
        out["timesteps_per_s"] = float(m.group(1))
    m = re.search(r"^Pair\s*\|.*\|\s*([0-9.]+)\s*$", txt, re.M)
    if m:
        out["pair_pct"] = float(m.group(1))
    m = re.search(r"^Comm\s*\|.*\|\s*([0-9.]+)\s*$", txt, re.M)
    if m:
        out["comm_pct"] = float(m.group(1))
    m = re.search(r"FullNghs:\s*([0-9.e+]+) ave\s+([0-9.e+]+) max", txt)
    if m:
        out["fullnghs_log"] = float(m.group(2))
    m = re.search(r"Nghost:\s*([0-9.e+]+) ave\s+([0-9.e+]+) max", txt)
    if m:
        out["nghost_log"] = float(m.group(2))
    m = re.search(r"Neighbor list builds = (\d+)", txt)
    if m:
        out["neigh_builds"] = int(m.group(1))
    m = re.search(r"Loop time of .* with (\d+) atoms", txt)
    if m:
        out["natoms"] = int(m.group(1))
    if "natoms" not in out:
        m = re.search(r"^\s*(\d+) atoms\s*$", txt, re.M)
        if m:
            out["natoms"] = int(m.group(1))
    return out


def parse_memprobe(case_dir):
    out = {}
    files = sorted(glob.glob(os.path.join(case_dir, "memprobe.rank*.jsonl")))
    if not files:
        return out
    peak_alloc = peak_res = 0
    per_rank_alloc = []
    nlocal_max = nghost_max = 0
    npairs_by_rank = {}
    tsteps = []
    for f in files:
        rank_alloc = 0
        with open(f) as fh:
            lines = fh.readlines()
        for line in lines:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get("type") == "header":
                out.setdefault("gpu_name", rec.get("gpu_name"))
                if rec.get("gpu_total_bytes"):
                    out.setdefault("gpu_total_gb", rec["gpu_total_bytes"] / 1e9)
                out.setdefault("dtype", rec.get("dtype"))
                out.setdefault("conv_fusion", str(rec.get("conv_fusion")))
            elif rec.get("type") == "step":
                nlocal_max = max(nlocal_max, rec["nlocal"])
                nghost_max = max(nghost_max, rec["nghost"])
                npairs_by_rank[rec["rank"]] = max(npairs_by_rank.get(rec["rank"], 0), rec["npairs"])
                tsteps.append(rec["t_step_ms"])
                rank_alloc = max(rank_alloc, rec.get("step_max_allocated", 0))
                peak_res = max(peak_res, rec.get("step_max_reserved", 0))
            elif rec.get("type") == "final":
                rank_alloc = max(rank_alloc, rec.get("cum_max_allocated", 0))
                peak_res = max(peak_res, rec.get("cum_max_reserved", 0))
        per_rank_alloc.append(rank_alloc)
        peak_alloc = max(peak_alloc, rank_alloc)
    out["nlocal_max"] = nlocal_max
    out["nghost_max"] = nghost_max
    out["npairs_sum"] = sum(npairs_by_rank.values())
    if nlocal_max:
        # per-rank E / N_local (uniform system: equal across ranks)
        out["npairs_per_local"] = max(npairs_by_rank.values()) / nlocal_max
    out["peak_alloc_gb"] = peak_alloc / 1e9
    out["peak_reserved_gb"] = peak_res / 1e9
    out["peak_alloc_per_rank_gb"] = ";".join(f"{a/1e9:.3f}" for a in per_rank_alloc)
    if tsteps:
        tsteps.sort()
        out["t_force_ms_median"] = tsteps[len(tsteps) // 2]
    return out


def parse_gpu_csv(path):
    peak = 0.0
    if not os.path.exists(path):
        return {}
    with open(path) as fh:
        for row in csv.reader(fh):
            try:
                peak = max(peak, float(row[2]))
            except Exception:
                pass
    return {"nvsmi_peak_gb": peak * 1.048576e-3}  # MiB -> GB


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case-dir", required=True)
    ap.add_argument("--status", required=True)
    ap.add_argument("--rc", default="")
    ap.add_argument("--model", default="")
    ap.add_argument("--data", default="")
    ap.add_argument("--steps", default="")
    ap.add_argument("--skin", default="")
    ap.add_argument("--ngpu", default="")
    ap.add_argument("--wall", default="")
    ap.add_argument("--summary", required=True)
    a = ap.parse_args()

    row = {c: "" for c in COLS}
    row.update(
        case=os.path.basename(a.case_dir.rstrip("/")), status=a.status, rc=a.rc, model=a.model,
        variant=re.sub(r".*\.(fp\w+?)-mliap_lammps\.pt$", r"\1", a.model) if a.model else "",
        data=os.path.basename(a.data), steps=a.steps, skin=a.skin, ngpu=a.ngpu, wall_s=a.wall, case_dir=a.case_dir,
    )
    m = re.search(r"_N(\d+)", os.path.basename(a.data))
    if m:
        row["natoms"] = int(m.group(1))
    row.update({k: v for k, v in parse_log(os.path.join(a.case_dir, "log.lammps")).items() if k in row})
    row.update({k: v for k, v in parse_memprobe(a.case_dir).items() if k in row})
    row.update(parse_gpu_csv(os.path.join(a.case_dir, "gpu.csv")))
    new = not os.path.exists(a.summary)
    with open(a.summary, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS, delimiter="\t")
        if new:
            w.writeheader()
        w.writerow(row)
    print("\t".join(f"{k}={row[k]}" for k in ("case", "status", "natoms", "npairs_per_local", "peak_alloc_gb", "peak_reserved_gb", "nvsmi_peak_gb", "timesteps_per_s")))


if __name__ == "__main__":
    main()
