#!/usr/bin/env bash
# Speed vs system size on one GPU: liquid water, 300 K, dt 1 fs, 3 repeats per size.
#   MACE (three-pass ML-IAP model, e.g. a distilled student) in LAMMPS (CUDA graphs, torch.compile, chunk 65536) and,
#   optionally, NEP89 in GPUMD (its own MD engine, NVE) as a reference.
#   N = 15,360 x rep^3 for rep = 1 2 4 8  (15,360 / 122,880 / 983,040 / 7,864,320 atoms).
#
# Required environment:
#   MODEL_LMP   three-pass ML-IAP model file (.pt) built with scripts/make_threepass_model.py
#   DATA        LAMMPS data file of the 15,360-atom water box (atom_style atomic, type 1 = O, type 2 = H)
# Optional:
#   LAMMPS_ENV  shell file to source first (puts the patched lmp, CUDA and the Python environment on PATH)
#   OUT         output directory (default ./runs/speed_benchmark)
#   GPUMD, NEP_MODEL   gpumd executable and NEP89 potential file; the NEP89 part is skipped unless both are set
#   REPS        replication factors (default "1 2 4 8")
set -uo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MODEL_LMP=${MODEL_LMP:?set MODEL_LMP to the three-pass ML-IAP .pt file}
DATA=${DATA:?set DATA to the LAMMPS data file of the 15,360-atom water box}
[ -n "${LAMMPS_ENV:-}" ] && source "$LAMMPS_ENV"
export PYTHONPATH=$REPO:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1 MKL_THREADING_LAYER=GNU OMP_NUM_THREADS=1
OUT=${OUT:-$PWD/runs/speed_benchmark}; mkdir -p "$OUT"; cd "$OUT"
REPS=${REPS:-"1 2 4 8"}
echo "started $(date)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
if [ -n "${GPUMD:-}" ] && [ -n "${NEP_MODEL:-}" ]; then
  echo "===== NEP89 / GPUMD"
  for rep in $REPS; do   # rep^3 copies of the 15,360-atom box
    D=$OUT/nep_r$rep; mkdir -p $D; cd $D
    python - "$DATA" $rep <<'PY'
import sys
from ase.io import read, write
a = read(sys.argv[1], format="lammps-data", atom_style="atomic", Z_of_type={1: 8, 2: 1})
r = int(sys.argv[2]); b = a.repeat((r, r, r)); b.pbc = True
write("model.xyz", b, format="extxyz")
print(len(b))
PY
    N=$(head -1 model.xyz)
    printf "potential %s\nvelocity 300\nensemble nve\ntime_step 1\ndump_thermo 1000\nrun 200\n" "$NEP_MODEL" > run.in
    for k in 1 2 3; do
      timeout 1200 "$GPUMD" > gpumd_$k.out 2>&1
      echo "   NEP89 N=$N rep$k: $(grep -i 'Speed of this run' gpumd_$k.out | tail -1)"
    done
    cd $OUT
  done
fi
echo "===== MACE three-pass / LAMMPS"
for rep in $REPS; do
  D=$OUT/mace_r$rep; mkdir -p $D; cd $D; cp -f "$REPO/inputs/in.water_scan.lmp" in.lmp
  for k in 1 2 3; do
    MACE3P_COMPILE=1 MACE3P_GRAPHS=1 MACE3P_CHUNK=65536 MACE3P_LOG=$D/m_$k.jsonl timeout 1500 lmp -k on g 1 -sf kk -pk kokkos newton on \
      neigh half -in in.lmp -log log_$k.lammps -var data_file "$DATA" -var model_file "$MODEL_LMP" \
      -var run_steps 60 -var skin 2.0 -var rep $rep -var dt 0.001 -var thermo_every 20 > lmp_$k.out 2>&1
    rc=$?
    python - $D/m_$k.jsonl "rep$rep run$k rc=$rc" <<'PY'
import json, sys
r = [json.loads(l) for l in open(sys.argv[1]) if '"t_total"' in l][5:]
if not r:
    print(f"   MACE [{sys.argv[2]}] no data"); sys.exit()
t = sorted(x["t_total"] for x in r)[len(r) // 2]; g = sum(x["t_gap_lammps"] or 0 for x in r) / len(r)
print(f"   MACE N={r[0]['nlocal']} [{sys.argv[2]}] steady {r[0]['nlocal']/(t+g):.3e} atom-step/s  peak {max(x['peak_gb'] for x in r):.1f} GB")
PY
  done
  cd $OUT
done
echo "finished $(date)"
