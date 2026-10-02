#!/usr/bin/env bash
# SevenNet (NequIP-type, L message-passing layers) inside LAMMPS with the exact L-layer receiver-chunked evaluation
# (mace_onegpu/lpass_sevennet.py), using the SAME LAMMPS build as for MACE.
#
# LAMMPS embeds the Python of the environment it was built with (the MACE environment, e3nn 0.4.4). SevenNet needs
# e3nn >= 0.5, so its packages come from a second environment placed FIRST on PYTHONPATH. Both environments must have
# the same Python minor version (tested: 3.11, torch 2.5.1 in both, sevenn 0.13 + e3nn 0.6 in the SevenNet one).
#   1. model file (only the checkpoint path and settings; the model is built inside LAMMPS)
#   2. 10 MD steps of the 192-atom water box, step-0 forces vs the original SevenNet in fp64 (PASS/FAIL)
#
# Environment:
#   SEVENN_ENV  prefix of the SevenNet environment (contains bin/python and lib/python3.X/site-packages)
#   LMP         LAMMPS executable built with ML-IAP + Python + Kokkos (default: lmp)
#   MODEL       SevenNet checkpoint or pretrained name (default: 7net-0)
#   OUT         work directory (default: ./runs/sevennet_lammps)
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SEVENN_ENV=${SEVENN_ENV:?set SEVENN_ENV to the prefix of the SevenNet environment}
LMP=${LMP:-lmp}
MODEL=${MODEL:-7net-0}
OUT=${OUT:-$PWD/runs/sevennet_lammps}; mkdir -p "$OUT"; cd "$OUT"
SEVPY=$SEVENN_ENV/bin/python
SITE=$("$SEVPY" -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
export PYTHONPATH=$SITE:$REPO
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
DATA=$REPO/examples/data/water64_1x1x1_N192.data

echo "== 1/2 model file and fp64 reference"
"$SEVPY" "$REPO/scripts/make_lpass_sevennet_model.py" "$MODEL" --out sevennet.lpass.pt --chunk 8192 --dtype float32
"$SEVPY" "$REPO/experimental/sevennet_lpass.py" --model "$MODEL" --write-ref ref.npz --data "$DATA" --base "$DATA" --dtype float64

echo "== 2/2 LAMMPS, 10 steps; forces at step 0 vs the reference"
rm -f forces.dump log.lammps
"$LMP" -k on g 1 -sf kk -pk kokkos newton on neigh half -in "$REPO/inputs/in.water_scan.lmp" \
  -var data_file "$DATA" -var model_file sevennet.lpass.pt -var elems "O H" \
  -var run_steps 10 -var dump_forces 10 -var dt 0.001 -var skin 2.0 -log log.lammps > lmp.out 2>&1 \
  || { tail -30 lmp.out; echo "FAIL: LAMMPS run failed (see $OUT/lmp.out)"; exit 1; }
grep -m1 "LPassSevenNetMLIAP" lmp.out
"$SEVPY" "$REPO/scripts/compare_forces.py" dump --a forces.dump --b ref.npz --log-a log.lammps --tol-max 1e-3
