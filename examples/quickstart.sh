#!/usr/bin/env bash
# Quick start and installation check (one GPU, about two minutes).
#   1. converts MACE-MPA-0 (medium) to the three-pass ML-IAP file;
#   2. runs 10 MD steps of the 192-atom water box in examples/data/ with LAMMPS (pair_style mliap unified);
#   3. compares the step-0 forces of LAMMPS (fp32 three-pass) with an ASE reference (fp64, e3nn) and prints
#      PASS or FAIL.  Expected: max|dF| about 1e-4 eV/A, i.e. the fp32 round-off of the model itself.
#
# Needs: a GPU, the Python environment of this package (pip install -e .), and a LAMMPS `lmp` built with
# PKG_ML-IAP + MLIAP_ENABLE_PYTHON + Kokkos CUDA (lammps/README.md; the patch is optional for this test).
#
# Environment:
#   TEACHER   path of mace-mpa-0-medium.model. If unset or missing, the file is downloaded from the MACE
#             foundation-model release (https://github.com/ACEsuit/mace-foundations, ~ 0.2 GB).
#   LMP       LAMMPS executable (default: lmp)
#   OUT       work directory (default: ./runs/quickstart)
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
OUT=${OUT:-$PWD/runs/quickstart}; mkdir -p "$OUT"
LMP=${LMP:-lmp}
TEACHER=${TEACHER:-$OUT/mace-mpa-0-medium.model}
export PYTHONPATH=$REPO:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1 MKL_THREADING_LAYER=GNU OMP_NUM_THREADS=1
cd "$OUT"
echo "== quickstart in $OUT ($(date))"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
command -v "$LMP" >/dev/null || { echo "FAIL: LAMMPS executable '$LMP' not found (set LMP)"; exit 2; }
python -c "import mace_onegpu, mace, torch; print('mace_onegpu', mace_onegpu.__version__, '| mace-torch', mace.__version__, '| torch', torch.__version__, '| cuda', torch.cuda.is_available())"

if [ ! -f "$TEACHER" ]; then
  echo "== downloading MACE-MPA-0 medium to $TEACHER"
  URL=https://github.com/ACEsuit/mace-foundations/releases/download/mace_mpa_0/mace-mpa-0-medium.model
  curl -L -o "$TEACHER" "$URL" 2>/dev/null || python -c "import sys, urllib.request; urllib.request.urlretrieve(sys.argv[1], sys.argv[2])" "$URL" "$TEACHER"
fi
DATA=$REPO/examples/data/water64_1x1x1_N192.data

echo "== 1/3 three-pass ML-IAP file"
python "$REPO/scripts/make_threepass_model.py" "$TEACHER" --out model.pt --chunk 16384 --spline-grid 512

echo "== 2/3 LAMMPS, 10 steps, forces dumped at step 0 and 10"
rm -f forces.dump log.lammps
MACE3P_GRAPHS=1 "$LMP" -k on g 1 -sf kk -pk kokkos newton on neigh half -in "$REPO/inputs/in.water_scan.lmp" \
  -var data_file "$DATA" -var model_file model.pt -var elems "O H" \
  -var run_steps 10 -var dump_forces 10 -var dt 0.001 -var skin 2.0 -log log.lammps > lmp.out 2>&1 \
  || { tail -30 lmp.out; echo "FAIL: LAMMPS run failed (see $OUT/lmp.out)"; exit 1; }
grep -E "^ +(0|10) " log.lammps | head -2

echo "== 3/3 reference forces (ASE, fp64, e3nn) and comparison"
python "$REPO/scripts/compare_forces.py" ref --model "$TEACHER" --data "$DATA" --out ref.npz
python "$REPO/scripts/compare_forces.py" dump --a forces.dump --b ref.npz --log-a log.lammps --tol-max 1e-3
