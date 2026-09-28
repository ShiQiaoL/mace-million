#!/usr/bin/env bash
# Long NPT run (default: 300 K, 1 bar, dt 1 fs, 10 segments x 100 ps = 1 ns) of a large system on ONE GPU. A small
# MACE student (torch.compile) drives the MD, and the teacher (e.g. MACE-MPA-0) supervises it on the fly: every
# --dump-every steps a frame is checked by a LAMMPS rerun of teacher and student, and the student is fine-tuned on
# teacher-labelled clusters when a threshold is exceeded (scripts/al_driver.py). In the paper this was 983,040 water
# atoms with a supervision overhead of about 1.3 %. Resumable: running the script again continues from
# $WORKDIR/state.json.
#
# Required environment:
#   TEACHER_MODEL  teacher MACE .model file (e.g. mace-mpa-0-medium.model)
#   TEACHER_LMP    three-pass ML-IAP .pt of the teacher (scripts/make_threepass_model.py)
#   STUDENT_MODEL  initial student .model (e.g. from examples/distill_student.sh)
#   TRAIN_XYZ      the student's training set (extxyz with REF_energy / REF_forces / REF_stress)
#   DATA           LAMMPS data file of the start configuration
#   E0S            teacher E0s as a mace_run_train dict, e.g. "{1: -1.11734008, 8: -1.54838784}"
#                  (python scripts/teacher_e0s.py $TEACHER_MODEL O H)
# Optional:
#   ELEMS          element names in LAMMPS type order (default "O H")
#   LAMMPS_ENV     shell file to source first (patched lmp, CUDA, Python environment)
#   WORKDIR        work directory of the run (default ./runs/supervised_run)
#   TIME_BUDGET_S  wall-time budget of the driver in s (default 345000); the outer timeout is 2 h longer
set -uo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TEACHER_MODEL=${TEACHER_MODEL:?set TEACHER_MODEL}
TEACHER_LMP=${TEACHER_LMP:?set TEACHER_LMP}
STUDENT_MODEL=${STUDENT_MODEL:?set STUDENT_MODEL}
TRAIN_XYZ=${TRAIN_XYZ:?set TRAIN_XYZ}
DATA=${DATA:?set DATA to the LAMMPS data file of the start configuration}
E0S=${E0S:?set E0S to the teacher E0s dict printed by scripts/teacher_e0s.py}
ELEMS=${ELEMS:-"O H"}
TIME_BUDGET_S=${TIME_BUDGET_S:-345000}
[ -n "${LAMMPS_ENV:-}" ] && source "$LAMMPS_ENV"
export PYTHONPATH=$REPO:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
F='FutureWarning|weights_only|_Jd, _W3j|torch.load\(|UserWarning|warnings.warn|atomic_numbers", torch|ResourceWarning|TemporaryDirectory'
AL=${WORKDIR:-$PWD/runs/supervised_run}
echo "started $(date)"
timeout -k 120 $((TIME_BUDGET_S + 7800)) python "$REPO/scripts/al_driver.py" --work-dir "$AL" \
   --teacher "$TEACHER_MODEL" \
   --teacher-lmp "$TEACHER_LMP" \
   --student "$STUDENT_MODEL" \
   --train-xyz "$TRAIN_XYZ" --data "$DATA" --elems $ELEMS \
   --e0s "$E0S" --temp 300 --press 1.01325 --segment-steps 100000 --dump-every 10000 \
   --dp-max 450 --al-weight 10 --compile --chunk 65536 --teacher-chunk 16384 --segment-timeout 32000 \
   --cycles 10 --time-budget-s "$TIME_BUDGET_S" 2>&1 | grep -vE "$F" | grep -E "^cycle|^done|Error|Traceback|RuntimeError"
python - "$AL" <<'PY'
import json, re, sys
AL = sys.argv[1]
st = json.load(open(f"{AL}/state.json"))
tot_steps = tot_md = 0.0
for h in st["history"]:
    c = h["cycle"]
    lp = open(f"{AL}/cycle{c:03d}/log.lammps", errors="replace").read()
    m = re.search(r"Loop time of ([\d.]+) on \d+ procs for (\d+) steps with (\d+) atoms", lp)
    if m:
        tot_md += float(m.group(1)); tot_steps += int(m.group(2)); n = int(m.group(3))
    print({k: h.get(k) for k in ("cycle", "halted", "worst", "density_mean", "retrained", "added_frames", "t_cycle_s")},
          f"MD {float(m.group(1)) / int(m.group(2)):.3f} s/step" if m else "")
wall = sum(h["t_cycle_s"] for h in st["history"])
if tot_steps:
    print(f"simulated {tot_steps/1e6:.3f} ns; MD-only {n*tot_steps/tot_md:.3e} atom-step/s; "
          f"including supervision/retraining {n*tot_steps/wall:.3e} atom-step/s; retrains "
          f"{sum(1 for h in st['history'] if h.get('retrained'))}")
PY
echo "finished $(date)"
