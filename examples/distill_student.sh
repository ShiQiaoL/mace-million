#!/usr/bin/env bash
# Distil a small MACE student from the teacher (e.g. MACE-MPA-0) for a new system, then validate it:
#   1 distillation data: teacher Langevin MD over a T x density grid (scripts/distill_data.py)
#   2 student with the V1 architecture (16x0e+16x1o, l=2, correlation 2, r 4.5 A), --loss stress (mace_run_train)
#   3 accuracy vs the teacher on held-out frames, NEP89 as an optional reference point (scripts/eval_vs_teacher.py)
#   4 NPT in LAMMPS: student vs teacher (both three-pass ML-IAP), density + RDF
#
# Built-in presets (SYS=sio2 or SYS=cuni) use the structures written by
#   python scripts/build_general.py --system sio2 --out-dir $DATA --reps 2 4
#   python scripts/build_general.py --system cuni --out-dir $DATA --reps 4 7
# For another system set SYS to a new name and give ELEMS, BASE, MID, TEMPS, SCALES and T_NPT yourself.
#
# Required environment:
#   SYS            system name (sio2, cuni, or your own)
#   TEACHER_MODEL  teacher MACE .model file (e.g. mace-mpa-0-medium.model)
#   TEACHER_LMP    three-pass ML-IAP .pt of the teacher (scripts/make_threepass_model.py $TEACHER_MODEL --out ...)
#   DATA           directory with the start structures (presets) -- not needed if BASE and MID are given
# Optional:
#   LAMMPS_ENV     shell file to source first (patched lmp, CUDA, Python environment)
#   NEP_MODEL      NEP89 potential file for the reference point in step 3 (needs calorine)
#   WORKDIR        output root (default ./runs/distill_$SYS)
set -uo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SYS=${SYS:?set SYS (sio2, cuni, or a name of your own with ELEMS/BASE/MID/TEMPS/SCALES/T_NPT)}
TEACHER=${TEACHER_MODEL:?set TEACHER_MODEL to the teacher .model file}
TEACHER_LMP=${TEACHER_LMP:?set TEACHER_LMP to the teacher three-pass ML-IAP .pt file}
[ -n "${LAMMPS_ENV:-}" ] && source "$LAMMPS_ENV"
export PYTHONPATH=$REPO:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1 MKL_THREADING_LAYER=GNU OMP_NUM_THREADS=1
case $SYS in
  sio2) : "${DATA:?set DATA to the directory written by build_general.py}"
        ELEMS=${ELEMS:-"Si O"}; BASE=${BASE:-$DATA/sio2_2x2x2_N192.extxyz}; MID=${MID:-$DATA/sio2_4x4x4_N1536.data}
        TEMPS=${TEMPS:-"300 1500 2500 3500"}; SCALES=${SCALES:-"1.03 1.0 0.97"}; T_NPT=${T_NPT:-3000} ;;
  cuni) : "${DATA:?set DATA to the directory written by build_general.py}"
        ELEMS=${ELEMS:-"Cu Ni"}; BASE=${BASE:-$DATA/cuni_4x4x4_N256.extxyz}; MID=${MID:-$DATA/cuni_7x7x7_N1372.data}
        TEMPS=${TEMPS:-"300 700 1100 1500"}; SCALES=${SCALES:-"1.02 1.0 0.98"}; T_NPT=${T_NPT:-1000} ;;
  *)    ELEMS=${ELEMS:?set ELEMS}; BASE=${BASE:?set BASE (start structure, extxyz)}; MID=${MID:?set MID (LAMMPS data file for NPT)}
        TEMPS=${TEMPS:?set TEMPS}; SCALES=${SCALES:?set SCALES}; T_NPT=${T_NPT:?set T_NPT} ;;
esac
WORKDIR=${WORKDIR:-$PWD/runs/distill_$SYS}
DS=$WORKDIR/data; ST=$WORKDIR/students; OUT=$WORKDIR/validate; mkdir -p "$DS" "$ST" "$OUT"; cd "$OUT"
F='FutureWarning|weights_only|_Jd, _W3j|torch.load\(|UserWarning|warnings.warn|atomic_numbers", torch|ResourceWarning|TemporaryDirectory'
echo "started $(date)   SYS=$SYS elems=$ELEMS"
E0S=$(python "$REPO/scripts/teacher_e0s.py" "$TEACHER" $ELEMS 2>/dev/null | tail -1); echo "E0s (teacher): $E0S"
echo "===== 1 distillation data"
[ -f "$DS/train_ref.extxyz" ] || timeout -k 30 2400 python "$REPO/scripts/distill_data.py" --model "$TEACHER" --structure "$BASE" \
   --temps $TEMPS --scales $SCALES --equil-ps 1 --sample-ps 4 --every-ps 0.1 --dtype float32 --out-dir "$DS" 2>&1 | grep -vE "$F"
S=V1_16x1_l2_c2_r45
echo "===== 2 student $S"
[ -f "$ST/$S/$S.model" ] || timeout -k 30 3000 mace_run_train --name "$S" --work_dir "$ST/$S" --model MACE \
    --interaction_first RealAgnosticDensityInteractionBlock --interaction RealAgnosticDensityResidualInteractionBlock \
    --distance_transform Agnesi --pair_repulsion --num_interactions 2 \
    --hidden_irreps "16x0e+16x1o" --max_ell 2 --correlation 2 --r_max 4.5 --num_radial_basis 8 --radial_MLP "[16, 16]" \
    --E0s "$E0S" --train_file "$DS/train_ref.extxyz" --valid_fraction 0.1 --test_file "$DS/test_ref.extxyz" \
    --energy_key REF_energy --forces_key REF_forces --stress_key REF_stress \
    --loss stress --energy_weight 1 --forces_weight 100 --stress_weight 10000 \
    --batch_size 4 --valid_batch_size 4 --max_num_epochs 120 --lr 0.01 --scheduler_patience 10 \
    --ema --ema_decay 0.99 --default_dtype float32 --device cuda --seed 1 2>&1 | grep -vE "$F" | grep -E "Epoch|Error|error" | tail -2
[ -f "$ST/$S/$S.model" ] || { echo "student training failed"; exit 1; }
echo "===== 3 accuracy vs the teacher (held-out frames)"
NEPARG=(); [ -n "${NEP_MODEL:-}" ] && NEPARG=(--nep "$NEP_MODEL")
python "$REPO/scripts/eval_vs_teacher.py" --frames "$DS/test_ref.extxyz" --models "$ST/$S/$S.model" ${NEPARG[@]+"${NEPARG[@]}"} \
   --json "$OUT/accuracy.json" 2>&1 | grep -vE "$F"
echo "===== 4 NPT $T_NPT K, 1 bar: student vs teacher"
TP=$ST/${SYS}_$S.threepass_spline512-mliap_lammps.pt
python "$REPO/scripts/make_threepass_model.py" "$ST/$S/$S.model" --out "$TP" --spline-grid 512 2>&1 | grep -vE "$F" | tail -1
npt () {  # name model
  mkdir -p "$OUT/npt_$1"; cd "$OUT/npt_$1"; cp -f "$REPO/inputs/in.npt_general.lmp" in.lmp
  MACE3P_GRAPHS=1 MACE3P_CHUNK=512 timeout 1500 lmp -k on g 1 -sf kk -pk kokkos newton on neigh half -in in.lmp -log log.lammps \
    -var elems "$ELEMS" -var data_file "$MID" -var model_file "$2" -var temp $T_NPT -var equil_steps 20000 -var prod_steps 30000 \
    -var dt 0.001 -var dump_every 500 -var thermo_every 100 > lammps.out 2>&1
  echo "   [$1] rc=$? $(grep 'Loop time' log.lammps | tail -1)"; grep -m1 -E "Fix halt|guard" lammps.out
  python "$REPO/scripts/analyze_md.py" npt log.lammps --dt-ps 0.001 2>&1 | tail -3 | sed "s/^/   /"
  cd "$OUT"
}
npt teacher "$TEACHER_LMP"
npt student "$TP"
python "$REPO/scripts/rdf.py" "$OUT/npt_teacher/traj.lammpstrj" "$OUT/npt_student/traj.lammpstrj" --types $ELEMS --rmax 6 \
   --out "$OUT/rdf.npz" 2>&1 | grep -vE "$F" | tail -8
echo "finished $(date)"
