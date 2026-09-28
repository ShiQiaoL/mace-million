#!/usr/bin/env bash
# Run ONE LAMMPS ML-IAP case with full instrumentation and append one row to a summary TSV.
#
# usage: run_case.sh <case_dir> <model_pt> <data_file> <run_steps> <skin> <ngpu> [dump_forces] [extra lmp -var ...]
# env:   KIT (repository root, holds inputs/ and scripts/), CASE_TIMEOUT (s, default 1500), SUMMARY (tsv path),
#        MEMPROBE_SNAPSHOT_STEP (default 0)
# Note:  the MACE_MEMPROBE* variables exported below are read by a separate memory-probe instrumentation
#        (sitecustomize.py) that is NOT part of this release; without it they are ignored and the memprobe columns
#        of the summary stay empty.  For ThreePassMLIAP use MACE3P_LOG=<file.jsonl> (per-call timings, peak memory).
#
# Collected per case: status (OK / OOM / FAIL / TIMEOUT), memprobe torch peaks (allocated / reserved, bytes),
# nlocal / nghost / npairs from memprobe, LAMMPS log stats (FullNghs, Nghost, Pair%, timesteps/s), nvidia-smi peak.
set -uo pipefail
CASE_DIR=$1; MODEL_PT=$2; DATA_FILE=$3; RUN_STEPS=$4; SKIN=$5; NGPU=$6; DUMP_FORCES=${7:-0}; shift 7 || shift $#
EXTRA_VARS="$*"
KIT=${KIT:?KIT not set}
CASE_TIMEOUT=${CASE_TIMEOUT:-1500}
SUMMARY=${SUMMARY:-$(dirname "$CASE_DIR")/summary.tsv}
SNAP=${MEMPROBE_SNAPSHOT_STEP:-0}

mkdir -p "$CASE_DIR"
cd "$CASE_DIR" || exit 1
cp -f "$KIT/inputs/in.water_scan.lmp" in.lmp

export MACE_MEMPROBE=1
export MACE_MEMPROBE_OUT="$CASE_DIR/memprobe"
export MACE_MEMPROBE_SNAPSHOT_STEP=$SNAP
export MACE_MEMPROBE_SNAPSHOT_ENTRIES=${MEMPROBE_SNAPSHOT_ENTRIES:-200000}

# GPU sampler (0.5 s) -> gpu.csv ; torch peaks from memprobe are the authoritative numbers
: > gpu.csv
( while true; do nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits >> gpu.csv 2>/dev/null || true; sleep 0.5; done ) &
SAMPLER=$!

# GPU_AWARE=on|off (optional) -> "-pk kokkos ... gpu/aware on|off"; unset = LAMMPS auto-detection.
# (ACEsuit/mace #1432: 2-GPU forward_exchange segfault with gpu/aware off, fixed by gpu/aware on.)
GA=""; [ -n "${GPU_AWARE:-}" ] && GA="gpu/aware $GPU_AWARE"
if [ "$NGPU" -gt 1 ]; then
  LAUNCH="mpirun -np $NGPU --bind-to none"
  KK="-k on g $NGPU -sf kk -pk kokkos newton on neigh half $GA"
else
  LAUNCH=""
  KK="-k on g 1 -sf kk -pk kokkos newton on neigh half $GA"
fi
# MACE_EDGE_CHUNK (optional, exact edge streaming via the memprobe edge_chunk_patch.py, not part of this release) is
# inherited from the environment.

START=$(date +%s.%N)
timeout --signal=TERM --kill-after=30 "$CASE_TIMEOUT" \
  $LAUNCH lmp $KK -in in.lmp -log log.lammps \
    -var data_file "$DATA_FILE" -var model_file "$MODEL_PT" -var run_steps "$RUN_STEPS" \
    -var skin "$SKIN" -var dump_forces "$DUMP_FORCES" $EXTRA_VARS > lammps.out 2>&1
RC=$?
END=$(date +%s.%N)
kill $SAMPLER 2>/dev/null; wait $SAMPLER 2>/dev/null

STATUS=OK
if [ $RC -eq 124 ] || [ $RC -eq 137 ]; then STATUS=TIMEOUT
elif grep -qiE "out of memory|OutOfMemoryError|CUBLAS_STATUS_ALLOC_FAILED|cudaErrorMemoryAllocation" lammps.out; then STATUS=OOM
elif [ $RC -ne 0 ]; then STATUS="FAIL($RC)"
elif ! grep -q "Total wall time" log.lammps 2>/dev/null; then STATUS="FAIL(nowall)"
fi

# status is recorded by bash first (the scan driver reads THIS file, not summary.tsv), then the TSV row is built by python;
# if python fails for any reason a minimal fallback row is appended so the scan never loses a case silently.
echo "$STATUS" > "$CASE_DIR/status.txt"
WALL=$(awk "BEGIN{print $END-$START}")
if ! python "$KIT/scripts/summarize_case.py" --case-dir "$CASE_DIR" --status "$STATUS" --rc "$RC" \
     --model "$MODEL_PT" --data "$DATA_FILE" --steps "$RUN_STEPS" --skin "$SKIN" --ngpu "$NGPU" \
     --wall "$WALL" --summary "$SUMMARY" 2> "$CASE_DIR/summarize.err"; then
  echo "[run_case] WARNING summarize_case.py failed (see $CASE_DIR/summarize.err); appending fallback row"
  [ -f "$SUMMARY" ] || printf 'case\tstatus\trc\tvariant\tdata\tnatoms\tsteps\tskin\tngpu\n' > "$SUMMARY"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$(basename "$CASE_DIR")" "$STATUS" "$RC" "" "$(basename "$DATA_FILE")" \
     "$(basename "$DATA_FILE" | sed -E 's/.*_N([0-9]+).*/\1/')" "$RUN_STEPS" "$SKIN" "$NGPU" >> "$SUMMARY"
fi
echo "[run_case] $(basename "$CASE_DIR") -> $STATUS (rc=$RC, wall=${WALL}s)"
# free GPU memory between cases is guaranteed: each case is a separate process
exit 0
