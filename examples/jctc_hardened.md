# Hardened SiO₂ supervision from the initial student

Run the commands below in Bash from the code repository root, with the patched one-GPU LAMMPS build on `PATH`. Use the MACE 0.3.15 per-atom force-weight overlay for masked cluster training. To create the overlay in a fresh directory:

```bash
mkdir -p runs/mace-overlay
cp -R "$(python -c 'import mace; print(next(iter(mace.__path__)))')" runs/mace-overlay/mace
python patches/patch_mace_nodeweights.py runs/mace-overlay/mace
export PYTHONPATH="$PWD/runs/mace-overlay:$PWD${PYTHONPATH:+:$PYTHONPATH}"
```

Set `DATA_ROOT` and `MODEL_ROOT` to the local checkouts of the companion dataset and model repositories. Set `TEACHER_MODEL` to the MACE-MPA-0 medium checkpoint. Create its LAMMPS model and a 35 × 35 × 35 starting supercell (1,029,000 atoms, type 1 Si and type 2 O). The release omits the original campaign's initial coordinate file. For a replay from those coordinates, set `START_DATA` to your original LAMMPS data file.

```bash
mkdir -p runs/start
export TEACHER_LMP="$PWD/runs/mpa0-lammps.pt"
python scripts/make_threepass_model.py "$TEACHER_MODEL" --out "$TEACHER_LMP" --spline-grid 512
python scripts/build_general.py --system sio2 --out-dir runs/start --reps 35
export START_DATA="$PWD/runs/start/sio2_35x35x35_N1029000.data"
```

Both commands below start from the released V1 student, target five accepted 2-ps NPT segments at 3000 K and compare forces every 1 ps. The initial 10% validation split is held fixed. Select one command for a campaign. Separate work directories keep the two examples independently resumable.

The nine preset values are recorded in `experiments/jctc_rev_al/sio2_anchored_rollback.json`. The cycle records retain which gates fired but omit the historical command line. The example uses a force RMSE limit of 0.10 eV/Å, a pressure limit of 1000 bar and a flagged fraction limit of 10⁻⁵ at 0.5 eV/Å. These choices reproduce the recorded gate decisions across the five archived segments, where only the flagged fraction triggered repair.

```bash
run=(--teacher "$TEACHER_MODEL" --teacher-lmp "$TEACHER_LMP"
     --student "$MODEL_ROOT/students/sio2/V1_r4.5/V1_16x1_l2_c2_r45.model"
     --train-xyz "$DATA_ROOT/training_sets/sio2/train_432.extxyz" --data "$START_DATA"
     --elems Si O --e0s "$(python scripts/teacher_e0s.py "$TEACHER_MODEL" Si O)"
     --temp 3000 --press 1.01325 --dt 0.001 --segment-steps 2000 --dump-every 1000 --cycles 5
     --f-rmse-max 0.10 --dp-max 1000 --f-max-max 0.5 --f-over-frac 1e-5
     --carve-radius 13 --carve-mode worst --carve-frames 2 --carve-per-frame 4 --valid-fraction 0.1
     --chunk 65536 --teacher-chunk 16384 --compile --time-budget-s 86400)

# Published preset
python scripts/al_driver.py --work-dir runs/sio2-preset "${run[@]}" --preset jctc-hardened

# Equivalent explicit settings
python scripts/al_driver.py --work-dir runs/sio2-explicit "${run[@]}" \
    --inner-radius 6.0 --cluster-energy 0 --al-weight 10.0 --short-range-anchors 1 \
    --sr-anchor-points 8 --sr-anchor-weight 5.0 --finetune-lr 0.002 --rollback-fmax 5.0 --fixed-val-split 1
```

Explicit flags override the preset in either argument order. `state.json` records the checked frames, worst errors, retraining result, accepted checkpoint and rollback reason for each attempted segment. The companion dataset records the published settings and cycle summaries in `experiments/jctc_rev_al/sio2_anchored_rollback.json`.
