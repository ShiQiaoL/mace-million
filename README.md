# mace-million

*Python package name: `mace_onegpu`.*

This repository runs the MACE foundation model MACE-MPA-0 for **millions of atoms on one GPU**, with exactly the
original MACE energies and forces. It contains the code for the paper *(paper in preparation)*.

The repository has three parts:

1. **Exact receiver-chunked three-pass evaluation** of two-layer MACE models (`mace_onegpu/`). The only features that
   couple different receiver atoms are the layer-1 node features `h1` (512 numbers per atom for MPA-0 medium). The
   energy and forces can therefore be computed chunk by chunk over the receiver atoms:
   - **pass A:** `H1 = layer1` (no gradient), then a ghost exchange;
   - **pass B:** layer-2 energy, which gives pair forces and `G1 = dE2/dh1`, then a reverse ghost exchange;
   - **pass C:** layer 1 again with gradient, loss `E1 + G1 . h1`, which gives the remaining pair forces.

   Only `H1`, `G1` and the per-pair arrays persist. Every wide intermediate (edge messages, symmetric-contraction
   tensors, the autograd graph) lives for one chunk only. The evaluator is integrated into LAMMPS through the
   **ML-IAP unified Python interface** with Kokkos (`pair_style mliap unified`). It uses exact speed-ups: species-block
   `skip_tp`, receiver-only node operations, fp64-sampled tabulated radial weights with Triton kernels, CUDA graphs
   and optionally `torch.compile`. Multi-GPU domain decomposition works, because the ghost exchange goes through
   LAMMPS.
2. **Small MACE students** distilled from the teacher, and **on-the-fly supervision** of the student by the teacher
   inside LAMMPS (`scripts/distill_data.py`, `scripts/al_driver.py`).
3. **Two changes to LAMMPS** `src/KOKKOS/mliap_data_kokkos.cpp` (`lammps/mliap_data_kokkos.patch`): no per-step host
   auto-sync of the ML-IAP pair data, and reallocation of the pair buffers with headroom.

## Tested stack

Python 3.11, mace-torch 0.3.15, PyTorch 2.5.1, cuEquivariance 0.9.1, Triton 3.1.0, CUDA 12.4.
LAMMPS 11 Feb 2026 (commit `5c6749b73a715615261e989ff15a70ce066ea8e5`) with Kokkos 5.0.2.
GPU: NVIDIA H200.

## Repository layout

```
mace_onegpu/      Python package (three-pass evaluator, LAMMPS ML-IAP wrapper, Triton kernels)
scripts/          command-line tools (model conversion, distillation, active learning, analysis)
inputs/           LAMMPS input decks (all knobs are -var variables)
examples/         bash examples: speed benchmark, distilling a student, supervised long run
lammps/           LAMMPS patch, build instructions, licence note (GPL-2.0)
tests/            GPU-free smoke tests (pytest)
```

## Installation

### 1. Python environment

```bash
conda create -n mace-onegpu python=3.11 && conda activate mace-onegpu
pip install -r requirements.txt       # plus the cuequivariance-ops-torch-cu12 wheel that matches your CUDA
```

The package is used from the source checkout. Put the repository root on `PYTHONPATH`:

```bash
export PYTHONPATH=/path/to/mace-onegpu:$PYTHONPATH
```

This is required for LAMMPS. The model file (`.pt`) pickles `mace_onegpu.mliap_threepass.ThreePassMLIAP` and
`mace_onegpu.fused_conv.FusedConvTP` by module path, so the Python interpreter embedded in LAMMPS must be able to
import `mace_onegpu`. The scripts in `scripts/` insert the repository root into `sys.path` themselves, so they also
run without it. `scripts/al_driver.py` adds the root to `PYTHONPATH` of the LAMMPS processes it launches.

At run time, `conv_spline.py` and `fused_edge_conv.py` write generated Triton kernel sources to
`mace_onegpu/_generated/`. That directory must be writable.

### 2. Patched LAMMPS

See [`lammps/README.md`](lammps/README.md). In short: check out LAMMPS commit `5c6749b7`, then run
`patch -p1 < lammps/mliap_data_kokkos.patch` and build with CMake: Kokkos CUDA (`Kokkos_ARCH_HOPPER90` for H100/H200),
`PKG_ML-IAP`, `MLIAP_ENABLE_PYTHON`, `PKG_PYTHON`, `PKG_ML-SNAP`, shared libraries, MPI. Unpatched LAMMPS also works;
the patch only removes per-step host copies and reallocations.

## Quick start

**1. Build the three-pass LAMMPS model** on a GPU node (the fused cuEquivariance conversion needs CUDA):

```bash
python scripts/make_threepass_model.py mace-mpa-0-medium.model \
    --out mace-mpa-0-medium.threepass-mliap_lammps.pt --chunk 16384 --spline-grid 512
```

The file holds the fp32 fused cuEquivariance model. The fast layers, spline tables and Triton kernels are rebuilt
inside LAMMPS on the first force call, for the elements that are present.

**2. Run LAMMPS** with `pair_style mliap unified`:

```
pair_style      mliap unified mace-mpa-0-medium.threepass-mliap_lammps.pt 0
pair_coeff      * * O H
```

For example, with the input decks in `inputs/`:

```bash
MACE3P_GRAPHS=1 MACE3P_CHUNK=16384 MACE3P_LOG=mace3p.jsonl \
  lmp -k on g 1 -sf kk -pk kokkos newton on neigh half -in inputs/in.water_scan.lmp \
      -var data_file water.data -var model_file mace-mpa-0-medium.threepass-mliap_lammps.pt \
      -var run_steps 100 -var skin 2.0 -var dt 0.001
```

For multi-GPU runs use `mpirun -np <n> lmp -k on g <n> ...`. The ghost exchange of `H1` and `G1` goes through LAMMPS.
If `forward_exchange` segfaults on two GPUs, add `gpu/aware on` to the `-pk kokkos` options (see ACEsuit/mace#1432).

### Environment variables (read by `mace_onegpu/mliap_threepass.py`)

| Variable | Default | Meaning |
|---|---|---|
| `MACE3P_CHUNK` | value stored in the `.pt` (`--chunk`, 16384) | receivers per chunk. Smaller chunks use less memory; larger chunks are faster. Students use 512 to 65536. |
| `MACE3P_GRAPHS` | `0` | `1`: capture the three per-chunk bodies of full chunks as CUDA graphs (static buffers, replayed per chunk). The partial last chunk runs eagerly. |
| `MACE3P_COMPILE` | `0` | `1`: `torch.compile` (inductor) the layer functions used inside the CUDA graphs (fixed shapes, compiled once). |
| `MACE3P_COMPILE_MODE` | unset (inductor default) | inductor mode for `MACE3P_COMPILE=1`, e.g. `max-autotune-no-cudagraphs` (the CUDA graphs are captured by the wrapper itself). |
| `MACE3P_CAP_MARGIN` | `1.25` | headroom factor of the CUDA-graph capacities (nodes, edges, atoms incl. ghosts). A recapture takes seconds, or tens of seconds with compile. |
| `MACE3P_CACHE_GRAPHS` | `0` | `1`: build the chunk topology once per force call (pass A) and reuse it in passes B and C (about 0.75 kB/atom). |
| `MACE3P_LOG` | unset | path of a JSONL log with one line per force call: step, nlocal/ntotal/npairs, per-pass times, LAMMPS gap time, peak GPU memory, CUDA-graph recaptures. |
| `MACE3P_PROFILE` | `0` | `1`: synchronise and time the local-graph construction (`graph_*` fields in the log). |
| `MACE3P_SKIN` | `2.0` | neighbour skin (A) used **only** by the int32 safety guard. Set it to the `neighbor` skin of the input. |
| `MACE3P_ALLOW_HUGE` | `0` | `1`: disable the int32 neighbour-slot guard (only deliberately; see below). |
| `MACE3P_FMAX_ABORT` | `100.0` | abort threshold for the largest per-pair force (eV/A). |
| `MACE3P_GC` | `1` | `0`: diagnostic only, freeze and disable Python's cyclic garbage collector for the run. |

Other variables: `MACE_RC_VERBOSE=1` prints per-chunk information in `receiver_chunk_mace.py`. `ACE_L1_LAYOUT`
selects the layout of the experimental `ace_layer1.py` (`graphed_three_pass.py --ace-layer1`). Recommended for LAMMPS
runs: `OMP_NUM_THREADS=1`, `MKL_THREADING_LAYER=GNU`. Do not set
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, which caused a SIGSEGV with this stack.

### Safety guards

`ThreePassMLIAP.compute_forces` aborts with a Python `RuntimeError` instead of letting LAMMPS/Kokkos crash the GPU:

- **int32 neighbour-slot guard.** LAMMPS/Kokkos indexes the full neighbour list with 32-bit integers. The wrapper
  estimates the number of neighbour slots including the skin as `npairs * ((r_max + skin) / r_max)^3`. If the
  estimate exceeds `0.95 * 2^31`, the run aborts. Beyond 2^31 slots Kokkos makes an illegal GPU memory access that
  can take a node down. On an H200, 8.96 M water atoms (0.89 of 2^31) ran and 9.88 M (0.98) crashed. Use fewer atoms
  per GPU, a smaller skin (and set `MACE3P_SKIN` to match), or more GPUs. `MACE3P_ALLOW_HUGE=1` overrides the guard.
- **Force guard.** If any per-atom energy is non-finite, or the largest per-pair force reaches
  `MACE3P_FMAX_ABORT` (default **100 eV/A**), the run aborts before LAMMPS integrates the step. NaN positions
  otherwise end in an illegal memory access inside Kokkos neighbour binning. The model is then usually extrapolating.

Two further measures avoid int32 limits: the `H1`/`G1` ghost exchanges are split by feature columns so that every
call stays below 2^31 elements, and the patched LAMMPS caps its pair-buffer headroom below 2^31.

## Students and on-the-fly supervision

The teacher (MACE-MPA-0) is exact but slow. A small MACE student (for example 16x0e+16x1o, l=2, correlation 2,
r 4.25-4.5 A) runs the MD much faster with the same three-pass machinery, and the teacher checks it on the fly.

1. **Teacher E0s:** `python scripts/teacher_e0s.py mace-mpa-0-medium.model Si O` prints the `--E0s` dict, so the
   student shares the teacher's energy zero.
2. **Distillation data:** `scripts/distill_data.py` runs teacher Langevin MD over a grid of temperatures and densities
   and writes `train_ref.extxyz` / `test_ref.extxyz` with `REF_energy`, `REF_forces` and `REF_stress`.
3. **Student training:** `mace_run_train` with `--loss stress` (see `examples/distill_student.sh` for the full command).
4. **Validation:** `scripts/eval_vs_teacher.py` compares against the teacher on held-out frames, optionally with
   NEP89 via `--nep` (needs calorine). `scripts/dft_bench.py` and `scripts/classical_eval.py` compare everything
   against VASP single points (`dft_bench.py` needs `VASP_PP_PATH`).
5. **Supervised MD** (`scripts/al_driver.py`): the student runs MD segments in LAMMPS (`inputs/in.al_segment.lmp`).
   At every `--dump-every` steps the frames are checked against the teacher. Small frames use ASE/MACE; large frames
   use a LAMMPS rerun of the teacher's three-pass model (`inputs/in.al_check.lmp`). The checks cover force RMSE, the
   fraction of atoms above a per-atom error, and the pressure difference. If a threshold is exceeded, or the segment
   stopped early, the student is fine-tuned on the teacher-labelled frames. For large systems these are clusters
   carved around the worst atoms. The run then continues. State is saved after every cycle in `state.json`, so the
   driver can be resumed.

Examples (plain bash; every path comes from an environment variable):

- `examples/speed_benchmark.sh`: speed vs N (15 k to 7.9 M water atoms), optionally against NEP89/GPUMD.
- `examples/distill_student.sh`: distil and validate a student for a new system (presets `SYS=sio2|cuni`).
- `examples/supervised_run.sh`: 1 ns NPT of a large system with the student under teacher supervision.

## Development tools

The package modules also run as benchmarks and self-tests on a GPU: `python -m mace_onegpu.receiver_chunk_mace
--validate ...`, `python -m mace_onegpu.graphed_three_pass ...`, `python -m mace_onegpu.fast_layers ...`,
`python -m mace_onegpu.conv_spline ...`. See each file's docstring. `scripts/run_case.sh` and
`scripts/summarize_case.py` are the harness used for the memory scans. Their `MACE_MEMPROBE*` columns need a separate
instrumentation that is not included here.

Tests (no GPU needed; module imports are skipped when torch/mace/triton are not installed):

```bash
python -m pytest tests
```

## Citation

*(paper in preparation)*

## License

MIT (see `LICENSE`), except `lammps/mliap_data_kokkos.patch`. That patch is a derivative of LAMMPS and is
distributed under GPL-2.0 (see `lammps/LICENSE-NOTE`).
