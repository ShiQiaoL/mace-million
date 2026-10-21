# mace-million

*Python package name: `mace_onegpu`.*

This repository runs the MACE foundation model MACE-MPA-0 for **millions of atoms on one GPU**, with exactly the
original MACE energies and forces. It contains the code for the paper *Foundation-Model Supervision of Distilled
Potentials for Million-Atom Molecular Dynamics* (X. Liu, Y. Wang and L. Shen, arXiv:2610.23371, 2026; see [Citation](#citation)).

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
pip install -r requirements.txt
pip install cuequivariance-ops-torch-cu12==0.9.1 cmake   # CUDA kernels for your CUDA major version; cmake for LAMMPS
pip install -e .
```

If large pip downloads are unreliable on your cluster, install PyTorch first from the conda channel and let pip
skip it: `conda create -n mace-onegpu python=3.11 pytorch=2.5.1 pytorch-cuda=12.4 torchtriton=3.1.0 -c pytorch -c nvidia`,
then the two `pip install` lines above.

`pip install -e .` makes `mace_onegpu` importable from any directory, including from the Python interpreter embedded
in LAMMPS. That is required: the model file (`.pt`) pickles `mace_onegpu.mliap_threepass.ThreePassMLIAP` and
`mace_onegpu.fused_conv.FusedConvTP` by module path. (Without the install, put the repository root on `PYTHONPATH`;
the scripts in `scripts/` insert it into `sys.path` themselves.)

At run time, `conv_spline.py` and `fused_edge_conv.py` write generated Triton kernel sources to
`mace_onegpu/_generated/`. That directory must be writable.

### 2. Patched LAMMPS

See [`lammps/README.md`](lammps/README.md). In short: check out LAMMPS commit `5c6749b7`, then run
`patch -p1 < lammps/mliap_data_kokkos.patch` and build with CMake: Kokkos CUDA (`Kokkos_ARCH_HOPPER90` for H100/H200),
`PKG_ML-IAP`, `MLIAP_ENABLE_PYTHON`, `PKG_PYTHON`, `PKG_ML-SNAP`, shared libraries, MPI. Unpatched LAMMPS also works;
the patch only removes per-step host copies and reallocations.

## Quick start

With a GPU, the Python environment and `lmp` on `PATH`, one script does the three steps below on a 192-atom water
box and checks the result against the original MACE (fp64) forces:

```bash
bash examples/quickstart.sh        # ~2 min; downloads mace-mpa-0-medium.model (0.2 GB) unless TEACHER points to it
```

It ends with `PASS` when the LAMMPS forces agree with the reference to 1e-3 eV/A (the fp32 round-off of the model is
about 1e-4 eV/A).

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
| `MACE3P_TOPO` | `batched` | how the receiver chunks' local graphs (node lists, local edge indices) are built: `batched` builds all chunks of a force call in a few passes over groups of consecutive chunks (one `torch.unique` per group); `chunk` builds them one chunk at a time (the earlier path, host-bound: about a dozen launches and a device sync per chunk). Both give identical graphs. |
| `MACE3P_TOPO_EDGES` | `30000000` | edges per group of the batched build (bounds its transient memory, about 50 B per edge). |
| `MACE3P_CACHE_GRAPHS` | `0` | with `MACE3P_TOPO=chunk` only: `1` keeps the per-chunk topologies of pass A for passes B and C (4 B per edge). The batched build always keeps them. |
| `MACE3P_LOG` | unset | path of a JSONL log with one line per force call: step, nlocal/ntotal/npairs, per-pass times (`A`, `B`, `C`; `A_topology` and `A_load` inside pass A), LAMMPS gap time, the peak torch allocation of each pass (`peak_A`, `peak_B`, `peak_C`, `peak_out`) and overall, CUDA-graph recaptures. |
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

## Other foundation models and SevenNet

**Other MACE models.** The same files run other two-layer MACE foundation models; tested on water (H200):
MACE-MATPES-r2SCAN-0 and MACE-MATPES-PBE-0 (architecture of MACE-MPA-0, fast path) and MACE-MH-1 (non-linear
interaction blocks; exact generic layers). Select the read-out head of multi-head models with `MACE3P_HEAD`, e.g.

```bash
MACE3P_HEAD=omat_pbe python scripts/make_threepass_model.py mace-mh-1.model --out mh1.pt
MACE3P_HEAD=omat_pbe lmp -k on g 1 -sf kk -pk kokkos newton on neigh half -in ... -var model_file mh1.pt
```

Models that keep e3nn TorchScript modules after the cuEquivariance conversion (MACE-MH-1) are stored as a rebuild
recipe and rebuilt inside LAMMPS without TorchScript, which cannot be deserialised in the LAMMPS process.

| model | replica check, fp64 (eV/A) | LAMMPS vs fp64 (eV/A) | 983,040 water atoms (atom-steps/s, peak GB) |
|---|---|---|---|
| MACE-MATPES-r2SCAN-0 | 4.2e-13 | 1.7e-5 | 1.97e5, 31 |
| MACE-MATPES-PBE-0 | 4.0e-13 | 1.6e-5 | 1.97e5, 31 |
| MACE-MH-1 (OMat24 head) | 4.4e-13 | 2.2e-5 | 1.5e4, 52 (chunk 4096) |
| SevenNet-0 (5 layers) | 4.4e-13 | 4.7e-6 | 1.3e4, 39 (e3nn); 8.5e4, 23 (cuEquivariance) |

**SevenNet (NequIP-type, L layers).** `mace_onegpu/lpass_sevennet.py` generalises the three passes to L layers
(L forward passes with one ghost exchange each, L recomputing backward passes with one reverse exchange each). The
original SevenNet-0 evaluation fits at most 21,504 water atoms on a 141-GB H200 with the e3nn backend (98,304 with cuEquivariance); the chunked evaluation runs 983,040
atoms in LAMMPS with a 39-GB peak and 3.1 million atoms standalone. It uses the same LAMMPS build: because MACE 0.3.15
requires e3nn 0.4.4 and SevenNet e3nn >= 0.5, the SevenNet environment is put first on `PYTHONPATH` (same Python
minor version required):

```bash
SEVENN_ENV=/path/to/sevennet-env bash examples/sevennet_lammps.sh     # builds the model file, checks forces: PASS
```

With SevenNet's cuEquivariance backend (`pip install sevenn[cueq12]` in the SevenNet environment, or
cuEquivariance 0.9.1 in a venv overlay) build the model file with `--cueq`:

```bash
python scripts/make_lpass_sevennet_model.py 7net-0 --out 7net-0.lpass_cueq.pt --cueq
```

cuEquivariance prepares its kernels per tensor shape, and inside LAMMPS the pair count changes every step, so the
chunk tensors are padded to shapes that only grow (`MACE3P_SEVENN_PAD`, on by default with `--cueq`; buckets
`MACE3P_NODE_BUCKET`=8192 nodes and `MACE3P_E_ALIGN`=65536 edges). The result is unchanged (forces within 4.5e-6 eV/A
of the fp64 reference, NVE drift 0.06 meV/atom over 5 ps); SevenNet-0 then runs at 8.5-9.4e4 atom-steps/s from 12,288
to 983,040 water atoms (23-GB peak at 983,040; e3nn backend: 1.3e4, 39 GB). SevenNet's own ML-IAP interface with cuEquivariance
reaches 5.5-9.5e4 atom-steps/s and at most 98,304 atoms on the H200. None of the kernels of the MACE fast path were
ported to the SevenNet path.

**eSEN (equivariant transformer, fairchem-core 1.10).** `mace_onegpu/lpass_esen.py` applies the same L-layer schedule to
eSEN checkpoints (eSEN-30M-OAM: ten blocks with SO(2) convolutions plus the edge-degree embedding = 11 message layers,
22 sweeps). It runs in a separate environment (Python 3.11, torch 2.5.1, `fairchem-core==1.10.0` installed with
`--no-deps` on that torch, `scipy<1.15`), put first on `PYTHONPATH` inside LAMMPS as for SevenNet. The original code
draws a random per-edge frame; `install_edge_frame()` fixes it (the released model is invariant to it to rounding, so
nothing changes but reproducibility). Measured on one H200 (water): the chunked evaluation reproduces the original forces
to 4.5e-13 eV/A (fp64, 12,288-atom replica) and 7.5e-5 eV/A in LAMMPS (fp32); it keeps 98 kB/atom resident, runs
98,304 atoms in LAMMPS at 42 GB and 6.2e2 atom-steps/s, whereas fairchem's OCPCalculator runs out of memory at 1,536
atoms and the same wrapper without chunking (`MACE3P_ESEN_FULL=1`) at 5,184.

```bash
python scripts/make_lpass_esen_model.py esen_30m_oam.pt --elements O H --out esen_oam.lpass.pt --chunk 2048
python experimental/esen_exactness_cpu.py --checkpoint esen_30m_oam.pt --xyz water.extxyz --chunk 64 --rep 1   # CPU, fp64
python experimental/esen_lpass.py --checkpoint esen_30m_oam.pt --base water64_1x1x1.extxyz --rep 4 4 4 --check --official
```

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

## Tests

```bash
python -m pytest tests                                           # smoke tests, no GPU: byte-compile and import
MACE_ONEGPU_TEACHER=/path/mace-mpa-0-medium.model python -m pytest tests/gpu   # GPU: exactness of the three passes
bash examples/quickstart.sh                                      # GPU + LAMMPS: end-to-end force parity
```

`tests/gpu/test_threepass.py` checks, on the 192-atom water box, that the fp64 three-pass evaluation reproduces the
full-graph model to 1e-9 eV/A, that the result does not depend on the chunk size, that a 2x2x2 periodic replica
reproduces the base-cell forces atom by atom, and that fp32 stays within 1e-3 eV/A of fp64. The tests are skipped
without CUDA or without `MACE_ONEGPU_TEACHER`. Last run (H200, mace-torch 0.3.15, PyTorch 2.5.1, 2026-09-30): 46 passed; `quickstart.sh` PASS with max|dF| = 2.1e-5 eV/A against the fp64 reference.

## Citation

Xiaoqing Liu, Yangshuai Wang and Lei Shen, *Foundation-Model Supervision of Distilled Potentials for Million-Atom
Molecular Dynamics*, arXiv:2610.23371 (2026), https://doi.org/10.48550/arXiv.2610.23371. The journal reference will be
added when the paper is published. The code snapshot used for the paper is commit `d572103` (GitHub release
`paper-d572103`; the same archive, `mace-onegpu-d572103.tar.gz`, is in the data repository).

```bibtex
@misc{liu2026supervision,
  author        = {Liu, Xiaoqing and Wang, Yangshuai and Shen, Lei},
  title         = {Foundation-Model Supervision of Distilled Potentials for Million-Atom Molecular Dynamics},
  year          = {2026},
  eprint        = {2610.23371},
  archivePrefix = {arXiv},
  primaryClass  = {physics.comp-ph},
  doi           = {10.48550/arXiv.2610.23371}
}
```

## License

MIT (see `LICENSE`), except the patches in `lammps/`. Those patches are derivatives of LAMMPS and are
distributed under GPL-2.0 (see `THIRD_PARTY_LICENSES.md` and `lammps/LICENSE-NOTE`).

## Benchmarks and source data

`benchmarks/data/` holds the numbers behind the paper's figures (accuracy against DFT and against the teacher,
throughput scaling, NVE energy logs, radial distribution functions, on-the-fly supervision histories), and
`benchmarks/figures/scripts/` the scripts that draw the figures from them:

```bash
cd benchmarks/figures/scripts && python fig3_throughput.py   # -> benchmarks/figures/fig3_throughput.pdf
```

The throughput figures report steady-state rates of short runs (compilation and CUDA-graph capture excluded). For
production runs, use the loop time that LAMMPS prints for a whole segment; it includes every neighbour-list rebuild
and any communication overhead. In our 983,040-atom water runs the neighbour skin (2-4 A, rebuilds every 16-50
steps) did not change the rate.

The training sets, the DFT reference frames, the check records of every supervised run and the student models are on
the Hugging Face Hub: dataset [shiqiao123/mace-foundation-model-supervision-data](https://huggingface.co/datasets/shiqiao123/mace-foundation-model-supervision-data)
and models [shiqiao123/mace-foundation-model-supervision-models](https://huggingface.co/shiqiao123/mace-foundation-model-supervision-models).
The dataset also holds a source archive of the code snapshot used for the paper (`code_snapshot/mace-onegpu-d572103.tar.gz`,
identical to release `paper-d572103`).
