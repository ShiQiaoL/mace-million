# Changelog

## Unreleased

- SevenNet + cuEquivariance inside LAMMPS: chunk tensors padded to per-chunk high-water-mark shapes (`SevenNetLayers.layer_padded`,
  `MACE3P_SEVENN_PAD`), because cuEquivariance prepares kernels per shape (~2 s per new shape) and LAMMPS changes the
  pair count every step. 8.5-9.4e4 atom-steps/s from 12,288 to 983,040 water atoms, 23 GB at 983,040 (was 1.3e4 / 39 GB with e3nn; 0.5-1.0e4 with
  cuEquivariance before padding); results unchanged (4.5e-6 eV/A vs fp64, NVE drift 0.06 meV/atom over 5 ps).
- SevenNet: capacity of the original evaluation measured with SevenNet's cuEquivariance backend as well
  (98,304 water atoms in LAMMPS; e3nn backend 21,504); the chunked evaluation with cuEquivariance is 5.9x faster
  than with e3nn standalone (0.135 s for 12,288 atoms) but not yet inside LAMMPS; `experimental/sevennet_shape_variation.py`.
- `scripts/al_driver.py --carve-radius` defaults to 13 A (was 7 A): cluster labels at 7 A deviate from the bulk
  correction by ~50% for water and ~3x for SiO2; at 13 A by ~1% (validate_clusters.py). In the 192,000-atom water
  test the two radii gave the same retrained error (19.8 / 19.9 meV/A); a fixed-student control shows 60% of the
  decrease from 23.8 meV/A comes from the relaxation of the box (benchmarks/data/al_state_V1_192k_*.json).
- SevenNet: `load_sevennet(..., cueq=True)`, `LPassSevenNetMLIAP(..., cueq=True)` and
  `make_lpass_sevennet_model.py --cueq` select SevenNet's cuEquivariance backend (sevenn[cueq12]);
  `official_sevennet_mliap()` builds SevenNet's own ML-IAP wrapper without TorchScript, as the reference for speed and
  memory in the same LAMMPS build (forces within 4.3e-6 eV/A of the fp64 reference; out of memory at 24,000 water atoms
  on a 141-GB H200).
- `experimental/validate_numerics.py`: error decomposition (reordering, radial tables, fp32 rounding, production path),
  convergence of the tabulated radial weights with the number of knots, finite differences of energy vs forces and of
  strain vs virial.
- `experimental/validate_clusters.py`: teacher forces in carved vacuum clusters (the production carving code) vs the
  periodic frame, as a function of cluster radius and depth below the cut surface.
- `experimental/bench_lpass_sevennet.py`, `experimental/profile_lpass_sevennet.py`.
- Figures: the 340-fold reduction is stated per added atom.

## 0.1.3 (2026-10-02)

- SevenNet inside LAMMPS: `mace_onegpu/lpass_sevennet.py` (exact L-layer receiver-chunked evaluation; ML-IAP
  wrapper `LPassSevenNetMLIAP` with one forward and one reverse ghost exchange per layer),
  `scripts/make_lpass_sevennet_model.py`, `examples/sevennet_lammps.sh`. SevenNet-0 on water (H200): forces within
  4.7e-6 eV/A of the fp64 reference, NVE drift 0.04 meV/atom over 5 ps (1,536 atoms), 983,040 atoms at 1.3e4
  atom-steps/s with a 39-GB peak; the original evaluation fits at most 21,504 atoms.
- `experimental/sevennet_lpass.py` now uses the package module and writes fp64 references (`--write-ref`).
- README: other foundation models and SevenNet.

## 0.1.2 (2026-10-02)

- Other MACE foundation models: readout-head selection for multi-head models (`MACE3P_HEAD`, e.g. `omat_pbe` for
  MACE-MH-1; energies, E0s and scale/shift use the same head); exact generic layers (`fast_layers.GenericLayers`) for
  interaction blocks that the specialised fast path does not cover (e.g. the non-linear blocks of MACE-MH-1).
- Models that keep e3nn TorchScript modules after the cuEquivariance conversion are stored as a rebuild recipe
  (class, config, state dict) and rebuilt inside LAMMPS with fx code generation only: unpickling TorchScript inside
  LAMMPS crashes (std::regex template instances clash with liblammps' C++11-ABI copies).
- CUDA-graph-safe replacement of MACE's multi-head `mask_head` (bit-identical output).
- `scripts/compare_forces.py ref --head`.
- `experimental/sevennet_lpass.py`: the receiver-chunked exact evaluation generalised to L message-passing layers,
  demonstrated on SevenNet-0 (NequIP architecture, 5 layers).
- `lammps/README.md`: results of LAMMPS's own ML-IAP tests on the patched and unpatched builds.

## 0.1.1 (2026-09-30)

- `pyproject.toml`: `pip install -e .`.
- `examples/quickstart.sh` with a 192-atom water box (`examples/data/`): model conversion, 10 LAMMPS steps, force parity
  against the fp64 reference with PASS/FAIL (`scripts/compare_forces.py --tol-max`).
- `tests/gpu/test_threepass.py`: exactness, chunk invariance, periodic replica and fp32/fp64 checks (skipped without a GPU).
- README: installation via pip, cmake from pip, tests section; consolidated benchmark figures.

## 0.1.0 (2026-09-28)

First release, matching the manuscript draft.

- Exact receiver-chunked three-pass evaluation of two-layer MACE models (`mace_onegpu/`), with species-block skip
  connections, tabulated radial weights (Triton), fused pass-A convolution, CUDA graphs and optional `torch.compile`.
- LAMMPS ML-IAP (unified, Kokkos) interface with ghost exchange of the layer-1 features and their gradient,
  32-bit neighbour-slot guard and force guard.
- LAMMPS patch for `src/KOKKOS/mliap_data_kokkos.cpp` (no host mirroring of device-only pair data; neighbour arrays
  grow with headroom), against commit `5c6749b` (11 Feb 2026).
- Distillation, evaluation, DFT-benchmark and classical-force-field scripts; on-the-fly supervision driver
  (`scripts/al_driver.py`) with LAMMPS rerun checks and bond-completed cluster carving.
- Benchmark source data and figure scripts (`benchmarks/`).
