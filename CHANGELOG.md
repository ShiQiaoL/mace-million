# Changelog

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
