# Changelog

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
