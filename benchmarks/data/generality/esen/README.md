# eSEN-30M-OAM through the exact receiver-chunked evaluation (third architecture family)

Model: eSEN-30M-OAM (fairchem-core 1.10 checkpoint, downloaded by the author from facebook/OMAT24; 10 eSEN blocks +
edge-degree embedding = 11 message layers, 22 chunk sweeps; lmax 3, mmax 2, 128 sphere/hidden channels, cutoff 6 Å,
gate activation, spectral MLP, conservative forces). Code: mace-million branch `esen`, `mace_onegpu/lpass_esen.py`,
`experimental/esen_lpass.py`, `experimental/esen_exactness_cpu.py`. Water boxes of the paper (192-atom base cell and
its replicas), one H200 (141 GB).

## Round 1 (2026-10-20; PBS 677503 standalone, 677504 LAMMPS) — files in job170_173_standalone/, job171_174_lammps/

| quantity | value | source |
|---|---|---|
| gauge noise of the original (random edge frame, fp64, 192 atoms, two calls) | max dF 4.7e-15 eV/Å, dE 1.1e-13 eV | check_fp64.out (NOISE) |
| chunked vs original, CPU fp64, 192-atom water frame | max dF 7e-15 eV/Å, dE 1e-13 eV | esen_cpu_real.log (hopper login node) |
| replica check 4×4×4 = 12,288 atoms, fp32 chunked vs fp64 original on the base cell | max dF 1.35e-4, RMS 1.5e-5 eV/Å; dE/atom 2e-7 eV | check_fp32.out |
| LAMMPS parity, 192 atoms, fp32 chunked vs fp64 reference | max dF 7.5e-5, RMS 8.2e-6 eV/Å | mw171 (step 2) |
| resident memory of the chunked evaluation | 98.3 kB/atom (fp32) | scan_fp32.out |
| chunked standalone fp32, chunks of 8,192 receivers | 12,288: 93 GB peak; 98,304: 103 GB; 192,000: 114 GB; 421,824: out of memory (working set of an 8,192-chunk ≈ 90 GB) | scan_fp32.out |
| chunked throughput (standalone and LAMMPS) | ≈ 6.35×10^2 atom-steps/s at 12k–192k atoms | scan_fp32.out, mw171 |
| original full-graph evaluation in LAMMPS (same wrapper, MACE3P_ESEN_FULL=1) | 192 atoms: 17 GB; 1,536 atoms: 129.6 GB peak, 8.2×10^2 atom-steps/s; 5,184 atoms: out of memory | mw171 (step 4) |
| official OCPCalculator cross-check / capacity | not obtained in round 1 (device mismatch in the driver; fixed for round 2) | — |

Round 2 (PBS 678235 standalone, 678236 LAMMPS): chunks of 1,024/2,048 receivers, official OCPCalculator as its own GPU
instance, fp64 replica check, capacity estimate from the measured working set, LAMMPS throughput at 12,288–98,304 atoms.

Notes
- The released model is invariant to fairchem's random per-edge frame to rounding (gate activation + spectral MLP);
  the fixed frame of lpass_esen.py is therefore only a reproducibility measure. A random-weight test model with the
  S2-grid activation is not invariant (0.1–0.3 eV/Å between two calls), see the CPU test log.
- Absolute cost: the exact eSEN-30M-OAM evaluation runs at ~6×10^2 atom-steps/s, about 350 times slower than
  three-pass MACE-MPA-0 (2.2×10^5); the chunked path is within ~25% of the full-graph path where both fit (1,536 atoms).

## Round 2 (2026-10-20 evening; PBS 678235 standalone, 678236 LAMMPS) — job173_*.out, mw174_esen_lmp2.o678236

| quantity | value | source |
|---|---|---|
| official OCPCalculator (fp32, fairchem's own graph, same frame) vs our fp64 reference, 192 atoms | max dF 1.1e-4, RMS 1.4e-5 eV/Å; dE −1.4e-3 eV; 7.6×10^2 atom-steps/s | job173_check_fp64.out |
| replica check 4×4×4 = 12,288 atoms, fp64 chunked (chunks of 1,024) vs fp64 original on the base cell | max dF 4.5e-13, RMS 7.7e-14 eV/Å; dE/atom −9e-16 eV | job173_check_fp64.out |
| chunked fp32, chunks of 1,024: 98,304 atoms | 23.1 GB peak = 98.3 kB/atom resident + 13.4 GB working set; 5.8×10^2 atom-steps/s | job173_scan_a.out |
| chunked fp32, 1,247,616 atoms (19×19×18) | out of memory (estimate 1.2 M was too optimistic; 983,040 and 1,105,920 in round 3) | job173_scan_b.out |
| LAMMPS, chunks of 2,048: 12,288 / 41,472 / 98,304 atoms | 624 / 616 / 620 atom-steps/s; peak 26.8 / 32.3 / 41.7 GB | mw174 |
| official evaluation capacity | N=192 only (19.7 GB peak, 4.3×10^2 atom-steps/s); the loop crashed in the driver, round 3 | job173_stock_fp32.out |

## Rounds 3–4 (2026-10-21; PBS 680340, 680411) — job176_178_capacity/

| quantity | value | source |
|---|---|---|
| official OCPCalculator capacity (fp32) | 192 atoms: 19.7 GB, 4.8×10^2 atom-steps/s; 384 atoms: 39.0 GB, 6.4×10^2; 1,536 atoms: out of memory | stock_fp32.out, stock_boundary.out |
| chunked fp32, chunks of 1,024 | 393,216 atoms: 61.6 GB allocated / 83.6 GB reserved, 5.9×10^2; 589,824: 92.3 / 112.9 GB, 5.9×10^2; **786,432: 123.0 / 146.7 GB, 5.9×10^2 (largest that ran)**; 983,040: out of memory (job176) | scan_large.out, mw176 |

Per-atom allocated peak 156.5 kB/atom = 98.3 kB resident node features + 16 kB of gradient arrays in the backward sweeps
+ the 13.4-GB working set spread over N; the reserved memory (PyTorch caching allocator) is 20–25 GB higher.

Numbers used in the paper (r3k, Table 2 and Section 3.5): own evaluation 384 atoms (ASE/OCPCalculator; 1,536 out of
memory) and 1,536 atoms (same wrapper, full graph, in LAMMPS; 5,184 out of memory); chunked 786,432 standalone (123 GB)
and ≥98,304 in LAMMPS (41.7 GB); throughput 6.2×10^2 (chunked, LAMMPS) vs 8.2×10^2 (full graph, 1,536 atoms) and
6.4×10^2 (OCPCalculator, 384 atoms); replica check 4.5×10^-13 eV/Å; resident 98.3 kB/atom.
