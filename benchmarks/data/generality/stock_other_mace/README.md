# "Original" evaluation of the other MACE foundation models (extended Table 2, author request 2026-10-19)

Water boxes of the paper, one H200 (141 GB), fp32. Jobs: 677505 (job172, unfused fp32 ML-IAP; superseded),
678237 (job175, ASE calculators for MH-1 and MPA-0), 680341 (job177, fp32_fused ML-IAP in LAMMPS + ASE for MATPES).

## Stock LAMMPS ML-IAP path (fp32_fused = cuEquivariance with conv_fusion, the variant behind the paper's MPA-0 stock numbers)

| model | 3-step evaluation fits | 30 MD steps fit | 30-step throughput | source |
|---|---|---|---|---|
| MACE-MATPES-r2SCAN-0 | 86,016 (98,304 out of memory) | 65,856 (86,016 out of memory at step 5) | 6.5×10^4 atom-steps/s at 65,856 | mw177 |
| MACE-MATPES-PBE-0 | 86,016 (98,304 out of memory) | 65,856 (86,016 out of memory) | 6.4×10^4 atom-steps/s at 65,856 | mw177 |
| MACE-MPA-0 (paper) | 86,016 | 41,472 for 300-step runs | 6.5×10^4 at 41,472 | SI Table S15 |
| MACE-MH-1 | the stock ML-IAP file aborts inside LAMMPS (TorchScript modules cannot be deserialised in the LAMMPS process) | — | — | mw172 |

The unfused fp32 variant (job172, what `mace.cli.create_lammps_model --format=mliap` produces) fits 24,000 atoms for 3 steps
and runs out of memory at 41,472; 30-step MD at 24,000: 7.9×10^4 (r2SCAN) / 6.8×10^4 (PBE) atom-steps/s.

## Official ASE calculator, standalone (mace.calculators.MACECalculator, cuEquivariance, fp32; second call timed)

| model | largest replica that fits | peak at that size | out of memory at | throughput (largest size) | source |
|---|---|---|---|---|---|
| MACE-MPA-0 | 65,856 | 101.6 GB (1.54 MB/atom) | 98,304 | 1.9×10^4 atom-steps/s | job175 mpa0_official.out |
| MACE-MATPES-r2SCAN-0 | 65,856 | 101.6 GB (1.54 MB/atom) | 98,304 | 1.9×10^4 | job177 matpes_r2scan_official.out |
| MACE-MATPES-PBE-0 | 65,856 | 101.6 GB (1.54 MB/atom) | 98,304 | 1.9×10^4 | job177 matpes_pbe_official.out |
| MACE-MH-1 (head omat_pbe) | 41,472 | 117.1 GB (2.82 MB/atom) | 65,856 | 1.6×10^4 | job175 mh1_official.out |

The 1.54 MB/atom of the ASE path agrees with the 1.55 MB/atom of the stock LAMMPS path quoted in the paper (Section 3.1).
