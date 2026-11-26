# LAMMPS patch: `src/KOKKOS/mliap_data_kokkos.cpp`

`mliap_data_kokkos.patch` makes two changes to the Kokkos ML-IAP data class. Both are performance changes; neither
alters the numbers LAMMPS passes to the Python model.

1. **nosync.** The ML-IAP neighbour and pair data (`numneighs`, `iatoms`, `ielems`, `elems`, `jatoms`, `pair_i`,
   `jelems`, `rij`, `ij`) are built on the device and read only there by the unified Python interface. The patch
   marks them as modified with `ignore_auto_sync = true`, and `sync(Device, ...)` no longer marks them host-modified.
   This removes a host-device round trip of every pair array on every step. Nsight measured about 90 ms/step of
   memcpy at 10^6 atoms.
2. **headroom.** The per-atom arrays (`natomneigh_max`) and per-pair arrays (`nneigh_max`) are reallocated with a
   10 % margin, not to the exact current size. The pair capacity is capped so that it never overflows int32 near
   2^31 pairs. Without the margin, the buffers (including the host mirror and memset) were reallocated almost every
   step while the pair count crept up. That cost about 0.25 s/step at 10^6 atoms.

## Reference version

The patch was made against **LAMMPS 11 Feb 2026** (git commit `5c6749b73a715615261e989ff15a70ce066ea8e5`), with its
bundled **Kokkos 5.0.2**.

## Applying the patch

The diff header uses `a/` and `b/` prefixes (`--- a/src/KOKKOS/mliap_data_kokkos.cpp`), so apply it from the top of
the LAMMPS source tree with `-p1`:

```bash
git clone https://github.com/lammps/lammps.git
cd lammps
git checkout 5c6749b73a715615261e989ff15a70ce066ea8e5
patch -p1 --dry-run < /path/to/mace-onegpu/lammps/mliap_data_kokkos.patch   # check first
patch -p1          < /path/to/mace-onegpu/lammps/mliap_data_kokkos.patch
# or, equivalently:  git apply /path/to/mace-onegpu/lammps/mliap_data_kokkos.patch
```

(The original diff was produced between two source trees named `lammps/` and `lammps-nosync/`. Only those two
prefixes were rewritten to `a/` and `b/`. The body of the diff is unchanged.)

## Build

This is the configuration used for the paper: CMake with the Kokkos CUDA preset, Hopper architecture (H100/H200,
`Kokkos_ARCH_HOPPER90`), ML-IAP with the Python interface, PYTHON and ML-SNAP. The compilers were GCC 13, OpenMPI 4.1
and CUDA 12.4. Run this in the Python environment where mace-torch, PyTorch and cuEquivariance are installed (see
`../requirements.txt`), so that LAMMPS embeds that interpreter.

```bash
cd lammps                                   # patched source tree
mkdir build-kokkos-cuda && cd build-kokkos-cuda
cp ../cmake/presets/kokkos-cuda.cmake .
echo 'set(Kokkos_ARCH_HOPPER90 ON CACHE BOOL "" FORCE)' >> kokkos-cuda.cmake
cmake -C ./kokkos-cuda.cmake \
  -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX="$HOME/opt/lammps-mace-onegpu" \
  -D CMAKE_CXX_STANDARD=17 -D BUILD_MPI=ON -D BUILD_SHARED_LIBS=ON \
  -D PKG_KOKKOS=ON -D Kokkos_ENABLE_CUDA=ON -D Kokkos_ENABLE_OPENMP=ON -D BUILD_OMP=ON \
  -D CMAKE_CXX_COMPILER="$PWD/../lib/kokkos/bin/nvcc_wrapper" \
  -D PKG_ML-IAP=ON -D PKG_ML-SNAP=ON -D MLIAP_ENABLE_PYTHON=ON -D PKG_PYTHON=ON \
  -D Python_EXECUTABLE="$(which python)" \
  ../cmake
cmake --build . -j 12
cmake --install .
```

For a different GPU, replace `Kokkos_ARCH_HOPPER90` with the matching Kokkos architecture flag, for example
`Kokkos_ARCH_AMPERE80` for A100.

If nvcc rejects `-Xcudafe --diag_suppress=unrecognized_pragma,--diag_suppress=128`, remove that option from the
generated `flags.make` files and rebuild:

```bash
find . -path '*/flags.make*' -type f -print0 | \
  xargs -0 -r sed -i 's/ -Xcudafe --diag_suppress=unrecognized_pragma,--diag_suppress=128//g'
cmake --build . -j 12
```

### Runtime environment notes

- Put the installed `bin/` on `PATH` and `lib64/` on `LD_LIBRARY_PATH`. Also add the Python environment's `lib/`.
- The pip-installed PyTorch ships its own cuBLAS. If LAMMPS picks up a different system cuBLAS, preload the one from
  the Python environment:
  `export LD_PRELOAD=$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cublas/lib/libcublas.so.12:$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cublas/lib/libcublasLt.so.12`
- Do not set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. It caused a SIGSEGV with this stack.
- `OMP_NUM_THREADS=1` and `MKL_THREADING_LAYER=GNU` are recommended.
- Put the repository root on `PYTHONPATH` so that LAMMPS can unpickle `mace_onegpu.*` from the model `.pt` file.

## Known defect of the reference LAMMPS version: `fix spring/self/kk`

`src/KOKKOS/fix_spring_self_kokkos.cpp` of LAMMPS 11 Feb 2026 calls `k_xoriginal.modify_host()` in `post_force`
without writing the host copy. After the device copy has been marked modified (by `DualView::resize` when
`atom->nmax` grows, or by the device-side exchange of the fix), this trips Kokkos' DualView consistency check
(`Kokkos_ENABLE_DEBUG_DUALVIEW_MODIFY_CHECK`, on by default) and the run aborts with

```
Kokkos::DualView::modify_host ERROR: Concurrent modification of host and device views in DualView "spring/self:xoriginal"
```

With the check disabled the stale host copy would silently overwrite the anchor positions instead. The defect is
independent of the ML-IAP patch above (reproduced with `pair_style lj/cut` on the patched and the unpatched build).
Upstream fixed it in September 2026 (commits 1b110d2, 3a87b51, f51d56f and 102d2d0 of lammps/lammps).
`fix_spring_self_kokkos_11Feb2026_backport.patch` brings the upstream `develop` version of
`fix_spring_self_kokkos.{cpp,h}` (as of 6 Oct 2026) to the reference source; apply it like the ML-IAP patch:

```bash
patch -p1 < /path/to/mace-onegpu/lammps/fix_spring_self_kokkos_11Feb2026_backport.patch
```

Until then, run `fix spring/self` on the host (`suffix off` before and `suffix on` after the fix command).
Other Kokkos fix and compute styles of this LAMMPS version received coherence fixes in the same upstream commits
(e.g. shake/kk, rigid/small/kk, neigh/history/kk, langevin/kk with `scale`, nh/kk, efield/kk, deform/kk,
wall/gran/kk); none of them is used by the runs of the paper.

## Known defect of the reference LAMMPS version: per-atom virial of `pair_style mliap` under Kokkos

`src/KOKKOS/pair_mliap_kokkos.cpp` of LAMMPS 11 Feb 2026 allocates its own per-atom arrays (it calls `ev_init` with
the base-class allocation turned off) but sizes and checks the per-atom virial `k_vatom` against `maxeatom` instead of
`maxvatom`, and labels it `pair:eatom`. `Pair::ev_setup` raises `maxeatom` only when a per-atom energy is requested, so
a per-atom virial alone, which is what `compute stress/atom` (or `centroid/stress/atom`) requests, leaves `k_vatom`
unallocated; the first tally writes through a null device pointer and the run dies with

```
cudaFree(arg_alloc_ptr) error( cudaErrorIllegalAddress): an illegal memory access was encountered
... LAMMPS_NS::update_pair_forces(LAMMPS_NS::MLIAPDataKokkosDevice*, double*)
```

(the fault is reported at the next `cudaFree`, not in the faulting kernel). Energies, forces and the global virial
(`compute pressure ... virial`) are unaffected, and no run of the paper requests a per-atom virial. Adding
`compute pe/atom` to the input makes the crash disappear only because it raises `maxeatom`; that masks the defect and
is not a fix. Upstream fixed it in lammps/lammps commit c5277f104 (4 Sep 2026, in `stable_30Sep2026`).
`pair_mliap_kokkos_vatom_11Feb2026_backport.patch` is that two-line change against the reference source; apply it like
the other patches:

```bash
patch -p1 < /path/to/mace-onegpu/lammps/pair_mliap_kokkos_vatom_11Feb2026_backport.patch
```

Reference source file before / after the patch: SHA256 `ed42506c80...` / `86b03ca307...`
(`src/KOKKOS/pair_mliap_kokkos.cpp`). The defect was found and confirmed by disassembly of a deployed build in a
collaborating project (per-atom stress of a Mg dislocation cell); see issue #1 of this repository.

## LAMMPS's own tests

`ctest -R mliap` (build with `-D ENABLE_TESTING=ON`) gives the same result on the patched and on the unpatched build
of the reference commit (H200 node, 2026-10-01): the five `ManybodyPairStyle:mliap_*` tests (nn, snap_chem,
snap_linear, snap_quadratic, so3) pass; in `TestMliapPyUnified`, `VersusLJMelt` passes and `VersusLJMeltGhost` fails
with bit-identical numbers on both builds (an upstream issue of the CPU path with ghost neighbours). The two Kokkos
variants of that test are skipped by LAMMPS itself in GPU builds; the Kokkos path changed by the patch is exercised by
`tests/gpu/` and `examples/quickstart.sh` of this repository. When linking the test executables, put GCC's own
`libstdc++` first (`-D CMAKE_EXE_LINKER_FLAGS="-L<gcc>/lib64 -Wl,-rpath,<gcc>/lib64"`) if the Python environment
ships an older one.

## License

See `LICENSE-NOTE`. The patches in this directory are derivatives of LAMMPS and are distributed under GPL-2.0.
