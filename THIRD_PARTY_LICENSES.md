# Third-party licence notes

Everything in this repository is distributed under the MIT License (see `LICENSE`), with one exception.

The patches in `lammps/` (`mliap_data_kokkos.patch`, `fix_spring_self_kokkos_11Feb2026_backport.patch` and
`pair_mliap_kokkos_vatom_11Feb2026_backport.patch`) modify source files of LAMMPS (https://www.lammps.org) and are
therefore derivative works of LAMMPS. Like LAMMPS, they are distributed under the GNU General Public License, version 2
(GPL-2.0); see `lammps/LICENSE-NOTE` and the LICENSE file of the LAMMPS source distribution. The MIT License does not
apply to these patches.
