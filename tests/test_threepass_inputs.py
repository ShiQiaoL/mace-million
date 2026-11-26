"""CPU regressions for cell geometry and chemical identities at the CLI input boundary."""
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
ase = pytest.importorskip("ase")
from ase.io import write

from mace_onegpu.receiver_chunk_mace import load_structure


def test_triclinic_base_is_rejected_before_replication(tmp_path):
    path = tmp_path / "tilted.extxyz"
    atoms = ase.Atoms("O", positions=[[1, 1, 1]], cell=[[12, 0, 0], [2, 12, 0], [0, 0, 12]], pbc=True)
    write(path, atoms)
    args = SimpleNamespace(base=path, structure=None, rep=(1, 2, 1))
    with pytest.raises(ValueError, match="axis-aligned orthorhombic cell"):
        load_structure(args, SimpleNamespace(atomic_numbers=[8]), torch.float64, "cpu")


def test_data_atom_types_and_replication(tmp_path):
    path = tmp_path / "oxide.data"
    atoms = ase.Atoms("OSi", positions=[[0, 0, 0], [2, 3, 4]], cell=[12, 13, 14], pbc=True)
    write(path, atoms, format="lammps-data", specorder=["O", "Si"])
    args = SimpleNamespace(base=path, structure=None, rep=(1, 2, 1), atom_types="1:O")
    model = SimpleNamespace(atomic_numbers=[8, 14])
    with pytest.raises(ValueError, match="Unknown LAMMPS atom type ID 2"):
        load_structure(args, model, torch.float64, "cpu")
    args.atom_types = "1:O,2:Si"
    pos, box, species, _ = load_structure(args, model, torch.float64, "cpu")
    expected = atoms.repeat(args.rep)
    np.testing.assert_allclose(pos.numpy(), expected.positions)
    np.testing.assert_allclose(box, expected.cell.lengths())
    assert species.tolist() == [0, 1, 0, 1]
