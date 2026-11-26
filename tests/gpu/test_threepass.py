"""GPU regression tests of the exact three-pass evaluation (no LAMMPS).

Skipped unless CUDA is available and MACE_ONEGPU_TEACHER points to mace-mpa-0-medium.model (or another two-layer
MACE model).  Structure: the 192-atom water box of examples/data.

  * fp64 three-pass against the standard full-graph evaluation of the same model: forces to 1e-9 eV/A;
  * chunk invariance: chunks of 48 and 192 receivers give the same fp64 forces to 1e-9 eV/A;
  * a 2x2x2 periodic replica (1,536 atoms) reproduces the base-cell forces atom by atom (fp64, 1e-9 eV/A), and the
    energy per atom;
  * fp32 (the production precision) against fp64: forces to 1e-3 eV/A.
Run:  MACE_ONEGPU_TEACHER=/path/mace-mpa-0-medium.model python -m pytest -q tests/gpu
"""
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mace")
from ase import io as aseio  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "examples", "data", "water64_1x1x1_N192.data")
TEACHER = os.environ.get("MACE_ONEGPU_TEACHER", "")
pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and os.path.isfile(TEACHER)),
    reason="needs CUDA and MACE_ONEGPU_TEACHER=<two-layer MACE .model>")
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")


def read_box():
    try:
        return aseio.read(DATA, format="lammps-data", atom_style="atomic", Z_of_type={1: 8, 2: 1}, units="metal")
    except TypeError:
        return aseio.read(DATA, format="lammps-data", style="atomic", Z_of_type={1: 8, 2: 1}, units="metal")


def three_pass(variant, atoms, chunk):
    """Energy and forces (in the original atom order) from the receiver-chunked evaluation."""
    from mace_onegpu import receiver_chunk_mace as rc

    model, dt = rc.build_model(TEACHER, variant, "cuda")
    zs = [int(z) for z in model.atomic_numbers]
    pos = torch.tensor(atoms.get_positions(), dtype=torch.float64, device="cuda")
    sp = torch.tensor([zs.index(int(z)) for z in atoms.get_atomic_numbers()], dtype=torch.int16, device="cuda")
    sysm = rc.System(pos, np.diag(np.array(atoms.get_cell())), sp, float(model.r_max), "cuda")
    sysm.build_neighbors()
    E, F, _, _ = rc.ReceiverChunkMACE(model, dt, chunk).compute(sysm, log=lambda *a: None)
    Fo = torch.empty_like(F)
    Fo[sysm.perm] = F                      # sorted -> original order
    return model, E, Fo.cpu().numpy()


def full_graph(model, atoms):
    from mace_onegpu.ase_gradient_probe import make_batch

    out = model(make_batch(model, atoms, "cuda").to_dict(), training=False, compute_force=True)
    return float(out["energy"].detach().double().sum()), out["forces"].detach().double().cpu().numpy()


@pytest.fixture(scope="module")
def box():
    return read_box()


def test_fp64_matches_full_graph(box):
    model, E, F = three_pass("fp64_fused", box, chunk=192)
    Eref, Fref = full_graph(model, box)
    assert abs(E - Eref) < 1e-7 * len(box)
    assert np.abs(F - Fref).max() < 1e-9


def test_chunk_invariance(box):
    _, E1, F1 = three_pass("fp64_fused", box, chunk=48)
    _, E2, F2 = three_pass("fp64_fused", box, chunk=192)
    assert abs(E1 - E2) < 1e-8 * len(box)
    assert np.abs(F1 - F2).max() < 1e-9


def test_replica_reproduces_base_cell(box):
    rep = box.repeat((2, 2, 2))
    model, E, F = three_pass("fp64_fused", rep, chunk=512)
    Eb, Fb = full_graph(model, box)
    n = len(box)
    assert abs(E / len(rep) - Eb / n) < 1e-8
    # atom i of copy k is atom i % n of the base cell (ase.repeat orders copies contiguously)
    assert np.abs(F - np.tile(Fb, (8, 1))).max() < 1e-9


def test_fp32_close_to_fp64(box):
    _, E32, F32 = three_pass("fp32_fused", box, chunk=192)
    _, E64, F64 = three_pass("fp64_fused", box, chunk=192)
    assert abs(E32 - E64) < 1e-3 * len(box)
    assert np.abs(F32 - F64).max() < 1e-3
