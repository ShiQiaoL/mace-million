"""CPU regression tests for segment acceptance, repair and rollback."""
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import al_driver as al


REQUIRED = ["--work-dir", "AL", "--teacher", "teacher", "--student", "student", "--train-xyz", "train.xyz",
            "--data", "start.data", "--elems", "H", "--e0s", "{1:0}"]


def test_sparse_single_atom_forces_retraining_and_rollback():
    n, error = 1_029_000, 6.0
    row = dict(f_rmse=error / np.sqrt(3 * n), frac_over=1 / n, f_max=error, dP_bar=0.0)
    assert row["f_rmse"] * 1000 == pytest.approx(3.415, abs=0.001)
    assert row["frac_over"] == pytest.approx(9.72e-7, abs=5e-10)
    _, bad, rollback = al.segment_gates([row], False, al.parse_args(REQUIRED))
    assert bad is True and rollback is True


def test_hardened_preset_and_explicit_overrides():
    a = al.parse_args(REQUIRED + ["--preset", "jctc-hardened"])
    assert (a.inner_radius, a.cluster_energy, a.al_weight, a.short_range_anchors, a.sr_anchor_points,
            a.sr_anchor_weight, a.finetune_lr, a.rollback_fmax, a.fixed_val_split) == (
                6.0, 0, 10.0, 1, 8, 5.0, 0.002, 5.0, 1)
    a = al.parse_args(REQUIRED + ["--sr-anchor-points", "3", "--preset", "jctc-hardened", "--fixed-val-split", "0"])
    assert a.sr_anchor_points == 3 and a.fixed_val_split == 0
    defaults = al.parse_args(REQUIRED)
    assert (defaults.inner_radius, defaults.cluster_energy, defaults.sr_anchor_points) == (0.0, 1, 7)


def dump(step, fields, atoms, count=None):
    return (f"ITEM: TIMESTEP\n{step}\nITEM: NUMBER OF ATOMS\n{len(atoms) if count is None else count}\n"
            "ITEM: BOX BOUNDS pp pp pp\n0 10\n0 10\n0 10\n"
            f"ITEM: ATOMS id {fields}\n" + "".join(line + "\n" for line in atoms))


def test_direct_checks_validate_forces_even_on_the_initial_frame():
    from ase import Atoms
    from ase.calculators.singlepoint import SinglePointCalculator

    at = Atoms("H", positions=[[0, 0, 0]], cell=[10, 10, 10], pbc=True)
    teacher = SinglePointCalculator(at, energy=0, forces=[[0, 0, 0]], stress=np.zeros(6))
    student = SinglePointCalculator(at, energy=0, forces=[[np.nan, 0, 0]], stress=np.zeros(6))
    with pytest.raises(al.FrameCheckError, match="non-finite force"):
        al.check_frames([at, at], teacher, student, 0.5, skip_first=True)


@pytest.fixture
def run_cycle(monkeypatch, tmp_path):
    """Use actual driver, dump readers and state file with CPU stand-ins for LAMMPS and training."""
    from ase import Atoms
    from ase.calculators.calculator import Calculator, all_changes
    from ase.io import write

    class Teacher(Calculator):
        implemented_properties = ["energy", "forces", "stress"]

        def calculate(self, atoms=None, properties=None, system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            self.results = dict(energy=0.0, forces=np.zeros((len(atoms), 3)), stress=np.zeros(6))

    for name in ["mace", "mace.modules", "mace.modules.loss", "mace.calculators"]:
        mod = ModuleType(name)
        monkeypatch.setitem(sys.modules, name, mod)
        if "." in name:
            parent, child = name.rsplit(".", 1)
            monkeypatch.setattr(sys.modules[parent], child, mod, raising=False)
    sys.modules["mace.calculators"].MACECalculator = lambda **kwargs: Teacher()

    student = tmp_path / "initial.model"
    student.write_bytes(b"student_c0")
    train = tmp_path / "train.extxyz"
    write(train, Atoms("H", positions=[[0, 0, 0]], cell=[10, 10, 10], pbc=True))
    wd = tmp_path / "run"
    invocation = ["--work-dir", str(wd), "--teacher", "teacher", "--teacher-lmp", "teacher.pt",
                  "--student", str(student), "--train-xyz", str(train), "--data", str(tmp_path / "start.data"),
                  "--elems", "H", "--e0s", "{1:0}", "--dump-every", "1", "--segment-steps", "1",
                  "--cycles", "1", "--max-train-atoms", "1", "--short-range-anchors", "0",
                  "--f-rmse-max", "100", "--f-over-frac", "1"]
    force_tail = ["2 0 0 0", "1 6 0 0"]  # reversed on purpose; largest error belongs to ID 1

    def run(fault="", train_rc=0, candidate=True, expect_stop=True, max_rollbacks=0):
        def md(cmd, cwd, **kwargs):
            cd = Path(cwd)
            (cd / "segment_end.restart").touch()
            (cd / "log.lammps").write_text("Step Density\n0 1.0\n1 1.0\n")
            pos = ["1 1 1 0 0", "2 1 2 0 0"]
            coords = dump(0, "type x y z", pos)
            if fault != "both_truncated":
                coords += dump(1, "type x y z", pos)
            if fault == "nan_position":
                coords = coords.replace("1 1 1 0 0", "1 1 nan 0 0", 1)
            (cd / "segment.lammpstrj").write_text(coords)
            return SimpleNamespace(returncode=0)

        def rerun(cd, st, model, tag, a, env):
            path = Path(cd) / f"{tag}.dump"
            zeros = ["1 0 0 0", "2 0 0 0"]
            initial = dump(0, "fx fy fz", zeros)
            end = dump(1, "fx fy fz", zeros if tag == "teacher" or fault == "clean" else force_tail)
            if tag == "student":
                if fault == "truncated":
                    end = dump(1, "fx fy fz", force_tail[:1], count=2)
                elif fault == "missing_frame":
                    end = ""
                elif fault == "timestep":
                    end = dump(2, "fx fy fz", force_tail)
                elif fault == "ids":
                    end = dump(1, "fx fy fz", ["3 0 0 0", "1 0 0 0"])
                elif fault == "count":
                    end = dump(1, "fx fy fz", zeros[:1])
                elif fault == "nan_force":
                    initial = dump(0, "fx fy fz", ["1 0 nan 0", "2 0 0 0"])
                elif fault == "rerun_position":
                    end = dump(1, "fx fy fz x y z", ["1 6 0 0 nan 0 0", "2 0 0 0 2 0 0"])
            path.write_text(initial + end)
            return str(path), {0: 0, 1: 0}

        def external(cmd, *args, **kwargs):
            if cmd[0] != "mace_run_train":
                Path(cmd[cmd.index("--out") + 1]).touch()
                return 0
            if candidate:
                root = Path(cmd[cmd.index("--work_dir") + 1])
                root.mkdir()
                name = cmd[cmd.index("--name") + 1]
                previous = Path(cmd[cmd.index("--foundation_model") + 1]).read_bytes()
                (root / f"{name}.model").write_bytes(previous if fault == "identical_model" else name.encode())
            return train_rc

        def carve(at, errors, *args, **kwargs):
            assert list(errors) == [6, 0]
            return [at[:1]]

        monkeypatch.setattr(al.subprocess, "run", md)
        monkeypatch.setattr(al, "lammps_check", rerun)
        monkeypatch.setattr(al, "sh", external)
        monkeypatch.setattr(al, "carve_clusters", carve)
        args = invocation + ["--max-rollbacks", str(max_rollbacks)]
        if expect_stop:
            with pytest.raises(RuntimeError, match="rollback_without_model_update"):
                al.main(args)
        else:
            al.main(args)
        return json.loads((wd / "state.json").read_text())

    return run


@pytest.mark.parametrize("fault", ["truncated", "missing_frame", "both_truncated", "timestep", "ids", "count",
                                    "nan_force", "nan_position", "rerun_position"])
def test_corrupt_frame_is_never_accepted(run_cycle, fault):
    state = run_cycle(fault)
    rec, = state["history"]
    assert rec["frame_mismatch_or_nonfinite"] and rec["frame_error"]
    assert rec["halted"] and rec["bad"] and rec["rolled_back"]
    assert rec["stop_reason"] == "rollback_without_model_update"
    assert state["accepted"] == 0 and state["start"] == rec["rewind_to"]


@pytest.mark.parametrize("train_rc,candidate", [(1, True), (0, False)])
def test_failed_rollback_retraining_stops_with_prior_checkpoint(run_cycle, train_rc, candidate):
    state = run_cycle(train_rc=train_rc, candidate=candidate)
    rec, = state["history"]
    assert rec["bad"] and rec["rolled_back"] and rec["carved"] == [1]
    assert rec["train_failed"] and rec["train_rc"] == train_rc and not rec["retrained"]
    assert rec["stop_reason"] == "rollback_without_model_update"
    assert state["start"] == rec["start_used"] == rec["rewind_to"]
    assert state["accepted"] == 0 and state["student"].endswith("student_c0.model")


def test_successful_rollback_switches_model_and_keeps_prior_restart(run_cycle):
    state = run_cycle(expect_stop=False)
    rec, = state["history"]
    assert rec["bad"] and rec["rolled_back"] and rec["retrained"] and not rec["train_failed"]
    assert rec["worst"]["f_max"] == 6.0 and rec["carved"] == [1]
    assert state["start"] == rec["rewind_to"] and state["accepted"] == 0
    assert Path(state["student"]).read_bytes() == b"student_c1"
    assert state["stop_reason"] == "max_rollbacks_exceeded"  # zero budget permits no replay


def test_zero_rollback_budget_accepts_clean_segment(run_cycle):
    state = run_cycle(fault="clean", expect_stop=False, max_rollbacks=0)
    assert state["accepted"] == state["cycle"] == 1
    assert state["rollbacks"] == 0 and state["stop_reason"] is None


def test_identical_candidate_at_new_path_stops_rollback(run_cycle):
    state = run_cycle(fault="identical_model", max_rollbacks=3)
    rec, = state["history"]
    previous = Path(state["student"])
    candidate = previous.parent / "cycle000" / "finetune" / "student_c1.model"
    assert previous.name == "student_c0.model"
    assert previous.read_bytes() == candidate.read_bytes() == b"student_c0"
    assert not (previous.parent / "student_c1.model").exists()
    assert rec["train_rc"] == 0 and not rec["retrained"] and rec["rolled_back"]
    assert state["stop_reason"] == rec["stop_reason"] == "rollback_without_model_update"
    assert state["accepted"] == 0 and state["start"] == rec["rewind_to"]


def test_max_rollbacks_stops_after_exactly_three(run_cycle):
    state = run_cycle(expect_stop=False, max_rollbacks=3)
    assert state["cycle"] == state["rollbacks"] == len(state["history"]) == 3
    assert all(rec["rolled_back"] and rec["retrained"] for rec in state["history"])
    assert state["accepted"] == 0 and state["stop_reason"] == "max_rollbacks_exceeded"
    assert state["history"][-1]["stop_reason"] == state["stop_reason"]
    assert run_cycle(expect_stop=False, max_rollbacks=3) == state
    assert not (Path(state["student"]).parent / "cycle003").exists()
