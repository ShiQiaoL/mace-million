"""Smoke tests that need no GPU.

* every .py file of the package and of scripts/ byte-compiles (py_compile);
* the package itself imports;
* every package module imports when its heavy dependencies (torch, mace, triton, cuequivariance, e3nn, ase) are
  installed. It is skipped otherwise. Importing does not touch the GPU.
Scripts are only compiled, not imported: several of them do their work at import time (argument parsing, model loading).
"""
import importlib
import os
import py_compile
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

PKG = os.path.join(ROOT, "mace_onegpu")
SCRIPTS = os.path.join(ROOT, "scripts")
PY_FILES = sorted(os.path.join(d, f) for d in (PKG, SCRIPTS) for f in os.listdir(d) if f.endswith(".py"))
MODULES = sorted(f[:-3] for f in os.listdir(PKG) if f.endswith(".py") and f != "__init__.py")

# heavy third-party dependencies per module (any missing -> the import test is skipped)
NEEDS = {
    "ace_layer1": ["torch"],
    "ase_gradient_probe": ["torch", "ase", "mace"],
    "conv_spline": ["torch", "triton"],
    "fast_layers": ["torch"],
    "fused_conv": ["torch"],
    "fused_edge_conv": ["torch", "triton"],
    "fused_radial": ["torch", "triton"],
    "graphed_three_pass": ["torch", "mace"],
    "mliap_threepass": ["torch"],
    "lpass_sevennet": ["torch", "sevenn"],
    "radial_3xtf32": ["torch"],
    "receiver_chunk_mace": ["torch", "numpy"],
    "spline_radial": ["torch"],
    "spline_w": ["torch", "triton"],
}


@pytest.mark.parametrize("path", PY_FILES, ids=lambda p: os.path.relpath(p, ROOT))
def test_py_compile(path, tmp_path):
    py_compile.compile(path, doraise=True, cfile=str(tmp_path / "out.pyc"))


def test_package_imports():
    import mace_onegpu

    assert mace_onegpu.__version__


def test_every_module_is_listed():
    assert set(MODULES) == set(NEEDS)


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name):
    for dep in NEEDS[name]:
        pytest.importorskip(dep)
    importlib.import_module(f"mace_onegpu.{name}")
