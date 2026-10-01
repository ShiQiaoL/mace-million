#!/usr/bin/env python
"""
Build the LAMMPS ML-IAP file for the exact three-pass MACE (mace_onegpu/mliap_threepass.py).
    python scripts/make_threepass_model.py mace-mpa-0-medium.model --out mace-mpa-0-medium.threepass-mliap_lammps.pt \
        [--chunk 16384] [--spline-grid 512]
Run on a GPU node (the fused cuEq conversion needs CUDA).  The saved object holds the fp32 fused cuEq model only;
fast layers / tables / Triton kernels are rebuilt inside LAMMPS on the first force call.
"""
import argparse
import os
import sys

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
# Fallback so that the script runs from a source checkout without installation: put the repository root (parent of
# scripts/) on sys.path so that the mace_onegpu package is importable.  The saved object references
# mace_onegpu.mliap_threepass, so LAMMPS needs the same root on PYTHONPATH when it loads the file.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import torch  # noqa: E402

from mace_onegpu import receiver_chunk_mace as rc  # noqa: E402
from mace_onegpu.mliap_threepass import ThreePassMLIAP  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--out", required=True)
ap.add_argument("--chunk", type=int, default=16384)
ap.add_argument("--spline-grid", type=int, default=512)
a = ap.parse_args()
model, dt = rc.build_model(a.model, "fp32_fused", "cuda")
# e3nn modules that the cuEquivariance conversion does not replace (e.g. the non-linear blocks of MACE-MH-1) keep
# TorchScript code; unpickling those inside LAMMPS crashes (C++ ABI clash), so store a rebuild recipe instead
n_script = sum(isinstance(x, torch.jit.ScriptModule) for x in model.modules())
recipe = None
if n_script:
    from mace.modules.wrapper_ops import CuEquivarianceConfig
    from mace.tools.scripts_utils import extract_config_mace_model

    torch.set_default_dtype(torch.float32)
    recipe = extract_config_mace_model(torch.load(a.model, map_location="cpu").float())
    recipe["cueq_config"] = CuEquivarianceConfig(enabled=True, layout="ir_mul", group="O3_e3nn", optimize_all=True,
                                                 conv_fusion=True)
    print(f"{n_script} TorchScript modules remain after conversion: the file stores a rebuild recipe")
obj = ThreePassMLIAP(model.to("cpu"), chunk=a.chunk, spline_grid=a.spline_grid, rebuild_config=recipe)
torch.save(obj, a.out)
print(f"saved {a.out}: chunk={a.chunk} spline_grid={a.spline_grid} dtype={dt} elements={len(obj.element_types)}")
