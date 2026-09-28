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
obj = ThreePassMLIAP(model.to("cpu"), chunk=a.chunk, spline_grid=a.spline_grid)
torch.save(obj, a.out)
print(f"saved {a.out}: chunk={a.chunk} spline_grid={a.spline_grid} dtype={dt} elements={len(obj.element_types)}")
