#!/usr/bin/env python
"""
Build the LAMMPS ML-IAP file for a SevenNet model with the exact L-layer receiver-chunked evaluation
(mace_onegpu/lpass_sevennet.py).
    python scripts/make_lpass_sevennet_model.py 7net-0 --out 7net-0.lpass-mliap_lammps.pt [--chunk 8192] [--dtype float32]
Run in the SevenNet environment. The file stores the checkpoint path and the settings; the model is built inside
LAMMPS on the first force call (run LAMMPS with the SevenNet environment's site-packages first on PYTHONPATH).
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import torch  # noqa: E402

from mace_onegpu.lpass_sevennet import LPassSevenNetMLIAP  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("checkpoint", help="SevenNet checkpoint path or pretrained name (e.g. 7net-0)")
ap.add_argument("--out", required=True)
ap.add_argument("--chunk", type=int, default=8192)
ap.add_argument("--dtype", default="float32", choices=["float32", "float64"])
ap.add_argument("--cueq", action="store_true", help="SevenNet's cuEquivariance backend (sevenn[cueq12])")
a = ap.parse_args()
obj = LPassSevenNetMLIAP(a.checkpoint, chunk=a.chunk, dtype=a.dtype, cueq=a.cueq)
torch.save(obj, a.out)
print(f"saved {a.out}: {obj.checkpoint}  elements={len(obj.element_types)}  r_cut={2 * obj.rcutfac}  "
      f"chunk={obj.chunk}  dtype={obj.dtype_name}")
