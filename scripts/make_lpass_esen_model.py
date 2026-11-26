#!/usr/bin/env python
"""
Build the LAMMPS ML-IAP file for an eSEN checkpoint (fairchem-core 1.10) with the exact receiver-chunked
evaluation (mace_onegpu/lpass_esen.py).
    python scripts/make_lpass_esen_model.py esen_30m_oam.pt --elements H O --out esen_oam.lpass-mliap_lammps.pt [--chunk 8192]
Run in the eSEN environment. The file stores the checkpoint path, the element list (the LAMMPS type order of
`pair_coeff * * <file> H O`) and the settings; the model is built inside LAMMPS on the first force call (run LAMMPS with
the eSEN environment's site-packages first on PYTHONPATH, as for SevenNet). MACE3P_ESEN_FULL=1 at run time selects
the stock full-graph evaluation for comparison; MACE3P_ESEN_FRAME=random restores fairchem's random edge frame.
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import torch  # noqa: E402

from mace_onegpu.lpass_esen import LPassESENMLIAP  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("checkpoint", help="eSEN checkpoint (.pt, fairchem-core 1.10 format)")
ap.add_argument("--elements", nargs="+", required=True, help="element symbols in LAMMPS type order, e.g. H O")
ap.add_argument("--out", required=True)
ap.add_argument("--chunk", type=int, default=8192)
ap.add_argument("--dtype", default="float32", choices=["float32", "float64"])
a = ap.parse_args()
obj = LPassESENMLIAP(a.checkpoint, a.elements, chunk=a.chunk, dtype=a.dtype)
torch.save(obj, a.out)
print(f"saved {a.out}: {obj.checkpoint}  elements={obj.element_types}  r_cut={2 * obj.rcutfac}  chunk={obj.chunk}  "
      f"dtype={obj.dtype_name}")
