#!/usr/bin/env python
"""Print the teacher's isolated-atom energies (E0s) for the given elements as a mace_run_train --E0s dict.
Element-general: read from the model's atomic_energies_fn, so the student shares the teacher's energy zero.

    python teacher_e0s.py mace-mpa-0-medium.model Si O      ->  {14: -..., 8: -...}
"""
import os
import sys

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
import torch  # noqa: E402
from ase.data import atomic_numbers  # noqa: E402

model = torch.load(sys.argv[1], map_location="cpu")
zs = [int(z) for z in model.atomic_numbers]
e0 = model.atomic_energies_fn.atomic_energies.detach().double().reshape(-1, len(zs))[0]
out = {}
for el in sys.argv[2:]:
    z = atomic_numbers[el]
    out[z] = float(e0[zs.index(z)])
print("{" + ", ".join(f"{z}: {e:.8f}" for z, e in out.items()) + "}")
