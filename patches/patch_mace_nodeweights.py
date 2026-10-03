"""Per-atom force weights for MACE 0.3.15 (overlay copy, not the installed package).

usage: python patch_mace_nodeweights.py <overlay>/mace

A training frame may carry an array key `forces_weight` (one value per atom, default 1). The force term of the
loss is multiplied by it, rescaled to mean 1 over the batch, so that masking atoms (weight 0: the vacuum buffer of a
carved cluster) neither changes the magnitude of the force term nor lets the buffer atoms' forces enter the fit.
Touches data/utils.py (read the array), modules/loss.py (apply it in the MSE, normed and conditional-Huber force
terms). AtomicData already turns unknown per-atom properties into [n_atoms, 1] tensors that batch along atoms.
"""
import os
import sys

root = sys.argv[1]
n = 0


def patch(rel, old, new, count=1):
    global n
    p = os.path.join(root, rel)
    s = open(p).read()
    assert s.count(old) == count, f"{rel}: expected {count} of {old[:60]!r}, found {s.count(old)}"
    open(p, "w").write(s.replace(old, new))
    n += 1


# ---- data/utils.py: per-atom force weights travel with the configuration as an unregistered per-atom property
patch("data/utils.py",
      """    for name, atoms_key in key_specification.arrays_keys.items():
        properties[name] = atoms.arrays.get(atoms_key, None)
        if not atoms_key in atoms.arrays:
            property_weights[name] = 0.0

    return Configuration(""",
      """    for name, atoms_key in key_specification.arrays_keys.items():
        properties[name] = atoms.arrays.get(atoms_key, None)
        if not atoms_key in atoms.arrays:
            property_weights[name] = 0.0

    # per-atom force weights (array key "forces_weight", default 1): the loss mask for the vacuum buffer of carved
    # clusters; always present so that every frame of a batch carries the attribute (AtomicData makes it [n, 1])
    properties["node_forces_weight"] = np.asarray(
        atoms.arrays.get("forces_weight", np.ones(len(atoms))), dtype=float
    ).reshape(-1)
    property_weights["node_forces_weight"] = 1.0      # bookkeeping only (dataset logging iterates over properties)

    return Configuration(""")

# ---- modules/loss.py: helper + three force terms
patch("modules/loss.py",
      """def reduce_loss(raw_loss: torch.Tensor, ddp: Optional[bool] = None) -> torch.Tensor:""",
      """def node_forces_weight(ref: Batch, like: torch.Tensor) -> Optional[torch.Tensor]:
    \"\"\"Per-atom force weights [n_atoms, 1] from the array key 'forces_weight' of the training file (None if absent),
    rescaled to mean 1 over the batch: masked atoms (weight 0) drop out of the force term without changing its size.\"\"\"
    w = getattr(ref, "node_forces_weight", None)
    if w is None:
        return None
    w = w.to(like.dtype).reshape(-1, 1)
    return w * (w.numel() / w.sum().clamp_min(1e-6))


def reduce_loss(raw_loss: torch.Tensor, ddp: Optional[bool] = None) -> torch.Tensor:""")

patch("modules/loss.py",
      """    raw_loss = (
        configs_weight
        * configs_forces_weight
        * torch.square(ref["forces"] - pred["forces"])
    )
    return reduce_loss(raw_loss, ddp)""",
      """    raw_loss = (
        configs_weight
        * configs_forces_weight
        * torch.square(ref["forces"] - pred["forces"])
    )
    w = node_forces_weight(ref, raw_loss)
    if w is not None:
        raw_loss = raw_loss * w
    return reduce_loss(raw_loss, ddp)""")

patch("modules/loss.py",
      """    raw_loss = configs_weight * configs_forces_weight * se
    return reduce_loss(raw_loss, ddp)""",
      """    raw_loss = configs_weight * configs_forces_weight * se
    w = node_forces_weight(ref, raw_loss)
    if w is not None:
        raw_loss = raw_loss * w
    return reduce_loss(raw_loss, ddp)""")

patch("modules/loss.py",
      """def conditional_huber_forces(
    ref_forces: torch.Tensor,
    pred_forces: torch.Tensor,
    huber_delta: float,
    ddp: Optional[bool] = None,
) -> torch.Tensor:""",
      """def conditional_huber_forces(
    ref_forces: torch.Tensor,
    pred_forces: torch.Tensor,
    huber_delta: float,
    ddp: Optional[bool] = None,
    weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:""")

patch("modules/loss.py",
      """    se[c4] = torch.nn.functional.huber_loss(
        ref_forces[c4], pred_forces[c4], reduction="none", delta=factors[3]
    )
    return reduce_loss(se, ddp)""",
      """    se[c4] = torch.nn.functional.huber_loss(
        ref_forces[c4], pred_forces[c4], reduction="none", delta=factors[3]
    )
    if weights is not None:
        se = se * weights
    return reduce_loss(se, ddp)""")

# the two call sites of the universal loss (ddp and non-ddp branches)
patch("modules/loss.py",
      """            loss_forces = conditional_huber_forces(
                configs_forces_weight * ref["forces"],
                configs_forces_weight * pred["forces"],
                huber_delta=self.huber_delta,
                ddp=ddp,
            )""",
      """            loss_forces = conditional_huber_forces(
                configs_forces_weight * ref["forces"],
                configs_forces_weight * pred["forces"],
                huber_delta=self.huber_delta,
                ddp=ddp,
                weights=node_forces_weight(ref, ref["forces"]),
            )""", count=2)

print(f"{n} patches applied under {root}")
