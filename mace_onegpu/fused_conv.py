"""
Pickle-safe fused convolution for MACE + cuEquivariance.

Diagnosis: mace 0.3.15's `with_cueq_conv_fusion` installs the fused call as an *instance attribute*
`conv_tp.forward = types.MethodType(forward, conv_tp)`.  A bound method pickles as getattr(obj, "forward"); when the
ML-IAP .pt is unpickled inside LAMMPS the instance attribute does not exist yet, so getattr resolves to the class
method `cuet.SegmentedPolynomial.forward`.  The interaction block still has `conv_fusion` set and therefore calls
`conv_tp(node_feats, edge_attrs, tp_weights, edge_index)`, which the class method interprets as
`forward(inputs=node_feats, input_indices=edge_attrs, ...)` ->
    AssertionError: the number of inputs must match the number of inputs of the polynomial
(observed for the fp64_fused / fp32_fused variants).

Fix: wrap the cuet.SegmentedPolynomial in a real nn.Module subclass defined in an importable module
(mace_onegpu.fused_conv; the repository root must be on PYTHONPATH inside LAMMPS), so the fused forward survives
pickling.
"""
import torch


class FusedConvTP(torch.nn.Module):
    """gather(sender) -> channel-wise tensor product -> scatter_add(receiver), fused in one cuEquivariance kernel.

    forward(node_feats [n_nodes, in1], edge_attrs [E, in2], tp_weights [E, W], edge_index [2, E]) -> [n_nodes, out]
    m_ji is never materialised: output rows are accumulated on the receiver index inside the kernel.
    """

    def __init__(self, segmented_polynomial, weight_numel, irreps_out=None):
        super().__init__()
        self.m = segmented_polynomial
        self.weight_numel = int(weight_numel)
        self.irreps_out = irreps_out

    def forward(self, node_feats, edge_attrs, tp_weights, edge_index):
        sender = edge_index[0]
        receiver = edge_index[1]
        out = self.m([tp_weights, node_feats, edge_attrs], {1: sender}, {0: node_feats}, {0: receiver})
        return out[0] if isinstance(out, (list, tuple)) else out


def repair_fused_blocks(model):
    """Replace monkeypatched fused conv_tp objects by pickle-safe FusedConvTP modules. Returns number repaired."""
    n = 0
    for blk in getattr(model, "interactions", []):
        if not hasattr(blk, "conv_fusion"):
            continue
        tp = blk.conv_tp
        if isinstance(tp, FusedConvTP):
            continue
        weight_numel = getattr(tp, "weight_numel", None)
        if weight_numel is None:
            weight_numel = blk.conv_tp_weights.hs[-1] if hasattr(blk.conv_tp_weights, "hs") else None
        # drop the instance-level monkeypatch so that tp(...) resolves to the class forward again
        for attr in ("forward", "original_forward"):
            if attr in tp.__dict__:
                del tp.__dict__[attr]
        irreps_out = getattr(blk, "irreps_out", None)
        blk.conv_tp = FusedConvTP(tp, weight_numel, irreps_out)
        n += 1
    return n
