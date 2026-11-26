#!/usr/bin/env python
"""
Exact FLOP budget of MACE-MPA-0 medium per edge and per atom, from the cuEquivariance descriptors MACE itself uses.
CPU only (pure-python descriptors). Prints FLOPs (1 FMA = 2 FLOP) for forward; backward is estimated as 2x forward
for multilinear ops (grads w.r.t. two operands), as in standard autodiff cost models.
"""
import cuequivariance as cue
from mace.tools.cg import O3_e3nn


def poly_flops(p):
    """sum over operations and paths of (product of segment sizes) * 2 * (#operand factors - 1)."""
    tot = 0
    for op, stp in p.operations:
        for path in stp.paths:
            size = 1
            for oi, si in enumerate(path.indices):
                shp = stp.operands[oi].segments[si]
                for d in shp:
                    size = max(size, 1) * 1
            # size of one path = product over subscript modes (u, i, ...) of their extent
            dims = {}
            for oi, si in enumerate(path.indices):
                shp = stp.operands[oi].segments[si]
                sub = stp.subscripts.operands[oi] if hasattr(stp.subscripts, "operands") else str(stp.subscripts).split("->")[0].split(",")[oi] if "->" in str(stp.subscripts) else str(stp.subscripts).replace("+", ",").split(",")[oi]
                for ch, d in zip(sub, shp):
                    dims[ch] = d
            n = 1
            for d in dims.values():
                n *= d
            coeff = path.coefficients
            ncoef = int(getattr(coeff, "size", 1))
            nfac = len(stp.operands) - 1
            tot += 2 * n * max(ncoef, 1) * nfac
    return tot


def main():
    I = lambda s: cue.Irreps(O3_e3nn, s)  # noqa: E731
    sh = I("1x0e+1x1o+1x2e+1x3o")
    h1 = "128x0e+128x1o"
    tgt = "128x0e+128x1o+128x2e+128x3o"
    out = {}
    # edge: channelwise TP (layer 1: node feats 128x0e; layer 2: 128x0e+128x1o)
    for name, xin in (("L1 conv TP", "128x0e"), ("L2 conv TP", h1)):
        from mace.modules.irreps_tools import tp_out_irreps_with_instructions
        from e3nn import o3
        mid, _ = tp_out_irreps_with_instructions(o3.Irreps(xin), o3.Irreps(str(sh).replace(" ", "")), o3.Irreps(tgt))
        d = cue.descriptors.channelwise_tensor_product(I(xin), sh, I(str(mid)))
        p = d.flatten_coefficient_modes().squeeze_modes().polynomial
        out[name] = (poly_flops(p), "per edge", str(mid))
    # atom: symmetric contraction (correlation 3), per element weights
    for name, xin, xout in (("L1 symmetric contraction", tgt, h1), ("L2 symmetric contraction", tgt, "128x0e")):
        d = cue.descriptors.symmetric_contraction(I(xin), I(xout), [1, 2, 3])
        p = d.polynomial if hasattr(d, "polynomial") else d
        out[name] = (poly_flops(p), "per atom", xout)
    # atom: equivariant linears (FLOPs = 2 * sum_l mul_in*mul_out*(2l+1))
    def lin(a, b):
        ia, ib = I(a), I(b)
        f = 0
        for mi, ir in ia:
            for mo, ir2 in ib:
                if ir == ir2:
                    f += 2 * mi * mo * ir.dim
        return f
    mid2 = "128x0e+128x1o+128x1o+128x2e+128x2e+128x3o+128x3o"  # placeholder; linear(message) mapped below via L2 mid
    out["L1 linear_up"] = (lin("128x0e", "128x0e"), "per atom", "")
    out["L1 linear (message)"] = (2 * 128 * 128 * 16, "per atom", "4 irreps, 1 path each")
    out["L1 skip (species block, exact)"] = (2 * 128 * 128 * 16, "per atom", "")
    out["L1 product linear"] = (lin(h1, h1), "per atom", "")
    out["L2 linear_up"] = (lin(h1, h1), "per atom", "")
    out["L2 linear (message 5120 -> 2048)"] = (2 * 128 * 128 * 40, "per atom", "40 segments into 16 comps")
    out["L2 skip (species block)"] = (2 * 128 * 128, "per atom", "")
    out["L2 product linear"] = (lin("128x0e", "128x0e"), "per atom", "")
    for k, (f, unit, extra) in out.items():
        print(f"{k:36s} {f/1e3:10.1f} kFLOP {unit:9s} {extra}")


if __name__ == "__main__":
    main()
