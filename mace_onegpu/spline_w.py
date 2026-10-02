#!/usr/bin/env python
"""
Tabulated radial TP weights W(r; ordered species pair) for MACE interaction blocks (element-general).

Replaces `blk.conv_tp_weights(edge_feats)` (a 4-layer MLP per edge) by a cubic-Hermite lookup in fp64-sampled
tables (conv_spline.build_tables).  The tensor product itself stays cuEquivariance (fast forward AND backward).
    forward : W[e, :]  = Hermite(TV, TG; r_e, pair_e)                      (Triton, one row per edge)
    backward: dL/dr_e  = sum_d dL/dW[e, d] * dW/dr[e, d]                    (Triton, same table)
Accuracy (fp64 sampling, 512 knots; experimental/validate_numerics.py): largest force error from the tables
1.6e-5 (water), 8e-6 (SiO2), 8e-7 eV/A (CuNi), comparable to fp32 roundoff (0.7-2.1e-5); it falls as knots^-3.
The backward kernel is the analytic derivative of the forward interpolant, so forces stay derivatives of the energy.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _coords(R, PAIR, e, me, r_lo, inv_h, M):
    r = tl.load(R + e, mask=me, other=0.0)
    p = tl.load(PAIR + e, mask=me, other=0)
    x = (r - r_lo) * inv_h
    x = tl.minimum(tl.maximum(x, 0.0), (M - 1).to(tl.float32) - 1e-4)
    kf = tl.floor(x)
    t = x - kf
    row = (p * M + kf.to(tl.int32)).to(tl.int64)
    return t, row


@triton.jit
def _w_fwd(R, PAIR, TV, TG, W, E, r_lo, inv_h, M, D: tl.constexpr, BE: tl.constexpr, BD: tl.constexpr):
    pid = tl.program_id(0)
    e = pid * BE + tl.arange(0, BE)
    me = e < E
    t, row = _coords(R, PAIR, e, me, r_lo, inv_h, M)
    t2 = t * t
    t3 = t2 * t
    b00 = (2.0 * t3 - 3.0 * t2 + 1.0)[:, None]
    b10 = (t3 - 2.0 * t2 + t)[:, None]
    b01 = (-2.0 * t3 + 3.0 * t2)[:, None]
    b11 = (t3 - t2)[:, None]
    mm = me[:, None]
    e64 = e.to(tl.int64)
    for d0 in tl.static_range(0, D, BD):
        od = d0 + tl.arange(0, BD)
        b0 = row[:, None] * D + od[None, :]
        b1 = b0 + D
        w = (b00 * tl.load(TV + b0, mask=mm, other=0.0) + b10 * tl.load(TG + b0, mask=mm, other=0.0)
             + b01 * tl.load(TV + b1, mask=mm, other=0.0) + b11 * tl.load(TG + b1, mask=mm, other=0.0))
        tl.store(W + e64[:, None] * D + od[None, :], w, mask=mm)


@triton.jit
def _w_bwd(R, PAIR, TV, TG, GW, GR, E, r_lo, inv_h, M, D: tl.constexpr, BE: tl.constexpr, BD: tl.constexpr):
    pid = tl.program_id(0)
    e = pid * BE + tl.arange(0, BE)
    me = e < E
    t, row = _coords(R, PAIR, e, me, r_lo, inv_h, M)
    t2 = t * t
    d00 = (6.0 * t2 - 6.0 * t)[:, None]
    d10 = (3.0 * t2 - 4.0 * t + 1.0)[:, None]
    d01 = (-6.0 * t2 + 6.0 * t)[:, None]
    d11 = (3.0 * t2 - 2.0 * t)[:, None]
    mm = me[:, None]
    e64 = e.to(tl.int64)
    acc = tl.zeros([BE, BD], dtype=tl.float32)
    for d0 in tl.static_range(0, D, BD):
        od = d0 + tl.arange(0, BD)
        b0 = row[:, None] * D + od[None, :]
        b1 = b0 + D
        dw = (d00 * tl.load(TV + b0, mask=mm, other=0.0) + d10 * tl.load(TG + b0, mask=mm, other=0.0)
              + d01 * tl.load(TV + b1, mask=mm, other=0.0) + d11 * tl.load(TG + b1, mask=mm, other=0.0))
        g = tl.load(GW + e64[:, None] * D + od[None, :], mask=mm, other=0.0)
        acc += g * dw
    tl.store(GR + e, tl.sum(acc, 1) * inv_h, mask=me)


BE, BD = 32, 128


class _SplineWFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, r, pair, tbl):
        E = r.numel()
        D = tbl.D
        W = torch.empty(E, D, dtype=torch.float32, device=r.device)
        _w_fwd[(triton.cdiv(E, BE),)](r, pair, tbl.TV, tbl.TG, W, E, tbl.r_lo, tbl.inv_h, tbl.M, D=D, BE=BE, BD=tbl.BD)
        ctx.save_for_backward(r, pair)
        ctx.tbl = tbl
        return W

    @staticmethod
    def backward(ctx, gW):
        r, pair = ctx.saved_tensors
        tbl = ctx.tbl
        E = r.numel()
        GR = torch.empty_like(r)
        _w_bwd[(triton.cdiv(E, BE),)](r, pair, tbl.TV, tbl.TG, gW.contiguous(), GR, E, tbl.r_lo, tbl.inv_h, tbl.M,
                                      D=tbl.D, BE=BE, BD=tbl.BD)
        return GR, None, None


class SplineW:
    """W = SplineW(r[E] fp32, pair[E] int32) -> [E, D], differentiable w.r.t. r."""

    def __init__(self, TV, TG, meta):
        self.TV, self.TG = TV, TG
        self.D = TV.shape[1]
        self.BD = next(b for b in (128, 64, 32, 16, 8, 4, 2, 1) if self.D % b == 0)   # any model width
        self.r_lo, self.inv_h, self.M = float(meta["r_lo"]), float(meta["inv_h"]), int(meta["M"])

    @torch._dynamo.disable   # our Triton kernels: graph break for torch.compile, run eagerly
    def __call__(self, r, pair):
        return _SplineWFn.apply(r.contiguous(), pair.contiguous(), self)


def make(model, species_z, n_grid=512, device="cuda"):
    from .conv_spline import build_tables

    meta = build_tables(model, species_z, n_grid=n_grid, device=device)
    return [SplineW(TV, TG, meta) for (TV, TG) in meta["tables"]], meta
