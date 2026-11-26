#!/usr/bin/env python
"""Numerical validation of the exact three-pass MACE evaluation, one frame at a time.

1. Error decomposition against the original model in fp64 (ASE calculator, e3nn):
     reordering     three-pass fp64 (plain, and with the fast layers) vs original
     interpolation  three-pass fp64 with fp64 cubic-Hermite tables of M knots vs the same without tables
     rounding       three-pass fp32 without tables vs original
     production     three-pass fp32 with the Triton tables of 512 knots (the path used in LAMMPS) vs original
   for the energy, the forces and the strain derivative S_ab = dE/d(eps_ab) = sum_edges dE/dv_a v_b (= V * ASE stress).
2. Convergence of the tabulated radial weights W(r; pair) and dW/dr with the number of knots, on a dense fp64 grid.
3. Finite differences in fp64: -dE/dx against the forces for a few atoms, and dE/d(eps_aa) against S_aa, for the
   three-pass evaluation without and with tables (the force must be the derivative of the energy that is computed).

    python validate_numerics.py --model mace-mpa-0-medium.model --xyz water64_1x1x1.extxyz --out water.json
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from mace_onegpu import fast_layers  # noqa: E402
from mace_onegpu import receiver_chunk_mace as rc  # noqa: E402
from mace_onegpu.spline_radial import _sample  # noqa: E402

DEV = "cuda"
F64 = torch.float64
R_LO = 0.5


# ----------------------------------------------------------------------------- structure
def load_atoms(path, index, r_max):
    from ase.io import read

    at = read(path, index=index)
    cell = np.array(at.get_cell())
    L = np.diag(cell).copy()
    if not np.allclose(cell - np.diag(L), 0.0, atol=1e-8):
        raise SystemExit(f"{path}[{index}]: orthorhombic cells only")
    rep = [max(1, math.ceil((2 * r_max + 0.5) / x)) for x in L]
    if any(r > 1 for r in rep):
        at = at.repeat(rep)
    at.wrap()
    return at, rep


# ----------------------------------------------------------------------------- fp64 cubic-Hermite tables (torch)
class Hermite64:
    """Same interpolant as spline_w.py (cubic Hermite on a uniform grid, clamped below r_lo), in fp64 torch ops."""

    def __init__(self, vals, ders, r_lo, r_hi):
        P, M, D = vals.shape
        self.M, self.D, self.r_lo = M, D, r_lo
        self.h = (r_hi - r_lo) / (M - 1)
        self.TV = vals.reshape(P * M, D).contiguous()
        self.TG = (ders * self.h).reshape(P * M, D).contiguous()

    def _coords(self, r, pair):
        x = ((r - self.r_lo) / self.h).clamp(0.0, self.M - 1 - 1e-9)
        k = torch.floor(x.detach())
        t = x - k
        row = pair.long() * self.M + k.long()
        return t, row

    def __call__(self, r, pair):
        t, row = self._coords(r, pair)
        t2, t3 = t * t, t * t * t
        b00, b10 = 2 * t3 - 3 * t2 + 1, t3 - 2 * t2 + t
        b01, b11 = -2 * t3 + 3 * t2, t3 - t2
        return (b00[:, None] * self.TV[row] + b10[:, None] * self.TG[row]
                + b01[:, None] * self.TV[row + 1] + b11[:, None] * self.TG[row + 1])

    def deriv(self, r, pair):
        t, row = self._coords(r, pair)
        t2 = t * t
        d00, d10 = 6 * t2 - 6 * t, 3 * t2 - 4 * t + 1
        d01, d11 = -6 * t2 + 6 * t, 3 * t2 - 2 * t
        return (d00[:, None] * self.TV[row] + d10[:, None] * self.TG[row]
                + d01[:, None] * self.TV[row + 1] + d11[:, None] * self.TG[row + 1]) / self.h


class FastLayersTab64(fast_layers.FastLayers):
    """Fast layers whose radial weights come from fp64 Hermite tables (never the fused fp32 pass-A kernel)."""

    def __init__(self, model, species_cols, tables):
        super().__init__(model, species_cols, spline_grid=0)
        n_elem = len(model.atomic_numbers)
        col2s = torch.zeros(n_elem, dtype=torch.long, device=DEV)
        col2s[torch.as_tensor(species_cols, device=DEV)] = torch.arange(len(species_cols), device=DEV)
        self.col2s64, self.S64, self.tab64 = col2s, len(species_cols), tables

    def _tp_weights(self, bi, blk, lengths, edge_feats, cutoff, node_attrs, edge_index):
        s = self.col2s64[node_attrs.argmax(1)]
        pair = s[edge_index[0].long()] * self.S64 + s[edge_index[1].long()]
        return self.tab64[bi](lengths.reshape(-1), pair)


def radial_fns(model):
    return [blk.conv_tp_weights for blk in model.interactions]


# ----------------------------------------------------------------------------- three-pass evaluator with virial
class Evaluator:
    def __init__(self, model, dt, layers, chunk):
        self.model, self.dt, self.layers = model, dt, layers
        self.ev = rc.ReceiverChunkMACE(model, dt, chunk)
        self._acc0 = self.ev._accum_forces

    def __call__(self, pos, box, spidx):
        if self.layers is None:
            rc.layer1, rc.layer2_energy = rc._layer1_orig, rc._layer2_orig
        else:
            rc.layer1, rc.layer2_energy = self.layers.layer1, self.layers.layer2_energy
        sysm = rc.System(torch.as_tensor(pos, dtype=F64, device=DEV), np.asarray(box, dtype=np.float64),
                         torch.as_tensor(spidx, dtype=torch.int16, device=DEV), float(self.model.r_max), DEV)
        sysm.build_neighbors()
        S = torch.zeros(3, 3, dtype=F64, device=DEV)

        def acc(F, g, gv):
            self._acc0(F, g, gv)
            n = g.nE
            # S_ab = sum_e dE/dv_a v_b.  detach: in fp64 the evaluator's vectors.to(dt).requires_grad_() is in place
            S.add_(gv[:n].detach().to(F64).T @ g.vectors[:n].detach().to(F64))

        self.ev._accum_forces = acc
        E, F, _, _ = self.ev.compute(sysm, log=lambda *a, **k: None)
        Fo = torch.empty_like(F)
        Fo[sysm.perm] = F.detach()
        return float(E), Fo.cpu().numpy(), S.detach().cpu().numpy()


def errors(a, b, N, V):
    (Ea, Fa, Sa), (Eb, Fb, Sb) = a, b
    dF = Fa - Fb
    return {"dE_per_atom": abs(Ea - Eb) / N, "max_dF": float(np.abs(dF).max()),
            "rms_dF": float(np.sqrt((dF ** 2).mean())), "max_dS_over_V": float(np.abs(Sa - Sb).max() / V)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--xyz", required=True)
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=4096)
    ap.add_argument("--grids", type=int, nargs="+", default=[64, 128, 256, 512, 1024, 2048])
    ap.add_argument("--dense", type=int, default=20001)
    ap.add_argument("--fd-atoms", type=int, default=4)
    ap.add_argument("--fd-h", type=float, default=1e-4)
    ap.add_argument("--fd-strain", type=float, default=1e-5)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t_start = time.time()
    res = {"model": os.path.basename(a.model), "xyz": os.path.basename(a.xyz), "index": a.index}

    # ---------------------------------------------------------------- reference: original model, fp64, e3nn
    torch.set_default_dtype(F64)
    from mace.calculators import MACECalculator

    calc = MACECalculator(model_paths=a.model, device=DEV, default_dtype="float64")
    r_max = float(calc.models[0].r_max)
    zs_model = [int(z) for z in calc.models[0].atomic_numbers]
    atoms, rep = load_atoms(a.xyz, a.index, r_max)
    atoms.calc = calc
    N, V = len(atoms), float(atoms.get_volume())
    pos, box = atoms.get_positions(), np.diag(np.array(atoms.get_cell()))
    Z = atoms.get_atomic_numbers()
    species_z = sorted(set(int(z) for z in Z))
    cols = [zs_model.index(z) for z in species_z]
    spidx = np.array([zs_model.index(int(z)) for z in Z])
    ref = (float(atoms.get_potential_energy()), atoms.get_forces(), V * atoms.get_stress(voigt=False))
    res.update(N=N, V=V, rep=rep, species=species_z, r_max=r_max, ref_E_per_atom=ref[0] / N,
               ref_F_rms=float(np.sqrt((ref[1] ** 2).mean())), ref_S_over_V=(ref[2] / V).tolist())
    del calc
    atoms.calc = None
    torch.cuda.empty_cache()
    print(f"{a.xyz}[{a.index}] x{rep}: N={N} species={species_z} E/N={ref[0] / N:.6f} eV  F_rms={res['ref_F_rms']:.4f} eV/A")

    # ---------------------------------------------------------------- fp64 three-pass paths
    m64, dt64 = rc.build_model(a.model, "fp64_fused", DEV)
    rc._layer1_orig = getattr(rc, "_layer1_orig", rc.layer1)
    rc._layer2_orig = getattr(rc, "_layer2_orig", rc.layer2_energy)
    plain64 = Evaluator(m64, dt64, None, a.chunk)
    fl64 = fast_layers.FastLayers(m64, cols, spline_grid=0)
    fast64 = Evaluator(m64, dt64, fl64, a.chunk)
    r_hi = float(m64.r_max)
    out_plain, out_fast = plain64(pos, box, spidx), fast64(pos, box, spidx)
    sgn = float(np.sum(out_fast[2] * ref[2]) / max(np.sum(ref[2] ** 2), 1e-30))
    res["S_vs_ASE_stress_projection"] = sgn            # +1: S equals V * ASE stress; -1: opposite sign convention
    if sgn < 0:
        ref = (ref[0], ref[1], -ref[2])
    dec = {"reorder_plain_fp64": errors(out_plain, ref, N, V), "reorder_fast_fp64": errors(out_fast, ref, N, V)}

    # tables: radial-weight convergence and end-to-end interpolation error
    fns = radial_fns(m64)
    with torch.no_grad():
        ex = _sample(m64, fns, species_z, a.dense, R_LO, r_hi, DEV, F64)
    r_dense = torch.linspace(R_LO, r_hi, a.dense, dtype=F64, device=DEV)
    conv, tabs_by_M = [], {}
    for M in a.grids:
        with torch.no_grad():
            tabs = [Hermite64(v, d, R_LO, r_hi) for (v, d) in _sample(m64, fns, species_z, M, R_LO, r_hi, DEV, F64)]
            row = {"M": M, "h": (r_hi - R_LO) / (M - 1), "blocks": []}
            for k, H in enumerate(tabs):
                vals, ders = ex[k]                                   # [P, dense, D]
                eW = eD = 0.0
                for p in range(vals.shape[0]):
                    pair = torch.full_like(r_dense, p, dtype=torch.long)
                    eW = max(eW, float((H(r_dense, pair) - vals[p]).abs().max()))
                    eD = max(eD, float((H.deriv(r_dense, pair) - ders[p]).abs().max()))
                row["blocks"].append({"max_abs_dW": eW, "rel_dW": eW / float(vals.abs().max()),
                                      "max_abs_dWdr": eD, "rel_dWdr": eD / float(ders.abs().max())})
        out_tab = Evaluator(m64, dt64, FastLayersTab64(m64, cols, tabs), a.chunk)(pos, box, spidx)
        row["vs_exact_fp64"] = errors(out_tab, out_fast, N, V)
        conv.append(row)
        tabs_by_M[M] = tabs
        b = row["blocks"]
        print(f"  M={M:5d}  W rel {b[0]['rel_dW']:.1e}/{b[1]['rel_dW']:.1e}  dW/dr rel {b[0]['rel_dWdr']:.1e}/"
              f"{b[1]['rel_dWdr']:.1e}  forces max {row['vs_exact_fp64']['max_dF']:.2e} eV/A  "
              f"E {row['vs_exact_fp64']['dE_per_atom']:.1e} eV/atom")
    res["tables"] = conv
    M_prod = 512 if 512 in tabs_by_M else a.grids[-1]
    tab64_eval = Evaluator(m64, dt64, FastLayersTab64(m64, cols, tabs_by_M[M_prod]), a.chunk)
    out_tab_prod = tab64_eval(pos, box, spidx)
    dec[f"interp_fp64_M{M_prod}"] = errors(out_tab_prod, out_fast, N, V)

    # ---------------------------------------------------------------- finite differences (fp64)
    a_r_max = float(m64.r_max)
    rng = np.random.default_rng(7)
    fmag = np.linalg.norm(out_fast[1], axis=1)
    pick = [int(np.argmax(fmag))] + [int(i) for i in rng.choice(N, size=a.fd_atoms - 1, replace=False)]
    fd = {}
    for tag, evf, base in (("exact_fp64", fast64, out_fast), (f"tables_fp64_M{M_prod}", tab64_eval, out_tab_prod)):
        dev_f, dev_s = [], []
        for i in pick:
            for c in range(3):
                p = pos.copy()
                p[i, c] += a.fd_h
                Ep = evf(p, box, spidx)[0]
                p[i, c] -= 2 * a.fd_h
                Em = evf(p, box, spidx)[0]
                dev_f.append(abs(-(Ep - Em) / (2 * a.fd_h) - base[1][i, c]))
        strain = {}
        for hs in (a.fd_strain, 10 * a.fd_strain):
            dev_s = []
            for c in range(3):                      # diagonal components: periodic cell and positions scaled
                sc = np.ones(3)
                sc[c] = 1 + hs
                Ep = evf(pos * sc, box * sc, spidx)[0]
                sc[c] = 1 - hs
                Em = evf(pos * sc, box * sc, spidx)[0]
                dev_s.append(abs((Ep - Em) / (2 * hs) - base[2][c, c]))
            strain[f"{hs:g}"] = {"max_dev_S_over_V": float(max(dev_s)) / V, "dev_S_diag_over_V": [float(d) / V for d in dev_s]}
        # shear components: the same atoms as an isolated cluster in a large orthogonal box (images out of reach),
        # deformed by F = I + h E_ab; dE/dh = S_ab = sum_edges dE/dv_a v_b (ASE stress = S / V)
        rmax = float(a_r_max)
        Lbig = float(np.max(box)) + 2 * rmax + 4.0
        pos_c = pos - pos.mean(0) + 0.5 * Lbig
        box_c = np.array([Lbig, Lbig, Lbig])
        base_c = evf(pos_c, box_c, spidx)
        shear = {}
        for hs in (a.fd_strain, 10 * a.fd_strain):
            dev = {}
            for (ia, ib) in ((0, 1), (0, 2), (1, 2), (0, 0), (1, 1), (2, 2)):
                Fp = np.eye(3)
                Fp[ia, ib] += hs
                Fm = np.eye(3)
                Fm[ia, ib] -= hs
                Ep = evf(pos_c @ Fp.T, box_c, spidx)[0]
                Em = evf(pos_c @ Fm.T, box_c, spidx)[0]
                dev[f"{ia}{ib}"] = abs((Ep - Em) / (2 * hs) - base_c[2][ia, ib])
            shear[f"{hs:g}"] = {"max_dev_S_eV": float(max(dev.values())), "dev_S_eV": {k: float(v) for k, v in dev.items()},
                                "max_dev_S_over_V": float(max(dev.values())) / V,
                                "max_abs_S_eV": float(np.abs(base_c[2]).max())}
        fd[tag] = {"atoms": pick, "h": a.fd_h, "max_dev_force": float(max(dev_f)), "median_dev_force": float(np.median(dev_f)),
                   "max_force_component": float(max(abs(base[1][i, c]) for i in pick for c in range(3))),
                   "strain": a.fd_strain, "max_dev_S_over_V": strain[f"{a.fd_strain:g}"]["max_dev_S_over_V"],
                   "S_diag_over_V": [float(base[2][c, c]) / V for c in range(3)],
                   "strain_diag_periodic": strain, "strain_all_cluster": shear,
                   "convention": "S_ab = dE/d(eps_ab) = sum_edges dE/dv_a v_b (eV); ASE stress = S/V; cell deformed by F = I + h E_ab"}
        print(f"  FD {tag}: force max dev {fd[tag]['max_dev_force']:.2e} eV/A (median {fd[tag]['median_dev_force']:.1e}); "
              f"diagonal strain (periodic) max dev {fd[tag]['max_dev_S_over_V']:.2e} eV/A^3 at h={a.fd_strain:g}, "
              f"{strain[f'{10 * a.fd_strain:g}']['max_dev_S_over_V']:.2e} at h={10 * a.fd_strain:g}; "
              f"all six components (cluster) max dev {shear[f'{a.fd_strain:g}']['max_dev_S_eV']:.2e} eV "
              f"(= {shear[f'{a.fd_strain:g}']['max_dev_S_over_V']:.2e} eV/A^3) at h={a.fd_strain:g}, "
              f"{shear[f'{10 * a.fd_strain:g}']['max_dev_S_eV']:.2e} eV at h={10 * a.fd_strain:g}; |S| max {shear[f'{a.fd_strain:g}']['max_abs_S_eV']:.3g} eV")
    res["finite_differences"] = fd
    del m64, plain64, fast64, tab64_eval, fl64
    torch.cuda.empty_cache()

    # ---------------------------------------------------------------- fp32 paths (rounding; production)
    m32, dt32 = rc.build_model(a.model, "fp32_fused", DEV)
    fl32 = fast_layers.FastLayers(m32, cols, spline_grid=0)
    out32 = Evaluator(m32, dt32, fl32, a.chunk)(pos, box, spidx)
    dec["rounding_fp32"] = errors(out32, ref, N, V)
    fl32s = fast_layers.FastLayers(m32, cols, spline_grid=512)
    out32s = Evaluator(m32, dt32, fl32s, a.chunk)(pos, box, spidx)
    dec["production_fp32_tables512"] = errors(out32s, ref, N, V)
    dec["triton_vs_fp64_tables512"] = errors(out32s, out_tab_prod, N, V) if M_prod == 512 else None
    res["decomposition"] = dec
    torch.set_default_dtype(torch.float32)

    import cuequivariance
    import mace

    res["versions"] = {"torch": torch.__version__, "mace": mace.__version__, "cuequivariance": cuequivariance.__version__,
                       "gpu": torch.cuda.get_device_name(0)}
    res["wall_s"] = time.time() - t_start
    print("  decomposition (vs original fp64 unless noted):")
    for k, v in dec.items():
        if v:
            print(f"    {k:28s} max|dF| {v['max_dF']:.2e}  rms {v['rms_dF']:.2e} eV/A   |dE|/N {v['dE_per_atom']:.1e} eV"
                  f"   max|dS|/V {v['max_dS_over_V']:.1e} eV/A^3")
    print(f"  S . V*stress(ASE) projection = {sgn:+.6f}   wall {res['wall_s']:.0f} s")
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
