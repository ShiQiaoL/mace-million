#!/usr/bin/env python
"""
Replace MACE's per-edge radial networks by per-species-pair cubic-Hermite splines.

Structural fact (MACE 0.3.15, RealAgnosticDensity* blocks, e.g. MACE-MPA-0):
    edge_feats   = Bessel(Agnesi(r; Z_send, Z_recv)) * f_cut(r)          [E, 10]
    tp_weights_k = conv_tp_weights_k(edge_feats)                           [E, 512 | 1280]
    dens_k       = density_fn_k(edge_feats)                                [E, 1]
so every per-edge quantity feeding the tensor product is a 1-D function of r for each ORDERED species pair.
We sample it once on a grid (value + dW/dr by central differences in fp64) and evaluate a cubic Hermite
spline at run time. The spline is a polynomial in r, so autograd gives consistent forces.

Usage as a library:
    from mace_onegpu.spline_radial import splineize
    info = splineize(model, species_z=[1, 8], n_grid=1024, r_lo=0.5, device="cuda")
Self-test / benchmark (GPU):
    python -m mace_onegpu.spline_radial --model mace-mpa-0-medium.model --data water64_4x4x4_N12288.data --grids 256 512 1024 2048
"""
import os
import time

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
import torch  # noqa: E402


class _Ctx:
    r = None      # [E, 1] raw edge lengths (carry autograd)
    pair = None   # [E] long, compact ordered-pair index


CTX = _Ctx()


SPLINE_CHUNK = int(os.environ.get("MACE_SPLINE_CHUNK", "262144"))


def _hermite_eval(V, G, M, r_lo, h, r, pair, deriv):
    """Evaluate (deriv=False) W(r) or (deriv=True) dW/dr for a chunk; no autograd."""
    x = ((r - r_lo) / h).clamp(0.0, M - 1 - 1e-9)
    k = x.floor().long()
    t = (x - k.to(x.dtype)).unsqueeze(-1)
    i0 = pair * M + k
    i1 = i0 + 1
    if not deriv:
        t2 = t * t
        t3 = t2 * t
        out = (2 * t3 - 3 * t2 + 1) * V[i0]
        out.add_((t3 - 2 * t2 + t) * G[i0])
        out.add_((-2 * t3 + 3 * t2) * V[i1])
        out.add_((t3 - t2) * G[i1])
        return out
    t2 = t * t
    out = (6 * t2 - 6 * t) * V[i0]
    out.add_((3 * t2 - 4 * t + 1) * G[i0])
    out.add_((-6 * t2 + 6 * t) * V[i1])
    out.add_((3 * t2 - 2 * t) * G[i1])
    return out / h


class _HermiteFn(torch.autograd.Function):
    """Spline lookup with O(E*D) output and O(E) saved state (r, pair); dL/dr computed chunk-wise."""

    @staticmethod
    def forward(ctx, r, pair, V, G, M, r_lo, h):
        ctx.save_for_backward(r, pair)
        ctx.V, ctx.G, ctx.M, ctx.r_lo, ctx.h = V, G, M, r_lo, h
        rr = r.reshape(-1)
        E = rr.numel()
        out = torch.empty(E, V.shape[1], dtype=V.dtype, device=V.device)
        with torch.no_grad():
            for s in range(0, E, SPLINE_CHUNK):
                e = min(s + SPLINE_CHUNK, E)
                out[s:e] = _hermite_eval(V, G, M, r_lo, h, rr[s:e], pair[s:e], False)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        r, pair = ctx.saved_tensors
        rr = r.reshape(-1)
        E = rr.numel()
        grad_r = torch.empty_like(rr)
        with torch.no_grad():
            for s in range(0, E, SPLINE_CHUNK):
                e = min(s + SPLINE_CHUNK, E)
                dW = _hermite_eval(ctx.V, ctx.G, ctx.M, ctx.r_lo, ctx.h, rr[s:e], pair[s:e], True)
                grad_r[s:e] = (dW * grad_out[s:e]).sum(dim=-1)
        return grad_r.reshape(r.shape), None, None, None, None, None, None


class HermiteTable(torch.nn.Module):
    """Cubic Hermite spline on a uniform grid, batched over ordered species pairs."""

    def __init__(self, r_lo, r_hi, values, derivs):
        super().__init__()
        P, M, D = values.shape
        self.r_lo, self.r_hi, self.M, self.D = float(r_lo), float(r_hi), int(M), int(D)
        self.h = (self.r_hi - self.r_lo) / (M - 1)
        self.register_buffer("V", values.reshape(P * M, D).contiguous())
        self.register_buffer("G", (derivs * self.h).reshape(P * M, D).contiguous())

    def forward(self, r, pair):
        return _HermiteFn.apply(r, pair, self.V, self.G, self.M, self.r_lo, self.h)


class SplineOut(torch.nn.Module):
    """Drop-in replacement for conv_tp_weights / density_fn: ignores its input, reads (r, pair) from CTX."""

    def __init__(self, table):
        super().__init__()
        self.table = table

    def forward(self, _edge_feats):
        return self.table(CTX.r, CTX.pair)


def _sample(model, block_fns, z_list, n_grid, r_lo, r_hi, device, dtype):
    """Evaluate each block function on the grid for every ordered pair (sender, receiver)."""
    re = model.radial_embedding
    zs = [int(z) for z in model.atomic_numbers]
    n_el = len(zs)
    idx = [zs.index(z) for z in z_list]
    r = torch.linspace(r_lo, r_hi, n_grid, dtype=torch.float64, device=device).unsqueeze(-1)
    eps = 1e-5 * (r_hi - r_lo)
    out_vals = [[] for _ in block_fns]
    out_ders = [[] for _ in block_fns]
    with torch.no_grad():
        for a in idx:          # sender species index
            for b in idx:      # receiver species index
                node_attrs = torch.zeros(2, n_el, dtype=torch.float64, device=device)
                node_attrs[0, a] = 1.0
                node_attrs[1, b] = 1.0
                ei = torch.zeros(2, n_grid, dtype=torch.long, device=device)
                ei[1] = 1   # sender 0 -> receiver 1

                def f(rr):
                    feats, cut = re(rr, node_attrs, ei, model.atomic_numbers)
                    outs = []
                    for fn in block_fns:
                        y = fn(feats)
                        if cut is not None:
                            y = y * cut
                        outs.append(y)
                    return outs

                v0 = f(r)
                vp = f(r + eps)
                vm = f(r - eps)
                for k in range(len(block_fns)):
                    out_vals[k].append(v0[k])
                    out_ders[k].append((vp[k] - vm[k]) / (2 * eps))
    return [(torch.stack(out_vals[k]).to(dtype), torch.stack(out_ders[k]).to(dtype)) for k in range(len(block_fns))]


def splineize(model, species_z, n_grid=1024, r_lo=0.5, device="cuda"):
    """Patch `model` in place. Returns dict with table sizes. Keep an unpatched copy if you need a reference."""
    dtype = next(model.parameters()).dtype
    r_hi = float(model.r_max)
    zs = [int(z) for z in model.atomic_numbers]
    n_el = len(zs)
    comp = {zs.index(z): i for i, z in enumerate(species_z)}
    nsp = len(species_z)
    lut = torch.full((n_el, n_el), -1, dtype=torch.long, device=device)
    for a, ia in comp.items():
        for b, ib in comp.items():
            lut[a, b] = ia * nsp + ib

    # sample in fp64 from a float64 copy of the radial parts, then cast
    fns = []
    for blk in model.interactions:
        fns.append(blk.conv_tp_weights)
        fns.append(blk.density_fn)
    orig_dtypes = {}
    for fn in fns + [model.radial_embedding]:
        orig_dtypes[fn] = next((p.dtype for p in fn.parameters()), None)
        fn.double()
    tables = _sample(model, fns, species_z, n_grid, r_lo, r_hi, device, dtype)
    for fn, dt in orig_dtypes.items():
        if dt is not None:
            fn.to(dt)
    model.radial_embedding.to(dtype)

    info = {"n_pairs": nsp * nsp, "n_grid": n_grid, "r_lo": r_lo, "r_hi": r_hi, "tables": []}
    for i, blk in enumerate(model.interactions):
        (vw, dw), (vd, dd) = tables[2 * i], tables[2 * i + 1]
        blk.conv_tp_weights = SplineOut(HermiteTable(r_lo, r_hi, vw, dw)).to(device)
        blk.density_fn = SplineOut(HermiteTable(r_lo, r_hi, vd, dd)).to(device)
        info["tables"].append({"block": i, "D_w": vw.shape[-1],
                               "bytes": int((vw.numel() + dw.numel() + vd.numel() + dd.numel()) * vw.element_size())})

    # hook radial_embedding to publish (r, pair) for the SplineOut modules
    re = model.radial_embedding
    orig_forward = re.forward

    def forward(edge_lengths, node_attrs, edge_index, atomic_numbers):
        sp = node_attrs.argmax(dim=-1)
        CTX.r = edge_lengths
        CTX.pair = lut[sp[edge_index[0]], sp[edge_index[1]]]
        if (CTX.pair < 0).any():
            raise RuntimeError("spline_radial: species pair not tabulated")
        return orig_forward(edge_lengths, node_attrs, edge_index, atomic_numbers)

    re.forward = forward
    return info


# ----------------------------------------------------------------------------- self-test / benchmark
def _main():
    import argparse
    import copy

    import numpy as np

    from .ase_gradient_probe import make_batch, read_structure
    from .fused_conv import repair_fused_blocks
    from mace.cli.convert_e3nn_cueq import run as run_e3nn_to_cueq

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--grids", nargs="+", type=int, default=[256, 512, 1024, 2048])
    ap.add_argument("--variants", nargs="+", default=["fp64", "fp32_fused"])
    ap.add_argument("--repeat", type=int, default=5)
    a = ap.parse_args()
    dev = "cuda"
    atoms = read_structure(a.data)
    species = sorted(set(int(z) for z in atoms.get_atomic_numbers()))
    print(f"structure {os.path.basename(a.data)}: N={len(atoms)} species={species}")

    def build(variant):
        dt = torch.float64 if variant.startswith("fp64") else torch.float32
        torch.set_default_dtype(dt)
        m = torch.load(a.model, map_location=dev)
        m = m.double() if dt == torch.float64 else m.float()
        if "fused" in variant:
            m = run_e3nn_to_cueq(copy.deepcopy(m).to(dev), device="cuda")
            repair_fused_blocks(m)
        else:
            m = run_e3nn_to_cueq(copy.deepcopy(m).to("cpu"), device="cpu")
        m = m.to(dev)
        for p in m.parameters():
            p.requires_grad_(False)
        return m

    def run(m, batch):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        out = m(batch.to_dict(), training=False, compute_force=True)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        return float(out["energy"].detach().double().sum()), out["forces"].detach().double().cpu().numpy(), dt, torch.cuda.max_memory_allocated() / 1e9

    def bench(m, batch):
        run(m, batch)  # warm-up / JIT
        ts, res = [], None
        for _ in range(a.repeat):
            res = run(m, batch)
            ts.append(res[2])
        return res[0], res[1], float(np.median(ts)), res[3]

    for variant in a.variants:
        print(f"\n=== variant {variant} ===")
        ref = build(variant)
        batch = make_batch(ref, atoms, dev)
        E = int(batch.edge_index.shape[1])
        e0, f0, t0, m0 = bench(ref, batch)
        print(f"reference   : E={E}  t={t0*1e3:8.1f} ms  peak={m0:7.2f} GB  energy={e0:.8f}")
        del ref
        torch.cuda.empty_cache()
        for g in a.grids:
            m = build(variant)
            info = splineize(m, species, n_grid=g, r_lo=0.5, device=dev)
            e1, f1, t1, m1 = bench(m, batch)
            d = f1 - f0
            tb = sum(x["bytes"] for x in info["tables"]) / 1e6
            print(f"spline M={g:5d}: t={t1*1e3:8.1f} ms ({t0/t1:4.2f}x)  peak={m1:7.2f} GB  table={tb:6.1f} MB  "
                  f"dE={e1-e0:+.2e} eV  max|dF|={np.abs(d).max():.2e}  RMS(dF)={np.sqrt((d**2).mean()):.2e} eV/A")
            del m
            torch.cuda.empty_cache()


if __name__ == "__main__":
    _main()
