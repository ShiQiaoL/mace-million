#!/usr/bin/env python
"""
Exact ACE-style ("density projection") rewrite of MACE layer 1 (RealAgnosticDensityInteractionBlock, first_layer).

Why it is exact: in layer 1 the sender features fed to the channelwise TP are  X_j = linear_up(node_embedding(onehot_j)),
i.e. they depend only on the sender's species s_j.  The radial MLP output is linear in its last hidden layer (no bias):
    W_ij[w, c] = cut_ij * sum_q h_ij[q] * A3[q, w*C + c]            (A3 = W_last / sqrt(fan_in))
and the TP is   m_i[o, c] = sum_terms coef * sum_j W_ij[w, c] * X_{s_j}[k, c] * Y_ij[m].
Hence
    m_i[o, c] = sum_terms coef * sum_s X_s[k, c] * sum_q A3[q, w*C + c] * B_{i,s}[q, m],
    B_{i,s}[q, m] = sum_{j in N(i), s_j = s} cut_ij h_ij[q] Y_ij[m]            (64 x 16 per atom per species)
Per edge only a 64x16 outer product remains; the 64 -> nW*C matmul moves from edges to atoms (x ~90 fewer in water).
The result equals the original up to fp32 re-association.

    from mace_onegpu import ace_layer1; ace_layer1.install(model, species_columns)   # replaces receiver_chunk_mace.layer1
"""
import math
import os

import torch



class ACELayer1:
    def __init__(self, model, species_cols):
        from .fused_edge_conv import path_table

        blk = model.interactions[0]
        assert type(blk).__name__ == "RealAgnosticDensityInteractionBlock", type(blk).__name__
        tab = path_table(blk)
        C, nW, nX, nY, nO = tab["C"], tab["nW"], tab["nX"], tab["nY"], tab["nO"]
        layers = list(blk.conv_tp_weights)
        last = layers[-1]
        assert last.act is None and last.var_in == 1 and last.var_out == 1
        H = last.h_in
        self.m, self.blk, self.tab = model, blk, tab
        self.C, self.nW, self.nX, self.nY, self.nO, self.H = C, nW, nX, nY, nO, H
        self.hidden = layers[:-1]
        self.A3 = (last.weight / math.sqrt(H)).view(H, nW, C)           # [q, w, c]
        dev = self.A3.device
        self.cols = torch.as_tensor(species_cols, dtype=torch.long, device=dev)
        self.S = len(species_cols)
        ncol = len(model.atomic_numbers)
        col2s = torch.full((ncol,), 0, dtype=torch.long, device=dev)
        col2s[self.cols] = torch.arange(self.S, device=dev)
        self.col2s = col2s
        # distinct (w, m) pairs and, per term, which pair / k / o / coef
        pairs = sorted({(w, m) for (w, k, m, o, c) in tab["terms"]})
        pidx = {p: i for i, p in enumerate(pairs)}
        self.pair_w = torch.tensor([p[0] for p in pairs], device=dev)
        self.pair_m = torch.tensor([p[1] for p in pairs], device=dev)
        self.terms = [(pidx[(w, m)], k, o, c) for (w, k, m, o, c) in tab["terms"]]
        self.layout = os.environ.get("ACE_L1_LAYOUT", "seg")   # 'seg' = [o][c] (ir_mul segments), 'mul' = [c][o]
        dt = self.A3.dtype
        eye = torch.zeros(self.S, ncol, dtype=dt, device=dev)     # built here: not allowed inside CUDA-graph capture
        eye[torch.arange(self.S, device=dev), self.cols] = 1.0
        self.eye = eye

    # ------------------------------------------------------------------ message = conv_tp(...) summed over receivers
    def message(self, Xs, Y, edge_feats, cutoff, recv, s_send, n_recv):
        h = edge_feats
        for l in self.hidden:
            h = l(h)
        if cutoff is not None:
            h = h * cutoff
        H, S, C = self.H, self.S, self.C
        # padding edges (exactly zero contribution) may point at nodes >= n_recv: send them to a trash row
        idx = torch.clamp(recv.long(), max=n_recv) * S + s_send
        P = (h.unsqueeze(2) * Y[:, self.pair_m].unsqueeze(1)).reshape(h.shape[0], -1)     # [E, H*P]
        B = torch.zeros((n_recv + 1) * S, P.shape[1], dtype=h.dtype, device=h.device).index_add_(0, idx, P)
        B = B[: n_recv * S].view(n_recv, S, H, len(self.pair_w))                           # [r, s, q, p]
        A = self.A3[:, self.pair_w, :]                                                      # [q, p, c]
        T = torch.einsum("rsqp,qpc->rspc", B, A)                                           # atom-level GEMM
        Xs = Xs.view(S, self.nX, C)
        out = [None] * self.nO
        for (p, k, o, coef) in self.terms:
            v = (T[:, :, p, :] * Xs[:, k, :]).sum(1) * coef                                  # [r, c]
            out[o] = v if out[o] is None else out[o] + v
        z = torch.zeros(n_recv, C, dtype=h.dtype, device=h.device)
        out = torch.stack([z if t is None else t for t in out], 1)                          # [r, o, c]
        if self.layout == "mul":
            out = out.transpose(1, 2)
        return out.reshape(n_recv, self.nO * C)

    # ------------------------------------------------------------------ drop-in for receiver_chunk_mace.layer1
    def layer1(self, model, g, node_attrs, vectors):
        from . import receiver_chunk_mace as rc

        _, edge_attrs, edge_feats, cutoff = rc.edge_embed(model, g, node_attrs, vectors)
        blk, nC = self.blk, g.nC
        Xs = blk.linear_up(model.node_embedding(self.eye))                                        # [S, nX*C]
        send, recv = g.edge_index[0], g.edge_index[1]
        s_send = self.col2s[node_attrs.argmax(1)][send.long()]
        msg = self.message(Xs, edge_attrs, edge_feats, cutoff, recv, s_send, nC)
        ed = torch.tanh(blk.density_fn(edge_feats) ** 2)
        if cutoff is not None:
            ed = ed * cutoff
        density = torch.zeros(nC + 1, 1, dtype=ed.dtype, device=ed.device).index_add_(
            0, torch.clamp(recv.long(), max=nC), ed)[:nC]
        na = node_attrs[:nC]
        message = blk.linear(msg) / (density + 1)
        message = blk.skip_tp(message, na)
        nf = blk.reshape(message)
        return model.products[0](node_feats=nf, sc=None, node_attrs=na)


def install(model, species_cols):
    from . import receiver_chunk_mace as rc

    ace = ACELayer1(model, species_cols)
    rc._layer1_orig = getattr(rc, "_layer1_orig", rc.layer1)
    rc.layer1 = ace.layer1
    return ace


# ----------------------------------------------------------------------------- self-test on a real water chunk
def _selftest():
    import argparse
    import time

    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    from . import receiver_chunk_mace as rc

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--rep", nargs=3, type=int, default=[8, 8, 8])
    ap.add_argument("--chunk", type=int, default=16384)
    a = ap.parse_args()
    dev = "cuda"
    model, dt = rc.build_model(a.model, "fp32_fused", dev)

    class A_:
        structure = None
        base = a.base
        rep = a.rep

    pos, box, sp, _ = rc.load_structure(A_, model, dt, dev)
    sysm = rc.System(pos, box, sp, float(model.r_max), dev)
    sysm.build_neighbors()
    cols = sorted(set(int(c) for c in torch.unique(sysm.spidx.long()).tolist()))
    print(f"N={sysm.N:,}  species columns present: {cols}", flush=True)
    ace = ACELayer1(model, cols)
    t = ace.tab
    print(f"layer-1 TP: C={t['C']} nW={t['nW']} nX={t['nX']} nY={t['nY']} nO={t['nO']} terms={len(t['terms'])} "
          f"pairs={len(ace.pair_w)} hidden H={ace.H}", flush=True)
    loc = torch.full((sysm.N,), -1, dtype=torch.int32, device=dev)
    g = rc.LocalGraph(sysm, 0, min(a.chunk, sysm.N), loc)
    na = rc.node_attrs_of(model, sysm.spidx[g.nodes_padded] if hasattr(g, "nodes_padded") else sysm.spidx[g.nodes], dt)
    v = g.vectors.to(dt)

    # (1) raw message vs original cuEq conv_tp (summed over receivers)
    blk = model.interactions[0]
    with torch.no_grad():
        _, Y, ef, cut = rc.edge_embed(model, g, na, v)
        X = blk.linear_up(model.node_embedding(na))
        tw = blk.conv_tp_weights(ef)
        if cut is not None:
            tw = tw * cut
        ref = blk.conv_tp(X, Y, tw, g.edge_index)[: g.nC]
        eye = torch.zeros(ace.S, na.shape[1], dtype=dt, device=dev)
        eye[torch.arange(ace.S, device=dev), ace.cols] = 1.0
        Xs = blk.linear_up(model.node_embedding(eye))
        s_send = ace.col2s[na.argmax(1)][g.edge_index[0].long()]
        best = None
        for lay in ("seg", "mul"):
            ace.layout = lay
            msg = ace.message(Xs, Y, ef, cut, g.edge_index[1], s_send, g.nC)
            err = float((msg - ref).abs().max() / ref.abs().max())
            print(f"  message layout={lay}: rel.max err vs cuEq conv_tp = {err:.3e}", flush=True)
            if best is None or err < best[1]:
                best = (lay, err)
        ace.layout = best[0]
        assert best[1] < 1e-5, f"no layout matches ({best})"
        print(f"  -> layout '{best[0]}' selected", flush=True)

    # (2) full layer-1 output h1 and d(sum h1*G)/dvec vs original
    G = torch.randn(g.nC, rc.layer1(model, g, na, v).shape[1], device=dev, dtype=dt)

    def run(fn):
        vv = v.detach().requires_grad_(True)
        h1 = fn(model, g, na, vv)
        (gv,) = torch.autograd.grad((h1 * G).sum(), [vv])
        return h1.detach(), gv

    h_ref, g_ref = run(rc.layer1)
    h_ace, g_ace = run(ace.layer1)
    eh = float((h_ace - h_ref).abs().max() / h_ref.abs().max())
    eg = float((g_ace - g_ref).abs().max() / g_ref.abs().max())
    print(f"  layer-1 h1: rel.max err {eh:.3e}    d/dvec: rel.max err {eg:.3e}", flush=True)

    def timeit(fn, n=10):
        for _ in range(2):
            run(fn)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            run(fn)
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / n * 1e3

    for name, fn in (("original", rc.layer1), ("ACE", ace.layer1)):
        torch.cuda.reset_peak_memory_stats()
        ms = timeit(fn)
        print(f"  {name:8s} layer-1 fwd+bwd on chunk ({g.nC} receivers, {int(g.nE):,} edges): {ms:8.2f} ms   "
              f"peak {torch.cuda.max_memory_allocated()/2**30:.2f} GiB", flush=True)
    with torch.no_grad():
        for name, fn in (("original", rc.layer1), ("ACE", ace.layer1)):
            for _ in range(2):
                fn(model, g, na, v)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(10):
                fn(model, g, na, v)
            torch.cuda.synchronize()
            print(f"  {name:8s} layer-1 forward only (pass A): {(time.perf_counter()-t0)/10*1e3:8.2f} ms", flush=True)


if __name__ == "__main__":
    _selftest()
