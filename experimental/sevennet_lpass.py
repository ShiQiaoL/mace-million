#!/usr/bin/env python
"""SPEC F2 (feasibility): exact receiver-chunked evaluation of an L-layer NequIP-type network (SevenNet-0, 5 layers).

The MACE three-pass scheme generalises to any message-passing network in which the node features of layer k depend
on the layer-(k-1) features of the receiver and its neighbours only:

  forward,  k = 1..L : X_k[C] = layer_k(X_{k-1}[C + N(C)], edges into C)     chunk by chunk, no autograd
  readout            : E = sum_i e(X_L[i]),  G_L = dE/dX_L                     per atom
  backward, k = L..1 : recompute layer_k on each chunk with autograd, loss = sum_{i in C} G_k[i] . X_k[i]
                       -> pair forces from d/d(edge vectors), and G_{k-1}[C + N(C)] += d/dX_{k-1}

Resident memory: the per-atom features X_0..X_L and one G at a time; edge-level intermediates and the autograd graph
exist for one chunk only. 2L chunk passes (MACE, L = 2, fuses the last forward with its backward: 3 passes).
In LAMMPS each layer would need one forward and one reverse ghost exchange, as for H1/G1 in mliap_threepass.py.

    python sevennet_lpass.py --model 7net-0 --base water64_1x1x1.extxyz --rep 4 4 4 --chunk 2048 --dtype float64 --check
Runs in an environment with sevenn (>= 0.10) and torch; mace is not needed.
"""
import argparse
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from mace_onegpu.receiver_chunk_mace import System, LocalGraph, _sync, _peak  # noqa: E402

GB = 1e9


def load_sevennet(name, dtype, device):
    import sevenn.util as su

    # e3nn's code-generated tensor products fix their dtype at construction: build the model under the target default
    # dtype. sevenn's patch for old checkpoints compares float32 tables with the new ones; let that comparison cast.
    prev, allclose = torch.get_default_dtype(), torch.allclose
    torch.set_default_dtype(dtype)
    torch.allclose = lambda x, y, *aa, **kw: allclose(x.to(y.dtype), y, *aa, **kw)
    try:
        model, cfg = su.model_from_checkpoint(su.pretrained_name_to_path(name) if not os.path.isfile(name) else name)
    finally:
        torch.allclose = allclose
        torch.set_default_dtype(prev)
    model = model.to(device=device, dtype=dtype).eval()
    # some sevenn/e3nn modules keep float tensors as plain attributes (not parameters/buffers): cast them as well
    for mod in model.modules():
        for name, val in list(vars(mod).items()):
            if isinstance(val, torch.Tensor) and torch.is_floating_point(val) and val.dtype != dtype:
                setattr(mod, name, val.to(device=device, dtype=dtype))
    for p in model.parameters():
        p.requires_grad_(False)
    return model, cfg


class LPassSevenNet:
    def __init__(self, model, cfg, dtype, chunk):
        self.m, self.dt, self.chunk = model, dtype, int(chunk)
        mods = model._modules
        self.L = int(cfg["num_convolution_layer"])
        self.r_max = float(cfg["cutoff"])
        self.edge_embedding = mods["edge_embedding"]
        self.embed = mods["onehot_to_feature_x"]
        self.layers = [[mods[n] for n in mods if n.startswith(f"{k}_")] for k in range(self.L)]
        self.readout = [mods["reduce_input_to_hidden"], mods["reduce_hidden_to_energy"], mods["rescale_atomic_energy"]]
        self.n_types = int(cfg.get("_number_of_species", len(model.type_map)))

    def _onehot(self, t):
        return torch.nn.functional.one_hot(t.long(), self.n_types).to(self.dt)

    def _graph(self, g, X, types, vectors):
        # SevenNet: edge_index[0] = receiver, [1] = sender, edge_vec = r_sender - r_receiver (= -LocalGraph.vectors)
        d = {"x": X[g.nodes], "node_attr": self._onehot(types[g.nodes]), "atom_type": types[g.nodes].long(),
             "edge_index": torch.stack([g.edge_index[1], g.edge_index[0]]), "edge_vec": -vectors}
        return self.edge_embedding(d)

    def _layer(self, k, d):
        for mod in self.layers[k]:
            d = mod(d)
        return d["x"]

    def compute(self, sysm, types, log=print):
        N, dev, dt = sysm.N, sysm.device, self.dt
        loc = torch.full((N,), -1, dtype=torch.int32, device=dev)
        F = torch.zeros(N, 3, dtype=torch.float64, device=dev)
        t0 = time.perf_counter()
        with torch.no_grad():
            X = [self.embed({"x": self._onehot(types), "node_attr": self._onehot(types)})["x"]]
            for k in range(self.L):                                        # forward passes
                Xk = None
                for a in range(0, N, self.chunk):
                    b = min(a + self.chunk, N)
                    g = LocalGraph(sysm, a, b, loc)
                    xk = self._layer(k, self._graph(g, X[k], types, g.vectors.to(dt)))[: g.nC]
                    if Xk is None:
                        Xk = torch.empty(N, xk.shape[1], dtype=dt, device=dev)
                    Xk[a:b] = xk
                    del g, xk
                X.append(Xk)
        t_fwd = time.perf_counter() - t0
        # readout and G_L
        XL = X[-1].clone().requires_grad_(True)
        d = {"x": XL, "atom_type": types.long()}
        for mod in self.readout:
            d = mod(d)
        e_at = d["atomic_energy"].reshape(-1)
        E = float(e_at.double().sum())
        (G,) = torch.autograd.grad(e_at.sum(), [XL])
        del XL, d, e_at
        t0 = time.perf_counter()
        for k in range(self.L - 1, -1, -1):                                 # backward passes (recompute)
            Gprev = torch.zeros_like(X[k]) if k > 0 else None
            for a in range(0, N, self.chunk):
                b = min(a + self.chunk, N)
                g = LocalGraph(sysm, a, b, loc)
                vec = g.vectors.to(dt).requires_grad_(True)
                xin = X[k].detach().clone().requires_grad_(k > 0)
                xk = self._layer(k, self._graph(g, xin, types, vec))[: g.nC]
                loss = (G[a:b] * xk).sum()
                if k > 0:
                    gv, gx = torch.autograd.grad(loss, [vec, xin])
                    Gprev += gx                                              # gradient w.r.t. X_{k-1} of all nodes
                else:
                    (gv,) = torch.autograd.grad(loss, [vec])
                # vec = r_recv - r_send:  F_recv -= dE/dvec,  F_send += dE/dvec
                F.index_add_(0, g.recv_g, -gv.double())
                F.index_add_(0, g.send_g, gv.double())
                del g, vec, xin, xk, loss, gv
            G = Gprev
        t_bwd = time.perf_counter() - t0
        resident = sum(x.numel() * x.element_size() for x in X)
        log(f"  L={self.L} forward {t_fwd:.1f}s backward {t_bwd:.1f}s  resident X_0..X_L {resident / N:.0f} B/atom")
        return E, F, resident


def reference(model, atoms, cutoff, device):
    """Original SevenNet evaluation (full graph, autograd through everything) on the same model and dtype."""
    import sevenn.nn.node_embedding as ne
    from sevenn.atom_graph_data import AtomGraphData
    from sevenn.train.dataload import unlabeled_atoms_to_graph

    dt = next(model.parameters()).dtype
    if not getattr(ne, "_f2_patched", False):        # the one-hot module casts to float32; follow the model dtype
        orig = ne.OnehotEmbedding.forward

        def fwd(self, data):
            out = orig(self, data)
            out[self.key_output] = out[self.key_output].to(dt)
            if self.key_additional_output is not None:
                out[self.key_additional_output] = out[self.key_additional_output].to(dt)
            return out
        ne.OnehotEmbedding.forward, ne._f2_patched = fwd, True
    gnp = unlabeled_atoms_to_graph(atoms, cutoff)
    data = AtomGraphData.from_numpy_dict(gnp).to(device)
    # edge vectors at full precision: r_sender - r_receiver by minimum image (orthorhombic, box > 2 r_cut)
    pos = torch.tensor(atoms.get_positions(), dtype=torch.float64, device=device)
    L = torch.tensor(np.diag(np.array(atoms.get_cell())), dtype=torch.float64, device=device)
    ei = data["edge_index"]
    v = pos[ei[1]] - pos[ei[0]]
    v = v - L * torch.round(v / L)
    print(f"reference: |edge_vec(input) - edge_vec(fp64)|max = {float((data['edge_vec'].double() - v).abs().max()):.2e} A", flush=True)
    data["edge_vec"] = v.to(dt)
    for k in list(data.keys()):
        v = data[k]
        if isinstance(v, torch.Tensor) and torch.is_floating_point(v) and v.dtype != dt:
            data[k] = v.to(dt)
    # from_numpy_dict stores positions/cell in float32: take them from the Atoms object at full precision
    if "pos" in data:
        data["pos"] = torch.tensor(atoms.get_positions(), dtype=dt, device=device)
    if "cell_lattice_vectors" in data:
        data["cell_lattice_vectors"] = torch.tensor(np.array(atoms.get_cell()), dtype=dt, device=device).reshape(data["cell_lattice_vectors"].shape)
    model.train(False)
    model.set_is_batch_data(False)          # as sevenn's calculator: single graph, no 'batch' key
    out = model(data)
    return float(out["inferred_total_energy"].double().sum()), out["inferred_force"].detach().double()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="7net-0")
    ap.add_argument("--base", required=True)
    ap.add_argument("--rep", nargs=3, type=int, default=[1, 1, 1])
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--dtype", default="float64", choices=["float32", "float64"])
    ap.add_argument("--check", action="store_true", help="compare every atom with the original model on the base cell")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--stock", action="store_true", help="also run the original model on the full replica (memory)")
    a = ap.parse_args()
    from ase.io import read

    dev = a.device
    dt = getattr(torch, a.dtype)
    model, cfg = load_sevennet(a.model, dt, dev)     # load under the float32 default (sevenn's checkpoint patching)
    torch.set_default_dtype(dt)
    ev = LPassSevenNet(model, cfg, dt, a.chunk)
    base = read(a.base)
    big = base.repeat(a.rep)
    tm = model.type_map
    types = torch.tensor([tm[int(z)] for z in big.get_atomic_numbers()], device=dev)
    cell = np.diag(np.array(big.get_cell()))
    pos = torch.tensor(big.get_positions(), dtype=torch.float64, device=dev)
    sysm = System(pos, cell, types.to(torch.int16), ev.r_max, dev)
    sysm.build_neighbors()
    types_s = types[sysm.perm]
    N = sysm.N
    print(f"SevenNet {a.model}: L={ev.L} r_max={ev.r_max} dtype={a.dtype} N={N:,} edges/atom={float(sysm.NN.float().mean()):.1f} "
          f"chunk={a.chunk}", flush=True)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    _sync(); t0 = time.perf_counter()
    E, F, resident = ev.compute(sysm, types_s)
    _sync(); tt = time.perf_counter() - t0
    print(f"L-pass: E={E:.6f} eV ({E / N:.6f}/atom)  t={tt:.2f}s ({N / tt:.3e} atom-step/s)  peak={_peak() / GB:.2f} GB "
          f"({_peak() / N / 1e3:.1f} kB/atom)", flush=True)
    Fo = torch.empty_like(F)
    Fo[sysm.perm] = F
    if a.check:
        nb = len(base)
        Eb, Fb = reference(model, base, ev.r_max, dev)
        k = torch.arange(N, device=dev) % nb                    # ase.repeat: copies are contiguous blocks of nb atoms
        d = Fo - Fb[k]
        print(f"CHECK vs original SevenNet on the {nb}-atom base cell: dE/atom={E / N - Eb / nb:+.3e} eV  "
              f"max|dF|={float(d.abs().max()):.3e}  RMS(dF)={math.sqrt(float((d * d).mean())):.3e} eV/A  (all {N:,} atoms)",
              flush=True)
    if a.stock:
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        try:
            _sync(); t0 = time.perf_counter()
            Es, Fs = reference(model, big, ev.r_max, dev)
            _sync(); ts = time.perf_counter() - t0
            print(f"stock SevenNet on N={N:,}: E={Es:.6f}  t={ts:.2f}s  peak={_peak() / GB:.2f} GB ({_peak() / N / 1e3:.1f} kB/atom)  "
                  f"max|F_stock - F_Lpass|={float((Fs - Fo).abs().max()):.3e}", flush=True)
        except torch.cuda.OutOfMemoryError:
            print(f"stock SevenNet on N={N:,}: OUT OF MEMORY", flush=True)


if __name__ == "__main__":
    main()
