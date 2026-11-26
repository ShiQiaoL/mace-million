"""Exercise the ML-IAP force callback with analytic layers and two local atoms on CPU."""
import os
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("ase")

from mace_onegpu import fast_layers, receiver_chunk_mace
from mace_onegpu.mliap_threepass import ThreePassMLIAP


class Model(torch.nn.Module):
    atomic_numbers = [1]
    r_max = 5

    def __init__(self, dtype):
        super().__init__()
        self.slope = torch.nn.Parameter(torch.tensor(1 + 2**-30, dtype=dtype))
        self.readouts = [lambda h, heads: h]

    def atomic_energies_fn(self, node_attrs):
        return torch.zeros_like(node_attrs)

    def scale_shift(self, energies, heads):
        return energies


class AnalyticLayers:
    def layer1(self, model, graph, attrs, vec):
        x, y, _ = vec[:graph.nC].T
        return (model.slope * x + y.square()).unsqueeze(1)

    def layer2_energy(self, model, graph, attrs, vec, h1, heads):
        x, _, z = vec[:graph.nC].T
        return -1.5 * x + z.square() + 0.5 * h1[:graph.nC, 0]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_pair_derivatives_preserve_model_precision(monkeypatch, dtype):
    for key in list(os.environ):
        if key.startswith("MACE3P_"):
            monkeypatch.delenv(key)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(fast_layers, "FastLayers", lambda *args, **kwargs: AnalyticLayers())
    model = Model(dtype)
    wrapper = ThreePassMLIAP(model, chunk=2, node_bucket=2, e_align=2)
    vec = torch.tensor([[1, 0.5 + 2**-30, 0.5 - 2**-31], [-1, -0.5 - 2**-30, -0.5 + 2**-31]],
                       dtype=torch.float64)
    data = SimpleNamespace(nlocal=2, ntotal=2, npairs=2, elems=[0, 0], pair_i=[1, 0], pair_j=[0, 1], rij=vec,
                           eatoms=torch.zeros(2, dtype=torch.float64))
    data.forward_exchange = data.reverse_exchange = lambda src, dst, width: dst.copy_(src)
    data.update_pair_forces_gpu = lambda pf: setattr(data, "pair_forces", pf)
    wrapper.compute_forces(data)
    expected = vec.to(dtype).clone()
    expected[:, 0] = 1.5 * (model.slope.item() - 1)
    expected[:, 1] *= 3
    expected[:, 2] *= 2
    assert data.pair_forces.dtype == torch.float64
    assert torch.equal(data.pair_forces, expected.double())


@pytest.mark.parametrize(("flags", "variant", "dtype", "knots"), [
    ([], "fp32_fused", torch.float32, 512),
    (["--dtype", "float64"], "fp64_fused", torch.float64, 0),
    (["--spline-grid", "0"], "fp32_fused", torch.float32, 0),
])
def test_export_cli_selects_precision(monkeypatch, flags, variant, dtype, knots):
    def build(path, selected, device):
        assert selected == variant
        return Model(dtype), dtype

    saved = []
    monkeypatch.setattr(receiver_chunk_mace, "build_model", build)
    monkeypatch.setattr(torch, "save", lambda obj, path: saved.append(obj))
    script = Path(__file__).resolve().parents[1] / "scripts" / "make_threepass_model.py"
    monkeypatch.setattr(sys, "argv", [str(script), "input.model", "--out", "out.pt", *flags])
    runpy.run_path(str(script), run_name="__main__")
    obj, = saved
    assert obj.dtype == dtype and obj.spline_grid == knots
