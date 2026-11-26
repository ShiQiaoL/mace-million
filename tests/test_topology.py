"""The batched topology build of the three-pass wrapper (ThreePassMLIAP._topology_all, MACE3P_TOPO=batched) must
reproduce the per-chunk reference (_topology) for every receiver chunk: same node list (receivers first, then the
outside senders in ascending order) and the same local sender index of every edge.  CPU only; no model needed."""
import os
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from mace_onegpu.mliap_threepass import ThreePassMLIAP  # noqa: E402


def pair_list(nl, nt, kmax, seed, self_pairs=False):
    """Receivers 0..nl-1 (sorted, random degree incl. one atom without neighbours), senders anywhere in 0..nt-1."""
    rng = np.random.default_rng(seed)
    deg = rng.integers(0, kmax + 1, size=nl)
    if nl > 3:
        deg[rng.integers(0, nl)] = 0
    pi = np.repeat(np.arange(nl, dtype=np.int32), deg)
    pj = rng.integers(0, nt, size=pi.size).astype(np.int32)
    if not self_pairs:
        bad = pj == pi
        pj[bad] = (pj[bad] + 1) % nt
    return torch.as_tensor(pi), torch.as_tensor(pj)


def wrapper(nt, C):
    o = ThreePassMLIAP.__new__(ThreePassMLIAP)
    o._dev = torch.device("cpu")
    o.chunk = C
    o._loc = torch.full((max(nt, 1),), -1, dtype=torch.int32)
    o._flag = None
    return o


CASES = [
    # nl, nt, C, kmax, seed, self_pairs
    (1000, 1300, 128, 12, 0, False),      # ghosts, partial tail chunk
    (1000, 1300, 1000, 12, 1, False),     # one full chunk: outside senders are ghosts only
    (1000, 1000, 4096, 12, 2, False),     # one chunk, no ghosts: no outside senders at all
    (1000, 1000, 100, 12, 3, False),      # no ghosts: outside senders are other chunks' atoms
    (777, 901, 64, 30, 4, False),         # odd sizes
    (5, 9, 2, 3, 5, False),               # tiny
    (1, 4, 16, 3, 6, False),              # a single local atom
    (3000, 3400, 256, 8, 7, True),        # pairs i == j allowed
    (20000, 23000, 2048, 20, 8, False),
]


@pytest.mark.parametrize("nl,nt,C,kmax,seed,self_pairs", CASES)
@pytest.mark.parametrize("group_edges", ["1", "700", "100000000"])   # one chunk per group ... everything in one group
def test_batched_topology_matches_reference(monkeypatch, nl, nt, C, kmax, seed, self_pairs, group_edges):
    monkeypatch.setenv("MACE3P_TOPO_EDGES", group_edges)
    pi, pj = pair_list(nl, nt, kmax, seed, self_pairs)
    o = wrapper(nt, C)
    rp = torch.searchsorted(pi, torch.arange(nl + 1, dtype=pi.dtype)).numpy()
    bounds = [(a, min(a + C, nl)) for a in range(0, nl, C)]
    got = o._topology_all(bounds, rp, pi, pj, nl, nt)
    assert set(got) == set(bounds)
    for a, b in bounds:
        e0, e1 = int(rp[a]), int(rp[b])
        nodes_ref, ls_ref = o._topology(a, b, e0, e1, pi, pj)
        nodes, ls = got[(a, b)]
        assert nodes.dtype == torch.int32 and ls.dtype == torch.int32
        assert torch.equal(nodes, nodes_ref)
        assert torch.equal(ls, ls_ref)
        # the local indices resolve to the right global atoms
        assert torch.equal(nodes.long()[ls.long()], pj[e0:e1].long())
        assert torch.equal(nodes.long()[(pi[e0:e1] - a).long()], pi[e0:e1].long())
        assert bool((o._loc == -1).all())
