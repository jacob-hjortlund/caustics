"""Critical curves and caustics traced through the adaptive mesh's band."""

import numpy as np
import pytest

from caustics.backend_obj import backend
from caustics.lenses.func import adaptive_critical as crit


def to_np(x):
    return backend.to_numpy(x)


# ---------------------------------------------------------------------------
# chain_order: list ranking by pointer jumping
# ---------------------------------------------------------------------------


def _random_chains(rng, sizes, cyclic):
    """A successor array over shuffled node ids, one chain per size.

    A chain of one node stays a path even when drawn cyclic: a one-node
    cycle would be a self-loop, which two distinct crossing edges never form.
    """
    k = int(sum(sizes))
    ids = rng.permutation(k)
    succ = -np.ones(k, dtype=np.int64)
    at = 0
    for n, cyc in zip(sizes, cyclic):
        chain = ids[at : at + n]
        succ[chain[:-1]] = chain[1:]
        if cyc and n > 1:
            succ[chain[-1]] = chain[0]
        at += n
    return succ


def _walk(succ):
    """Reference order: each path from its start, each cycle from its smallest node."""
    k = succ.size
    has_pred = np.zeros(k, dtype=bool)
    has_pred[succ[succ >= 0]] = True
    seen = np.zeros(k, dtype=bool)
    curves = []
    for s in range(k):
        if has_pred[s]:
            continue
        chain = [s]
        while succ[chain[-1]] >= 0:
            chain.append(int(succ[chain[-1]]))
        seen[chain] = True
        curves.append((s, chain, False))
    for s in range(k):
        if seen[s]:
            continue
        chain = [s]
        seen[s] = True
        while succ[chain[-1]] != s:
            chain.append(int(succ[chain[-1]]))
            seen[chain[-1]] = True
        curves.append((s, chain, True))
    return sorted(curves, key=lambda curve: curve[0])


@pytest.mark.parametrize("seed", range(6))
def test_chain_order_matches_a_python_walk(seed):
    """Pointer jumping gives the plain walk's curves, order and ``closed``."""
    rng = np.random.default_rng(seed)
    sizes = rng.integers(1, 40, size=12)
    cyclic = rng.random(12) < 0.5
    succ = _random_chains(rng, sizes, cyclic)
    order, offsets, closed = (
        to_np(x) for x in crit.chain_order(backend.as_array(succ, dtype=backend.int64))
    )
    want = _walk(succ)
    assert offsets.tolist() == np.cumsum([0] + [len(c) for _, c, _ in want]).tolist()
    assert closed.tolist() == [cl for _, _, cl in want]
    assert order.tolist() == [n for _, c, _ in want for n in c]


@pytest.mark.parametrize(
    "succ, order, offsets, closed",
    [
        ([], [], [0], []),
        ([-1], [0], [0, 1], [False]),
        ([1, 0], [0, 1], [0, 2], [True]),
        ([2, -1, 1], [0, 2, 1], [0, 3], [False]),
    ],
    ids=["empty", "lone node", "two-cycle", "path"],
)
def test_chain_order_small_cases(succ, order, offsets, closed):
    got = crit.chain_order(backend.as_array(np.asarray(succ, dtype=np.int64)))
    assert to_np(got[0]).tolist() == order
    assert to_np(got[1]).tolist() == offsets
    assert to_np(got[2]).tolist() == closed


def test_no_numpy_import_in_the_module():
    source = open(crit.__file__).read()
    assert "import numpy" not in source
