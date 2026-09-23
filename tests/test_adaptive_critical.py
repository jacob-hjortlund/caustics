"""Critical curves and caustics traced through the adaptive mesh's band."""

import itertools

import numpy as np
import pytest

from caustics.backend_obj import backend
from caustics.lenses.func import adaptive as new
from caustics.lenses.func import adaptive_critical as crit


def to_np(x):
    return backend.to_numpy(x)


def _arr(x):
    return backend.as_array(np.asarray(x, dtype=np.float64), dtype=backend.float64)


def _band(samples, lens, det, source=None):
    """A hand-built `CriticalBand`; ``source`` defaults to ``lens``."""
    samples = np.asarray(samples, dtype=np.int64)
    return new.CriticalBand(
        leaves=backend.as_array(np.arange(samples.shape[0]), dtype=backend.int64),
        samples=backend.as_array(samples, dtype=backend.int64),
        lens=_arr(lens),
        source=_arr(lens if source is None else source),
        det=_arr(det),
    )


# The positively oriented unit leaf theta_1 theta_2 theta_3 = (0,0), (1,0),
# (0,1), followed by its midpoints m_1 m_2 m_3, m_i opposite theta_i.
UNIT_LENS = np.array(
    [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [0.0, 0.5], [0.5, 0.0]]
)


def _unit_band(det):
    return _band([[0, 1, 2, 3, 4, 5]], UNIT_LENS, det)


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


# ---------------------------------------------------------------------------
# Tracer kernels on hand-built bands
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("signs", list(itertools.product((1.0, -1.0), repeat=6)))
def test_child_segments_keep_positive_det_on_the_left(signs):
    """Every segment, for all 64 sign patterns of a leaf, has ``det > 0`` on its left.

    Between them the 64 patterns give each of the four children all six of
    its mixed sign patterns. For each segment, the child it crosses is the
    three distinct samples of its two edges; every positive corner of that
    child must lie strictly left of the segment and every negative one
    strictly right. Magnitudes are varied so the crossings are not all
    midpoints of their edges.
    """
    det = np.asarray(signs) * np.array([1.0, 2.0, 0.5, 3.0, 1.5, 0.25])
    band = _unit_band(det)
    start, end = crit.child_segments(band.samples, band.det)
    start, end = to_np(start), to_np(end)
    p, _ = crit.crossing_points(
        backend.as_array(start), band.lens, band.source, band.det
    )
    q, _ = crit.crossing_points(backend.as_array(end), band.lens, band.source, band.det)
    p, q = to_np(p), to_np(q)
    for a, b, s, e in zip(p, q, start, end):
        corners = set(s.tolist()) | set(e.tolist())
        assert len(corners) == 3
        for c in corners:
            d = b - a
            r = UNIT_LENS[c] - a
            side = d[0] * r[1] - d[1] * r[0]
            assert side > 0 if det[c] > 0 else side < 0


def test_child_segments_skip_a_leaf_of_one_class():
    for det in (np.ones(6), -np.ones(6)):
        band = _unit_band(det)
        start, end = crit.child_segments(band.samples, band.det)
        assert start.shape[0] == 0 and end.shape[0] == 0


def test_crossing_points_interpolate_the_zero_linearly_in_both_planes():
    """``t = det_p / (det_p - det_q)``, applied to ``lens`` and ``source`` alike."""
    lens = np.array([[0.0, 0.0], [4.0, 0.0]])
    source = np.array([[1.0, 1.0], [1.0, 9.0]])
    edges = backend.as_array(np.array([[0, 1]]), dtype=backend.int64)
    got_lens, got_source = crit.crossing_points(
        edges, _arr(lens), _arr(source), _arr([1.0, -3.0])
    )
    assert to_np(got_lens).tolist() == [[1.0, 0.0]]
    assert to_np(got_source).tolist() == [[1.0, 3.0]]


def test_an_exact_zero_counts_as_positive_and_puts_the_crossing_on_it():
    """``det == 0`` at ``theta_1`` with every other sample negative.

    Zero counts positive, so ``theta_1`` is the lone positive corner of its
    child and every crossing of that child sits on ``theta_1`` itself.
    """
    det = np.array([0.0, -1.0, -1.0, -1.0, -1.0, -1.0])
    band = _unit_band(det)
    curves = crit.trace_band(band)
    lens = to_np(curves.lens)
    assert lens.shape[0] > 0
    assert np.array_equal(lens, np.zeros_like(lens))


def test_two_leaves_sharing_an_edge_chain_through_one_crossing():
    """``det = x - 1.5`` across the square ``[0, 2]**2`` split on its diagonal.

    The line crosses the lower-right leaf from ``y = 0`` to the diagonal and
    the upper-left one from the diagonal to ``y = 2``; the two share the
    crossing at ``(1.5, 1.5)``, which must appear once, with the chain running
    through it. ``det`` is affine, so the linear interpolation is exact, and
    ``det > 0`` on the right of the line puts it on the left of downward
    travel.
    """
    lens = np.array(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [2.0, 2.0],
            [0.0, 2.0],
            [2.0, 1.0],
            [1.0, 1.0],
            [1.0, 0.0],
            [1.0, 2.0],
            [0.0, 1.0],
        ]
    )
    samples = [[0, 1, 2, 4, 5, 6], [0, 2, 3, 7, 8, 5]]
    curves = crit.trace_band(_band(samples, lens, lens[:, 0] - 1.5))
    assert to_np(curves.offsets).tolist() == [0, 5]
    assert to_np(curves.closed).tolist() == [False]
    assert to_np(curves.lens).tolist() == [
        [1.5, 2.0],
        [1.5, 1.5],
        [1.5, 1.0],
        [1.5, 0.5],
        [1.5, 0.0],
    ]
    assert np.array_equal(to_np(curves.source), to_np(curves.lens))


def test_trace_band_rejects_a_crossing_with_two_successors():
    """The same leaf twice gives every crossing a second successor."""
    det = np.array([1.0, -1.0, -1.0, 1.0, -1.0, 1.0])
    band = _band([[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]], UNIT_LENS, det)
    with pytest.raises(AssertionError, match="two successors"):
        crit.trace_band(band)


@pytest.mark.parametrize(
    "band",
    [new.empty_band(), _unit_band(np.ones(6))],
    ids=["empty band", "one class"],
)
def test_a_band_without_crossings_has_no_curves(band):
    curves = crit.trace_band(band)
    assert to_np(curves.offsets).tolist() == [0]
    assert tuple(curves.lens.shape) == (0, 2)
    assert tuple(curves.source.shape) == (0, 2)
    assert tuple(curves.closed.shape) == (0,)
