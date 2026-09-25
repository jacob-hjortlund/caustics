"""Critical curves and caustics traced through the adaptive mesh's band."""

import itertools
import math
from types import SimpleNamespace

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
    # Every mixed child -- not all three corners of one class -- returns
    # exactly one segment, and no other child returns any.
    children = np.asarray(new.CHILD_VERTEX_INDICES)
    classes = det[children] >= 0
    mixed = classes.any(axis=1) & ~classes.all(axis=1)
    assert start.shape[0] == int(mixed.sum())
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


def test_trace_band_rejects_a_crossing_with_two_predecessors():
    """The two-leaf fixture above, with leaf A's orientation reversed.

    Swapping ``theta_2`` and ``theta_3`` -- and so ``m_2`` and ``m_3``, to
    keep ``m_i`` opposite ``theta_i`` -- makes leaf A clockwise, so both of
    its segments run backwards along the edge it shares with leaf B: instead
    of one segment ending where the other starts, both end on the shared
    node, giving it two predecessors and leaving no node with two
    successors.
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
    samples = [[0, 2, 1, 4, 6, 5], [0, 2, 3, 7, 8, 5]]
    band = _band(samples, lens, lens[:, 0] - 1.5)
    with pytest.raises(AssertionError, match="two predecessors"):
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
    assert tuple(curves.hole.shape) == (0,)
    assert curves.holes.centres.shape[0] == 0


# ---------------------------------------------------------------------------
# End to end, against known answers
# ---------------------------------------------------------------------------


def _stack_2x2(a, b, c, d):
    """``[[a, b], [c, d]]`` at every point, shape ``(N, 2, 2)``."""
    return backend.stack(
        (backend.stack((a, b), dim=-1), backend.stack((c, d), dim=-1)), dim=-2
    )


def _cored_isothermal(x, y):
    """``alpha = 1.2 theta / sqrt(theta**2 + 0.05)``: two circular critical curves."""
    r = (x * x + y * y + 0.05) ** 0.5
    return x - 1.2 * x / r, y - 1.2 * y / r


def _cored_isothermal_jacobian(x, y):
    r = (x * x + y * y + 0.05) ** 0.5
    k = 1.2 / r**3
    return _stack_2x2(
        1.0 - 1.2 / r + k * x * x, k * x * y, k * x * y, 1.0 - 1.2 / r + k * y * y
    )


def _row_fold(x, y):
    """``det A = 1 - 2y``: zero on ``y = 0.5``, a lattice row for ``fov=4``."""
    return x * 1.0, y - y * y


def _row_fold_jacobian(x, y):
    one, zero = backend.ones_like(x), backend.zeros_like(x)
    return _stack_2x2(one, zero, zero, 1.0 - 2.0 * y)


def _broken_fold(x, y):
    """``det A = 0.6 + 2y``, zero on ``y = -0.3``, and NaN wherever ``x > 1``."""
    nan = backend.where(
        x > 1.0, backend.zeros_like(x) + float("nan"), backend.zeros_like(x)
    )
    return x + nan, 0.6 * y + y * y + nan


def _broken_fold_jacobian(x, y):
    nan = backend.where(
        x > 1.0, backend.zeros_like(x) + float("nan"), backend.zeros_like(x)
    )
    one, zero = backend.ones_like(x), backend.zeros_like(x)
    return _stack_2x2(one + nan, zero, zero, 0.6 + 2.0 * y + nan)


CORED = SimpleNamespace(
    raytrace=_cored_isothermal, jacobian_lens_equation=_cored_isothermal_jacobian
)
ROW_FOLD = SimpleNamespace(
    raytrace=_row_fold, jacobian_lens_equation=_row_fold_jacobian
)
BROKEN_FOLD = SimpleNamespace(
    raytrace=_broken_fold, jacobian_lens_equation=_broken_fold_jacobian
)

# Tangential: 1 - 1.2 / r = 0 at r = sqrt(theta**2 + 0.05) = 1.2. Radial:
# 1 - 1.2 * 0.05 / r**3 = 0. The tangential caustic is the origin; the radial
# one is a circle of radius theta * |1 - 1.2 / r| there.
TANGENTIAL = math.sqrt(1.2**2 - 0.05)
_R_RADIAL = (1.2 * 0.05) ** (1.0 / 3.0)
RADIAL = math.sqrt(_R_RADIAL**2 - 0.05)
RADIAL_CAUSTIC = RADIAL * abs(1.0 - 1.2 / _R_RADIAL)


def _curves(mesh):
    """``[(lens, source, closed), ...]`` per curve, as numpy."""
    curves = crit.mesh_critical_curves(mesh)
    off = to_np(curves.offsets)
    lens, source, closed = (
        to_np(curves.lens),
        to_np(curves.source),
        to_np(curves.closed),
    )
    return [
        (lens[a:b], source[a:b], bool(c)) for a, b, c in zip(off[:-1], off[1:], closed)
    ]


def _signed_area(p):
    x, y = p[:, 0], p[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)


def test_cored_isothermal_curves_match_the_analytic_answers():
    """Two loops at the analytic radii, oriented, with the analytic caustics.

    The lens-plane tolerance is the guaranteed one, a quarter of the
    ``min_img_sep`` passed. The caustic has no guaranteed bound; measured on
    this exact fixture the tangential caustic lies within 3.5e-7 of the origin
    and the radial caustic radius within 1.5e-6, so 1e-4 is headroom, not
    slack. ``det A > 0`` outside the tangential curve and inside the radial
    one, so keeping it on the left runs the first clockwise and the second
    counter-clockwise.
    """
    min_img_sep = 1e-2
    mesh = new.build_adaptive_mesh(CORED, fov=4.0, init_res=16, min_img_sep=min_img_sep)
    curves = _curves(mesh)
    assert len(curves) == 2 and all(closed for _, _, closed in curves)
    (t_lens, t_src, _), (r_lens, r_src, _) = sorted(
        curves, key=lambda c: -np.hypot(*c[0].T).mean()
    )
    assert np.abs(np.hypot(*t_lens.T) - TANGENTIAL).max() < min_img_sep / 4
    assert np.abs(np.hypot(*r_lens.T) - RADIAL).max() < min_img_sep / 4
    assert _signed_area(t_lens) < 0 < _signed_area(r_lens)
    assert np.hypot(*t_src.T).max() < 1e-4
    assert np.abs(np.hypot(*r_src.T) - RADIAL_CAUSTIC).max() < 1e-4


def test_without_holes_the_curves_are_trace_band_s_and_follow_no_hole():
    mesh = new.build_adaptive_mesh(CORED, fov=4.0, init_res=16, min_img_sep=1e-2)
    got = crit.mesh_critical_curves(mesh)
    raw = crit.trace_band(mesh.critical_band)
    for field in ("lens", "source", "offsets", "closed"):
        assert np.array_equal(
            to_np(getattr(got, field)), to_np(getattr(raw, field))
        ), field
    hole = to_np(got.hole)
    assert hole.dtype == np.int64 and hole.shape == (got.lens.shape[0],)
    assert (hole == -1).all()
    assert got.holes.centres.shape[0] == 0
    assert to_np(got.holes.offsets).tolist() == [0]


def test_the_fov_cuts_the_tangential_circle_into_four_open_arcs():
    """The square ``[-1, 1]**2`` meets the tangential circle only near its corners.

    ``TANGENTIAL`` lies between 1 and ``sqrt(2)``, so four arcs of the circle
    are inside the fov, each ending on it, while the radial loop is whole.
    Every end sits on a boundary edge, whose two samples share the boundary
    coordinate exactly, so the end is on the boundary exactly.
    """
    mesh = new.build_adaptive_mesh(CORED, fov=2.0, init_res=8, min_img_sep=1e-2)
    curves = _curves(mesh)
    arcs = [lens for lens, _, closed in curves if not closed]
    loops = [lens for lens, _, closed in curves if closed]
    assert len(arcs) == 4 and len(loops) == 1
    for lens in arcs:
        for end in (lens[0], lens[-1]):
            assert np.abs(end).max() == 1.0


def test_a_curve_through_lattice_points_is_traced_without_any_parity_flag():
    """Every sample on ``y = 0.5`` has ``det A`` exactly zero.

    So no leaf carries ``LEAF_JACOBIAN_PARITY_UNRESOLVED`` -- a mask on that
    flag finds nothing -- yet the band's zero-is-positive rule traces one open
    curve along the row, exactly. ``det A > 0`` below the row puts it on the
    left of travel in ``-x``, so the curve runs from ``x = 2`` to ``x = -2``.
    This also pins the spec's rule that coincident consecutive points --
    which tracing a curve through lattice points produces -- are kept, not
    removed.
    """
    mesh = new.build_adaptive_mesh(ROW_FOLD, fov=4.0, init_res=8, min_img_sep=2e-2)
    status = to_np(mesh.leaf_status)
    assert not ((status & new.LEAF_JACOBIAN_PARITY_UNRESOLVED) != 0).any()
    curves = _curves(mesh)
    assert len(curves) == 1
    lens, _, closed = curves[0]
    assert not closed
    assert np.abs(lens[:, 1] - 0.5).max() < 1e-12
    assert lens[0, 0] == 2.0 and lens[-1, 0] == -2.0
    assert (np.diff(lens, axis=0) == 0).all(axis=1).any()


def test_a_curve_ends_where_the_lens_turns_nonfinite():
    """The fold ``y = -0.3`` runs into a region where the lens is NaN.

    No band leaf has a non-finite sample, so the curve is open: one end on the
    fov boundary at ``x = -2``, the other where the band stops, within a leaf
    edge of ``x = 1``.
    """
    min_img_sep = 0.05
    mesh = new.build_adaptive_mesh(
        BROKEN_FOLD, fov=4.0, init_res=4, min_img_sep=min_img_sep
    )
    curves = _curves(mesh)
    assert len(curves) == 1
    lens, _, closed = curves[0]
    assert not closed
    assert np.abs(lens[:, 1] + 0.3).max() < min_img_sep / 4
    left, right = sorted([lens[0, 0], lens[-1, 0]])
    assert left == -2.0
    assert 1.0 - min_img_sep < right <= 1.0
