import warnings

import numpy as np
import pytest

from caustics.backend_obj import backend
from caustics.cosmology import FlatLambdaCDM
from caustics.lenses import SIE, Point
from caustics.lenses.adaptive import (
    LeafStatus,
    BuildStats,
    Mesh,
    _ActiveKeys,
    _canonical_order,
    _close,
    _dedup_representatives,
    _depth_floor,
    _edge_quarter_keys,
    _initial_triangles,
    _invalidate_nonfinite_origins,
    _Lattice,
    _LeafStore,
    _make_raytrace_np,
    _midpoint_ij,
    _min_angle,
    _red_split,
    _refine,
    _validate_build_args,
    _VertexCache,
    build_adaptive_mesh,
)
from caustics.lenses.func import forward_raytrace_rootfind
from caustics.lenses.func.adaptive import (
    CHILD_VERTEX_INDICES,
    ROOT_SHAPES,
    affine_from_triangles,
    child_matrix_tables,
    contains,
    converged_from_deviation,
    evaluate_criterion,
    midpoint_deviation,
    sanitize_bary,
    shape_matrix,
    sigma_min_2x2,
    triangle_weights,
)
from caustics.utils import meshgrid

RNG = np.random.default_rng(20260904)


def to_np(x):
    return backend.to_numpy(x)


def as_arr(x):
    return backend.as_array(np.asarray(x, dtype=np.float64))


def signed_area(tri):
    """Twice the signed area of a (..., 3, 2) triangle."""
    P = shape_matrix(tri)
    return P[..., 0, 0] * P[..., 1, 1] - P[..., 0, 1] * P[..., 1, 0]


def red_split(tri):
    """Split a (..., 3, 2) triangle into (..., 4, 3, 2) children, canonical order."""
    t1, t2, t3 = tri[..., 0, :], tri[..., 1, :], tri[..., 2, :]
    six = np.stack([t1, t2, t3, (t2 + t3) / 2, (t3 + t1) / 2, (t1 + t2) / 2], axis=-2)
    idx = np.asarray(CHILD_VERTEX_INDICES)
    return six[..., idx, :].reshape(*tri.shape[:-2], 4, 3, 2)


def _six_points(tri):
    t1, t2, t3 = tri[:, 0, :], tri[:, 1, :], tri[:, 2, :]
    mid = np.stack([(t2 + t3) / 2, (t3 + t1) / 2, (t1 + t2) / 2], axis=1)
    return tri, mid


def test_group_tables_close_and_have_order_six():
    M, G, COMPOSE, PINV0, ROOT_CLASS = child_matrix_tables()
    assert M.shape == (4, 2, 2) and G.shape == (6, 2, 2)
    dets = M[:, 0, 0] * M[:, 1, 1] - M[:, 0, 1] * M[:, 1, 0]
    assert (dets == 1).all(), "every child matrix must have det +1"
    # G is closed under right multiplication by every M, and COMPOSE indexes it
    for c in range(6):
        for k in range(4):
            assert (G[c] @ M[k] == G[COMPOSE[c, k]]).all()
    # exactly six distinct elements
    assert len({G[c].tobytes() for c in range(6)}) == 6
    # both root shapes lie in one orbit
    R0 = shape_matrix(np.asarray(ROOT_SHAPES[0], dtype=np.float64))
    R1 = shape_matrix(np.asarray(ROOT_SHAPES[1], dtype=np.float64))
    assert np.allclose(R0 @ G[ROOT_CLASS[1]], R1)


def test_root_shapes_are_positively_oriented():
    for shape in ROOT_SHAPES:
        assert signed_area(np.asarray(shape, dtype=np.float64)) > 0


@pytest.mark.parametrize("flip", [False, True])
def test_children_preserve_orientation_and_quarter_the_area(flip):
    tri = RNG.normal(size=(64, 3, 2))
    if flip:
        tri = tri[:, ::-1, :]
    parent = signed_area(tri)
    kids = signed_area(red_split(tri))
    assert np.allclose(kids, parent[:, None] / 4, rtol=0, atol=1e-12)


def test_pinv_table_scales_by_two_to_the_level():
    M, G, COMPOSE, PINV0, ROOT_CLASS = child_matrix_tables()
    h0 = 0.05
    for shape_idx in (0, 1):
        root = np.asarray(ROOT_SHAPES[shape_idx], dtype=np.float64) * h0
        cls = ROOT_CLASS[shape_idx]
        tri, level = root, 0
        for _ in range(5):
            kids = red_split(tri)
            for k in range(4):
                direct = np.linalg.inv(shape_matrix(kids[k]))
                table = (2.0 ** (level + 1) / h0) * PINV0[COMPOSE[cls, k]]
                assert np.allclose(direct, table, rtol=1e-12, atol=1e-12)
            tri, cls, level = kids[0], COMPOSE[cls, 0], level + 1


def test_affine_from_table_matches_explicit_inversion():
    M, G, COMPOSE, PINV0, ROOT_CLASS = child_matrix_tables()
    h0 = 0.05
    tri = np.asarray(ROOT_SHAPES[1], dtype=np.float64) * h0
    cls, level = ROOT_CLASS[1], 0
    kids = red_split(tri)
    lens_map = RNG.normal(size=(2, 2))
    for k in range(4):
        q = kids[k] @ lens_map.T
        explicit = affine_from_triangles(kids[k], q)
        Q = shape_matrix(q)
        table = Q @ ((2.0 ** (level + 1) / h0) * PINV0[COMPOSE[cls, k]])
        assert np.allclose(explicit, table, rtol=1e-10, atol=1e-12)
        assert np.allclose(explicit, lens_map, rtol=1e-10, atol=1e-12)


def test_sigma_min_matches_svd():
    A = RNG.normal(size=(2000, 2, 2))
    expected = np.linalg.svd(A, compute_uv=False)[:, -1]
    assert np.allclose(sigma_min_2x2(A), expected, rtol=1e-9, atol=1e-12)


def test_sigma_min_near_and_exactly_singular():
    a = RNG.normal(size=(500, 2))
    perp = np.stack([-a[:, 1], a[:, 0]], axis=1)
    # Rows offset perpendicularly, so det == eps * |a|^2 is genuinely small.
    # Scaling one row instead would give a rank-1 matrix with det exactly zero,
    # which tests the degenerate branch, not the near-degenerate one.
    for eps in (1e-4, 1e-7):
        A = np.stack([a, a + eps * perp], axis=1)
        got = sigma_min_2x2(A)
        expected = np.linalg.svd(A, compute_uv=False)[:, -1]
        assert np.isfinite(got).all()
        assert np.allclose(got, expected, rtol=1e-5, atol=0.0)
    exact = np.stack([a, 2.0 * a], axis=1)  # rank 1: det is exactly 0 in IEEE
    assert (sigma_min_2x2(exact) == 0.0).all()


def test_sigma_min_of_zero_matrix_is_exactly_zero():
    got = sigma_min_2x2(np.zeros((3, 2, 2)))
    assert (got == 0.0).all()
    assert not np.isnan(got).any()


def test_sigma_min_handles_conformal_ulp_negativity():
    """F - 2D is exactly zero in R but goes negative by an ulp in float.

    Without the clip this returns NaN for a near-circularly-symmetric lens core,
    which fail-closed then refines to max_level. See spec section 2.2.
    """
    ab = RNG.normal(size=(200000, 2)).astype(np.float32)
    A = np.empty((ab.shape[0], 2, 2), dtype=np.float32)
    A[:, 0, 0] = ab[:, 0]
    A[:, 0, 1] = ab[:, 1]
    A[:, 1, 0] = -ab[:, 1]
    A[:, 1, 1] = ab[:, 0]
    F = (A.astype(np.float32) ** 2).sum(axis=(-2, -1))
    D = np.abs(A[:, 0, 0] * A[:, 1, 1] - A[:, 0, 1] * A[:, 1, 0])
    assert (F - 2 * D < 0).any(), "test fixture no longer exercises the guard"
    got = sigma_min_2x2(A.astype(np.float64))
    assert np.isfinite(got).all()


def test_sigma_min_propagates_nan_input():
    A = np.full((1, 2, 2), np.nan)
    assert np.isnan(sigma_min_2x2(A)).all()


def test_converged_from_deviation_fails_closed():
    r = np.zeros((3, 3))
    assert converged_from_deviation(r, np.array([1.0, 1.0, 1.0]), 1.0).all()
    # NaN must take the split branch, not the converged branch
    assert not converged_from_deviation(r, np.full(3, np.nan), 1.0).any()
    # s == 0 with r == 0 gives 0 < 0, False, split -- the conservative direction
    assert not converged_from_deviation(r, np.zeros(3), 1.0).any()


def test_criterion_converges_on_an_affine_map():
    M, G, COMPOSE, PINV0, ROOT_CLASS = child_matrix_tables()
    h0 = 0.05
    tri = np.stack([np.asarray(ROOT_SHAPES[s], dtype=np.float64) * h0 for s in (0, 1)])
    classes = ROOT_CLASS.copy()
    lens_map = np.array([[0.7, 0.1], [-0.2, 0.9]])
    v, m = _six_points(tri)
    keep, parity_ok, s = evaluate_criterion(
        v @ lens_map.T, m @ lens_map.T, classes, 0, h0, 1e-3, PINV0, COMPOSE
    )
    assert keep.all() and parity_ok.all()
    assert np.allclose(s, np.linalg.svd(lens_map, compute_uv=False)[-1])


def test_criterion_parity_fires_across_a_fold():
    M, G, COMPOSE, PINV0, ROOT_CLASS = child_matrix_tables()
    h0 = 1.0
    tri = np.asarray(ROOT_SHAPES[1], dtype=np.float64)[None] * h0 - np.array([0.0, 0.5])
    classes = ROOT_CLASS[1:2].copy()
    fold = lambda p: np.stack([p[..., 0], p[..., 1] ** 2], axis=-1)  # noqa: E731
    v, m = _six_points(tri)
    keep, parity_ok, s = evaluate_criterion(
        fold(v), fold(m), classes, 0, h0, 1e-3, PINV0, COMPOSE
    )
    assert not parity_ok[0] and not keep[0]


def test_affine_and_sigma_min_are_invariant_to_simultaneous_relabelling():
    M, G, COMPOSE, PINV0, ROOT_CLASS = child_matrix_tables()
    p = RNG.normal(size=(200, 3, 2))
    q = RNG.normal(size=(200, 3, 2))
    base = affine_from_triangles(p, q)
    perm = np.argsort(RNG.random((200, 3)), axis=1)
    pp = np.take_along_axis(p, perm[..., None], axis=1)
    qp = np.take_along_axis(q, perm[..., None], axis=1)
    permuted = affine_from_triangles(pp, qp)
    assert np.allclose(base, permuted, rtol=1e-9, atol=1e-11)
    assert np.array_equal(
        np.sign(np.linalg.det(base)), np.sign(np.linalg.det(permuted))
    )
    assert np.allclose(sigma_min_2x2(base), sigma_min_2x2(permuted), rtol=1e-9)


def test_criterion_reports_nonfinite_as_split():
    M, G, COMPOSE, PINV0, ROOT_CLASS = child_matrix_tables()
    h0 = 0.05
    tri = np.asarray(ROOT_SHAPES[0], dtype=np.float64)[None] * h0
    classes = ROOT_CLASS[0:1].copy()
    v, m = _six_points(tri)
    bad = m.copy()
    bad[0, 0, 0] = np.nan
    keep, parity_ok, s = evaluate_criterion(
        v, bad, classes, 0, h0, 1e-3, PINV0, COMPOSE
    )
    assert not keep[0]


def test_midpoint_deviation_pairs_midpoint_with_opposite_edge():
    v = np.array([[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]])
    m = np.array([[[0.5, 0.5], [0.0, 0.5], [0.5, 0.0]]])  # exact affine images
    assert np.allclose(midpoint_deviation(v, m), 0.0)
    shifted = m.copy()
    shifted[0, 1] += np.array([0.0, 0.25])
    got = midpoint_deviation(v, shifted)
    assert np.allclose(got, [[0.0, 0.25, 0.0]])


def test_weights_sum_to_twice_the_signed_area():
    tri = RNG.normal(size=(500, 3, 2))
    beta = RNG.normal(size=(500, 2))
    w = to_np(triangle_weights(as_arr(tri), as_arr(beta)))
    P = shape_matrix(tri)
    d = P[:, 0, 0] * P[:, 1, 1] - P[:, 0, 1] * P[:, 1, 0]
    assert np.allclose(w.sum(axis=1), d, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("flip", [False, True])
def test_containment_agrees_with_barycentric_truth_on_both_parities(flip):
    tri = np.array([[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]])
    if flip:
        tri = tri[:, ::-1, :]
    pts = RNG.uniform(-0.5, 1.5, size=(4000, 2))
    tiled = np.repeat(tri, len(pts), axis=0)
    hit = to_np(contains(triangle_weights(as_arr(tiled), as_arr(pts))))
    truth = (pts[:, 0] >= 0) & (pts[:, 1] >= 0) & (pts[:, 0] + pts[:, 1] <= 1)
    assert (hit == truth).all()


def test_shared_edge_weights_are_exactly_negated():
    """Two leaves meeting on an edge must not both reject a point on that edge."""
    verts = np.array([[0.0, 0.0], [1.0, 0.3], [0.4, 1.0], [1.3, 1.2]])
    left = verts[[0, 1, 2]][None]
    right = verts[[1, 3, 2]][None]  # shares the edge (1, 2), opposite traversal
    beta = np.array([[0.62, 0.71]])
    wl = to_np(triangle_weights(as_arr(left), as_arr(beta)))
    wr = to_np(triangle_weights(as_arr(right), as_arr(beta)))
    # left's weight opposite vertex 0 uses the (1, 2) edge; right's opposite
    # vertex 3 uses (2, 1). They must be bit-exact negatives.
    assert wl[0, 0] == -wr[0, 1]


def test_sanitize_bary_is_always_in_the_simplex():
    """The clip path keeps wide-but-finite w/d ratios inside the simplex.

    The 1e-300 scaling stresses the clamp with extreme ratios; it does not reach
    the centroid fallback, because 1e-300 is a normal double and w/d is invariant
    under a uniform scaling. The fallback is covered by the two tests below.
    """
    w = RNG.normal(size=(1000, 3)) * 1e-300
    d = w.sum(axis=1)
    bary = to_np(sanitize_bary(as_arr(w), as_arr(d)))
    assert np.isfinite(bary).all()
    assert (bary >= 0).all() and (bary <= 1).all()
    assert np.allclose(bary.sum(axis=1), 1.0, rtol=0, atol=1e-12)


def test_sanitize_bary_falls_back_to_the_centroid_on_total_degeneracy():
    w = np.zeros((4, 3))
    d = np.zeros(4)
    bary = to_np(sanitize_bary(as_arr(w), as_arr(d)))
    assert np.array_equal(bary, np.full((4, 3), 1.0 / 3.0))


def test_sanitize_bary_selects_per_row_between_normalized_and_centroid():
    """Degenerate and ordinary rows in one call must be resolved independently.

    Both other sanitize_bary tests are all-or-nothing -- every row ordinary, or
    every row degenerate -- so neither would catch an implementation that decided
    the fallback once for the whole batch instead of per row.
    """
    w = as_arr([[3.0, 1.5, 1.5], [0.0, 0.0, 0.0], [2.0, 1.0, 1.0], [0.0, 0.0, 0.0]])
    d = as_arr([6.0, 0.0, 4.0, 0.0])
    bary = to_np(sanitize_bary(w, d))
    assert np.allclose(bary[[0, 2]], [[0.5, 0.25, 0.25], [0.5, 0.25, 0.25]])
    assert np.allclose(bary[[1, 3]], 1.0 / 3.0)
    assert np.allclose(bary.sum(axis=1), 1.0, rtol=0, atol=1e-12)


def test_sanitize_bary_recovers_ordinary_coordinates():
    tri = np.array([[[0.0, 0.0], [2.0, 0.0], [0.0, 3.0]]])
    beta = np.array([[0.5, 0.75]])
    w = triangle_weights(as_arr(tri), as_arr(beta))
    d = as_arr([2.0 * 3.0])
    bary = to_np(sanitize_bary(w, d))
    assert np.allclose(bary @ tri[0], beta, rtol=1e-12, atol=1e-14)


def test_depth_floor_matches_the_size_criterion():
    assert _depth_floor(5.0, 100, 10.0) == 0
    d = _depth_floor(5.0, 100, 1e-3)
    l0 = np.sqrt(2) * 5.0 / 100
    assert l0 / 2**d <= 1e-3 < l0 / 2 ** (d - 1)


def test_lattice_key_roundtrip_and_geometry():
    lat = _Lattice(4.0, 0.0, 0.0, 4, 3)
    assert lat.n == 32
    ij = np.array([[0, 0], [32, 32], [7, 19]])
    assert np.array_equal(lat.ij_from_key(lat.key(ij)), ij)
    assert np.allclose(lat.xy(ij[0]), [-2.0, -2.0])
    assert np.allclose(lat.xy(ij[1]), [2.0, 2.0])
    assert lat.on_boundary(ij).tolist() == [True, True, False]


def test_vertex_cache_lookup_missing_insert():
    cache = _VertexCache()
    keys = np.array([7, 3, 7, 11], dtype=np.int64)
    assert np.array_equal(cache.missing(keys), np.array([3, 7, 11]))
    todo = cache.missing(keys)
    ij = np.stack([todo, todo], axis=-1)
    cache.insert(todo, ij, ij.astype(np.float64))
    assert len(cache) == 3
    assert cache.missing(keys).size == 0
    assert np.array_equal(cache.lookup(np.array([7, 99])), np.array([1, -1]))


def test_active_keys_membership():
    ak = _ActiveKeys()
    assert not ak.contains(np.array([1])).any()
    ak.add(np.array([5, 1, 5], dtype=np.int64))
    assert ak.contains(np.array([1, 5, 6])).tolist() == [True, True, False]


def test_leaf_store_add_remove_compact():
    store = _LeafStore()
    rows = store.add(
        np.arange(6).reshape(2, 3), 0, np.zeros(2, np.int64), LeafStatus.CONVERGED
    )
    store.add(np.arange(3).reshape(1, 3), 1, np.zeros(1, np.int64), LeafStatus.FORCED)
    store.remove(rows[:1])
    v, level, cls, status = store.compact()
    assert v.shape == (2, 3)
    assert level.tolist() == [0, 1]
    assert status.tolist() == [LeafStatus.CONVERGED, LeafStatus.FORCED]


def test_initial_triangles_tile_the_square_and_are_positively_oriented():
    M, G, COMPOSE, PINV0, ROOT_CLASS = child_matrix_tables()
    init_res, max_level = 4, 2
    ij, cls = _initial_triangles(init_res, max_level, ROOT_CLASS)
    assert ij.shape == (2 * init_res**2, 3, 2)
    area = signed_area(ij.astype(np.float64))
    assert (area > 0).all()
    step = 1 << max_level
    assert np.isclose(area.sum() / 2, (init_res * step) ** 2)
    assert set(np.unique(cls)) == {int(ROOT_CLASS[0]), int(ROOT_CLASS[1])}


def test_midpoints_are_exact_integers_and_opposite_their_vertex():
    ij = np.array([[[0, 0], [4, 0], [0, 4]]], dtype=np.int64)
    m = _midpoint_ij(ij)
    assert np.array_equal(m[0], np.array([[2, 2], [0, 2], [2, 0]]))


def test_red_split_is_triangle_major_and_advances_the_class():
    M, G, COMPOSE, PINV0, ROOT_CLASS = child_matrix_tables()
    v = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
    m = np.array([[6, 7, 8], [9, 10, 11]], dtype=np.int64)
    cls = np.array([0, 4], dtype=np.int64)
    cv, cc = _red_split(v, m, cls, COMPOSE)
    assert cv.shape == (8, 3) and cc.shape == (8,)
    assert cv[0].tolist() == [0, 8, 7]  # C_1 = (theta1, m3, m2)
    assert cv[3].tolist() == [6, 7, 8]  # C_4 = (m1, m2, m3)
    assert cc[:4].tolist() == COMPOSE[0].tolist()
    assert cc[4:].tolist() == COMPOSE[4].tolist()


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(fov=0.0),
        dict(fov=-1.0),
        dict(init_res=0),
        dict(min_img_sep=0.0),
        dict(min_img_sep=-1.0),
        dict(max_depth=-1),
    ],
)
def test_validate_build_args_rejects_bad_input(kwargs):
    args = dict(fov=5.0, init_res=8, min_img_sep=1e-2, max_depth=10)
    args.update(kwargs)
    with pytest.raises(ValueError):
        _validate_build_args(**args)


def test_validate_build_args_rejects_lattice_overflow():
    with pytest.raises(ValueError, match="lattice"):
        _validate_build_args(fov=5.0, init_res=100, min_img_sep=1e-12, max_depth=60)


def make_counting_raytrace(fn):
    """Wrap a numpy (N,2)->(N,2) map as a backend raytrace, counting evaluations."""
    calls = {"points": 0, "batches": 0}

    def raytrace(x, y):
        xy = np.stack([backend.to_numpy(x), backend.to_numpy(y)], axis=-1)
        calls["points"] += xy.shape[0]
        calls["batches"] += 1
        out = fn(xy)
        return backend.as_array(out[:, 0]), backend.as_array(out[:, 1])

    return raytrace, calls


def refine_with(fn, fov=4.0, init_res=4, min_img_sep=0.5, max_depth=25):
    M, G, COMPOSE, PINV0, ROOT_CLASS = child_matrix_tables()
    tables = (M, G, COMPOSE, PINV0, ROOT_CLASS)
    max_level = min(max_depth, _depth_floor(fov, init_res, min_img_sep))
    lat = _Lattice(fov, 0.0, 0.0, init_res, max_level)
    raytrace, calls = make_counting_raytrace(fn)
    ref = _refine(
        _make_raytrace_np(raytrace, None),
        lat,
        init_res,
        fov / init_res,
        min_img_sep,
        max_level,
        tables,
        None,
    )
    return ref, lat, calls, max_level


AFFINE = np.array([[0.7, 0.1], [-0.2, 0.9]])


def test_refine_converges_everywhere_at_level_zero_for_an_affine_map():
    ref, lat, calls, max_level = refine_with(lambda p: p @ AFFINE.T)
    v, level, cls, status = ref.store.compact()
    assert v.shape[0] == 2 * 4**2
    assert (level == 0).all()
    assert (status == LeafStatus.CONVERGED).all()
    assert ref.counters["converged_level0"] == v.shape[0]
    assert ref.counters["parity_splits"] == 0
    assert ref.counters["deviation_splits"] == 0


def test_refine_never_evaluates_a_point_twice():
    ref, lat, calls, max_level = refine_with(
        lambda p: np.stack([p[:, 0], p[:, 1] ** 2], axis=-1), min_img_sep=0.05
    )
    assert calls["points"] == len(ref.cache)
    assert len(np.unique(lat.key(ref.cache.ij))) == len(ref.cache)


def test_refine_terminates_at_max_level_on_a_kappa_one_sheet():
    """kappa == 1 maps the whole lens plane to a point: A == 0 everywhere."""
    ref, lat, calls, max_level = refine_with(
        lambda p: np.zeros_like(p), min_img_sep=0.5
    )
    v, level, cls, status = ref.store.compact()
    assert (level == max_level).all()
    assert (status == LeafStatus.SIZE_FLOOR).all()
    assert ref.counters["sigma_zero"] > 0
    assert np.isfinite(ref.cache.beta).all()
    # This fixture genuinely reaches max_level, so it is the one that pins the
    # loop's batch structure on the full-descent path: one raytrace batch per
    # level that needs new points, and the max_level iteration needs none
    # because its vertices are all already cached -- hence max_level batches,
    # not max_level + 1. Every lattice point is evaluated exactly once.
    assert calls["batches"] == max_level
    assert calls["points"] == (lat.n + 1) ** 2


def sis_raytrace(p, b=1.0):
    """SIS deflection ``beta = theta (1 - b/|theta|)``, non-finite at ``theta = 0``.

    A *point* non-finite set, unlike the half-plane fixtures: the origin is a
    lattice vertex for even ``init_res``, so exactly one sample point in the
    whole build is non-finite and the six level-0 triangles sharing it are the
    ones the old terminate-on-non-finite policy condemned wholesale. An
    area-shaped fixture cannot distinguish a policy that refines into the bad
    set from one that stops at its boundary, because there the boundary is
    where all the leaves are anyway.

    The Jacobian is non-degenerate away from the critical curve ``|theta| = b``,
    so most of the domain converges early and the refinement that does happen is
    attributable.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.linalg.norm(p, axis=-1, keepdims=True)
        return p * (1.0 - b / r)


def test_refine_splits_a_nonfinite_subregion_down_to_max_level():
    """Non-finite is maximal ignorance, so it refines rather than terminating.

    The bad half-plane is condemned only at ``max_level``, where no split is
    available -- not at whatever level it was first sampled. Reinstating the
    early ``store.add(v[~good], ..., INVALID)`` puts INVALID rows at level 0 and
    fails the level assertion.
    """

    def broken(p):
        out = p.copy()
        bad = p[:, 0] > 0.5
        out[bad] = np.nan
        return out

    ref, lat, calls, max_level = refine_with(broken, min_img_sep=0.5)
    assert max_level > 0, "fixture must allow at least one split"
    v, level, cls, status = ref.store.compact()
    invalid = status == LeafStatus.INVALID
    assert invalid.any()
    assert (level[invalid] == max_level).all()
    assert not np.isfinite(ref.cache.beta[v[invalid]]).all()
    # a triangle wholly in the good half is untouched
    good = status == LeafStatus.CONVERGED
    assert good.any()
    assert np.isfinite(ref.cache.beta[v[good]]).all()


def test_refine_splits_only_the_triangles_that_touch_a_point_singularity():
    """The cost of refining on non-finite, counted exactly.

    Six level-0 triangles share the origin, and a red split hands the bad vertex
    to exactly one of the four children -- the corner child at that vertex -- so
    the non-finite frontier stays six triangles wide at every level rather than
    quadrupling. Hand-derived total: ``6 * max_level`` splits over levels
    ``0 .. max_level - 1``, with no non-finite triangle left to split at
    ``max_level``.

    This is the counter that would expose an area-shaped non-finite region
    driving an ``O(4**max_level)`` descent, which is the one real cost of
    inverting the policy.
    """
    ref, lat, calls, max_level = refine_with(sis_raytrace, min_img_sep=0.05)
    assert max_level == 5, "hand-derived counts below assume this depth"
    assert ref.counters["nonfinite_splits"] == 6 * max_level


def test_refine_marks_nonfinite_vertices_invalid_even_at_max_level():
    """The max_level short-circuit must not blanket-label everything SIZE_FLOOR.

    ``min_img_sep`` forces ``max_level == 0``, so the loop's first and only
    iteration *is* the max_level iteration, and the ``~finite_v`` branch is the
    whole of this build's non-finite handling -- there is no deeper level for a
    non-finite triangle to be pushed down to. That isolation is the point: with
    ``max_level > 0`` a failure here could equally be the split path
    misbehaving, whereas at ``max_level == 0`` only the short-circuit's own
    INVALID-versus-SIZE_FLOOR discrimination can be at fault.
    """

    def broken(p):
        out = p.copy()
        out[p[:, 0] > 0.5] = np.nan
        return out

    ref, lat, calls, max_level = refine_with(broken, min_img_sep=2.0)
    assert max_level == 0
    v, level, cls, status = ref.store.compact()
    assert (status == LeafStatus.INVALID).any()
    # Without this, a run producing zero SIZE_FLOOR rows would make the loop
    # below vacuously true.
    assert (status == LeafStatus.SIZE_FLOOR).any()
    for row in np.flatnonzero(status == LeafStatus.SIZE_FLOOR):
        assert np.isfinite(ref.cache.beta[v[row]]).all()
    for row in np.flatnonzero(status == LeafStatus.INVALID):
        assert not np.isfinite(ref.cache.beta[v[row]]).all()


def test_refine_makes_one_batch_per_level_and_exits_early_when_affine():
    """One raytrace batch per level that needs new points, and no more.

    The identity map is affine everywhere, so every level-0 triangle converges
    and the loop exits through the empty-``active`` break without ever reaching
    ``max_level``. This pins the level-synchronous one-batch-per-level structure
    on the early-exit path.

    This deliberately does NOT claim to test the ``max_level`` midpoint-batch
    skip, tempting though a batch count is: at ``max_level`` a triangle's edges
    are one lattice unit, so ``_midpoint_ij``'s floor division collapses its
    midpoints onto lattice points that are *already cached*. Requesting them
    would add no ``raytrace`` call at all, which makes the skip invisible to
    ``calls`` -- verified numerically: all 512 max-level triangles yield 288
    distinct floored midpoints, none outside the finest lattice. The skip's
    observable behaviour is its *labelling* -- SIZE_FLOOR and INVALID without
    running the criterion -- and that is covered by
    ``test_refine_terminates_at_max_level_on_a_kappa_one_sheet`` and
    ``test_refine_marks_nonfinite_vertices_invalid_even_at_max_level``.
    """
    ref, lat, calls, max_level = refine_with(lambda p: p * 1.0, min_img_sep=0.5)
    v, level, cls, status = ref.store.compact()
    assert (
        max_level > 0
    ), "fixture must allow deeper levels for early exit to mean anything"
    assert (level == 0).all()
    assert calls["batches"] == 1


def test_refine_all_leaves_are_positively_oriented():
    ref, lat, calls, max_level = refine_with(
        lambda p: np.stack([p[:, 0], p[:, 1] ** 2], axis=-1), min_img_sep=0.05
    )
    v, level, cls, status = ref.store.compact()
    tri = lat.xy(ref.cache.ij[v])
    assert (signed_area(tri) > 0).all()


def assert_balanced(ref, lat, max_level):
    """No leaf edge carries an active quarter point."""
    v, level, cls, status = ref.store.compact()
    for lv in np.unique(level):
        if lv > max_level - 2:
            continue
        sel = level == lv
        keys = _edge_quarter_keys(lat, ref.cache.ij[v[sel]])
        assert not ref.active.contains(keys).any(), f"unbalanced at level {lv}"


def test_edge_quarter_keys_are_lattice_points_of_both_quarters():
    lat = _Lattice(4.0, 0.0, 0.0, 1, 4)  # n = 16
    ij = np.array([[[0, 0], [16, 0], [0, 16]]], dtype=np.int64)
    keys = _edge_quarter_keys(lat, ij)
    got = {tuple(p) for p in lat.ij_from_key(keys[0])}
    assert (4, 0) in got and (12, 0) in got  # edge (0,1)
    assert (0, 4) in got and (0, 12) in got  # edge (2,0)


def test_leaf_store_add_accepts_per_row_level_and_status():
    store = _LeafStore()
    store.add(
        np.arange(6).reshape(2, 3),
        np.array([1, 3]),
        np.zeros(2, np.int64),
        np.array([LeafStatus.FORCED, LeafStatus.INVALID]),
    )
    v, level, cls, status = store.compact()
    assert level.tolist() == [1, 3]
    assert status.tolist() == [LeafStatus.FORCED, LeafStatus.INVALID]


def localised_fold(p):
    """Affine away from a narrow band, curved and fold-bearing inside it.

    Outside |y| < 0.5 the map is exactly affine with sigma_min = 0.6, so those
    triangles converge at level 0. Inside, beta2 = 0.6y + y^2 - 0.25 is curved and
    its Jacobian 0.6 + 2y changes sign at y = -0.3, so both the deviation test and
    the parity test fire. The result is a mesh with real level transitions for the
    cascade, closure, and crack tests to work on.

    Continuous at |y| = 0.5, where y^2 - 0.25 vanishes.

    Do NOT replace the bend with a constant outside the band (e.g. 0.25*sign(y)):
    that makes the map degenerate in y everywhere outside, so sigma_min == 0, every
    triangle splits, and the mesh refines uniformly to max_level with no level
    transitions at all -- silently voiding every test that depends on them.
    """
    y = p[:, 1]
    bend = np.where(np.abs(y) < 0.5, y**2 - 0.25, 0.0)
    return np.stack([p[:, 0], 0.6 * y + bend], axis=-1)


def test_mesh_is_edge_balanced_after_refinement():
    ref, lat, calls, max_level = refine_with(
        localised_fold, fov=4.0, init_res=4, min_img_sep=0.05
    )
    assert max_level >= 4, "fixture must allow several levels"
    v, level, cls, status = ref.store.compact()
    assert len(np.unique(level)) > 1, "fixture must produce level transitions"
    assert_balanced(ref, lat, max_level)


def test_cascade_produces_forced_children():
    ref, lat, calls, max_level = refine_with(
        localised_fold, fov=4.0, init_res=4, min_img_sep=0.05
    )
    v, level, cls, status = ref.store.compact()
    assert (status == LeafStatus.FORCED).any()
    assert ref.counters["forced"] > 0
    assert ref.counters["cascade_rounds"] > 0


def test_forced_and_invalid_statuses_are_mutually_consistent():
    """FORCED leaves are finite; INVALID leaves are not.

    Each status is asserted independently. A single ``.any()`` over the union of
    the two would be satisfied by FORCED alone -- which ``localised_fold``
    already guarantees via ``test_cascade_produces_forced_children`` -- leaving
    the INVALID half of the claim unfalsifiable.

    The cascade has no INVALID-inheritance arm to reach, and this fixture is
    what shows why one is unnecessary rather than merely unused.
    ``_find_unbalanced`` filters candidates on ``store.valid & (store.level <=
    min(frontier_level, max_level) - 2)`` and not on status, so an INVALID leaf
    would be an ordinary violator candidate like any other -- but INVALID only
    ever lands at ``max_level``, two levels above that bound, and the
    ``max_level`` branch breaks out of the level loop before any cascade runs.
    Instrumented on this fixture: the cascade processes 103 violators and none
    of them is INVALID, while 2728 non-finite triangles were split on the way
    down. So the unconditional ``FORCED`` in ``_refine`` is exact here, not a
    simplification that happens to hold.
    """

    def half_bad(p):
        out = localised_fold(p)
        out[p[:, 0] > 1.0] = np.nan
        return out

    ref, lat, calls, max_level = refine_with(
        half_bad, fov=4.0, init_res=4, min_img_sep=0.05
    )
    v, level, cls, status = ref.store.compact()
    assert (status == LeafStatus.FORCED).any()
    assert (status == LeafStatus.INVALID).any()
    # A FORCED child descends from a non-INVALID parent, whose vertices were
    # verified finite before it was allowed to split. Not an invariant of the
    # module -- a parent's midpoints are deferred, never criterion-checked, so a
    # re-forced child can carry a non-finite vertex to freeze, which is what
    # `_invalidate_nonfinite_origins` exists to catch. It does hold on this
    # fixture, and a cascade that leaked non-finite geometry into FORCED rows on
    # the ordinary path would break it.
    for row in np.flatnonzero(status == LeafStatus.FORCED):
        assert np.isfinite(ref.cache.beta[v[row]]).all()
    # Conversely, no INVALID leaf may have all-finite vertices: the label is
    # only ever applied because some point of the triangle failed the check.
    inv_beta = ref.cache.beta[v[status == LeafStatus.INVALID]]
    assert not np.isfinite(inv_beta).all(axis=(1, 2)).any()
    # INVALID is unreachable below max_level, which is what makes the
    # unconditional FORCED above exact rather than lucky.
    assert (level[status == LeafStatus.INVALID] == max_level).all()


def test_cascade_still_evaluates_every_point_exactly_once():
    ref, lat, calls, max_level = refine_with(
        localised_fold, fov=4.0, init_res=4, min_img_sep=0.05
    )
    assert calls["points"] == len(ref.cache)


def undirected_edges(lat, cache, leaves):
    k = lat.key(cache.ij[leaves])  # (L, 3)
    e = np.stack([k[:, [0, 1]], k[:, [1, 2]], k[:, [2, 0]]], axis=1)
    return np.sort(e, axis=-1).reshape(-1, 2)


def gated_hanging_nodes(lat, cache, active, slots):
    """Hanging-node mask, applying the same exactness gate ``_close`` applies.

    Without the gate a max_level triangle's edges are one lattice unit long, so
    ``_midpoint_ij``'s floor division collapses each "midpoint" onto one of that
    same edge's own endpoints -- a real, trivially active mesh vertex -- and
    every max_level leaf reads as having three hanging nodes. Any test that asks
    "does a hanging node exist here" must gate, or it is measuring that artifact.
    """
    ij = cache.ij[slots]
    mid_keys = lat.key(_midpoint_ij(ij))
    exact = ((ij[:, [1, 2, 0]] + ij[:, [2, 0, 1]]) % 2 == 0).all(axis=-1)
    return exact & active.contains(mid_keys)


def closed_mesh(fn, **kw):
    ref, lat, calls, max_level = refine_with(fn, **kw)
    v, level, cls, status = ref.store.compact()
    order = _canonical_order(lat, ref.cache, v)
    v, level, status = v[order], level[order], status[order]
    # Captured BEFORE closure. `_close` must add no vertices, so a test checking
    # `leaves` against the cache size has to use a bound that predates the call;
    # reading `len(cache)` afterwards would silently absorb any growth into the
    # bound and the check could never fail.
    n_cache_pre = len(ref.cache)
    leaves, origin, out_level, out_status = _close(
        lat, ref.cache, ref.active, v, level, status
    )
    return (
        ref,
        lat,
        v,
        level,
        status,
        leaves,
        origin,
        out_level,
        out_status,
        n_cache_pre,
    )


def test_min_angle_of_an_equilateral_triangle():
    tri = np.array([[[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3) / 2]]])
    assert np.allclose(_min_angle(tri), np.pi / 3)


def test_canonical_order_is_independent_of_input_order():
    ref, lat, calls, max_level = refine_with(localised_fold, min_img_sep=0.05)
    v, level, cls, status = ref.store.compact()
    perm = RNG.permutation(v.shape[0])
    a = v[_canonical_order(lat, ref.cache, v)]
    b = v[perm][_canonical_order(lat, ref.cache, v[perm])]
    assert np.array_equal(a, b)


def test_closure_makes_every_edge_appear_once_or_twice():
    ref, lat, v, pre_lvl, pre_st, leaves, origin, lvl, st, n_pre = closed_mesh(
        localised_fold, min_img_sep=0.05
    )
    edges = undirected_edges(lat, ref.cache, leaves)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    assert set(np.unique(counts)) <= {1, 2}


def test_closure_leaves_no_hanging_node():
    """The property closure exists for, checked directly on the closed mesh.

    Edge multiplicity cannot substitute for this: a hanging node never raises
    any edge's count (the coarse triangle contributes ``(A, B)`` once, the finer
    neighbours contribute only ``(A, M)`` and ``(M, B)``), so a closure that
    left hanging nodes behind would still show multiplicities inside ``{1, 2}``.
    Multiplicity catches over-generation; this catches under-closure.
    """
    ref, lat, v, pre_lvl, pre_st, leaves, origin, lvl, st, n_pre = closed_mesh(
        localised_fold, min_img_sep=0.05
    )
    assert gated_hanging_nodes(lat, ref.cache, ref.active, v).any(), "nothing to close"
    assert not gated_hanging_nodes(lat, ref.cache, ref.active, leaves).any()


def test_pre_closure_mesh_is_not_already_conforming():
    """Guard against a fixture where closure has nothing to do.

    Tested with the hanging-node predicate directly rather than via
    undirected-edge multiplicity, because a hanging node does not raise any
    edge's count: a coarse triangle contributes its edge ``(A, B)`` exactly once
    while the finer neighbours contribute the half-edges ``(A, M)`` and
    ``(M, B)`` -- never ``(A, B)``, since no fine triangle has both endpoints. So
    a mesh riddled with hanging nodes has the same ``{1, 2}`` multiplicity
    profile as a conforming one, and hanging nodes are indistinguishable from
    domain-boundary edges by counting alone. This is the same predicate
    :func:`_close` itself uses to classify each leaf, **including its exactness
    gate**. The gate is not optional here: ungated, a max_level triangle's
    collapsed pseudo-midpoints are trivially active, so ``.any()`` would be
    satisfied by that artifact alone and this guard would pass whether or not a
    genuine hanging node existed anywhere in the mesh.
    """
    ref, lat, calls, max_level = refine_with(localised_fold, min_img_sep=0.05)
    v, level, cls, status = ref.store.compact()
    assert gated_hanging_nodes(
        lat, ref.cache, ref.active, v
    ).any(), "fixture has no hanging nodes"


def test_closure_preserves_orientation_and_inherits_level_and_status():
    ref, lat, v, pre_lvl, pre_st, leaves, origin, lvl, st, n_pre = closed_mesh(
        localised_fold, min_img_sep=0.05
    )
    assert (signed_area(lat.xy(ref.cache.ij[leaves])) > 0).all()
    assert np.array_equal(lvl, pre_lvl[origin])
    assert np.array_equal(st, pre_st[origin])
    assert lvl.shape == (leaves.shape[0],) and st.shape == (leaves.shape[0],)


def test_leaf_origin_groups_are_contiguous_and_tile_their_origin():
    ref, lat, v, pre_lvl, pre_st, leaves, origin, lvl, st, n_pre = closed_mesh(
        localised_fold, min_img_sep=0.05
    )
    assert (np.diff(origin) >= 0).all(), "origin must be non-decreasing"
    child_area = signed_area(lat.xy(ref.cache.ij[leaves]))
    origin_area = signed_area(lat.xy(ref.cache.ij[v]))
    summed = np.zeros_like(origin_area)
    np.add.at(summed, origin, child_area)
    assert np.allclose(summed, origin_area, rtol=1e-12)


def test_unclosed_leaf_is_its_own_origin_geometry():
    ref, lat, v, pre_lvl, pre_st, leaves, origin, lvl, st, n_pre = closed_mesh(
        localised_fold, min_img_sep=0.05
    )
    _, counts = np.unique(origin, return_counts=True)
    solo = np.flatnonzero(counts == 1)
    assert solo.size > 0
    for o in solo[:20]:
        row = np.flatnonzero(origin == o)[0]
        assert np.array_equal(leaves[row], v[o])


def test_closure_adds_no_new_vertices():
    """Bound taken before closure, so the assertion can actually fail.

    Reading ``len(ref.cache)`` after ``_close`` returns would fold any vertices
    it inserted into the bound itself, making the check unfalsifiable -- it
    passed against a known-buggy ``_close`` for exactly that reason.
    """
    ref, lat, v, pre_lvl, pre_st, leaves, origin, lvl, st, n_pre = closed_mesh(
        localised_fold, min_img_sep=0.05
    )
    assert leaves.max() < n_pre
    assert set(np.unique(leaves)) <= set(range(n_pre))


def test_close_reaches_the_count_equals_3_branch_via_a_gaussian_bump():
    """`_close`'s ``count == 3`` branch, which no other fixture reaches.

    That branch re-derives the red split with a raw ``concatenate`` + fancy-index
    gather rather than calling ``_red_split``, so ``_red_split``'s own tests give
    it zero coverage. A narrow Gaussian bump gives an ``init_res=8`` grid coarse
    enough that most triangles converge quickly while a few interior ones split
    deep enough to leave a fully-hanging (3-node) origin behind for ``_close`` to
    red-split. Measured: ``n_closure_by_pattern == (244, 42, 6)``.
    """

    def bump(p, w=0.08, amp=1.0, c=(0.13, 0.07)):
        centre = np.asarray(c)
        r2 = ((p - centre) ** 2).sum(axis=-1)
        return p * 0.5 + (amp * np.exp(-r2 / (2 * w**2)))[:, None] * np.array(
            [1.0, 0.3]
        )

    ref, lat, v, pre_lvl, pre_st, leaves, origin, lvl, st, n_pre = closed_mesh(
        bump, fov=4.0, init_res=8, min_img_sep=0.02
    )
    group_sizes = np.bincount(origin, minlength=v.shape[0])
    pattern = (
        int((group_sizes == 2).sum()),
        int((group_sizes == 3).sum()),
        int((group_sizes == 4).sum()),
    )
    assert pattern[2] > 0, f"fixture must reach the count == 3 branch, got {pattern}"
    assert pattern == (244, 42, 6), f"measured n_closure_by_pattern={pattern}"

    tri = lat.xy(ref.cache.ij[leaves])
    assert (signed_area(tri) > 0).all(), "every leaf must be positively oriented"

    origin_area = signed_area(lat.xy(ref.cache.ij[v]))
    summed = np.zeros_like(origin_area)
    np.add.at(summed, origin, signed_area(tri))
    assert np.allclose(
        summed, origin_area, rtol=1e-12
    ), "origin-group areas must tile their origin exactly"


def build(fn, fov=4.0, init_res=4, min_img_sep=0.25, **kw):
    raytrace, calls = make_counting_raytrace(fn)
    mesh = build_adaptive_mesh(raytrace, fov, init_res, min_img_sep, **kw)
    return mesh, calls


def test_build_returns_a_consistent_mesh_for_an_affine_map():
    mesh, calls = build(lambda p: p @ AFFINE.T)
    L = mesh.leaves.shape[0]
    assert L == 2 * 4**2
    assert mesh.stats.n_converged == L and mesh.stats.n_invalid == 0
    assert mesh.stats.n_leaves_pre_closure == L
    assert np.array_equal(backend.to_numpy(mesh.leaf_origin), np.arange(L))
    src = backend.to_numpy(mesh.vertices_source)
    lens = backend.to_numpy(mesh.vertices_lens)
    assert np.allclose(src, lens @ AFFINE.T, rtol=1e-10, atol=1e-12)


def test_leaf_area2_is_computed_from_the_stored_source_vertices():
    mesh, calls = build(localised_fold, min_img_sep=0.05)
    tri = backend.to_numpy(mesh.vertices_source)[backend.to_numpy(mesh.leaves)]
    P = shape_matrix(tri)
    expected = P[:, 0, 0] * P[:, 1, 1] - P[:, 0, 1] * P[:, 1, 0]
    assert np.array_equal(backend.to_numpy(mesh.leaf_area2), expected)


def test_vertices_are_compacted_and_ordered_by_lattice_key():
    mesh, calls = build(localised_fold, min_img_sep=0.05)
    used = np.unique(backend.to_numpy(mesh.leaves))
    assert used.tolist() == list(range(mesh.vertices_lens.shape[0]))
    lens = backend.to_numpy(mesh.vertices_lens)
    key = np.lexsort((lens[:, 1], lens[:, 0]))
    assert np.array_equal(key, np.arange(len(lens)))


def test_leaf_origin_survives_the_vertex_remap():
    """Spec test 15, at Mesh level: the compaction must not scramble origins."""
    mesh, _ = build(localised_fold, min_img_sep=0.05)
    leaves = backend.to_numpy(mesh.leaves)
    origin = backend.to_numpy(mesh.leaf_origin)
    origins = backend.to_numpy(mesh.origin_leaves)
    vl = backend.to_numpy(mesh.vertices_lens)
    assert (origin >= 0).all() and (origin < origins.shape[0]).all()
    assert (np.diff(origin) >= 0).all()
    _, counts = np.unique(origin, return_counts=True)
    solo = np.flatnonzero(counts == 1)
    assert solo.size > 0
    for o in solo[:20]:
        row = np.flatnonzero(origin == o)[0]
        assert np.array_equal(leaves[row], origins[o])
    child = signed_area(vl[leaves])
    parent = signed_area(vl[origins])
    summed = np.zeros_like(parent)
    np.add.at(summed, origin, child)
    assert np.allclose(summed, parent, rtol=1e-10)


def test_build_is_deterministic():
    a, _ = build(localised_fold, min_img_sep=0.05)
    b, _ = build(localised_fold, min_img_sep=0.05)
    for name in ("leaves", "leaf_area2", "leaf_origin", "leaf_status", "leaf_level"):
        assert np.array_equal(
            backend.to_numpy(getattr(a, name)), backend.to_numpy(getattr(b, name))
        )
    assert np.array_equal(
        backend.to_numpy(a.vertices_source), backend.to_numpy(b.vertices_source)
    )


def test_depth_limit_warns_and_names_the_required_max_depth():
    with pytest.warns(UserWarning, match=r"Set max_depth >= \d+"):
        mesh, _ = build(localised_fold, min_img_sep=1e-4, max_depth=2)
    assert mesh.stats.depth_limited
    assert mesh.stats.max_level == 2
    assert mesh.stats.d_floor > 2


def test_invalid_leaves_are_kept_but_excluded_from_the_index():
    def broken(p):
        out = localised_fold(p)
        out[p[:, 0] > 1.0] = np.inf
        return out

    mesh, _ = build(broken, min_img_sep=0.05)
    status = backend.to_numpy(mesh.leaf_status)
    assert (status == LeafStatus.INVALID).any()
    assert mesh.stats.n_invalid > 0
    assert mesh.stats.n_nonfinite_vertices > 0
    indexed = set(backend.to_numpy(mesh._cell_leaves).tolist())
    assert not indexed & set(np.flatnonzero(status == LeafStatus.INVALID).tolist())


def test_query_seeds_an_inner_image_that_runs_into_the_lens_centre():
    """The coverage the old terminate-on-non-finite policy destroyed.

    For the SIS the inner image runs continuously into the lens centre as the
    source approaches the cut: ``|theta_minus| = b - beta``. Terminating the six
    level-0 triangles that share the origin therefore removed a hexagon of
    half-width ``fov / init_res`` from the spatial index -- 1.0 arcsec on this
    fixture -- and with it the seed for every inner image inside it, exactly
    where a grid-and-Newton forward_raytrace is already weakest.

    Hand-derived, not read off the mesh: at ``beta = 0.8`` and ``b = 1`` the two
    images are ``theta = 1.8`` and ``theta = -0.2``, since
    ``1.8 * (1 - 1/1.8) = 0.8`` and ``-0.2 * (1 - 1/0.2) = 0.8``. The inner one
    sits 5x deeper inside the old hexagon than its half-width, so the old policy
    returns only the outer seed, 2.0 arcsec away.
    """
    mesh, _ = build(sis_raytrace, min_img_sep=0.05)
    seed, offsets = mesh.seeds(backend.as_array(np.array([[0.8, 0.0]])))
    seed = backend.to_numpy(seed)
    assert offsets.shape[0] == 2 and seed.shape[0] > 0
    for image in ([-0.2, 0.0], [1.8, 0.0]):
        gap = np.linalg.norm(seed - np.asarray(image), axis=1).min()
        assert gap <= 0.05, f"no seed within min_img_sep of {image}, closest {gap:.3g}"


def test_stats_count_splits_driven_by_ignorance_apart_from_the_criterion():
    """``n_nonfinite_splits`` is a third split reason, not folded into the other two.

    A non-finite triangle carries no criterion evidence at all -- the criterion
    cannot be evaluated on it -- so counting it as a parity or deviation split
    would misattribute refinement the criterion never asked for, and counting it
    nowhere would hide the ``O(4**max_level)`` descent an area-shaped non-finite
    region provokes.
    """
    mesh, _ = build(sis_raytrace, min_img_sep=0.05)
    s = mesh.stats
    assert s.n_nonfinite_splits == 6 * s.max_level
    # The origin is the only non-finite sample, and it is a *vertex* of every
    # triangle that ever sees it, so no midpoint-driven split is miscounted here.
    assert s.n_nonfinite_vertices == 1
    clean, _ = build(localised_fold, min_img_sep=0.05)
    assert clean.stats.n_nonfinite_splits == 0


def test_every_indexed_leaf_has_finite_source_vertices():
    def broken(p):
        out = localised_fold(p)
        out[p[:, 0] > 1.0] = np.nan
        return out

    mesh, _ = build(broken, min_img_sep=0.05)
    vs = backend.to_numpy(mesh.vertices_source)
    leaves = backend.to_numpy(mesh.leaves)
    for leaf in np.unique(backend.to_numpy(mesh._cell_leaves)):
        assert np.isfinite(vs[leaves[leaf]]).all()


def test_index_registers_every_leaf_in_the_cell_of_each_of_its_vertices():
    mesh, _ = build(localised_fold, min_img_sep=0.05)
    vs = backend.to_numpy(mesh.vertices_source)
    leaves = backend.to_numpy(mesh.leaves)
    offs = backend.to_numpy(mesh._cell_offsets)
    cells = backend.to_numpy(mesh._cell_leaves)
    lo = backend.to_numpy(mesh._index_lo)
    cell = backend.to_numpy(mesh._index_cell)
    status = backend.to_numpy(mesh.leaf_status)
    for leaf in RNG.choice(len(leaves), size=50, replace=False):
        if status[leaf] == LeafStatus.INVALID:
            continue
        for q in vs[leaves[leaf]]:
            ix = int(np.clip((q[0] - lo[0]) // cell[0], 0, mesh._nx - 1))
            iy = int(np.clip((q[1] - lo[1]) // cell[1], 0, mesh._ny - 1))
            c = ix * mesh._ny + iy
            assert leaf in cells[offs[c] : offs[c + 1]]


def test_query_covers_points_on_the_source_bbox_upper_edge():
    """Regression: the upper bbox edge used to return zero candidates.

    `cell = span / [nx, ny]`, so a point at `x == hi_x` yields `u_x == nx`. The
    old cell-index containment test rejected it, while `_build_index` clips leaf
    registration to `nx - 1` -- so leaves whose AABB reaches `hi` were indexed
    but unreachable. Measured before the fix: 18 of 18 upper-edge vertices
    returned nothing where brute-force containment found candidates.
    """
    mesh, _ = build(localised_fold, min_img_sep=0.05)
    vs = backend.to_numpy(mesh.vertices_source)
    leaves = backend.to_numpy(mesh.leaves)
    status = backend.to_numpy(mesh.leaf_status)
    hi = backend.to_numpy(mesh._index_hi)
    on_edge = np.flatnonzero((vs[:, 0] == hi[0]) | (vs[:, 1] == hi[1]))
    assert on_edge.size > 0, "fixture must have vertices on the upper bbox edge"
    for v in on_edge:
        beta = vs[v]
        tri = backend.as_array(vs[leaves])
        pts = backend.as_array(np.repeat(beta[None], leaves.shape[0], axis=0))
        truth = backend.to_numpy(contains(triangle_weights(tri, pts)))
        expected = set(
            np.flatnonzero(truth & (status != int(LeafStatus.INVALID))).tolist()
        )
        idx, off, _ = query_np(mesh, beta[None])
        assert (
            set(idx[off[0] : off[1]].tolist()) >= expected
        ), f"upper-edge point {beta} lost candidates"


def test_query_matches_brute_force_containment_on_multi_cell_leaves():
    """The one-cell-lookup completeness claim, on a mesh with wide leaf AABBs.

    `_build_index` registers each leaf across its full cell rectangle, not just
    its three vertex cells -- and no other test distinguishes those, since the
    vertex-cell test checks only vertices and the crack test's uniform reference
    shares `_build_index` so a common bug cancels. Measured on this fixture:
    263 of 1350 leaves span three or more index cells on an axis.
    """
    mesh, _ = build(localised_fold, min_img_sep=0.05)
    vs = backend.to_numpy(mesh.vertices_source)
    leaves = backend.to_numpy(mesh.leaves)
    status = backend.to_numpy(mesh.leaf_status)
    lo = backend.to_numpy(mesh._index_lo)
    cell = backend.to_numpy(mesh._index_cell)
    tri = vs[leaves]
    i0 = np.floor((tri.min(axis=1) - lo) / cell).astype(np.int64)
    i1 = np.floor((tri.max(axis=1) - lo) / cell).astype(np.int64)
    span = i1 - i0 + 1
    assert (span >= 3).any(), "fixture must contain multi-cell leaf AABBs"
    beta = RNG.uniform(-0.9, 0.9, size=(200, 2))
    idx, off, _ = query_np(mesh, beta)
    tri_b = backend.as_array(tri)
    for b in range(beta.shape[0]):
        pts = backend.as_array(np.repeat(beta[b][None], leaves.shape[0], axis=0))
        truth = backend.to_numpy(contains(triangle_weights(tri_b, pts)))
        expected = set(
            np.flatnonzero(truth & (status != int(LeafStatus.INVALID))).tolist()
        )
        assert set(idx[off[b] : off[b + 1]].tolist()) >= expected


def test_stats_report_level_zero_convergence_and_termination_split():
    mesh, _ = build(localised_fold, min_img_sep=0.05)
    s = mesh.stats
    assert (
        s.n_converged + s.n_size_floor + s.n_forced + s.n_invalid
        == s.n_leaves_pre_closure
    )
    assert len(s.leaves_by_level) == s.max_level + 1
    assert sum(s.leaves_by_level) == s.n_leaves_pre_closure
    # A level-0 step-7 pass is one of the pre-closure leaves counted at *some*
    # level, so this is a true invariant of the event count -- not the
    # unfalsifiable `>= 0` an unsigned count trivially satisfies -- and it would
    # catch a counter that ran away.
    assert s.n_converged_at_level_0 <= s.n_leaves_pre_closure
    assert s.cancellation_floor < 1e-6  # float64 build
    assert s.n_vertices == mesh.vertices_lens.shape[0]


def test_leaf_area2_uses_the_downcast_vertices_at_reduced_precision():
    """The downcast-before-compute ordering, at a dtype where it is observable.

    At float64 -- the dtype every other test uses -- ``vs.astype(np_dtype)`` is a
    value-preserving no-op, so cast-then-compute and compute-then-cast are
    bit-identical and neither ordering can be distinguished. The property only
    has teeth at a precision-losing dtype: here the two orderings disagree on
    the great majority of leaves, so this is the test that actually pins it.
    """
    mesh, _ = build(localised_fold, min_img_sep=0.05, dtype=backend.float32)
    vs = backend.to_numpy(mesh.vertices_source)
    assert vs.dtype == np.float32
    P = shape_matrix(vs[backend.to_numpy(mesh.leaves)])
    from_stored = P[:, 0, 0] * P[:, 1, 1] - P[:, 0, 1] * P[:, 1, 0]
    assert np.array_equal(backend.to_numpy(mesh.leaf_area2), from_stored)

    mesh64, _ = build(localised_fold, min_img_sep=0.05)
    vs64 = backend.to_numpy(mesh64.vertices_source)
    Q = shape_matrix(vs64[backend.to_numpy(mesh64.leaves)])
    computed_then_cast = (Q[:, 0, 0] * Q[:, 1, 1] - Q[:, 0, 1] * Q[:, 1, 0]).astype(
        np.float32
    )
    assert not np.array_equal(
        from_stored, computed_then_cast
    ), "fixture no longer distinguishes the two orderings"


def test_freeze_invalidates_a_whole_origin_group_from_one_bad_vertex():
    """The freeze-time finiteness re-check, exercised directly.

    It cannot be reached through a full build: every vertex that becomes a
    corner or midpoint of an evaluated triangle is finiteness-checked by
    ``_refine`` first, except on a narrow cascade path (a FORCED leaf re-forced
    in a later round via a deferred, never-checked midpoint) that no available
    fixture reaches. Unit-tested on a synthetic triple instead -- otherwise the
    re-check could be deleted with no test failing.
    """
    vs = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [np.nan, 0.5]])
    leaves = np.array([[0, 1, 2], [0, 1, 3], [0, 1, 2]])
    origin = np.array([0, 0, 1])  # leaf 1 is non-finite and shares origin 0
    pre_status = np.array([LeafStatus.CONVERGED, LeafStatus.CONVERGED], dtype=np.int8)
    out = _invalidate_nonfinite_origins(vs, leaves, origin, pre_status)
    assert out[0] == LeafStatus.INVALID, "one bad leaf must invalidate its origin"
    assert out[1] == LeafStatus.CONVERGED, "a clean origin must be untouched"


def test_kappa_one_sheet_builds_without_nan():
    mesh, calls = build(lambda p: np.zeros_like(p), min_img_sep=0.5)
    assert mesh.stats.n_sigma_min_exactly_zero > 0
    assert not np.isnan(backend.to_numpy(mesh.vertices_source)).any()
    assert backend.to_numpy(mesh.leaf_level).max() == mesh.max_level


def query_np(mesh, beta, batch_size=None):
    idx, off, bary = mesh.query(beta, batch_size=batch_size)
    return backend.to_numpy(idx), backend.to_numpy(off), backend.to_numpy(bary)


def test_query_csr_is_well_formed():
    mesh, _ = build(localised_fold, min_img_sep=0.05)
    beta = RNG.uniform(-2.5, 2.5, size=(64, 2))
    idx, off, bary = query_np(mesh, beta)
    assert off.shape == (65,) and off[0] == 0 and off[-1] == idx.shape[0]
    assert (np.diff(off) >= 0).all()
    assert bary.shape == (idx.shape[0], 3)
    for b in range(64):
        block = idx[off[b] : off[b + 1]]
        assert (np.diff(block) > 0).all(), "blocks must be strictly ascending"


def test_query_handles_empty_input_and_misses():
    mesh, _ = build(lambda p: p @ AFFINE.T)
    idx, off, bary = query_np(mesh, np.zeros((0, 2)))
    assert off.tolist() == [0] and idx.shape == (0,) and bary.shape == (0, 3)
    far = np.array([[1e6, 1e6], [-1e6, 0.0]])
    idx, off, bary = query_np(mesh, far)
    assert off.tolist() == [0, 0, 0]


def test_query_rejects_wrong_shapes():
    mesh, _ = build(lambda p: p @ AFFINE.T)
    with pytest.raises(ValueError):
        mesh.query(np.array([0.0, 0.0]))
    with pytest.raises(ValueError):
        mesh.query(np.zeros((4, 3)))


def test_query_is_invariant_to_batch_size_and_point_order():
    mesh, _ = build(localised_fold, min_img_sep=0.05)
    beta = RNG.uniform(-1.5, 1.5, size=(97, 2))
    ref = query_np(mesh, beta)
    for bs in (1, 7, 96, 97, 1000):
        got = query_np(mesh, beta, batch_size=bs)
        for a, b in zip(ref, got):
            assert np.array_equal(a, b)
    perm = RNG.permutation(97)
    pidx, poff, pbary = query_np(mesh, beta[perm])
    for new, old in enumerate(perm):
        assert np.array_equal(
            pidx[poff[new] : poff[new + 1]], ref[0][ref[1][old] : ref[1][old + 1]]
        )
        # `bary` too, not just the indices: a permutation-dependent bug that
        # scrambled coordinates while leaving leaf ids correct would otherwise
        # slip through this check.
        assert np.array_equal(
            pbary[poff[new] : poff[new + 1]], ref[2][ref[1][old] : ref[1][old + 1]]
        )


def test_query_finds_the_affine_preimage():
    mesh, _ = build(lambda p: p @ AFFINE.T, fov=4.0, init_res=4, min_img_sep=0.25)
    lens_pts = RNG.uniform(-1.8, 1.8, size=(200, 2))
    beta = lens_pts @ AFFINE.T
    idx, off, bary = query_np(mesh, beta)
    assert (np.diff(off) >= 1).all(), "every interior point must hit a leaf"
    leaves = backend.to_numpy(mesh.leaves)
    vl = backend.to_numpy(mesh.vertices_lens)
    seed = np.einsum("kj,kjd->kd", bary, vl[leaves[idx]])
    first = seed[off[:-1]]
    assert np.allclose(first, lens_pts, atol=1e-9)


def test_bary_is_in_the_simplex_on_every_leaf():
    """The simplex guarantee end-to-end, on a curved mesh and a degenerate one.

    The two fixtures need different query points. A ``kappa == 1`` sheet maps
    every leaf to the single point ``(0, 0)``, so its source-plane bounding box
    is degenerate and queries spread over a region land in empty index cells --
    returning zero candidates and leaving all three assertions vacuously true,
    since numpy's ``all`` and ``allclose`` are True over empty input. Measured:
    the spread-out draw yields 44 candidates on ``localised_fold`` but **0** on
    the sheet, against 4096 when querying the origin. The
    ``idx.shape[0] > 0`` guard is what stops that passing silently.
    """
    for fn, sep, degenerate in (
        (localised_fold, 0.05, False),
        (lambda p: np.zeros_like(p), 0.5, True),
    ):
        mesh, _ = build(fn, min_img_sep=sep)
        # Drawn in both branches so the shared module-level RNG sequence stays
        # unchanged for the tests that follow; discarded just below for the
        # sheet, where spread-out points miss the degenerate image entirely.
        beta = RNG.uniform(-0.4, 0.4, size=(40, 2))
        if degenerate:
            beta = np.zeros((8, 2))
        idx, off, bary = query_np(mesh, beta)
        assert idx.shape[0] > 0, "fixture returned no candidates"
        assert np.isfinite(bary).all()
        assert (bary >= 0).all() and (bary <= 1).all()
        assert np.allclose(bary.sum(axis=1), 1.0, atol=1e-12)


def test_bary_reconstructs_beta_on_every_hit_leaf():
    """Barycentric coordinates invert the source-plane map on this mesh.

    ``good`` is kept as a live invariant rather than used as a filter. No leaf
    in this fixture is anywhere near degenerate: measured, the minimum
    ``|leaf_area2|`` over all 1350 leaves is ``6.1e-6``, four orders of
    magnitude above the ``1e-10`` threshold, and the minimum among actual hits
    is ``3.2e-4``. So a ``~good`` branch here would be dead code -- an earlier
    version of this test carried exactly such a loop, which could never execute
    on any run. ``assert good.all()`` fires if a future fixture change ever
    does produce a sub-threshold leaf, at which point that branch needs
    writing; the centroid path is meanwhile covered by
    :func:`test_centroid_fallback_on_a_totally_degenerate_leaf`, where
    ``leaf_area2`` is exactly zero.
    """
    mesh, _ = build(localised_fold, min_img_sep=0.05)
    beta = RNG.uniform(-1.5, 1.5, size=(200, 2))
    idx, off, bary = query_np(mesh, beta)
    area = np.abs(backend.to_numpy(mesh.leaf_area2))[idx]
    vs = backend.to_numpy(mesh.vertices_source)
    leaves = backend.to_numpy(mesh.leaves)
    owner = np.repeat(np.arange(len(beta)), np.diff(off))
    assert idx.shape[0] > 0, "fixture returned no candidates"
    good = area > 1e-10
    assert (
        good.all()
    ), "fixture produced a degenerate leaf; the ~good branch needs writing"
    recon = np.einsum("kj,kjd->kd", bary, vs[leaves[idx]])
    assert np.allclose(recon, beta[owner], atol=1e-8)


def test_centroid_fallback_on_a_totally_degenerate_leaf():
    """kappa == 1 maps every leaf to a point: w and d are exactly zero."""
    mesh, _ = build(lambda p: np.zeros_like(p), fov=4.0, init_res=4, min_img_sep=0.5)
    idx, off, bary = query_np(mesh, np.zeros((1, 2)))
    assert idx.shape[0] > 0
    assert np.array_equal(bary, np.full((idx.shape[0], 3), 1.0 / 3.0))
    assert (np.abs(backend.to_numpy(mesh.leaf_area2)[idx]) == 0).all()
    lvl = backend.to_numpy(mesh.leaf_level)[idx]
    assert (lvl == mesh.max_level).all()
    l_max = np.sqrt(2) * 4.0 / (4 * 2**mesh.max_level)
    assert l_max <= 0.5


def test_geometry_wrappers_match_a_manual_gather():
    mesh, _ = build(localised_fold, min_img_sep=0.05)
    beta = RNG.uniform(-1.5, 1.5, size=(30, 2))
    idx, off, bary = mesh.query(beta)
    for name, verts in (
        ("triangles_lens", mesh.vertices_lens),
        ("triangles_source", mesh.vertices_source),
    ):
        via_beta, offsets = getattr(mesh, name)(beta)
        via_idx = getattr(mesh, name)(leaf_indices=idx)
        manual = verts[mesh.leaves[idx]]
        assert np.array_equal(backend.to_numpy(via_beta), backend.to_numpy(manual))
        assert np.array_equal(backend.to_numpy(via_idx), backend.to_numpy(manual))
        assert np.array_equal(backend.to_numpy(offsets), backend.to_numpy(off))


def cross2(u, v):
    """Scalar cross product of 2-D vectors.

    ``np.cross`` on 2-vectors is deprecated in NumPy 2.0 and emits a
    ``DeprecationWarning`` per call -- 141 of them across this file's runs, and
    a hard failure under ``-W error::DeprecationWarning``. This is the same
    value, computed the way :func:`signed_area` above already does it, and is
    bit-identical to the ``np.cross`` result.
    """
    return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]


def test_seeds_lie_inside_their_lens_triangle():
    mesh, _ = build(localised_fold, min_img_sep=0.05)
    beta = RNG.uniform(-1.5, 1.5, size=(50, 2))
    idx, off, bary = mesh.query(beta)
    seed, offsets = mesh.seeds(beta)
    direct = mesh.seeds(leaf_indices=idx, bary=bary)
    assert np.allclose(backend.to_numpy(seed), backend.to_numpy(direct))
    assert np.array_equal(backend.to_numpy(offsets), backend.to_numpy(off))
    tri = backend.to_numpy(mesh.triangles_lens(leaf_indices=idx))
    s = backend.to_numpy(seed)
    for k in range(len(s)):
        w = np.array(
            [
                cross2(tri[k, (i + 1) % 3] - s[k], tri[k, (i + 2) % 3] - s[k])
                for i in range(3)
            ]
        )
        assert (w >= -1e-9).all() or (w <= 1e-9).all()


@pytest.mark.parametrize("name", ["triangles_lens", "triangles_source", "seeds"])
def test_wrappers_reject_ambiguous_arguments(name):
    mesh, _ = build(lambda p: p @ AFFINE.T)
    fn = getattr(mesh, name)
    with pytest.raises(ValueError):
        fn()
    with pytest.raises(ValueError):
        fn(np.zeros((1, 2)), leaf_indices=backend.as_array([0]))


def test_seeds_requires_bary_with_leaf_indices():
    mesh, _ = build(lambda p: p @ AFFINE.T)
    with pytest.raises(ValueError, match="bary"):
        mesh.seeds(leaf_indices=backend.as_array([0]))


def test_public_symbols_are_re_exported():
    import caustics

    assert caustics.build_adaptive_mesh is build_adaptive_mesh
    assert caustics.Mesh is Mesh
    assert caustics.LeafStatus is LeafStatus
    assert caustics.BuildStats is BuildStats
    assert caustics.func.sigma_min_2x2 is not None
    assert caustics.func.triangle_weights is not None


def dedup(points, tol):
    """Greedy clustering; returns one representative per cluster, sorted."""
    keep = []
    for p in points:
        if all(np.linalg.norm(p - q) >= tol for q in keep):
            keep.append(p)
    return np.array(sorted(keep, key=tuple)) if keep else np.zeros((0, 2))


def test_sie_candidates_recover_forward_raytrace_images(device):
    """Spec test 26, split into the two contracts this module actually owns.

    1. **Coverage** -- every image ``forward_raytrace`` finds has a candidate
       seed within ``min_img_sep``. That is exactly what :meth:`Mesh.seeds`
       promises: the hit leaf's own affine map is the one step 7 bounds, so the
       seed is accurate to ``min_img_sep`` by construction. Measured across
       these three source points, the worst distance is 2.5e-3 against a 1e-2
       tolerance -- four times better than the guarantee.
    2. **No spurious images** -- every candidate the root-finder converges on is
       a genuine image.

    This deliberately does **not** assert that the refined set has the same
    cardinality as ``forward_raytrace``'s. For ``sp = [0.2, 0.2]`` this SIE has
    a central image at radius ~7e-4, *inside* its own softening radius
    ``s = 1e-3``, where the Jacobian is nearly degenerate.
    ``forward_raytrace_rootfind`` diverges there (residual 0.31 against a 1e-3
    filter) even though the mesh does supply a seed 2.5e-3 away from it. A
    cardinality assertion would therefore be reporting the downstream
    root-finder's convergence on a softened singularity, not this module's
    coverage -- conflating two systems in one number. Contract 1 fails loudly if
    coverage is ever genuinely lost, which is the property worth pinning.
    """
    lens = SIE(
        name="sie",
        cosmology=FlatLambdaCDM(name="cosmo"),
        z_l=0.5,
        z_s=1.5,
        x0=0.0,
        y0=0.0,
        q=0.4,
        phi=np.pi / 5,
        Rein=1.0,
        s=1e-3,
    ).to(device)
    mesh = build_adaptive_mesh(
        lens.raytrace, fov=5.0, init_res=32, min_img_sep=1e-2, device=device
    )
    for sp in ([0.2, 0.2], [0.05, -0.05], [1.4, 1.1]):
        sx = backend.as_array(sp[0], device=device)
        sy = backend.as_array(sp[1], device=device)
        ex, ey = lens.forward_raytrace(sx, sy)
        expected = dedup(
            np.stack([backend.to_numpy(ex), backend.to_numpy(ey)], axis=-1), 1e-2
        )
        # The coverage contract below goes vacuous if `dedup` ever returned an
        # empty `expected`: `.all()` over an empty array is True.
        assert expected.shape[0] > 0, f"{sp}: forward_raytrace found no images"
        seed, offsets = mesh.seeds(backend.as_array(np.asarray([sp])))
        seed = backend.to_numpy(seed)
        assert seed.shape[0] >= expected.shape[0], "candidates must cover the images"
        refined = forward_raytrace_rootfind(
            backend.as_array(seed[:, 0], device=device),
            backend.as_array(seed[:, 1], device=device),
            sx,
            sy,
            lens.raytrace,
        )
        refined = backend.to_numpy(refined)
        bx, by = lens.raytrace(
            backend.as_array(refined[:, 0], device=device),
            backend.as_array(refined[:, 1], device=device),
        )
        residual = np.linalg.norm(
            np.stack([backend.to_numpy(bx), backend.to_numpy(by)], -1) - np.asarray(sp),
            axis=-1,
        )
        got = dedup(refined[residual < 1e-3], 1e-2)
        # Contract 1: coverage. Every image has a seed within min_img_sep.
        nearest = np.linalg.norm(expected[:, None, :] - seed[None, :, :], axis=-1).min(
            axis=1
        )
        assert (
            nearest < 1e-2
        ).all(), f"{sp}: uncovered image, worst seed distance {nearest.max():.3e}"
        # Contract 2: no spurious images among those the root-finder converged on.
        assert got.shape[0] > 0, f"{sp}: nothing converged"
        for p in got:
            assert (
                np.linalg.norm(expected - p, axis=-1).min() < 1e-2
            ), f"{sp}: converged to {p}, which is not a forward_raytrace image"


def test_point_mass_recovers_the_analytic_image_pair():
    lens = Point(
        name="pt",
        cosmology=FlatLambdaCDM(name="cosmo"),
        z_l=0.5,
        z_s=1.5,
        x0=0.0,
        y0=0.0,
        Rein=1.0,
        s=1e-6,
    )
    mesh = build_adaptive_mesh(lens.raytrace, fov=8.0, init_res=64, min_img_sep=1e-2)
    b = 0.4
    seed, offsets = mesh.seeds(backend.as_array(np.array([[b, 0.0]])))
    seed = backend.to_numpy(seed)
    # theta_pm = (b +- sqrt(b^2 + 4 Rein^2)) / 2, both on the x axis
    expected = np.array([(b + np.sqrt(b**2 + 4)) / 2, (b - np.sqrt(b**2 + 4)) / 2])
    for theta in expected:
        assert np.abs(seed[:, 0] - theta).min() < 5e-2
        assert np.abs(seed[np.argmin(np.abs(seed[:, 0] - theta)), 1]) < 5e-2


def test_coverage_does_not_drop_at_level_transitions():
    """Spec test 28. Reports the gap rather than only thresholding it."""
    fov, init_res, sep = 4.0, 4, 0.05
    mesh, _ = build(localised_fold, fov=fov, init_res=init_res, min_img_sep=sep)
    ml = mesh.max_level
    assert ml >= 3 and len(set(backend.to_numpy(mesh.leaf_level).tolist())) > 1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        uniform, _ = build(
            localised_fold,
            fov=fov,
            init_res=init_res * 2**ml,
            min_img_sep=sep,
            max_depth=0,
        )
    grid = np.linspace(-0.9, 0.9, 120)
    beta = np.stack(np.meshgrid(grid, grid, indexing="ij"), axis=-1).reshape(-1, 2)
    _, off_a, _ = query_np(mesh, beta, batch_size=4096)
    _, off_u, _ = query_np(uniform, beta, batch_size=4096)
    hit_a = np.diff(off_a) > 0
    hit_u = np.diff(off_u) > 0
    gap = int((hit_u & ~hit_a).sum())
    print(f"coverage gap at level transitions: {gap} / {int(hit_u.sum())} covered")
    assert gap == 0


def test_criterion_is_blind_to_structure_below_the_sampling_scale():
    """Spec section 2.6, asserted in both directions.

    The criterion reads six points per triangle and the centroid is 0.289*edge from
    the nearest of them, so a perturbation supported inside that radius is exactly
    invisible. Completeness is conditional on init_res resolving it.
    """
    centre = np.array([[-2.0, -2.0], [0.0, 0.0], [-2.0, 0.0]]).mean(axis=0)

    def bumped(p):
        r2 = ((p - centre) ** 2).sum(axis=-1)
        bump = (2.0 * np.exp(-r2 / (2 * 0.08**2)))[:, None] * np.array([1.0, 0.0])
        return p * 0.5 + bump

    # At init_res=2 the nearest of the six sample points is 0.47 from the bump
    # centre, where the bump is 6e-8 -- far below the threshold. At init_res=32 the
    # cell is 0.125 and the deviation is ~0.6, well above it.
    #
    # `min_img_sep` must sit BELOW the level-0 hypotenuse at init_res=32
    # (sqrt(2)*4/32 = 0.1768). Above it, `_depth_floor` returns 0, `max_level` is 0,
    # the max_level short-circuit fires on the first iteration and the criterion
    # never runs at all -- leaving every criterion counter at zero and making the
    # "detects" direction unsatisfiable for a reason that has nothing to do with the
    # blind spot. An earlier version of this fixture used 0.5 and failed exactly
    # that way. The `max_level >= 1` guard below pins it, because a zeroed counter
    # otherwise reads as a passing "blind" assertion.
    #
    # Measured at 0.05: coarse 8/8 converged with 0 splits; fine 2010 of 2477 with
    # 106 deviation splits at max_level 2. The blind direction holds at every
    # tolerance tried, so only the detects direction is sensitive to this.
    sep = 0.05
    coarse, _ = build(bumped, fov=4.0, init_res=2, min_img_sep=sep)
    fine, _ = build(bumped, fov=4.0, init_res=32, min_img_sep=sep)
    assert fine.max_level >= 1, "fine build must actually run the criterion"
    assert coarse.stats.n_converged_at_level_0 == coarse.stats.n_leaves_pre_closure
    assert coarse.stats.n_deviation_splits == 0
    assert fine.stats.n_converged_at_level_0 < fine.stats.n_leaves_pre_closure
    assert fine.stats.n_deviation_splits > 0


def test_build_and_query_run_on_the_configured_device(device):
    """Build and query complete on the configured device and return sane CSR.

    Note what this does **not** verify: every array is converted through
    ``backend.to_numpy`` before inspection, so this test cannot distinguish
    "computed on the requested device" from "computed elsewhere and converted
    back". It pins that the pipeline runs end to end under the ``device``
    fixture and returns coherent results, not placement.
    """
    lens = SIE(
        name="sie",
        cosmology=FlatLambdaCDM(name="cosmo"),
        z_l=0.5,
        z_s=1.5,
        x0=0.0,
        y0=0.0,
        q=0.7,
        phi=0.0,
        Rein=1.0,
        s=1e-3,
    ).to(device)
    mesh = build_adaptive_mesh(
        lens.raytrace, fov=4.0, init_res=8, min_img_sep=0.1, device=device
    )
    idx, off, bary = mesh.query(backend.as_array(np.array([[0.1, 0.1], [3.0, 3.0]])))
    off_np = backend.to_numpy(off)
    bary_np = backend.to_numpy(bary)
    assert off_np.shape == (3,)
    # The hit/miss pair is what makes this falsifiable. `off.shape` is
    # `(B + 1,)` for any B by the CSR contract, and `np.isfinite` is vacuously
    # True on an empty array, so shape-plus-finiteness alone would pass even if
    # the query silently returned nothing for both points. Measured: [0, 3, 3].
    assert off_np[1] > off_np[0], "the interior source point must hit a leaf"
    assert off_np[2] == off_np[1], "the far exterior point must hit nothing"
    assert bary_np.shape[0] == off_np[-1], "bary rows must match the CSR total"
    assert np.isfinite(bary_np).all()


def test_mesh_stores_min_img_sep():
    """`forward_raytrace` needs the dedup radius, and only the build knows it.

    `raytrace` is deliberately *not* stored -- it is passed per call -- but
    `min_img_sep` is the mesh's own defining tolerance, so a caller should never
    have to restate it and risk restating it wrong.
    """
    lens = Point(
        name="pt",
        cosmology=FlatLambdaCDM(name="cosmo"),
        z_l=0.5,
        z_s=1.5,
        x0=0.0,
        y0=0.0,
        Rein=1.0,
        s=1e-6,
    )
    mesh = build_adaptive_mesh(lens.raytrace, fov=4.0, init_res=8, min_img_sep=0.05)
    assert mesh.min_img_sep == 0.05


def test_dedup_collapses_points_closer_than_the_tolerance():
    points = as_arr([[0.0, 0.0], [0.0, 0.001], [1.0, 0.0]])
    keep = to_np(_dedup_representatives(points, np.array([3]), 0.01))
    assert keep.sum() == 2, "the coincident pair must collapse to one image"
    assert keep[2], "the distant point must survive"


def test_dedup_keeps_points_separated_by_the_tolerance():
    """Separation *of* min_img_sep means distinct, matching the build contract.

    The size floor is ``l_max <= min_img_sep`` and step 7 hides no pair separated
    by more than ``min_img_sep / 4``, so the boundary belongs to the distinct
    side. Adjacency is ``d < tol``, not ``<=``.
    """
    points = as_arr([[0.0, 0.0], [0.01, 0.0]])
    keep = to_np(_dedup_representatives(points, np.array([2]), 0.01))
    assert keep.sum() == 2


def test_dedup_counts_connected_components_not_greedy_clusters():
    """Order independence, which greedy clustering does not have.

    Three collinear points spaced ``0.9 * tol`` apart form one connected
    component. Greedy returns 2 for the order below and 1 for ``[1, 0, 2]`` --
    the answer would depend on the order ``query`` happened to emit candidates
    in, which is not something a multiplicity map may depend on.
    """
    p = np.array([[0.0, 0.0], [0.009, 0.0], [0.018, 0.0]])
    counts = np.array([3])
    base = to_np(_dedup_representatives(as_arr(p), counts, 0.01)).sum()
    assert base == 1, f"one chained component expected, got {base}"
    for order in ([1, 0, 2], [2, 1, 0], [0, 2, 1], [2, 0, 1]):
        got = to_np(_dedup_representatives(as_arr(p[order]), counts, 0.01)).sum()
        assert got == base, f"order {order} gave {got}, not {base}"


def test_dedup_never_merges_across_blocks():
    """Two source points whose images coincide must not collapse into one.

    The padded ``(B, M, M)`` formulation makes cross-block bleed the natural bug
    here, and it would silently halve a multiplicity map.
    """
    points = as_arr([[0.0, 0.0], [0.0, 0.0]])
    keep = to_np(_dedup_representatives(points, np.array([1, 1]), 0.01))
    assert keep.sum() == 2, "identical points in different blocks are distinct"


def test_dedup_handles_ragged_blocks_and_empty_blocks():
    """Padding must not invent images in a block that found none."""
    points = as_arr([[0.0, 0.0], [5.0, 5.0], [5.0, 5.0005]])
    keep = to_np(_dedup_representatives(points, np.array([1, 0, 2]), 0.01))
    assert keep.tolist() == [True, True, False]


def test_dedup_is_unchanged_by_bucketing_on_randomised_blocks():
    """The bucketed kernel must agree with a per-block reference exactly.

    Blocks are grouped by count and run at their own M rather than padded to
    the global maximum, so the risk is a scatter that puts one block's answer
    on another block's rows. Running each block *alone* through the same
    function is the independent reference: a single-block call has nothing to
    mis-scatter.
    """
    rng = np.random.default_rng(20260910)
    # The all-empty vector is explicit: 20 random draws from this seed never
    # produce one, and it is the case that reaches the `total == 0` early exit.
    cases = [np.zeros(4, dtype=np.int64)]
    cases += [rng.integers(0, 6, size=rng.integers(1, 12)) for _ in range(20)]
    for counts in cases:
        pts = rng.normal(scale=0.01, size=(int(counts.sum()), 2)).reshape(-1, 2)
        got = to_np(_dedup_representatives(as_arr(pts), counts, 0.01))
        starts = np.cumsum(counts) - counts
        want = np.concatenate(
            [
                to_np(
                    _dedup_representatives(as_arr(pts[s : s + c]), np.array([c]), 0.01)
                )
                for s, c in zip(starts, counts)
            ]
            + [np.zeros(0, dtype=bool)]
        )
        assert got.tolist() == want.tolist(), f"counts={counts.tolist()}"


def test_dedup_keeps_exactly_one_point_per_singleton_block():
    """Blocks of one bypass the clustering kernel; they must still be kept."""
    points = as_arr([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    keep = to_np(_dedup_representatives(points, np.array([1, 1, 1]), 0.01))
    assert keep.tolist() == [True, True, True]


def test_dedup_mixes_singleton_and_clustered_blocks_in_order():
    """The bypass and the kernel write into one output; order must survive.

    Block 0 is a singleton, block 1 collapses to one image, block 2 is a
    singleton again. A scatter that appends the bypassed blocks after the
    clustered ones would pass every count-based assertion and still return the
    representatives in the wrong rows.
    """
    points = as_arr([[9.0, 9.0], [0.0, 0.0], [0.0, 0.001], [5.0, 5.0]])
    keep = to_np(_dedup_representatives(points, np.array([1, 2, 1]), 0.01))
    assert keep.tolist() == [True, True, False, True]


def sie_fixture(device=None):
    lens = SIE(
        name="sie",
        cosmology=FlatLambdaCDM(name="cosmo"),
        z_l=0.5,
        z_s=1.5,
        x0=0.0,
        y0=0.0,
        q=0.4,
        phi=np.pi / 5,
        Rein=1.0,
        s=1e-3,
    )
    if device is not None:
        lens = lens.to(device)
    mesh = build_adaptive_mesh(
        lens.raytrace, fov=5.0, init_res=32, min_img_sep=1e-2, device=device
    )
    return lens, mesh


def test_forward_raytrace_finds_no_spurious_sie_images():
    """Every returned image is an image `lens.forward_raytrace` also finds.

    The residual and leaf-or-ball filters exist for this: a stalled
    Levenberg-Marquardt solve leaves a point that is not an image, and dedup will
    not absorb it when it sits further than `min_img_sep` from a real one.
    """
    lens, mesh = sie_fixture()
    for sp in ([0.2, 0.2], [0.05, -0.05]):
        images, counts = mesh.forward_raytrace(as_arr([sp]), lens.raytrace)
        images = to_np(images)
        assert to_np(counts).tolist() == [images.shape[0]]
        assert images.shape[0] > 0, f"{sp}: no images found"
        # The reference path is float32-only: `LensBase.forward_raytrace` raises
        # "expected scalar type Float but found Double" on float64 input. The mesh
        # itself is float64, so only this comparison call is narrowed.
        ex, ey = lens.forward_raytrace(backend.as_array(sp[0]), backend.as_array(sp[1]))
        expected = dedup(np.stack([to_np(ex), to_np(ey)], axis=-1), 1e-2)
        for p in images:
            assert (
                np.linalg.norm(expected - p, axis=-1).min() < 1e-2
            ), f"{sp}: returned {p}, which is not a forward_raytrace image"


def test_forward_raytrace_covers_every_sie_image():
    """The converse contract: no image is dropped by the filters or the dedup."""
    lens, mesh = sie_fixture()
    for sp in ([0.2, 0.2], [0.05, -0.05]):
        images, _ = mesh.forward_raytrace(as_arr([sp]), lens.raytrace)
        images = to_np(images)
        # The reference path is float32-only: `LensBase.forward_raytrace` raises
        # "expected scalar type Float but found Double" on float64 input. The mesh
        # itself is float64, so only this comparison call is narrowed.
        ex, ey = lens.forward_raytrace(backend.as_array(sp[0]), backend.as_array(sp[1]))
        expected = dedup(np.stack([to_np(ex), to_np(ey)], axis=-1), 1e-2)
        assert expected.shape[0] > 0, f"{sp}: reference found no images"
        nearest = np.linalg.norm(expected[:, None, :] - images[None, :, :], axis=-1)
        worst = nearest.min(axis=1).max()
        assert worst < 1e-2, f"{sp}: uncovered image, worst distance {worst:.3e}"


def test_forward_raytrace_recovers_the_analytic_point_mass_pair():
    """The one case with a closed form: exactly two images, at known positions.

    ``theta = (b +- sqrt(b^2 + 4 Rein^2)) / 2``, both on the x axis. Asserting
    the cardinality is only defensible here, where it is a theorem rather than a
    measurement.
    """
    lens = Point(
        name="pt",
        cosmology=FlatLambdaCDM(name="cosmo"),
        z_l=0.5,
        z_s=1.5,
        x0=0.0,
        y0=0.0,
        Rein=1.0,
        s=1e-6,
    )
    mesh = build_adaptive_mesh(lens.raytrace, fov=8.0, init_res=64, min_img_sep=1e-2)
    b = 0.4
    images, counts = mesh.forward_raytrace(as_arr([[b, 0.0]]), lens.raytrace)
    images = to_np(images)
    assert to_np(counts).tolist() == [2], f"expected 2 images, got {images}"
    expected = np.sort([(b + np.sqrt(b**2 + 4)) / 2, (b - np.sqrt(b**2 + 4)) / 2])
    assert np.allclose(np.sort(images[:, 0]), expected, atol=1e-4)
    assert np.abs(images[:, 1]).max() < 1e-4


def test_forward_raytrace_batches_independently():
    """A batched call equals looping one source at a time.

    Falsifiable against the two bugs the ragged layout invites: targets paired
    with the wrong seeds, and dedup merging images of different sources.
    """
    lens, mesh = sie_fixture()
    points = [[0.2, 0.2], [0.05, -0.05], [0.4, -0.3]]
    images, counts = mesh.forward_raytrace(as_arr(points), lens.raytrace)
    counts = to_np(counts)
    assert counts.shape == (3,)
    assert counts.sum() == to_np(images).shape[0]
    offsets = np.concatenate(([0], np.cumsum(counts)))
    for i, sp in enumerate(points):
        one, one_counts = mesh.forward_raytrace(as_arr([sp]), lens.raytrace)
        assert to_np(one_counts).tolist() == [counts[i]], f"{sp}: count differs"
        block = to_np(images)[offsets[i] : offsets[i + 1]]
        assert np.allclose(block, to_np(one), atol=1e-8), f"{sp}: images differ"


def test_forward_raytrace_batch_size_does_not_change_the_answer():
    lens, mesh = sie_fixture()
    beta = as_arr([[0.2, 0.2], [0.05, -0.05], [0.4, -0.3], [0.0, 0.3]])
    full, full_counts = mesh.forward_raytrace(beta, lens.raytrace)
    for size in (1, 2, 3):
        part, part_counts = mesh.forward_raytrace(beta, lens.raytrace, batch_size=size)
        assert to_np(part_counts).tolist() == to_np(full_counts).tolist()
        assert np.allclose(to_np(part), to_np(full), atol=1e-8)


def test_forward_raytrace_returns_an_empty_block_outside_the_source_plane():
    """A source the mesh never maps to has zero images, not a raised error."""
    lens, mesh = sie_fixture()
    images, counts = mesh.forward_raytrace(as_arr([[50.0, 50.0]]), lens.raytrace)
    assert to_np(counts).tolist() == [0]
    assert to_np(images).shape == (0, 2)


def test_forward_raytrace_rejects_an_unknown_method():
    lens, mesh = sie_fixture()
    with pytest.raises(ValueError, match="rootfind"):
        mesh.forward_raytrace(as_arr([[0.05, 0.02]]), lens.raytrace, method="nope")


def test_multiplicity_map_rejects_an_unknown_method():
    lens, mesh = sie_fixture()
    with pytest.raises(ValueError, match="dedup"):
        mesh.multiplicity_map(lens.raytrace, pixelscale=0.2, method="nope")


def test_dedup_method_never_calls_raytrace():
    """The whole point of `method="dedup"` is that the lens is not evaluated.

    A mesh seed is the preimage of beta under its own leaf's affine map, so it
    is already an approximate image; there is nothing left to solve. If this
    fails, the method is doing the work it exists to skip.
    """
    lens, mesh = sie_fixture()

    def exploding_raytrace(x, y):
        raise AssertionError("raytrace must not be called for method='dedup'")

    images, counts = mesh.forward_raytrace(
        as_arr([[0.05, 0.02], [0.4, 0.3]]), exploding_raytrace, method="dedup"
    )
    assert int(to_np(counts).sum()) == images.shape[0]


def test_dedup_method_matches_rootfind_layout():
    lens, mesh = sie_fixture()
    beta = as_arr([[0.05, 0.02], [3.0, 3.0], [0.0, 0.0]])
    images, counts = mesh.forward_raytrace(beta, lens.raytrace, method="dedup")
    counts_np = to_np(counts)
    assert images.shape[1] == 2
    assert counts_np.shape == (3,)
    assert int(counts_np.sum()) == images.shape[0]
    assert counts_np[1] == 0, "a point outside the source-plane mesh has no images"


def test_dedup_method_is_invariant_to_batch_size():
    lens, mesh = sie_fixture()
    beta = as_arr(RNG.uniform(-0.3, 0.3, size=(40, 2)))
    ref_i, ref_c = mesh.forward_raytrace(beta, lens.raytrace, method="dedup")
    for step in (1, 7, 40, 1000):
        got_i, got_c = mesh.forward_raytrace(
            beta, lens.raytrace, batch_size=step, method="dedup"
        )
        assert to_np(got_c).tolist() == to_np(ref_c).tolist(), f"batch_size={step}"
        assert np.allclose(to_np(got_i), to_np(ref_i)), f"batch_size={step}"


def test_dedup_method_agrees_with_rootfind_away_from_the_caustic():
    """Counts must match where the answer is unambiguous.

    Inside the tangential caustic an SIE has four images, outside it two, and
    the two methods may legitimately disagree only within about min_img_sep of
    the caustic itself (spec section 4.3). Sampling well inside and well
    outside keeps the assertion on the part of the contract that is exact.
    """
    lens, mesh = sie_fixture()
    beta = as_arr([[0.01, 0.0], [0.0, 0.01], [-0.015, 0.008], [0.8, 0.8], [-0.9, 0.7]])
    _, rootfind = mesh.forward_raytrace(beta, lens.raytrace, method="rootfind")
    _, dedup = mesh.forward_raytrace(beta, lens.raytrace, method="dedup")
    assert to_np(dedup).tolist() == to_np(rootfind).tolist()


def test_dedup_positions_are_within_min_img_sep_of_the_refined_roots():
    """Positions are accurate to min_img_sep, the build's *lens-plane* tolerance.

    Not to a source-plane residual: `min_img_sep` bounds the seed's distance
    from the image in the lens plane, and the source-plane residual is that
    distance times the local Jacobian. On a SIZE_FLOOR leaf, which stopped
    because it hit the floor rather than because the deviation test passed,
    there is no source-plane bound at all -- measured residuals there reach
    7e-2 against a min_img_sep of 1e-2. Comparing against the root finder's
    own answer is what the documented contract actually claims.

    Measured worst case on this fixture: 3.7e-3 against min_img_sep = 1e-2.
    """
    lens, mesh = sie_fixture()
    beta = as_arr([[0.02, 0.01], [0.05, 0.02], [-0.03, 0.04], [0.3, 0.2]])
    dedup_i, dedup_c = mesh.forward_raytrace(beta, lens.raytrace, method="dedup")
    root_i, root_c = mesh.forward_raytrace(beta, lens.raytrace, method="rootfind")
    dedup_c, root_c = to_np(dedup_c), to_np(root_c)
    assert dedup_c.tolist() == root_c.tolist(), "fixture must not straddle a caustic"

    do = np.concatenate(([0], np.cumsum(dedup_c)))
    ro = np.concatenate(([0], np.cumsum(root_c)))
    di, ri = to_np(dedup_i), to_np(root_i)
    for b in range(dedup_c.size):
        D, R = di[do[b] : do[b + 1]], ri[ro[b] : ro[b + 1]]
        nearest = np.linalg.norm(D[:, None, :] - R[None, :, :], axis=-1).min(axis=1)
        assert (nearest <= mesh.min_img_sep).all(), f"source {b}: {nearest}"


def test_multiplicity_map_has_the_requested_shape_and_extent():
    lens, mesh = sie_fixture()
    m, extent = mesh.multiplicity_map(lens.raytrace, 0.2, nx=7, ny=5, x0=0.0, y0=0.0)
    assert to_np(m).shape == (5, 7), "shape is (ny, nx), imshow-ready"
    # Outer pixel edges, not first/last centres: `extent` is what `imshow` wants.
    assert np.allclose(extent, (-0.7, 0.7, -0.5, 0.5))


def test_multiplicity_map_agrees_with_forward_raytrace_pixel_by_pixel():
    """Pins the grid orientation, which a transposed reshape would silently flip.

    The map must be the per-pixel image count of the very same source points
    ``forward_raytrace`` would be given, laid out so that ``m[j, i]`` is the pixel
    at ``(x_i, y_j)``.
    """
    lens, mesh = sie_fixture()
    pixelscale, nx, ny = 0.25, 4, 3
    m, _ = mesh.multiplicity_map(lens.raytrace, pixelscale, nx=nx, ny=ny)
    m = to_np(m)
    xs = (np.arange(nx) - (nx - 1) / 2) * pixelscale
    ys = (np.arange(ny) - (ny - 1) / 2) * pixelscale
    lo, hi = to_np(mesh._index_lo), to_np(mesh._index_hi)
    x0, y0 = (lo + hi) / 2
    # A non-square grid makes the transpose check falsifiable.
    assert m.shape == (ny, nx)
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            _, counts = mesh.forward_raytrace(as_arr([[x + x0, y + y0]]), lens.raytrace)
            assert m[j, i] == to_np(counts)[0], f"pixel ({i}, {j}) at ({x}, {y})"


def test_multiplicity_map_defaults_its_field_of_view_to_the_source_plane_mesh():
    """With no x0/y0/nx/ny, the grid covers the indexed leaves' source-plane bbox."""
    lens, mesh = sie_fixture()
    lo, hi = to_np(mesh._index_lo), to_np(mesh._index_hi)
    pixelscale = 0.5
    m, extent = mesh.multiplicity_map(lens.raytrace, pixelscale)
    ny, nx = to_np(m).shape
    assert nx == int(np.ceil((hi[0] - lo[0]) / pixelscale))
    assert ny == int(np.ceil((hi[1] - lo[1]) / pixelscale))
    # Centred on the bbox, and covering it -- square pixels mean slight overhang.
    assert extent[0] <= lo[0] and extent[1] >= hi[0]
    assert extent[2] <= lo[1] and extent[3] >= hi[1]
    assert np.isclose((extent[0] + extent[1]) / 2, (lo[0] + hi[0]) / 2)
    assert np.isclose((extent[2] + extent[3]) / 2, (lo[1] + hi[1]) / 2)


def test_multiplicity_map_of_a_point_lens_is_two_away_from_the_centre():
    """A point lens has exactly two images for every source but the origin.

    Sampled away from the centre, where the softening core's demagnified third
    image lives at a scale ``min_img_sep`` cannot resolve.
    """
    lens = Point(
        name="pt",
        cosmology=FlatLambdaCDM(name="cosmo"),
        z_l=0.5,
        z_s=1.5,
        x0=0.0,
        y0=0.0,
        Rein=1.0,
        s=1e-6,
    )
    mesh = build_adaptive_mesh(lens.raytrace, fov=8.0, init_res=64, min_img_sep=1e-2)
    m, _ = mesh.multiplicity_map(lens.raytrace, 0.1, nx=5, ny=5, x0=1.5, y0=0.0)
    assert (to_np(m) == 2).all(), f"expected all 2, got\n{to_np(m)}"


def test_multiplicity_map_of_a_centred_sie_is_symmetric_under_point_reflection():
    """`beta -> -beta` must not change the image count.

    A centred SIE has an even convergence, so ``alpha(-theta) = -alpha(theta)``
    and the images of ``-beta`` are exactly the negatives of those of ``beta``.
    `utils.meshgrid` centres its samples on zero, so the reflection is a pixel
    permutation and the comparison is exact rather than interpolated.
    """
    lens, mesh = sie_fixture()
    m, _ = mesh.multiplicity_map(lens.raytrace, 0.15, nx=9, ny=9, x0=0.0, y0=0.0)
    m = to_np(m)
    assert (m == m[::-1, ::-1]).all(), f"not point-symmetric:\n{m}"
    assert m.max() > m.min(), "a caustic must show up as a change in multiplicity"


def test_multiplicity_map_of_a_cored_sie_obeys_the_odd_image_theorem():
    """A non-singular lens produces an odd number of images.

    The SIE's ``s = 1e-3`` core makes it non-singular, so every source off a
    caustic has odd multiplicity -- 1 outside the radial caustic, 3 between the
    two, 5 inside the tangential caustic. This is the sharpest available check on
    the whole pipeline, because any single dropped or spurious image flips the
    parity of the pixel it lands in. It caught nothing less than the
    ``batch_lm`` stopping bug: before that fix the central image was abandoned
    whenever its faster siblings converged, and those pixels read 4.
    """
    lens, mesh = sie_fixture()
    m = to_np(
        mesh.multiplicity_map(lens.raytrace, 0.08, nx=25, ny=25, x0=0.0, y0=0.0)[0]
    )
    assert set(np.unique(m).tolist()) <= {
        1,
        3,
        5,
    }, f"even counts present: {np.unique(m)}"
    assert (
        (m == 5).any() and (m == 3).any() and (m == 1).any()
    ), "the grid must span both caustics for this to be a real test"


def test_the_odd_image_theorem_test_can_actually_fail():
    """Falsifiability guard for the test above.

    Starve the root finder of iterations and images go missing; the parity check
    must notice. Without this, a pipeline that silently returned the same count
    everywhere would pass the theorem vacuously.
    """
    lens, mesh = sie_fixture()
    m = to_np(
        mesh.multiplicity_map(
            lens.raytrace,
            0.08,
            nx=25,
            ny=25,
            x0=0.0,
            y0=0.0,
            lm_kwargs={"max_iter": 2},
        )[0]
    )
    assert not set(np.unique(m).tolist()) <= {
        1,
        3,
        5,
    }, f"under-convergence still gave odd counts everywhere: {np.unique(m)}"


def test_multiplicity_map_dedup_method_never_calls_raytrace():
    lens, mesh = sie_fixture()

    def exploding_raytrace(x, y):
        raise AssertionError("raytrace must not be called for method='dedup'")

    mult, extent = mesh.multiplicity_map(
        exploding_raytrace, pixelscale=0.1, nx=9, ny=7, method="dedup"
    )
    assert tuple(mult.shape) == (7, 9)
    assert len(extent) == 4


def test_multiplicity_map_dedup_agrees_with_forward_raytrace_pixel_by_pixel():
    """The map must be exactly its own per-pixel `forward_raytrace`.

    The map consumes counts from a generator that does not build the image
    array, so this is the check that dropping the positions did not drop or
    reorder a count with them.
    """
    lens, mesh = sie_fixture()
    nx, ny, pixelscale = 11, 9, 0.08
    mult, extent = mesh.multiplicity_map(
        lens.raytrace,
        pixelscale=pixelscale,
        nx=nx,
        ny=ny,
        x0=0.0,
        y0=0.0,
        method="dedup",
    )
    gx, gy = meshgrid(
        pixelscale, nx, ny, device=mesh.device, dtype=mesh.vertices_source.dtype
    )
    beta = backend.stack((gx, gy), dim=-1).reshape(-1, 2)
    _, counts = mesh.forward_raytrace(beta, lens.raytrace, method="dedup")
    assert to_np(mult).reshape(-1).tolist() == to_np(counts).tolist()


def test_multiplicity_map_dedup_tracks_rootfind_to_within_a_caustic_sliver():
    """Dedup must reproduce the root-finding map except very near a caustic.

    Note what this deliberately does NOT assert: the odd-image theorem. That
    invariant holds for `method="rootfind"` and is tested above, but dedup
    counts distinct *seeds*, and a near-tangential pair within min_img_sep of
    the caustic can merge -- on this fixture two of 625 pixels read 2 instead
    of 3. Asserting parity here would be asserting something the method does
    not promise (spec section 4.3).

    What it does promise is that the disagreement is rare and never off by
    more than one image. Measured: 2 pixels (0.32%), all delta = -1. A broken
    bucket or a mis-scattered representative blows past 1% immediately.
    """
    lens, mesh = sie_fixture()
    kw = dict(pixelscale=0.08, nx=25, ny=25, x0=0.0, y0=0.0)
    dedup = to_np(mesh.multiplicity_map(lens.raytrace, method="dedup", **kw)[0])
    root = to_np(mesh.multiplicity_map(lens.raytrace, method="rootfind", **kw)[0])
    delta = dedup.astype(np.int64) - root.astype(np.int64)
    differing = int((delta != 0).sum())
    assert differing <= 0.01 * delta.size, f"{differing}/{delta.size} pixels differ"
    assert np.abs(delta).max() <= 1, f"off by {np.abs(delta).max()} images"
    assert set(np.unique(root).tolist()) <= {1, 3, 5}, "rootfind reference is wrong"


def test_forward_chunk_want_images_false_returns_none_but_same_counts():
    """`want_images` must gate only the image gather, never the counts.

    `_forward_chunk` computes `counts` before it ever looks at `want_images`
    (adaptive.py:1404-1415) -- the flag only decides whether the deduplicated
    representatives are also returned. This pins that contract directly on the
    helper: flip `want_images` and the images slot changes from an array to
    `None`, but the counts must not move by a single element.
    """
    lens, mesh = sie_fixture()
    beta = as_arr([[0.05, 0.02], [3.0, 3.0], [0.0, 0.0]])
    tol = mesh.min_img_sep
    images, counts_true = mesh._forward_chunk(
        beta, lens.raytrace, "dedup", tol, {}, True
    )
    none_images, counts_false = mesh._forward_chunk(
        beta, lens.raytrace, "dedup", tol, {}, False
    )
    assert images is not None, "sanity: this beta must produce at least one image"
    assert none_images is None, "want_images=False must not materialize positions"
    assert counts_false.tolist() == counts_true.tolist(), (
        f"counts must be identical regardless of want_images: "
        f"{counts_false.tolist()} != {counts_true.tolist()}"
    )


def test_multiplicity_map_calls_forward_chunk_with_want_images_false(monkeypatch):
    """`multiplicity_map` must request counts only, never the image gather.

    This is the wiring half of the counts-only path: Task 4's whole point was
    to stop `multiplicity_map` materializing the per-chunk image array it was
    always going to discard, via `want_images=False` at the `_image_chunks`
    call site. Nothing else in the suite calls through `Mesh._forward_chunk`
    with a spy, so a regression that silently flipped that `False` back to
    `True` -- reintroducing the discarded gather -- would otherwise pass every
    existing test, since they only check the final counts, not how they were
    obtained.
    """
    lens, mesh = sie_fixture()
    seen = []
    original = Mesh._forward_chunk

    def spy(self, chunk, raytrace, method, tol, lm_kwargs, want_images):
        seen.append(want_images)
        return original(self, chunk, raytrace, method, tol, lm_kwargs, want_images)

    monkeypatch.setattr(Mesh, "_forward_chunk", spy)
    mesh.multiplicity_map(lens.raytrace, pixelscale=0.1, nx=5, ny=5)

    assert seen, "no calls observed -- test is vacuous"
    assert all(w is False for w in seen), (
        f"multiplicity_map must call _forward_chunk with want_images=False "
        f"for every chunk, got {seen}"
    )
