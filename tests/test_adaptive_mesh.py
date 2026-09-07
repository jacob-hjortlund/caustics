import numpy as np
import pytest

from caustics.backend_obj import backend
from caustics.lenses.adaptive import (
    LeafStatus,
    _ActiveKeys,
    _depth_floor,
    _initial_triangles,
    _Lattice,
    _LeafStore,
    _make_raytrace_np,
    _midpoint_ij,
    _red_split,
    _refine,
    _validate_build_args,
    _VertexCache,
)
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


def test_criterion_is_invariant_to_simultaneous_relabelling():
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


def test_refine_marks_a_nonfinite_subregion_invalid_and_stops():
    def broken(p):
        out = p.copy()
        bad = p[:, 0] > 0.5
        out[bad] = np.nan
        return out

    ref, lat, calls, max_level = refine_with(broken, min_img_sep=0.5)
    v, level, cls, status = ref.store.compact()
    assert (status == LeafStatus.INVALID).any()
    invalid_beta = ref.cache.beta[v[status == LeafStatus.INVALID]]
    assert not np.isfinite(invalid_beta).all()
    # a triangle wholly in the good half is untouched
    good = status == LeafStatus.CONVERGED
    assert good.any()
    assert np.isfinite(ref.cache.beta[v[good]]).all()


def test_refine_marks_nonfinite_vertices_invalid_even_at_max_level():
    """The max_level short-circuit must not blanket-label everything SIZE_FLOOR.

    ``min_img_sep`` forces ``max_level == 0``, so the loop's first and only
    iteration *is* the max_level iteration. That is the only way a triangle can
    reach this branch carrying a non-finite vertex: below ``max_level`` a triangle
    splits only if all six of its points are finite, and a child's vertices are
    drawn from exactly those six, so every triangle at level >= 1 has finite
    vertices by construction. With ``max_level > 0`` the ``~finite_v`` branch adds
    no rows at all and the test cannot fail for the reason it names.
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
