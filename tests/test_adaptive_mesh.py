import numpy as np
import pytest

from caustics.backend_obj import backend
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


def test_sanitize_bary_recovers_ordinary_coordinates():
    tri = np.array([[[0.0, 0.0], [2.0, 0.0], [0.0, 3.0]]])
    beta = np.array([[0.5, 0.75]])
    w = triangle_weights(as_arr(tri), as_arr(beta))
    d = as_arr([2.0 * 3.0])
    bary = to_np(sanitize_bary(w, d))
    assert np.allclose(bary @ tri[0], beta, rtol=1e-12, atol=1e-14)
