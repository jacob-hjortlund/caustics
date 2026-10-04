"""Triangle maths, the red-split tables, the lattice, and the spatial index."""

import math

import numpy as np
import pytest

from caustics.backend_obj import backend
from caustics.lenses.func.adaptive.mesh_backend import mesh_backend
from caustics.lenses.func.adaptive.criterion import (
    child_shape_matrices,
    converged_from_deviation,
    midpoint_deviation,
    parity_from_children,
)
from caustics.lenses.func.adaptive.geometry import (
    CHILD_VERTEX_INDICES,
    COMPOSE,
    PINV0,
    ROOT_CLASS,
    ROOT_SHAPES,
    area2,
    child_matrix_tables,
    contains,
    csr_offsets,
    is_member,
    min_angle,
    sanitize_bary,
    shape_matrix,
    sigma_min_2x2,
    to_device,
    triangle_weights,
)
from caustics.lenses.func.adaptive.index import as_points, build_index
from caustics.lenses.func.adaptive.lattice import (
    check_lattice_keys,
    depth_floor,
    extend_lattice,
    initial_triangles,
    lattice_fov,
    lattice_h0,
    lattice_ij_from_key,
    lattice_init_res,
    lattice_key,
    lattice_on_boundary,
    lattice_xy,
    make_lattice,
    midpoint_ij,
    ring_triangles,
    warn_depth_limited,
)

from adaptive_maps import f64, i64, to_np

RNG = np.random.default_rng(20260918)


def _arr(x):
    return f64(x)


def _i64(x):
    return i64(x)


def signed_area(tri):
    """Twice the signed area of a (..., 3, 2) numpy triangle."""
    e1 = tri[..., 1, :] - tri[..., 0, :]
    e2 = tri[..., 2, :] - tri[..., 0, :]
    return e1[..., 0] * e2[..., 1] - e1[..., 1] * e2[..., 0]


def np_red_split(tri):
    """Split a (..., 3, 2) numpy triangle into (..., 4, 3, 2) children, canonical order."""
    t1, t2, t3 = tri[..., 0, :], tri[..., 1, :], tri[..., 2, :]
    six = np.stack([t1, t2, t3, (t2 + t3) / 2, (t3 + t1) / 2, (t1 + t2) / 2], axis=-2)
    idx = np.asarray(CHILD_VERTEX_INDICES)
    return six[..., idx, :].reshape(*tri.shape[:-2], 4, 3, 2)


def _six_points(tri):
    t1, t2, t3 = tri[:, 0, :], tri[:, 1, :], tri[:, 2, :]
    mid = np.stack([(t2 + t3) / 2, (t3 + t1) / 2, (t1 + t2) / 2], axis=1)
    return tri, mid


def _grid(n):
    """Every point of an ``(n + 1) x (n + 1)`` lattice, shape ``((n + 1)**2, 2)``."""
    axis = np.arange(n + 1)
    return np.stack(np.meshgrid(axis, axis, indexing="ij"), axis=-1).reshape(-1, 2)


def _same_lattice(a, b):
    return (a.level, a.n, a.scale, a.origin) == (
        b.level,
        b.n,
        b.scale,
        b.origin,
    ) and np.array_equal(to_np(a.lo), to_np(b.lo))


def test_sigma_min_of_zero_matrix_is_exactly_zero():
    A = np.zeros((4, 2, 2))
    got = to_np(sigma_min_2x2(_arr(A)))
    assert (got == 0.0).all()
    assert not np.isnan(got).any()


def test_sigma_min_propagates_nan():
    A = np.zeros((1, 2, 2))
    A[0, 0, 0] = np.nan
    assert np.isnan(to_np(sigma_min_2x2(_arr(A)))).all()


def test_sigma_min_matches_svd():
    A = RNG.normal(size=(2000, 2, 2))
    expected = np.linalg.svd(A, compute_uv=False)[:, -1]
    assert np.allclose(to_np(sigma_min_2x2(_arr(A))), expected, rtol=1e-9, atol=1e-12)


def test_sigma_min_near_and_exactly_singular():
    a = RNG.normal(size=(500, 2))
    perp = np.stack([-a[:, 1], a[:, 0]], axis=1)
    # Rows offset perpendicularly, so det == eps * |a|^2 is genuinely small.
    # Scaling one row instead would give a rank-1 matrix with det exactly zero,
    # which tests the degenerate branch, not the near-degenerate one.
    for eps in (1e-4, 1e-7):
        A = np.stack([a, a + eps * perp], axis=1)
        got = to_np(sigma_min_2x2(_arr(A)))
        expected = np.linalg.svd(A, compute_uv=False)[:, -1]
        assert np.isfinite(got).all()
        assert np.allclose(got, expected, rtol=1e-5, atol=0.0)
    exact = np.stack([a, 2.0 * a], axis=1)  # rank 1: det is exactly 0 in IEEE
    assert (to_np(sigma_min_2x2(_arr(exact))) == 0.0).all()


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
    got = to_np(sigma_min_2x2(_arr(A.astype(np.float64))))
    assert np.isfinite(got).all()


def test_group_tables_close_and_have_order_six():
    M, G, COMPOSE, PINV0, ROOT_CLASS = child_matrix_tables()
    assert M.dtype == mesh_backend.int64
    assert G.dtype == mesh_backend.int64
    assert COMPOSE.dtype == mesh_backend.int64
    assert PINV0.dtype == mesh_backend.float64
    assert ROOT_CLASS.dtype == mesh_backend.int64
    M, G, COMPOSE, ROOT_CLASS = (
        to_np(M),
        to_np(G),
        to_np(COMPOSE),
        to_np(ROOT_CLASS),
    )
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
    R0 = to_np(shape_matrix(_arr(ROOT_SHAPES[0])))
    R1 = to_np(shape_matrix(_arr(ROOT_SHAPES[1])))
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
    kids = signed_area(np_red_split(tri))
    assert np.allclose(kids, parent[:, None] / 4, rtol=0, atol=1e-12)


def test_pinv_table_scales_by_two_to_the_level():
    M, G, COMPOSE, PINV0, ROOT_CLASS = child_matrix_tables()
    COMPOSE, PINV0, ROOT_CLASS = (
        to_np(COMPOSE),
        to_np(PINV0),
        to_np(ROOT_CLASS),
    )
    h0 = 0.05
    for shape_idx in (0, 1):
        root = np.asarray(ROOT_SHAPES[shape_idx], dtype=np.float64) * h0
        cls = ROOT_CLASS[shape_idx]
        tri, level = root, 0
        for _ in range(5):
            kids = np_red_split(tri)
            for k in range(4):
                direct = to_np(mesh_backend.linalg.inv(shape_matrix(_arr(kids[k]))))
                table = (2.0 ** (level + 1) / h0) * PINV0[COMPOSE[cls, k]]
                assert np.allclose(direct, table, rtol=1e-12, atol=1e-12)
            tri, cls, level = kids[0], COMPOSE[cls, 0], level + 1


def test_the_inverse_table_recovers_the_affine_map():
    M, G, COMPOSE, PINV0, ROOT_CLASS = child_matrix_tables()
    COMPOSE, PINV0, ROOT_CLASS = (
        to_np(COMPOSE),
        to_np(PINV0),
        to_np(ROOT_CLASS),
    )
    h0 = 0.05
    tri = np.asarray(ROOT_SHAPES[1], dtype=np.float64) * h0
    cls, level = ROOT_CLASS[1], 0
    kids = np_red_split(tri)
    lens_map = RNG.normal(size=(2, 2))
    for k in range(4):
        q = kids[k] @ lens_map.T
        Q = to_np(shape_matrix(_arr(q)))
        table = Q @ ((2.0 ** (level + 1) / h0) * PINV0[COMPOSE[cls, k]])
        assert np.allclose(table, lens_map, rtol=1e-10, atol=1e-12)


def test_converged_from_deviation_fails_closed():
    r = np.zeros((3, 3))
    assert to_np(converged_from_deviation(_arr(r), _arr([1.0, 1.0, 1.0]), 1.0)).all()
    # NaN must take the split branch, not the converged branch
    assert not to_np(
        converged_from_deviation(_arr(r), _arr(np.full(3, np.nan)), 1.0)
    ).any()
    # s == 0 with r == 0 gives 0 < 0, False, split -- the conservative direction
    assert not to_np(converged_from_deviation(_arr(r), _arr(np.zeros(3)), 1.0)).any()


def test_child_shape_matrices_match_shape_matrix_of_each_child():
    """Q_k must be exactly the edge matrix of child k, bit for bit.

    Both forms subtract the same operands in the same order, so this is an
    equality check and not an allclose: a reassociation that changed rounding
    would change the sign of a near-degenerate det and silently move the parity
    verdict, which is the whole thing this kernel decides.
    """
    rng = np.random.default_rng(0)
    beta_v = rng.normal(size=(5, 3, 2))
    beta_m = rng.normal(size=(5, 3, 2))
    Q = to_np(child_shape_matrices(_arr(beta_v), _arr(beta_m)))
    assert Q.shape == (5, 4, 2, 2)
    six = np.concatenate([beta_v, beta_m], axis=1)
    for k, idx in enumerate(CHILD_VERTEX_INDICES):
        expected = to_np(shape_matrix(_arr(six[:, list(idx)])))
        assert np.array_equal(Q[:, k], expected)


def test_parity_from_children_accepts_a_constant_sign():
    Q = np.tile(np.eye(2), (2, 4, 1, 1))
    assert to_np(parity_from_children(_arr(Q))).tolist() == [
        True,
        True,
    ]


def test_parity_from_children_rejects_a_sign_change():
    Q = np.tile(np.eye(2), (1, 4, 1, 1))
    Q[0, 2] = np.array([[0.0, 1.0], [1.0, 0.0]])  # det -1 among three det +1
    assert to_np(parity_from_children(_arr(Q))).tolist() == [False]


def test_parity_from_children_condemns_a_lone_degenerate_child():
    Q = np.tile(np.eye(2), (1, 4, 1, 1))
    Q[0, 3] = 0.0  # sign 0 differs from +1
    assert to_np(parity_from_children(_arr(Q))).tolist() == [False]


def test_parity_from_children_passes_an_entirely_degenerate_triangle():
    """All four dets exactly zero is a constant sign, so there is no parity
    *change* and the triangle passes. Spec section 4.6: condemning this case
    would be the deviation test in disguise, which is out of scope. A kappa == 1
    sheet is the fixture that reaches it.
    """
    Q = np.zeros((1, 4, 2, 2))
    assert to_np(parity_from_children(_arr(Q))).tolist() == [True]


@pytest.mark.parametrize(
    "determinants",
    [
        [np.nan, np.nan, np.nan, np.nan],
        [np.nan, 0.0, 0.0, 0.0],
    ],
)
def test_parity_from_children_rejects_nan_determinants(determinants):
    Q = np.zeros((1, 4, 2, 2))
    Q[0, :, 0, 0] = determinants
    Q[0, :, 1, 1] = 1.0
    assert to_np(parity_from_children(_arr(Q))).tolist() == [False]


def test_midpoint_deviation_pairs_midpoint_with_opposite_edge():
    v = np.array([[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]])
    m = np.array([[[0.5, 0.5], [0.0, 0.5], [0.5, 0.0]]])  # exact affine images
    assert np.allclose(to_np(midpoint_deviation(_arr(v), _arr(m))), 0.0)
    shifted = m.copy()
    shifted[0, 1] += np.array([0.0, 0.25])
    got = to_np(midpoint_deviation(_arr(v), _arr(shifted)))
    assert np.allclose(got, [[0.0, 0.25, 0.0]])


def test_weights_sum_to_twice_the_signed_area():
    tri = RNG.normal(size=(500, 3, 2))
    beta = RNG.normal(size=(500, 2))
    w = to_np(triangle_weights(_arr(tri), _arr(beta)))
    P = to_np(shape_matrix(_arr(tri)))
    d = P[:, 0, 0] * P[:, 1, 1] - P[:, 0, 1] * P[:, 1, 0]
    assert np.allclose(w.sum(axis=1), d, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("flip", [False, True])
def test_containment_agrees_with_barycentric_truth_on_both_parities(flip):
    tri = np.array([[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]])
    if flip:
        tri = tri[:, ::-1, :]
    pts = RNG.uniform(-0.5, 1.5, size=(4000, 2))
    tiled = np.repeat(tri, len(pts), axis=0)
    hit = to_np(contains(triangle_weights(_arr(tiled), _arr(pts))))
    truth = (pts[:, 0] >= 0) & (pts[:, 1] >= 0) & (pts[:, 0] + pts[:, 1] <= 1)
    assert (hit == truth).all()


def test_shared_edge_weights_are_exactly_negated():
    """Two leaves meeting on an edge must not both reject a point on that edge."""
    verts = np.array([[0.0, 0.0], [1.0, 0.3], [0.4, 1.0], [1.3, 1.2]])
    left = verts[[0, 1, 2]][None]
    right = verts[[1, 3, 2]][None]  # shares the edge (1, 2), opposite traversal
    beta = np.array([[0.62, 0.71]])
    wl = to_np(triangle_weights(_arr(left), _arr(beta)))
    wr = to_np(triangle_weights(_arr(right), _arr(beta)))
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
    bary = to_np(sanitize_bary(_arr(w), _arr(d)))
    assert np.isfinite(bary).all()
    assert (bary >= 0).all() and (bary <= 1).all()
    assert np.allclose(bary.sum(axis=1), 1.0, rtol=0, atol=1e-12)


def test_sanitize_bary_falls_back_to_the_centroid_on_total_degeneracy():
    w = np.zeros((4, 3))
    d = np.zeros(4)
    bary = to_np(sanitize_bary(_arr(w), _arr(d)))
    assert np.array_equal(bary, np.full((4, 3), 1.0 / 3.0))


def test_sanitize_bary_selects_per_row_between_normalized_and_centroid():
    """Degenerate and ordinary rows in one call must be resolved independently.

    Both other sanitize_bary tests are all-or-nothing -- every row ordinary, or
    every row degenerate -- so neither would catch an implementation that decided
    the fallback once for the whole batch instead of per row.
    """
    w = _arr([[3.0, 1.5, 1.5], [0.0, 0.0, 0.0], [2.0, 1.0, 1.0], [0.0, 0.0, 0.0]])
    d = _arr([6.0, 0.0, 4.0, 0.0])
    bary = to_np(sanitize_bary(w, d))
    assert np.allclose(bary[[0, 2]], [[0.5, 0.25, 0.25], [0.5, 0.25, 0.25]])
    assert np.allclose(bary[[1, 3]], 1.0 / 3.0)
    assert np.allclose(bary.sum(axis=1), 1.0, rtol=0, atol=1e-12)


def test_sanitize_bary_recovers_ordinary_coordinates():
    tri = np.array([[[0.0, 0.0], [2.0, 0.0], [0.0, 3.0]]])
    beta = np.array([[0.5, 0.75]])
    w = triangle_weights(_arr(tri), _arr(beta))
    d = _arr([2.0 * 3.0])
    bary = to_np(sanitize_bary(w, d))
    assert np.allclose(bary @ tri[0], beta, rtol=1e-12, atol=1e-14)


def test_depth_floor_matches_the_size_criterion():
    assert depth_floor(5.0 / 100, 10.0) == 0
    d = depth_floor(5.0 / 100, 1e-3)
    l0 = np.sqrt(2) * 5.0 / 100
    assert l0 / 2**d <= 1e-3 < l0 / 2 ** (d - 1)


def test_lattice_key_roundtrip_and_geometry():
    lat = make_lattice(4.0, 0.0, 0.0, 4, 3)
    assert lat.n == 32
    ij_np = np.array([[0, 0], [32, 32], [7, 19]], dtype=np.int64)
    ij = mesh_backend.as_array(ij_np, dtype=mesh_backend.int64)
    key = lattice_key(lat, ij)
    assert to_np(lattice_ij_from_key(lat, key)).tolist() == ij_np.tolist()
    assert np.allclose(to_np(lattice_xy(lat, ij[0])), [-2.0, -2.0])
    assert np.allclose(to_np(lattice_xy(lat, ij[1])), [2.0, 2.0])
    assert to_np(lattice_on_boundary(lat, ij)).tolist() == [
        True,
        True,
        False,
    ]


def test_widening_the_lattice_does_not_move_any_vertex():
    """Bit-identical coordinates, not merely close ones.

    `scale' = fov / (2n)` equals `fl(fov / n) / 2` exactly, because binary
    floating point is scale-invariant under powers of two, and `(2 * ij) *
    scale'` then rounds the same exact real as `ij * scale`. If this ever fails,
    the widened lattice has perturbed the frozen mesh's geometry and every
    downstream bit-exactness argument in the module is void.
    """
    fov, init_res, max_level = 4.0, 4, 3
    narrow = make_lattice(fov, 0.0, 0.0, init_res, max_level)
    wide = make_lattice(fov, 0.0, 0.0, init_res, max_level + 1)
    assert wide.n == 2 * narrow.n
    assert wide.level == max_level + 1

    ij_np = np.stack(
        np.meshgrid(np.arange(narrow.n + 1), np.arange(narrow.n + 1), indexing="ij"),
        axis=-1,
    ).reshape(-1, 2)
    ij = mesh_backend.as_array(ij_np, dtype=mesh_backend.int64)
    assert np.array_equal(
        to_np(lattice_xy(narrow, ij)),
        to_np(lattice_xy(wide, 2 * ij)),
    )


def test_initial_triangles_tile_the_square_and_are_positively_oriented():
    """Ported from `test_adaptive_mesh.py`: geometric properties that the
    oracle-comparison test above does not check -- full square coverage and
    a consistent positive orientation -- rather than element-wise equality
    with the oracle.
    """
    _, _, _, _, root_class = child_matrix_tables()
    init_res, max_level = 4, 2
    ij, cls = initial_triangles(init_res, max_level, root_class)
    assert ij.shape == (2 * init_res**2, 3, 2)
    P = shape_matrix(mesh_backend.to(ij, dtype=mesh_backend.float64))
    area = P[..., 0, 0] * P[..., 1, 1] - P[..., 0, 1] * P[..., 1, 0]
    assert bool(mesh_backend.all(area > 0))
    step = 1 << max_level
    assert np.isclose(float(mesh_backend.sum(area)) / 2, (init_res * step) ** 2)
    assert set(to_np(cls).tolist()) == set(to_np(root_class).tolist())


def test_midpoints_are_exact_integers_and_opposite_their_vertex():
    ij = np.array([[[0, 0], [4, 0], [0, 4]]], dtype=np.int64)
    m = midpoint_ij(_i64(ij))
    assert to_np(m)[0].tolist() == [[2, 2], [0, 2], [2, 0]]


def test_midpoints_are_exact_at_max_level_on_the_widened_lattice():
    """The reason the lattice is one level finer than max_level.

    With the lattice at max_level a triangle's edges are one unit long and
    `midpoint_ij`'s floor division collapses each "midpoint" onto one of that
    edge's own endpoints. One level finer, every edge vector is even at every
    level up to and including max_level, so the midpoints are genuine lattice
    points -- and they are exactly the points with an odd coordinate, which is
    what guarantees they can never collide with a cached vertex.
    """
    _, _, _, _, root_class = child_matrix_tables()
    max_level = 3
    lat = make_lattice(4.0, 0.0, 0.0, 2, max_level + 1)
    ij, cls = initial_triangles(2, lat.level, root_class)
    # Descend to max_level by taking child C_4 (the middle child) each time.
    for _ in range(max_level):
        ij = midpoint_ij(ij)
    assert bool(mesh_backend.all(ij % 2 == 0)), "max_level vertices are even"

    mid = midpoint_ij(ij)
    # Exact: the floor division threw nothing away.
    assert bool(
        mesh_backend.all((ij[:, [1, 2, 0]] + ij[:, [2, 0, 1]]) % 2 == 0)
    ), "edge endpoint sums must be even for the midpoint to be exact"
    # Every midpoint has an odd coordinate, so it is not a vertex of any level.
    assert bool(mesh_backend.all(mesh_backend.any(mid % 2 == 1, dim=-1)))


def test_check_lattice_keys_rejects_a_lattice_too_fine_to_key():
    with pytest.raises(ValueError, match="lattice too fine"):
        check_lattice_keys(2**20, 21, "Raise min_img_sep.")


def test_a_fresh_lattice_places_points_exactly_as_before():
    lat = make_lattice(4.0, 0.3, -0.1, 3, 4)
    ij = _grid(lat.n)
    assert lat.origin == 0
    want = to_np(lat.lo) + ij.astype(np.float64) * lat.scale
    assert np.array_equal(to_np(lattice_xy(lat, i64(ij))), want)


def test_extend_lattice_keeps_every_old_position_bit_for_bit():
    """A non-dyadic center on purpose: nothing may be recomputed from a new fov."""
    lat = make_lattice(4.0, 0.3, -0.1, 3, 4)
    ext = extend_lattice(lat, 2)
    pad = 2 << lat.level
    ij = _grid(lat.n)
    assert np.array_equal(
        to_np(lattice_xy(ext, i64(ij + pad))),
        to_np(lattice_xy(lat, i64(ij))),
    )


def test_check_lattice_keys_rejects_exactly_the_lattices_int64_cannot_key():
    """``45 * 2**26 + 1`` points per axis still key in int64; ``46 * 2**26 + 1`` do not."""
    check_lattice_keys(45, 25, "unused")
    with pytest.raises(ValueError, match="lattice too fine.*Rebuild coarser"):
        check_lattice_keys(46, 25, "Rebuild coarser.")


def test_ring_triangles_are_the_level0_triangles_outside_the_central_block():
    level = 3
    root_class = child_matrix_tables()[4]
    every_ij, every_cls = initial_triangles(6, level, root_class)
    ring_ij, ring_cls = ring_triangles(6, 1, level, root_class)

    def cells(ij):
        return to_np(ij)[:, 0, :] // (1 << level)

    def inner(ij):
        c = cells(ij)
        return ((c >= 1) & (c < 5)).all(axis=1)

    def as_set(ij, cls):
        return {(tuple(t.reshape(-1)), c) for t, c in zip(to_np(ij), to_np(cls))}

    assert ring_ij.shape[0] == 2 * (6 * 6 - 4 * 4)
    assert not inner(ring_ij).any()
    ring = as_set(ring_ij, ring_cls)
    every = as_set(every_ij, every_cls)
    assert ring <= every
    assert len(every - ring) == int(inner(every_ij).sum())


def test_extend_lattice_grows_the_extent_and_keeps_key_order():
    lat = make_lattice(4.0, 0.3, -0.1, 3, 4)
    ext = extend_lattice(lat, 2)
    pad = 2 << lat.level
    assert (ext.level, ext.scale) == (lat.level, lat.scale)
    assert ext.n == lat.n + 2 * pad
    assert ext.origin == pad
    assert np.array_equal(to_np(ext.lo), to_np(lat.lo))
    ij = _grid(lat.n)
    old = to_np(lattice_key(lat, i64(ij)))
    got = to_np(lattice_key(ext, i64(ij + pad)))
    assert np.array_equal(np.argsort(old), np.argsort(got))


def test_an_extended_dyadic_lattice_is_the_fresh_lattice_of_the_larger_fov():
    """The premise of every equivalence test below.

    With a dyadic fov, center and cell size, a fresh lattice over the larger
    fov places every point exactly where the extended one does.
    """
    lat = make_lattice(4.0, 0.5, -0.25, 8, 3)
    ext = extend_lattice(lat, 2)
    fresh = make_lattice(6.0, 0.5, -0.25, 12, 3)
    assert (ext.n, ext.scale) == (fresh.n, fresh.scale)
    ij = i64(_grid(ext.n))
    assert np.array_equal(to_np(lattice_xy(ext, ij)), to_np(lattice_xy(fresh, ij)))


def test_extend_lattice_chains_and_accepts_zero():
    lat = make_lattice(4.0, 0.0, 0.0, 2, 3)
    assert _same_lattice(extend_lattice(lat, 0), lat)
    assert _same_lattice(
        extend_lattice(extend_lattice(lat, 1), 2), extend_lattice(lat, 3)
    )


def test_min_angle_of_an_equilateral_triangle():
    tri = mesh_backend.as_array(
        np.array([[[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3) / 2]]]),
        dtype=mesh_backend.float64,
    )
    assert to_np(min_angle(tri))[0] == pytest.approx(np.pi / 3)


def test_min_angle_of_a_degenerate_triangle_is_zero_not_nan():
    tri = mesh_backend.as_array(
        np.array([[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]]), dtype=mesh_backend.float64
    )
    assert to_np(min_angle(tri))[0] == 0.0


def test_the_table_constants_are_child_matrix_tables():
    _, _, compose, pinv0, root_class = child_matrix_tables()
    assert np.array_equal(to_np(COMPOSE), to_np(compose))
    assert np.array_equal(to_np(PINV0), to_np(pinv0))
    assert np.array_equal(to_np(ROOT_CLASS), to_np(root_class))


def test_area2_is_twice_the_signed_area():
    tri = f64([[[0, 0], [2, 0], [0, 1]], [[0, 0], [0, 1], [2, 0]]])
    assert to_np(area2(tri)).tolist() == [2.0, -2.0]


def test_csr_offsets_start_at_zero_and_accumulate():
    assert to_np(csr_offsets(i64([2, 0, 3]))).tolist() == [0, 2, 2, 5]
    assert to_np(csr_offsets(i64([]))).tolist() == [0]


def test_is_member_reads_an_ascending_array():
    sorted_values = i64([1, 4, 9])
    assert to_np(is_member(sorted_values, i64([0, 1, 5, 9, 10]))).tolist() == [
        False,
        True,
        False,
        True,
        False,
    ]
    assert not to_np(is_member(i64([]), i64([3]))).any()


def test_to_device_maps_every_array_field_of_nested_named_tuples():
    lat = make_lattice(4.0, 0.0, 0.0, 2, 3)
    moved = to_device(lat, mesh_backend.device(lat.lo))
    assert moved.n == lat.n and moved.scale == lat.scale
    assert np.array_equal(to_np(moved.lo), to_np(lat.lo))


def test_the_derived_lattice_sizes_match_the_build_arguments():
    lat = make_lattice(4.0, 0.25, -0.5, 8, 5)
    assert lattice_init_res(lat) == 8
    assert lattice_h0(lat) == 0.5
    assert lattice_fov(lat) == 4.0
    grown = extend_lattice(lat, 2)
    assert lattice_init_res(grown) == 12 and lattice_fov(grown) == 6.0
    assert lattice_h0(grown) == 0.5


def test_keys_round_trip_without_a_stride_field():
    lat = make_lattice(4.0, 0.0, 0.0, 3, 2)
    ij = i64([[0, 0], [lat.n, lat.n], [5, 7]])
    assert np.array_equal(
        to_np(lattice_ij_from_key(lat, lattice_key(lat, ij))), to_np(ij)
    )
    assert to_np(lattice_key(lat, ij)).tolist()[2] == 5 * (lat.n + 1) + 7


def test_depth_floor_reads_the_level0_hypotenuse():
    assert depth_floor(1.0, 2.0) == 0
    assert depth_floor(1.0, math.sqrt(2.0) / 4) == 2


def test_warn_depth_limited_names_the_request_and_the_needed_depth():
    with pytest.warns(
        UserWarning,
        match=r"Lens mesh is depth-limited.*min_img_sep=0.1.*max_depth >= 4",
    ):
        warn_depth_limited("Lens mesh", "min_img_sep=0.1 arcsec", 0.5, 0.05, 2)


def test_warn_depth_limited_is_silent_when_the_floor_is_reached(recwarn):
    warn_depth_limited("Lens mesh", "x", 0.5, 0.05, 4)
    assert not recwarn.list


def test_build_index_of_an_empty_leaf_set_is_a_single_cell():
    idx = build_index(f64(np.zeros((3, 2))), i64(np.zeros((0, 3))), i64(np.zeros(0)))
    assert idx.nx == 1 and idx.ny == 1
    assert to_np(idx.cell_leaves).size == 0


@pytest.mark.skipif(
    backend.backend != "torch", reason="needs torch's meta default device"
)
@pytest.mark.parametrize("n_rows", [0, 60], ids=["empty", "filled"])
def test_build_index_allocates_on_its_inputs_device_not_the_default_one(n_rows):
    """A mesh on another device is indexed at query time (``band_cover``).

    With the default device set to ``meta``, anything ``build_index``
    allocates without its inputs' device lands there and fails to mix.
    """
    import torch

    rng = np.random.default_rng(13)
    vs = f64(rng.normal(size=(50, 2)))
    leaves = i64(rng.integers(0, 50, (60, 3)))
    rows = i64(np.arange(n_rows))
    with torch.device("meta"):
        idx = build_index(vs, leaves, rows)
    for name in ("lo", "hi", "cell", "cell_offsets", "cell_leaves"):
        assert getattr(idx, name).device == vs.device, name


def test_build_index_leaves_are_ascending_within_every_cell():
    rng = np.random.default_rng(13)
    vs = rng.normal(size=(50, 2))
    leaves = rng.integers(0, 50, (60, 3))
    idx = build_index(f64(vs), i64(leaves), i64(np.arange(60)))
    offsets = to_np(idx.cell_offsets)
    cell_leaves = to_np(idx.cell_leaves)
    assert offsets[0] == 0 and offsets[-1] == cell_leaves.size
    exercised = 0
    for a, b in zip(offsets[:-1], offsets[1:]):
        block = cell_leaves[a:b]
        if block.size > 1:
            exercised += 1
            assert (np.diff(block) > 0).all()
    assert exercised > 0, "fixture must exercise a cell containing multiple leaves"


def test_as_points_flattens_any_shape_into_float64_rows():
    got = as_points(2.0, 3.0, None)
    assert got.dtype == mesh_backend.float64 and to_np(got).tolist() == [[2.0, 3.0]]
    got = as_points(f64([[0.0, 1.0]]), f64([[2.0, 3.0]]), None)
    assert to_np(got).tolist() == [[0.0, 2.0], [1.0, 3.0]]
    assert as_points(f64([]), f64([]), None).shape == (0, 2)
