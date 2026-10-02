import math

import numpy as np
import pytest

from caustics.backend_obj import backend
from caustics.lenses.func.adaptive import build_adaptive_mesh
from caustics.lenses.func.adaptive.geometry import shape_matrix, triangle_weights
from caustics.lenses.func.adaptive.magnification import (
    band_cover,
    band_magnification_floor,
    counts_once,
    hit_magnification,
    mesh_total_magnification,
    sheet_edges,
)
from caustics.lenses.func.adaptive.query import mesh_query, mesh_seeds


def to_np(x):
    return backend.to_numpy(x)


def _arr(x):
    return backend.as_array(np.asarray(x, dtype=np.float64), dtype=backend.float64)


def _stack_2x2(a, b, c, d):
    return backend.stack(
        (backend.stack((a, b), dim=-1), backend.stack((c, d), dim=-1)), dim=-2
    )


def _half(x, y):
    """``beta = theta / 2``: every lattice point maps to a dyadic, exactly."""
    return 0.5 * x, 0.5 * y


def _half_jacobian(x, y):
    half, zero = 0.5 + 0.0 * x, 0.0 * x
    return _stack_2x2(half, zero, zero, half)


def _row_fold(x, y):
    """``det A = 1 - 2y``: a fold on ``y = 0.5``, its caustic ``beta_y = 1/4``."""
    return x * 1.0, y - y * y


def _row_fold_jacobian(x, y):
    one, zero = backend.ones_like(x), backend.zeros_like(x)
    return _stack_2x2(one, zero, zero, 1.0 - 2.0 * y)


def _sis(x, y):
    """A singular isothermal sphere of Einstein radius 1 at the origin."""
    r = backend.sqrt(x * x + y * y)
    return x - x / r, y - y / r


def _sis_jacobian(x, y):
    r = backend.sqrt(x * x + y * y)
    k = 1.0 / r**3
    return _stack_2x2(
        1.0 - 1.0 / r + k * x * x, k * x * y, k * x * y, 1.0 - 1.0 / r + k * y * y
    )


def _collapse(x, y):
    """A ``kappa == 1`` sheet: every point maps to the origin."""
    return 0.0 * x, 0.0 * y


def _collapse_jacobian(x, y):
    zero = 0.0 * x
    return _stack_2x2(zero, zero, zero, zero)


def _half_but_nan_jacobian_near_the_origin(x, y):
    """``_half``'s Jacobian, NaN within 0.3 of the origin, where ``_half`` is finite."""
    bad = (x * x + y * y) < 0.09
    return backend.where(bad[:, None, None], backend.nan, _half_jacobian(x, y))


@pytest.fixture(scope="module")
def half_mesh():
    # Affine: all 32 level-0 leaves converge, each mapping to a dyadic triangle.
    return build_adaptive_mesh(_half, _half_jacobian, 4.0, 4, 0.25)


@pytest.fixture(scope="module")
def fold_mesh():
    return build_adaptive_mesh(_row_fold, _row_fold_jacobian, 4.0, 4, 0.25)


def test_every_point_inside_an_affine_sheet_counts_once_with_its_exact_magnification(
    half_mesh,
):
    # Multiples of 1/4 hit leaf-image vertices (multiples of 1/2), edge
    # points and points on the cells' diagonals, all exactly.
    k = np.arange(-3, 4) / 4.0
    beta = _arr(np.stack(np.meshgrid(k, k, indexing="ij"), axis=-1).reshape(-1, 2))
    mu, n = mesh_total_magnification(half_mesh, beta)
    assert to_np(n).tolist() == [1] * beta.shape[0]
    assert to_np(mu).tolist() == [4.0] * beta.shape[0]


def _counts(tri):
    """Whether the point (0, 0.5), on ``tri``'s edge x == 0, counts for ``tri``."""
    tri = _arr([tri])
    w = triangle_weights(tri, _arr([[0.0, 0.5]]))
    P = shape_matrix(tri)
    area2 = P[:, 0, 0] * P[:, 1, 1] - P[:, 0, 1] * P[:, 1, 0]
    return bool(to_np(counts_once(tri, w, area2))[0])


def test_a_shared_edge_counts_for_exactly_one_of_two_neighbours():
    right = [[0.0, 0.0], [1.0, 0.5], [0.0, 1.0]]  # counter-clockwise, x > 0
    left = [[0.0, 0.0], [0.0, 1.0], [-1.0, 0.5]]  # counter-clockwise, x < 0
    assert (_counts(right), _counts(left)) == (True, False)


def test_a_fold_edge_counts_for_both_triangles_or_for_neither():
    # Both on the side the shift (1, eps) points into; the second clockwise.
    assert _counts([[0.0, 0.0], [1.0, 0.5], [0.0, 1.0]])
    assert _counts([[0.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
    # Both on the other side.
    assert not _counts([[0.0, 0.0], [0.0, 1.0], [-1.0, 0.5]])
    assert not _counts([[0.0, 0.0], [-0.5, 0.5], [0.0, 1.0]])


def test_points_in_the_critical_band_images_are_infinitely_magnified(fold_mesh):
    cover = band_cover(fold_mesh)
    assert cover.triangles.shape[0] > 0
    centroid = backend.sum(cover.vertices[cover.triangles], dim=1) / 3.0
    mu, _ = mesh_total_magnification(fold_mesh, centroid)
    assert np.isposinf(to_np(mu)).all()


def test_a_mesh_without_a_critical_band_has_no_band_limit(half_mesh):
    assert band_magnification_floor(half_mesh.critical_band) == math.inf
    assert band_cover(half_mesh).triangles.shape[0] == 0


def test_the_band_limit_is_the_smallest_reciprocal_of_a_leafs_largest_det(fold_mesh):
    band = fold_mesh.critical_band
    det = np.abs(to_np(band.det))[to_np(band.samples)]
    assert band_magnification_floor(band) == float((1.0 / det.max(axis=1)).min())


def test_results_do_not_depend_on_the_batch_size(fold_mesh):
    rng = np.random.default_rng(3)
    beta = _arr(rng.uniform([-1.5, -1.0], [1.5, 0.3], (200, 2)))
    mu, n = mesh_total_magnification(fold_mesh, beta)
    for size in (1, 7, 64):
        mu_b, n_b = mesh_total_magnification(fold_mesh, beta, batch_size=size)
        assert np.array_equal(to_np(mu_b), to_np(mu))
        assert np.array_equal(to_np(n_b), to_np(n))


def test_a_lens_that_is_not_finite_at_a_vertex_still_samples_finite_magnifications():
    # init_res = 4 puts the SIS centre on a lattice vertex, where the map is NaN.
    mesh = build_adaptive_mesh(_sis, _sis_jacobian, 4.0, 4, 0.02)
    beta = _arr([[0.3, 0.0], [0.0, -0.45], [0.2, 0.2]])
    mu, _ = mesh_total_magnification(mesh, beta)
    assert np.isfinite(to_np(mu)).all()
    # mu_tot = 2 / |beta| inside the SIS's Einstein radius. Interpolated det A
    # errs by 0.06-0.6% at these three points (measured 2026-10-02).
    expected = 2.0 / np.linalg.norm(to_np(beta), axis=1)
    np.testing.assert_allclose(to_np(mu), expected, rtol=0.02)


def test_a_float32_mesh_samples_the_float64_mesh_magnifications():
    m64 = build_adaptive_mesh(_row_fold, _row_fold_jacobian, 4.0, 4, 0.25)
    m32 = build_adaptive_mesh(
        _row_fold, _row_fold_jacobian, 4.0, 4, 0.25, dtype=backend.float32
    )
    converged = np.flatnonzero(to_np(m64.leaf_status) == 0)
    centroid = to_np(m64.vertices_source)[to_np(m64.leaves)[converged]].mean(axis=1)
    mu64, n64 = mesh_total_magnification(m64, _arr(centroid))
    mu32, n32 = mesh_total_magnification(m32, _arr(centroid))
    assert np.array_equal(to_np(n32), to_np(n64))
    finite = np.isfinite(to_np(mu64))
    assert np.array_equal(np.isfinite(to_np(mu32)), finite)
    np.testing.assert_allclose(to_np(mu32)[finite], to_np(mu64)[finite], rtol=1e-4)


def test_an_affine_sheet_is_bounded_by_its_fov_edges_alone(half_mesh):
    edges = sheet_edges(half_mesh)
    src, dn = to_np(edges.source), to_np(edges.dn)
    assert to_np(edges.fov).all()
    assert src.shape == (16, 2, 2)
    mid = src.mean(axis=1)
    bottom, top = np.isclose(mid[:, 1], -1.0), np.isclose(mid[:, 1], 1.0)
    left, right = np.isclose(mid[:, 0], -1.0), np.isclose(mid[:, 0], 1.0)
    assert (bottom | top | left | right).all()
    # Vertices are in lattice-key order, x first: a bottom or top edge runs
    # in +x, a left or right one in +y, and dn is +1 with the sheet on the
    # left of that direction.
    assert (dn[bottom] == 1).all() and (dn[top] == -1).all()
    assert (dn[left] == -1).all() and (dn[right] == 1).all()


def test_the_fold_mesh_has_band_edges_hugging_the_caustic(fold_mesh):
    edges = sheet_edges(fold_mesh)
    inner = ~to_np(edges.fov)
    assert inner.any()
    beta_y = to_np(edges.source)[inner][..., 1]
    assert (beta_y <= 0.25 + 1e-12).all() and (beta_y >= 0.25 - 0.1).all()
    assert set(np.abs(to_np(edges.dn)[inner]).tolist()) <= {1, 2}


def _half_but_nan_top_left(x, y):
    nan = backend.where(y - x > 0.5, 0.0 * x + float("nan"), 0.0 * x)
    return 0.5 * x + nan, 0.5 * y + nan


def _half_but_nan_top_left_jacobian(x, y):
    nan = backend.where(y - x > 0.5, 0.0 * x + float("nan"), 0.0 * x)
    half, zero = 0.5 + nan, 0.0 * x
    return _stack_2x2(half, zero, zero, half)


def test_the_corner_diagonal_of_a_one_cell_mesh_is_not_a_fov_edge():
    # One cell, two leaves: the upper-left one has a NaN vertex, so the
    # diagonal it shares with the converged lower-right one is a sheet edge
    # with both endpoints on the lattice boundary -- yet not a fov edge.
    mesh = build_adaptive_mesh(
        _half_but_nan_top_left, _half_but_nan_top_left_jacobian, 2.0, 1, 6.0
    )
    edges = sheet_edges(mesh)
    src, fov = to_np(edges.source), to_np(edges.fov)
    diagonal = np.isclose(src[:, 0, 0], src[:, 0, 1]) & np.isclose(
        src[:, 1, 0], src[:, 1, 1]
    )
    assert diagonal.sum() == 1
    assert not fov[diagonal][0]
    assert fov[~diagonal].all() and (~diagonal).sum() == 2


def test_each_image_reads_det_a_interpolated_at_its_affine_preimage(fold_mesh):
    """``det A = 1 - 2y`` is linear, so interpolating it is exact: each image
    contributes ``1 / |1 - 2 y|`` at the seed `mesh_seeds` gives it."""
    rng = np.random.default_rng(5)
    beta = _arr(rng.uniform([-1.5, -1.5], [1.5, 0.2], (300, 2)))
    mu, _ = mesh_total_magnification(fold_mesh, beta)
    idx, off, bary = mesh_query(fold_mesh, beta)
    y = to_np(mesh_seeds(fold_mesh, idx, bary))[:, 1]
    owner = np.repeat(np.arange(300), np.diff(to_np(off)))
    want = np.bincount(owner, 1.0 / np.abs(1.0 - 2.0 * y), minlength=300)
    mu = to_np(mu)
    finite = np.isfinite(mu)
    assert finite.sum() > 200 and (want[finite] > 0).sum() > 100
    np.testing.assert_allclose(mu[finite], want[finite], rtol=1e-12)


def test_at_a_vertex_image_an_image_reads_that_vertex_s_det_a_exactly(half_mesh):
    """With ``det A`` linear over the lens plane, every point reads it exactly
    to rounding, and a point at a vertex's image -- on several leaves, one of
    which counts it -- reads that vertex's own value, bit for bit."""
    lens = to_np(half_mesh.vertices_lens)
    det = 0.25 + 0.05 * lens[:, 0] + 0.03 * lens[:, 1]
    mesh = half_mesh._replace(vertices_det=_arr(det))
    inner = np.flatnonzero(np.abs(lens).max(axis=1) < 2.0)
    mu, n = mesh_total_magnification(
        mesh, _arr(to_np(half_mesh.vertices_source)[inner])
    )
    assert to_np(n).tolist() == [1] * inner.size
    assert np.array_equal(to_np(mu), 1.0 / det[inner])
    rng = np.random.default_rng(4)
    beta = rng.uniform(-0.9, 0.9, (400, 2))
    mu, _ = mesh_total_magnification(mesh, _arr(beta))
    theta = 2.0 * beta
    np.testing.assert_allclose(
        to_np(mu), 1.0 / (0.25 + 0.05 * theta[:, 0] + 0.03 * theta[:, 1]), rtol=1e-12
    )


def test_a_leaf_whose_vertex_det_a_is_not_of_one_strict_sign_reads_its_area_ratio(
    half_mesh,
):
    """``det A = 0.5`` everywhere interpolates to ``mu = 2``, where the leaves'
    area ratio is 4. One vertex set to ``-0.5``, NaN or 0 sends every leaf
    around it to its area ratio, and no other."""
    rng = np.random.default_rng(9)
    beta = _arr(rng.uniform(-0.9, 0.9, (400, 2)))
    idx, off, _ = mesh_query(half_mesh, beta)
    assert (np.diff(to_np(off)) == 1).all()
    leaves = to_np(half_mesh.leaves)[to_np(idx)]
    v = int(leaves[0, 0])
    touches = (leaves == v).any(axis=1)
    assert touches.any() and not touches.all()
    for bad in (-0.5, np.nan, 0.0):
        det = np.full(half_mesh.vertices_det.shape[0], 0.5)
        det[v] = bad
        mu, n = mesh_total_magnification(
            half_mesh._replace(vertices_det=_arr(det)), beta
        )
        assert to_np(n).tolist() == [1] * 400
        np.testing.assert_allclose(to_np(mu), np.where(touches, 4.0, 2.0), rtol=1e-12)


def test_magnification_is_continuous_across_a_shared_edge_within_a_sheet(fold_mesh):
    """Two converged leaves of one parity share an edge's two vertices, so
    their interpolants agree along it: just either side of its midpoint, mu
    differs by its gradient times the step, where area ratios jumped."""
    leaves = to_np(fold_mesh.leaves)
    side = np.sign(to_np(fold_mesh.leaf_area2))
    owners = {}
    for t in np.flatnonzero(to_np(fold_mesh.leaf_status) == 0):
        for e in range(3):
            edge = tuple(sorted((leaves[t, e], leaves[t, (e + 1) % 3])))
            owners.setdefault(edge, []).append(t)
    shared = [
        e for e, ts in owners.items() if len(ts) == 2 and side[ts[0]] == side[ts[1]]
    ]
    assert len(shared) > 50
    pick = np.random.default_rng(2).choice(len(shared), 50, replace=False)
    a = np.array([shared[k][0] for k in pick])
    b = np.array([shared[k][1] for k in pick])
    src = to_np(fold_mesh.vertices_source)
    mid, d = 0.5 * (src[a] + src[b]), src[b] - src[a]
    normal = np.stack([-d[:, 1], d[:, 0]], axis=1) / np.linalg.norm(d, axis=1)[:, None]
    left = [
        to_np(x) for x in mesh_total_magnification(fold_mesh, _arr(mid + 1e-9 * normal))
    ]
    right = [
        to_np(x) for x in mesh_total_magnification(fold_mesh, _arr(mid - 1e-9 * normal))
    ]
    same = (left[1] == right[1]) & np.isfinite(left[0]) & np.isfinite(right[0])
    assert same.sum() >= 40
    np.testing.assert_allclose(left[0][same], right[0][same], rtol=1e-6, atol=0)


def test_a_jacobian_not_finite_where_the_raytrace_is_leaves_magnifications_finite():
    """``vertices_det`` holds the NaN the lens returned, and no converged
    leaf's image reads it: every point reads 4 or, where its image's leaf
    failed, 0."""
    mesh = build_adaptive_mesh(
        _half, _half_but_nan_jacobian_near_the_origin, 4.0, 4, 0.25
    )
    r2 = (to_np(mesh.vertices_lens) ** 2).sum(axis=1)
    assert np.isnan(to_np(mesh.vertices_det)[r2 < 0.09]).any()
    beta = np.random.default_rng(6).uniform(-0.9, 0.9, (400, 2))
    mu, n = (to_np(x) for x in mesh_total_magnification(mesh, _arr(beta)))
    assert np.isfinite(mu).all()
    assert (mu[n == 0] == 0).all() and (n == 0).any()
    np.testing.assert_allclose(mu[n == 1], 4.0, rtol=1e-12)


def test_a_mesh_with_no_converged_leaf_has_no_magnification_anywhere():
    mesh = build_adaptive_mesh(_collapse, _collapse_jacobian, 4.0, 4, 0.5)
    assert (to_np(mesh.leaf_status) != 0).all()
    assert (to_np(mesh.vertices_det) == 0).all()
    mu, n = mesh_total_magnification(mesh, _arr([[0.0, 0.0], [0.3, -0.2]]))
    assert to_np(mu).tolist() == [0.0, 0.0] and to_np(n).tolist() == [0, 0]


def test_a_hit_reads_the_same_magnification_in_a_call_of_any_size(fold_mesh):
    """Each hit sums its own three terms in a fixed order, so a chunk of hits
    reads exactly what the whole batch does -- which keeps
    `build_magnification_mesh`'s samples equal to `mesh_total_magnification`
    at the same points. A row reduction would not: past about 1366 rows
    (4096 elements) jax changes how it sums a row, and a third of the rows
    then round differently. 5000 hits against chunks of at most 1000."""
    converged = np.flatnonzero(to_np(fold_mesh.leaf_status) == 0)
    rng = np.random.default_rng(8)
    leaves = rng.choice(converged, 5000)
    bary = rng.dirichlet(np.ones(3), 5000)

    def hits(lo, hi):
        idx = backend.as_array(leaves[lo:hi], dtype=backend.int64)
        return to_np(hit_magnification(fold_mesh, idx, _arr(bary[lo:hi])))

    full = hits(0, 5000)
    for lo in range(0, 5000, 1000):
        assert np.array_equal(hits(lo, lo + 1000), full[lo : lo + 1000]), lo
