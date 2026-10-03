"""Total magnification, the magnification map, and magnified regions."""

import math
import warnings

import numpy as np
import pytest

from caustics.backend_obj import backend
from caustics.lenses.func.adaptive.geometry import (
    _CHILD_VERTEX_INDEX_TABLE,
    area2,
    child_matrix_tables,
    shape_matrix,
    triangle_weights,
)
from caustics.lenses.func.adaptive.lattice import (
    initial_triangles,
    lattice_fov,
    lattice_on_boundary,
    make_lattice,
    midpoint_ij,
)
from caustics.lenses.func.adaptive.lens_mesh import build_lens_mesh
from caustics.lenses.func.adaptive.magnification import (
    band_cover,
    band_magnification_floor,
    counts_once,
    hit_magnification,
    sheet_edges,
    total_magnification,
)
from caustics.lenses.func.adaptive.magnification_map import (
    build_magnification_map,
    level_cells,
    segment_cells,
    split_mask,
    triangle_cells,
)
from caustics.lenses.func.adaptive.query import mesh_query, mesh_seeds
from caustics.lenses.func.adaptive.regions import (
    in_magnified_region,
    magnified_area,
    magnified_regions,
)

from adaptive_maps import f64, stack_2x2, to_np


def _arr(x):
    return f64(x)


_stack_2x2 = stack_2x2


def total_mu(mesh, beta, batch_size=None):
    """`total_magnification` at ``(B, 2)`` points."""
    beta = backend.as_array(beta, dtype=backend.float64)
    return total_magnification(beta[:, 0], beta[:, 1], mesh, batch_size=batch_size)


def in_region(mag, mu_min, beta):
    """`in_magnified_region` at ``(B, 2)`` points."""
    beta = backend.as_array(beta, dtype=backend.float64)
    return in_magnified_region(beta[:, 0], beta[:, 1], mag, mu_min)


def leaf_status(mesh):
    return to_np(mesh.origin_status)[to_np(mesh.leaf_origin)]


def leaf_levels(mag):
    """Each map leaf's level from its lattice area; a closure quarter reads one level finer."""
    ij = to_np(mag.vertices_ij)[to_np(mag.leaves)]
    e1, e2 = ij[:, 1] - ij[:, 0], ij[:, 2] - ij[:, 0]
    a2 = np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]).astype(np.float64)
    return mag.lattice.level - np.ceil(np.log(a2) / np.log(4.0) - 1e-9).astype(np.int64)


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


def _half_but_nan_top_left(x, y):
    nan = backend.where(y - x > 0.5, 0.0 * x + float("nan"), 0.0 * x)
    return 0.5 * x + nan, 0.5 * y + nan


def _half_but_nan_top_left_jacobian(x, y):
    nan = backend.where(y - x > 0.5, 0.0 * x + float("nan"), 0.0 * x)
    half, zero = 0.5 + nan, 0.0 * x
    return _stack_2x2(half, zero, zero, half)


def _cored(x, y):
    """``alpha = 1.2 theta / sqrt(theta**2 + 0.05)``: a point tangential caustic
    at the origin and a radial caustic of radius 0.6637."""
    r = (x * x + y * y + 0.05) ** 0.5
    return x - 1.2 * x / r, y - 1.2 * y / r


def _cored_jacobian(x, y):
    r = (x * x + y * y + 0.05) ** 0.5
    k = 1.2 / r**3
    return _stack_2x2(
        1.0 - 1.2 / r + k * x * x, k * x * y, k * x * y, 1.0 - 1.2 / r + k * y * y
    )


def _cored_alpha(t):
    return 1.2 * t / np.sqrt(t * t + 0.05)


def _cored_dalpha(t):
    return 1.2 * 0.05 / (t * t + 0.05) ** 1.5


def _cored_mu(b):
    """Exact total magnification of ``_cored`` at source radius ``b``, from the
    images on the axis through the source: roots of ``theta - alpha = b``."""
    theta = np.linspace(-3.0, 3.0, 600001)
    f = theta - _cored_alpha(theta) - b
    s = np.flatnonzero(np.sign(f[:-1]) != np.sign(f[1:]))
    roots = theta[s] - f[s] * (theta[s + 1] - theta[s]) / (f[s + 1] - f[s])
    radial = 1.0 - _cored_alpha(roots) / roots
    return np.sum(1.0 / np.abs(radial * (1.0 - _cored_dalpha(roots))))


def _cored_crossings(mu_min):
    """Radii where ``_cored_mu`` crosses ``mu_min`` inside the radial caustic,
    and the caustic's own radius."""
    tr = np.sqrt(0.06 ** (2.0 / 3.0) - 0.05)
    caustic = abs(tr - 1.2 * tr / np.sqrt(tr * tr + 0.05))
    b = np.linspace(1e-3, caustic - 1e-4, 4000)
    mu = np.array([_cored_mu(x) for x in b])
    s = np.flatnonzero(np.sign(mu[:-1] - mu_min) != np.sign(mu[1:] - mu_min))
    return b[s[0]], b[s[1]], caustic


def _curves(regions):
    pts, off = to_np(regions.source), to_np(regions.offsets)
    return [pts[off[c] : off[c + 1]] for c in range(len(off) - 1)], to_np(
        regions.closed
    )


def _point_segment_distance(p, a, b):
    d = b - a
    t = np.clip(
        ((p[:, None] - a[None]) * d[None]).sum(-1) / (d * d).sum(-1)[None], 0, 1
    )
    return np.linalg.norm(p[:, None] - (a[None] + t[..., None] * d[None]), axis=-1)


def _shoelace(curve):
    x, y = curve[:, 0], curve[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)


def _winding(points, curves):
    total = np.zeros(points.shape[0])
    for c in curves:
        a = c[None, :, :] - points[:, None, :]
        b = np.roll(c, -1, axis=0)[None, :, :] - points[:, None, :]
        cross = a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
        total += np.arctan2(cross, (a * b).sum(-1)).sum(axis=1)
    return total / (2.0 * np.pi)


def _brute_cells(a, b, n, size, grow=0.0):
    """Cells of ``[0, n * size]**2`` whose closed square, grown by ``grow``, the
    segment ``a``-``b`` meets: Liang-Barsky clipping, cell by cell."""
    out = set()
    d = b - a
    for i in range(n):
        for j in range(n):
            x0, x1 = i * size - grow, (i + 1) * size + grow
            y0, y1 = j * size - grow, (j + 1) * size + grow
            t0, t1, ok = 0.0, 1.0, True
            for p, q in (
                (-d[0], a[0] - x0),
                (d[0], x1 - a[0]),
                (-d[1], a[1] - y0),
                (d[1], y1 - a[1]),
            ):
                if p == 0:
                    if q < 0:
                        ok = False
                        break
                elif p < 0:
                    t0 = max(t0, q / p)
                else:
                    t1 = min(t1, q / p)
            if ok and t0 <= t1:
                out.add(i * n + j)
    return out


@pytest.fixture(scope="module")
def half_mesh():
    # Affine: all 32 level-0 leaves converge, each mapping to a dyadic triangle.
    return build_lens_mesh(_half, _half_jacobian, 4.0, 4, 0.25)


@pytest.fixture(scope="module")
def fold_mesh():
    return build_lens_mesh(_row_fold, _row_fold_jacobian, 4.0, 4, 0.25)


@pytest.fixture(scope="module")
def fold_lens():
    # Lens plane x in [-1, 1], y in [-0.5, 1.5]: source x in [-1, 1],
    # beta_y in [-0.75, 0.25]; mu_tot >= 4 on the strip 3/16 <= beta_y <= 1/4.
    return build_lens_mesh(_row_fold, _row_fold_jacobian, 2.0, 8, 0.02, y0=0.5)


@pytest.fixture(scope="module")
def fold_mag(fold_lens):
    # A window clear of the fov's image, which runs along x = +-1 and beta_y = -0.75.
    return build_magnification_map(
        fold_lens, 8, 0.02, mu_min=4.0, fov=1.0, x0=0.0, y0=0.0
    )


@pytest.fixture(scope="module")
def wide_mag(fold_lens):
    # The default window takes in the image of the lens fov's boundary.
    return build_magnification_map(fold_lens, 8, 0.05, mu_min=1.5)


@pytest.fixture(scope="module")
def fold_lens_fine():
    return build_lens_mesh(_row_fold, _row_fold_jacobian, 2.0, 8, 0.002, y0=0.5)


@pytest.fixture(scope="module")
def cored_lens():
    return build_lens_mesh(_cored, _cored_jacobian, 4.5, 9, 0.005)


@pytest.fixture(scope="module")
def sis_lens():
    return build_lens_mesh(_sis, _sis_jacobian, 4.0, 8, 0.005, centers=[(0.0, 0.0)])


@pytest.fixture(scope="module")
def sis_mag(sis_lens):
    return build_magnification_map(
        sis_lens, 8, 0.005, mu_min=4.0, fov=1.5, x0=0.0, y0=0.0
    )


@pytest.fixture(scope="module")
def sis_lens_coarse():
    return build_lens_mesh(_sis, _sis_jacobian, 4.0, 8, 0.01, centers=[(0.0, 0.0)])


@pytest.fixture(scope="module")
def sis_mag_coarse(sis_lens_coarse):
    # mu_tot >= 4 on the disk |beta| <= 0.5; the window stays inside the
    # image of the lens fov's boundary.
    return build_magnification_map(
        sis_lens_coarse, 8, 0.02, mu_min=4.0, fov=1.5, x0=0.0, y0=0.0
    )


def test_every_point_inside_an_affine_sheet_counts_once_with_its_exact_magnification(
    half_mesh,
):
    # Multiples of 1/4 hit leaf-image vertices (multiples of 1/2), edge
    # points and points on the cells' diagonals, all exactly.
    k = np.arange(-3, 4) / 4.0
    beta = _arr(np.stack(np.meshgrid(k, k, indexing="ij"), axis=-1).reshape(-1, 2))
    mu, n = total_mu(half_mesh, beta)
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
    triangles, _ = band_cover(fold_mesh)
    assert triangles.shape[0] > 0
    source = fold_mesh.critical_band.source
    centroid = backend.sum(source[triangles], dim=1) / 3.0
    mu, _ = total_mu(fold_mesh, centroid)
    assert np.isposinf(to_np(mu)).all()


def test_a_mesh_without_a_critical_band_has_no_band_limit(half_mesh):
    assert band_magnification_floor(half_mesh.critical_band) == math.inf
    assert band_cover(half_mesh)[0].shape[0] == 0


def test_the_band_limit_is_the_smallest_reciprocal_of_a_leafs_largest_det(fold_mesh):
    band = fold_mesh.critical_band
    det = np.abs(to_np(band.det))[to_np(band.samples)]
    assert band_magnification_floor(band) == float((1.0 / det.max(axis=1)).min())


def test_results_do_not_depend_on_the_batch_size(fold_mesh):
    rng = np.random.default_rng(3)
    beta = _arr(rng.uniform([-1.5, -1.0], [1.5, 0.3], (200, 2)))
    mu, n = total_mu(fold_mesh, beta)
    for size in (1, 7, 64):
        mu_b, n_b = total_mu(fold_mesh, beta, batch_size=size)
        assert np.array_equal(to_np(mu_b), to_np(mu))
        assert np.array_equal(to_np(n_b), to_np(n))


def test_a_lens_that_is_not_finite_at_a_vertex_still_samples_finite_magnifications():
    # init_res = 4 puts the SIS center on a lattice vertex, where the map is NaN.
    mesh = build_lens_mesh(_sis, _sis_jacobian, 4.0, 4, 0.02)
    beta = _arr([[0.3, 0.0], [0.0, -0.45], [0.2, 0.2]])
    mu, _ = total_mu(mesh, beta)
    assert np.isfinite(to_np(mu)).all()
    # mu_tot = 2 / |beta| inside the SIS's Einstein radius. Interpolated det A
    # errs by 0.06-0.6% at these three points (measured 2026-10-02).
    expected = 2.0 / np.linalg.norm(to_np(beta), axis=1)
    np.testing.assert_allclose(to_np(mu), expected, rtol=0.02)


def test_an_affine_sheet_is_bounded_by_its_fov_edges_alone(half_mesh):
    segments, on_fov = sheet_edges(half_mesh)
    src = to_np(segments)
    assert to_np(on_fov).all()
    assert src.shape == (16, 2, 2)
    mid = src.mean(axis=1)
    bottom, top = np.isclose(mid[:, 1], -1.0), np.isclose(mid[:, 1], 1.0)
    left, right = np.isclose(mid[:, 0], -1.0), np.isclose(mid[:, 0], 1.0)
    assert (bottom | top | left | right).all()


def test_the_fold_mesh_has_band_edges_hugging_the_caustic(fold_mesh):
    segments, on_fov = sheet_edges(fold_mesh)
    inner = ~to_np(on_fov)
    assert inner.any()
    beta_y = to_np(segments)[inner][..., 1]
    assert (beta_y <= 0.25 + 1e-12).all() and (beta_y >= 0.25 - 0.1).all()


def test_the_corner_diagonal_of_a_one_cell_mesh_is_not_a_fov_edge():
    # One cell, two leaves: the upper-left one has a NaN vertex, so the
    # diagonal it shares with the converged lower-right one is a sheet edge
    # with both endpoints on the lattice boundary -- yet not a fov edge.
    mesh = build_lens_mesh(
        _half_but_nan_top_left, _half_but_nan_top_left_jacobian, 2.0, 1, 6.0
    )
    segments, on_fov = sheet_edges(mesh)
    src, fov = to_np(segments), to_np(on_fov)
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
    mu, _ = total_mu(fold_mesh, beta)
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
    mu, n = total_mu(mesh, _arr(to_np(half_mesh.vertices_source)[inner]))
    assert to_np(n).tolist() == [1] * inner.size
    assert np.array_equal(to_np(mu), 1.0 / det[inner])
    rng = np.random.default_rng(4)
    beta = rng.uniform(-0.9, 0.9, (400, 2))
    mu, _ = total_mu(mesh, _arr(beta))
    theta = 2.0 * beta
    np.testing.assert_allclose(
        to_np(mu), 1.0 / (0.25 + 0.05 * theta[:, 0] + 0.03 * theta[:, 1]), rtol=1e-12
    )


def test_magnification_is_continuous_across_a_shared_edge_within_a_sheet(fold_mesh):
    """Two converged leaves of one parity share an edge's two vertices, so
    their interpolants agree along it: just either side of its midpoint, mu
    differs by its gradient times the step."""
    leaves = to_np(fold_mesh.leaves)
    side = np.sign(to_np(area2(fold_mesh.vertices_source[fold_mesh.leaves])))
    owners = {}
    for t in np.flatnonzero(leaf_status(fold_mesh) == 0):
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
    left = [to_np(x) for x in total_mu(fold_mesh, _arr(mid + 1e-9 * normal))]
    right = [to_np(x) for x in total_mu(fold_mesh, _arr(mid - 1e-9 * normal))]
    same = (left[1] == right[1]) & np.isfinite(left[0]) & np.isfinite(right[0])
    assert same.sum() >= 40
    np.testing.assert_allclose(left[0][same], right[0][same], rtol=1e-6, atol=0)


def test_a_jacobian_not_finite_where_the_raytrace_is_leaves_magnifications_finite():
    """``vertices_det`` holds the NaN the lens returned, and no converged
    leaf's image reads it: every point reads 4 or, where its image's leaf
    failed, 0."""
    mesh = build_lens_mesh(_half, _half_but_nan_jacobian_near_the_origin, 4.0, 4, 0.25)
    r2 = (to_np(mesh.vertices_lens) ** 2).sum(axis=1)
    assert np.isnan(to_np(mesh.vertices_det)[r2 < 0.09]).any()
    beta = np.random.default_rng(6).uniform(-0.9, 0.9, (400, 2))
    mu, n = (to_np(x) for x in total_mu(mesh, _arr(beta)))
    assert np.isfinite(mu).all()
    assert (mu[n == 0] == 0).all() and (n == 0).any()
    np.testing.assert_allclose(mu[n == 1], 4.0, rtol=1e-12)


def test_a_mesh_with_no_converged_leaf_has_no_magnification_anywhere():
    mesh = build_lens_mesh(_collapse, _collapse_jacobian, 4.0, 4, 0.5)
    assert (leaf_status(mesh) != 0).all()
    assert (to_np(mesh.vertices_det) == 0).all()
    mu, n = total_mu(mesh, _arr([[0.0, 0.0], [0.3, -0.2]]))
    assert to_np(mu).tolist() == [0.0, 0.0] and to_np(n).tolist() == [0, 0]


def test_a_hit_reads_the_same_magnification_in_a_call_of_any_size(fold_mesh):
    """Each hit sums its own three terms in a fixed order, so a chunk of hits
    reads exactly what the whole batch does -- which keeps
    `build_magnification_map`'s samples equal to `total_magnification`
    at the same points. A row reduction would not: past about 1366 rows
    (4096 elements) jax changes how it sums a row, and a third of the rows
    then round differently. 5000 hits against chunks of at most 1000."""
    converged = np.flatnonzero(leaf_status(fold_mesh) == 0)
    rng = np.random.default_rng(8)
    leaves = rng.choice(converged, 5000)
    bary = rng.dirichlet(np.ones(3), 5000)

    def hits(lo, hi):
        idx = backend.as_array(leaves[lo:hi], dtype=backend.int64)
        return to_np(hit_magnification(fold_mesh, idx, _arr(bary[lo:hi])))

    full = hits(0, 5000)
    for lo in range(0, 5000, 1000):
        assert np.array_equal(hits(lo, lo + 1000), full[lo : lo + 1000]), lo


def test_the_fold_strip_has_its_exact_edges_and_area(fold_lens_fine):
    mag = build_magnification_map(
        fold_lens_fine, 8, 0.005, mu_min=4.0, fov=1.0, x0=0.0, y0=0.0
    )
    b = to_np(magnified_regions(mag, 4.0).source)[:, 1]
    lower = b < 0.22
    assert lower.any() and (~lower).any()
    assert np.abs(b[lower] - 0.1875).max() <= 0.005
    # A crossing on an edge across the caustic lies between its two samples.
    assert (np.abs(b[~lower] - 0.25) <= 0.005).all()
    area, complete = magnified_area(mag, 4.0)
    assert not bool(to_np(complete))
    assert abs(float(to_np(area)) - 0.0625) < 0.0015


def test_the_sis_region_is_the_disk_of_radius_two_over_mu_min(sis_mag):
    regions = magnified_regions(sis_mag, 4.0)
    # One loop: the sampled field is continuous within each sheet, so it
    # crosses mu_min once along every ray from the center.
    assert to_np(regions.offsets).tolist() == [0, to_np(regions.source).shape[0]]
    assert to_np(regions.closed).all()
    r = np.linalg.norm(to_np(regions.source), axis=1)
    assert np.abs(r - 0.5).max() <= 0.005
    area, complete = magnified_area(sis_mag, 4.0)
    assert bool(to_np(complete))
    assert abs(float(to_np(area)) / (np.pi * 0.25) - 1.0) < 1e-3


def test_sweep_and_threshold_modes_agree_on_the_sis_disk(sis_lens, sis_mag):
    # Sweep mode resolves log(1 + mu) to rtol rather than a threshold to the
    # floor, so its boundary sits about rtol / |d log(1 + mu) / d beta| =
    # 0.02 / 1.6 from the threshold mesh's: some 5% of the disk's area.
    sweep = build_magnification_map(
        sis_lens, 8, 0.01, rtol=0.02, fov=1.5, x0=0.0, y0=0.0
    )
    a_threshold, _ = magnified_area(sis_mag, 4.0)
    a_sweep, _ = magnified_area(sweep, 4.0)
    assert abs(float(to_np(a_sweep)) / float(to_np(a_threshold)) - 1.0) < 0.06


def test_the_cored_region_is_a_disk_and_an_annulus_inside_the_radial_caustic(
    cored_lens,
):
    mag = build_magnification_map(
        cored_lens, 8, 0.005, mu_min=6.0, fov=1.6, x0=0.0, y0=0.0
    )
    r_disk, r_annulus, r_caustic = _cored_crossings(6.0)
    regions = magnified_regions(mag, 6.0)
    # The disk's edge, and the annulus's inner and outer edges.
    assert to_np(regions.offsets).shape[0] == 4 and to_np(regions.closed).all()
    r = np.linalg.norm(to_np(regions.source), axis=1)
    radii = np.array([r_disk, r_annulus, r_caustic])
    assert np.abs(r[:, None] - radii[None, :]).min(axis=1).max() < 0.01
    area, _ = magnified_area(mag, 6.0)
    exact = np.pi * (r_disk**2 + r_caustic**2 - r_annulus**2)
    assert abs(float(to_np(area)) / exact - 1.0) < 3e-4


def test_converged_leaves_have_det_a_of_one_strict_sign_at_their_vertices(
    fold_lens_fine, sis_lens, cored_lens
):
    """So every converged leaf's interpolated det A is nonzero."""
    for mesh in (fold_lens_fine, sis_lens, cored_lens):
        converged = np.flatnonzero(leaf_status(mesh) == 0)
        d3 = to_np(mesh.vertices_det)[to_np(mesh.leaves)[converged]]
        assert np.isfinite(d3).all()
        assert ((d3 > 0).all(axis=1) | (d3 < 0).all(axis=1)).all()


def test_the_sis_total_magnification_is_accurate_to_a_fraction_of_a_per_cent(
    sis_lens,
):
    """Measured 2026-10-02 on this mesh: median 0.026%, 95th percentile 0.11%."""
    rng = np.random.default_rng(11)
    r = np.sqrt(rng.uniform(0.05**2, 0.9**2, 4000))
    a = rng.uniform(0.0, 2.0 * np.pi, 4000)
    beta = np.stack([r * np.cos(a), r * np.sin(a)], axis=1)
    mu, n = total_mu(sis_lens, backend.as_array(beta))
    two = to_np(n) == 2
    assert two.mean() > 0.95
    err = np.abs(to_np(mu)[two] * r[two] / 2.0 - 1.0)
    assert np.median(err) < 1e-3 and np.percentile(err, 95) < 4e-3


def test_the_inside_is_on_the_left(fold_mag):
    curves, _ = _curves(magnified_regions(fold_mag, 4.0))
    for c in sorted(curves, key=len)[-2:]:
        travel = c[-1, 0] - c[0, 0]
        # Inside lies above the level line and below the caustic.
        if c[:, 1].mean() < 0.22:
            assert travel > 0
        else:
            assert travel < 0


def test_open_curves_end_on_the_window_boundary(fold_mag):
    curves, closed = _curves(magnified_regions(fold_mag, 4.0))
    ends = np.array([p for c, k in zip(curves, closed) if not k for p in (c[0], c[-1])])
    assert ends.shape[0] > 0
    assert np.isclose(np.abs(ends).max(axis=1), 0.5).all()


def test_curves_stop_where_the_data_is_incomplete(wide_mag):
    curves, closed = _curves(magnified_regions(wide_mag, 1.5))
    assert not closed.all()
    ends = np.array([p for c, k in zip(curves, closed) if not k for p in (c[0], c[-1])])
    lo = to_np(wide_mag.lattice.lo)
    hi = lo + lattice_fov(wide_mag.lattice)
    on_window = (
        np.isclose(ends[:, 0], lo[0])
        | np.isclose(ends[:, 0], hi[0])
        | np.isclose(ends[:, 1], lo[1])
        | np.isclose(ends[:, 1], hi[1])
    )
    xy = to_np(wide_mag.vertices)
    inc = to_np(wide_mag.leaves)[to_np(wide_mag.incomplete)]
    a = xy[inc].reshape(-1, 2)
    b = xy[inc[:, [1, 2, 0]]].reshape(-1, 2)
    on_incomplete = _point_segment_distance(ends, a, b).min(axis=1) < 1e-9
    assert (on_window | on_incomplete).all()


def test_a_threshold_beyond_the_band_limit_leaves_only_the_band_strip(fold_mag):
    b = to_np(magnified_regions(fold_mag, 1e9).source)[:, 1]
    assert b.shape[0] > 0
    assert (np.abs(b - 0.25) <= 0.02).all()


def test_the_area_is_the_signed_area_the_closed_loops_bound(sis_mag_coarse):
    curves, closed = _curves(magnified_regions(sis_mag_coarse, 4.0))
    assert closed.all()
    area, complete = magnified_area(sis_mag_coarse, 4.0)
    assert bool(to_np(complete))
    np.testing.assert_allclose(
        float(to_np(area)), sum(_shoelace(c) for c in curves), rtol=1e-9
    )


def test_membership_is_the_side_of_the_loops_a_point_is_on(sis_mag_coarse):
    curves, _ = _curves(magnified_regions(sis_mag_coarse, 4.0))
    points = np.random.default_rng(11).uniform(-0.74, 0.74, (2000, 2))
    inside, complete = in_region(sis_mag_coarse, 4.0, _arr(points))
    assert to_np(complete).all()
    assert np.array_equal(to_np(inside), _winding(points, curves) > 0.5)


def test_points_outside_the_window_are_not_known(sis_mag_coarse):
    inside, complete = in_region(sis_mag_coarse, 4.0, _arr([[5.0, 5.0], [0.0, 0.0]]))
    assert to_np(complete).tolist() == [False, True]
    assert to_np(inside).tolist() == [False, True]


def test_the_area_takes_the_shape_of_mu_min(sis_mag_coarse):
    area, complete = magnified_area(sis_mag_coarse, [4.0, 8.0])
    assert tuple(area.shape) == (2,) and tuple(complete.shape) == (2,)
    single, _ = magnified_area(sis_mag_coarse, 4.0)
    assert tuple(single.shape) == ()
    assert float(to_np(area)[0]) == float(to_np(single))
    assert to_np(area)[1] < to_np(area)[0]


def test_a_zero_threshold_takes_the_whole_window(sis_mag_coarse):
    area, complete = magnified_area(sis_mag_coarse, 0.0)
    np.testing.assert_allclose(float(to_np(area)), 1.5**2, rtol=1e-12)
    assert not bool(to_np(complete))
    assert to_np(magnified_regions(sis_mag_coarse, 0.0).offsets).tolist() == [0]


def test_a_window_that_misses_every_image_has_no_region(sis_lens_coarse):
    far = build_magnification_map(
        sis_lens_coarse, 4, 0.1, mu_min=4.0, fov=1.0, x0=50.0, y0=0.0
    )
    assert (to_np(far.mu) == 0).all()
    assert not to_np(far.incomplete).any()
    assert to_np(magnified_regions(far, 4.0).offsets).tolist() == [0]
    area, complete = magnified_area(far, 4.0)
    assert float(to_np(area)) == 0.0 and bool(to_np(complete))
    inside, known = in_region(far, 4.0, _arr([[50.0, 0.0]]))
    assert to_np(inside).tolist() == [False] and to_np(known).tolist() == [True]


def test_segment_cells_flag_every_cell_a_segment_touches_and_no_far_one():
    rng = np.random.default_rng(5)
    n, size = 16, 1.0 / 16
    a = rng.uniform(-0.2, 1.2, (60, 2))
    b = a + rng.normal(0.0, 0.15, (60, 2))
    a[:5, 0] = b[:5, 0]  # vertical segments
    a[5:10, 1] = b[5:10, 1]  # horizontal segments
    got = set(
        to_np(segment_cells(_arr(a), _arr(b), _arr([0.0, 0.0]), size, n, 1e-9)).tolist()
    )
    touched = set().union(*(_brute_cells(a[e], b[e], n, size) for e in range(60)))
    near = set().union(*(_brute_cells(a[e], b[e], n, size, 1e-6) for e in range(60)))
    assert touched <= got <= near


def test_segment_cells_of_no_segment_or_of_segments_outside_the_grid_are_empty():
    lo = _arr([0.0, 0.0])
    empty = backend.zeros((0, 2), dtype=backend.float64)
    assert segment_cells(empty, empty, lo, 0.1, 10, 1e-9).shape[0] == 0
    a, b = _arr([[5.0, 5.0], [-3.0, 0.5]]), _arr([[6.0, 5.5], [-2.0, 0.6]])
    assert segment_cells(a, b, lo, 0.1, 10, 1e-9).shape[0] == 0


def test_every_lattice_triangle_is_half_of_its_level_cell():
    root_class = child_matrix_tables()[4]
    lat = make_lattice(4.0, 0.0, 0.0, 3, 4)
    ij, _ = initial_triangles(3, lat.level, root_class)
    for level in range(4):
        side = 1 << (lat.level - level)
        n_cells = 3 << level
        cell = to_np(triangle_cells(lat, ij, level))
        pts = to_np(ij)
        ci, cj = cell // n_cells, cell % n_cells
        assert (pts[..., 0] >= (ci * side)[:, None]).all()
        assert (pts[..., 0] <= ((ci + 1) * side)[:, None]).all()
        assert (pts[..., 1] >= (cj * side)[:, None]).all()
        assert (pts[..., 1] <= ((cj + 1) * side)[:, None]).all()
        e1, e2 = pts[:, 1] - pts[:, 0], pts[:, 2] - pts[:, 0]
        assert (e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0] == side * side).all()
        six = backend.concatenate((ij, midpoint_ij(ij)), dim=1)
        ij = six[:, _CHILD_VERTEX_INDEX_TABLE].reshape(-1, 3, 2)


def test_level_cells_rasterize_on_the_lattice_level_grid():
    lat = make_lattice(1.0, 0.0, 0.0, 2, 3)
    # A segment along y = 0.1 from x = -0.4 to x = 0.4, inside row 0 of the
    # 2 x 2 level-0 grid and crossing both of its columns.
    seg = _arr([[[-0.4, 0.1], [0.4, 0.1]]])
    assert to_np(level_cells(lat, seg, 0.0, 0)).tolist() == [1, 3]


def test_vertex_values_are_the_sampler_at_the_vertices(fold_lens, fold_mag):
    mu, n = total_mu(fold_lens, fold_mag.vertices)
    assert np.array_equal(to_np(fold_mag.mu), to_np(mu))
    assert np.array_equal(to_np(fold_mag.n), to_np(n))


def test_the_closed_leaves_are_conforming_and_positively_oriented(fold_mag):
    leaves, xy = to_np(fold_mag.leaves), to_np(fold_mag.vertices)
    e1 = xy[leaves[:, 1]] - xy[leaves[:, 0]]
    e2 = xy[leaves[:, 2]] - xy[leaves[:, 0]]
    assert (e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0] > 0).all()
    a, b = leaves.reshape(-1), leaves[:, [1, 2, 0]].reshape(-1)
    directed = set(zip(a.tolist(), b.tolist()))
    assert len(directed) == a.shape[0]
    on_edge = to_np(lattice_on_boundary(fold_mag.lattice, fold_mag.vertices_ij))
    for p, q in directed:
        if (q, p) not in directed:
            assert on_edge[p] and on_edge[q]


def test_every_leaf_straddling_a_target_is_at_the_floor(fold_mag):
    inside = to_np(fold_mag.mu)[to_np(fold_mag.leaves)] >= 4.0
    mixed = inside.any(axis=1) & ~inside.all(axis=1)
    assert mixed.any()
    # A straddling leaf is a finest-level triangle: twice its lattice area is 4.
    assert (leaf_levels(fold_mag)[mixed] == fold_mag.lattice.level - 1).all()


def test_no_leaf_below_the_floor_is_touched_by_a_band_edge(fold_lens, fold_mag):
    segments, on_fov = sheet_edges(fold_lens)
    inner = _arr(to_np(segments)[~to_np(on_fov)])
    level = leaf_levels(fold_mag)
    ij = fold_mag.vertices_ij[fold_mag.leaves]
    for d in sorted(set(level.tolist()) - {fold_mag.lattice.level - 1}):
        at = backend.as_array(np.flatnonzero(level == d), dtype=backend.int64)
        cells = to_np(level_cells(fold_mag.lattice, inner, 0.0, d))
        own = to_np(triangle_cells(fold_mag.lattice, ij[at], d))
        assert not np.isin(own, cells).any()


def test_only_leaves_the_fov_image_touches_are_incomplete(fold_lens, fold_mag):
    assert not to_np(fold_mag.incomplete).any()
    wide = build_magnification_map(fold_lens, 8, 0.05, mu_min=4.0)
    inc = to_np(wide.incomplete)
    assert inc.any()
    # Each incomplete leaf's pre-closure cell meets the image of the fov's
    # boundary. That cell is at most twice the leaf's longest edge across.
    segments, on_fov = sheet_edges(fold_lens)
    outer = to_np(segments)[to_np(on_fov)]
    tri = to_np(wide.vertices)[to_np(wide.leaves)[inc]]
    longest = np.linalg.norm(tri - tri[:, [1, 2, 0]], axis=-1).max(axis=1)
    distance = np.stack(
        [
            _point_segment_distance(tri[:, k], outer[:, 0], outer[:, 1]).min(axis=1)
            for k in range(3)
        ],
        axis=1,
    ).min(axis=1)
    assert (distance <= 2.0 * longest + 1e-9).all()


def test_the_build_does_not_depend_on_the_batch_size(fold_lens, fold_mag):
    other = build_magnification_map(
        fold_lens, 8, 0.02, mu_min=4.0, fov=1.0, x0=0.0, y0=0.0, batch_size=37
    )
    for name in (
        "vertices",
        "vertices_ij",
        "mu",
        "n",
        "leaves",
        "incomplete",
    ):
        assert np.array_equal(
            to_np(getattr(other, name)), to_np(getattr(fold_mag, name))
        ), name


def test_a_capped_depth_warns(fold_lens):
    with pytest.warns(UserWarning, match="depth-limited"):
        build_magnification_map(
            fold_lens, 8, 0.02, max_depth=1, mu_min=4.0, fov=1.0, x0=0.0, y0=0.0
        )


def test_a_threshold_above_the_band_limit_warns(fold_lens):
    with pytest.warns(UserWarning, match="exceeds mu_band"):
        build_magnification_map(fold_lens, 8, 0.05, mu_min=1e6, fov=1.0, x0=0.0, y0=0.0)


def test_a_resolved_build_raises_neither_warning(fold_lens):
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        build_magnification_map(fold_lens, 8, 0.05, mu_min=4.0, fov=1.0, x0=0.0, y0=0.0)
    messages = [str(w.message) for w in record]
    assert not [m for m in messages if "depth-limited" in m or "mu_band" in m]


def test_total_magnification_reads_a_scalar_point_and_empty_points(half_mesh):
    mu, n = total_magnification(0.1, -0.2, half_mesh)
    assert mu.shape == (1,) and n.shape == (1,)
    mu, n = total_magnification(f64([]), f64([]), half_mesh)
    assert mu.shape == (0,) and n.shape == (0,) and n.dtype == backend.int64


def test_in_magnified_region_reads_a_scalar_point_and_empty_points(sis_mag_coarse):
    inside, complete = in_magnified_region(0.05, 0.0, sis_mag_coarse, 4.0)
    assert to_np(inside).tolist() == [True] and to_np(complete).tolist() == [True]
    inside, complete = in_magnified_region(f64([]), f64([]), sis_mag_coarse, 4.0)
    assert inside.shape == (0,) and complete.shape == (0,)


def test_split_mask_splits_where_a_target_is_straddled_or_a_cell_touched():
    mu6 = f64([[1, 1, 1, 1, 1, 1], [1, 1, 5, 1, 3, 3]])
    none = backend.zeros((2,), dtype=backend.bool)
    every = backend.ones((2,), dtype=backend.bool)
    assert to_np(split_mask(mu6, none, (4.0,), None)).tolist() == [False, True]
    assert to_np(split_mask(mu6, every, (), None)).tolist() == [True, True]


def test_split_mask_splits_on_a_log_deviation_and_on_nan():
    mu6 = f64(
        [[1, 1, 1, 1, 1, 1], [1, 1, 1, 9, 1, 1], [np.inf, np.inf, 1, 1, 1, np.inf]]
    )
    none = backend.zeros((3,), dtype=backend.bool)
    assert to_np(split_mask(mu6, none, (), 0.1)).tolist() == [False, True, True]
