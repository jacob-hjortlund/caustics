"""The lens mesh: its criterion and band, its build, extension and closed build."""

import pickle
import warnings
import numpy as np
import pytest

from caustics.backend_obj import backend
from caustics.cosmology import FlatLambdaCDM
from caustics.lenses import SIE
from caustics.lenses.func.adaptive.band import (
    build_band,
    in_band,
)
from caustics.lenses.func.adaptive.criterion import (
    LEAF_APPROX_PARITY_UNRESOLVED,
    LEAF_CONVERGED,
    LEAF_CONVERGENCE_FAILED,
    LEAF_JACOBIAN_NONFINITE,
    LEAF_JACOBIAN_PARITY_UNRESOLVED,
    LEAF_RAYTRACE_NONFINITE,
    child_shape_matrices,
    lens_status,
    parity_from_children,
)
from caustics.lenses.func.adaptive.geometry import (
    ROOT_CLASS,
    ROOT_SHAPES,
)
from caustics.lenses.func.adaptive.index import index_hits
from caustics.lenses.func.adaptive.lattice import (
    lattice_xy,
    make_lattice,
)
from caustics.lenses.func.adaptive.lens_mesh import (
    build_lens_mesh,
    make_sampler,
)

from adaptive_maps import (
    sis_jacobian,
    sis_raytrace,
    broken_where,
    collapse,
    collapse_jacobian,
    lens,
    numpy_lens,
    AFFINE,
    affine,
    affine_jacobian,
    assert_same,
    build,
    f64,
    i64,
    localised_fold,
    localised_fold_jacobian,
    row_fold,
    row_fold_jacobian,
    stack_2x2,
    to_np,
)

H0 = 0.05


def six(tri):
    """Vertices ``(n, 3, 2)`` to the six samples ``(n, 6, 2)``: vertices, then ``m_i`` opposite vertex ``i``."""
    m = np.stack(
        (
            (tri[:, 1] + tri[:, 2]) / 2,
            (tri[:, 2] + tri[:, 0]) / 2,
            (tri[:, 0] + tri[:, 1]) / 2,
        ),
        axis=1,
    )
    return np.concatenate((tri, m), axis=1)


def roots(shift=(0.0, 0.0), h0=H0):
    """The six samples of both level-0 root triangles of cell size ``h0``."""
    return six(h0 * np.asarray(ROOT_SHAPES, dtype=np.float64) + np.asarray(shift))


def status(theta6, beta, det, level=0, h0=H0, min_img_sep=0.01):
    """``lens_status`` of the two root triangles, ``beta`` and ``det`` numpy maps of ``(..., 2)``."""
    n = theta6.shape[0]
    return to_np(
        lens_status(
            f64(beta(theta6)),
            f64(det(theta6)),
            ROOT_CLASS,
            i64([level] * n),
            h0,
            min_img_sep,
        )
    )


A = AFFINE


def affine_det(p):
    return np.full(p.shape[:-1], np.linalg.det(A))


def test_an_affine_map_converges():
    assert status(roots(), affine, affine_det).tolist() == [LEAF_CONVERGED] * 2


def test_a_reflecting_affine_map_converges_too():
    def flip(p):
        return p * np.array([1.0, -1.0])

    assert status(roots(), flip, lambda p: -np.ones(p.shape[:-1])).tolist() == [0, 0]


def test_a_fold_sets_both_parity_flags():
    def fold(p):
        return np.stack((p[..., 0], p[..., 1] ** 2), axis=-1)

    got = status(roots((0.0, -0.4 * H0)), fold, lambda p: 2.0 * p[..., 1])
    assert all(s & LEAF_APPROX_PARITY_UNRESOLVED for s in got)
    assert all(s & LEAF_JACOBIAN_PARITY_UNRESOLVED for s in got)


def test_a_nonfinite_image_is_the_row_s_only_flag():
    def beta(p):
        out = affine(p)
        out[0, 4] = np.nan
        return out

    def det(p):

        return np.where(np.arange(6) == 1, np.nan, 1.0) * np.ones(p.shape[:-1])

    got = status(roots(), beta, det)
    assert got.tolist() == [LEAF_RAYTRACE_NONFINITE, LEAF_JACOBIAN_NONFINITE]


@pytest.mark.parametrize("sample", range(6))
def test_one_flipped_det_sign_flags_jacobian_parity_on_its_row_alone(sample):
    def det(p):
        out = np.full(p.shape[:-1], np.linalg.det(A))
        out[1, sample] *= -1
        return out

    assert status(roots(), affine, det).tolist() == [
        LEAF_CONVERGED,
        LEAF_JACOBIAN_PARITY_UNRESOLVED,
    ]


@pytest.mark.parametrize("bad", [np.nan, np.inf, 0.0])
def test_a_nonfinite_or_zero_det_flags_jacobian_nonfinite_not_parity(bad):
    def det(p):
        out = np.full(p.shape[:-1], np.linalg.det(A))
        out[0, 3] = bad
        return out

    assert status(roots(), affine, det).tolist() == [LEAF_JACOBIAN_NONFINITE, 0]


def test_the_sign_comes_from_det_a_itself_however_small():
    def det(p):
        out = np.full(p.shape[:-1], 1e-300)
        out[0, 2] = -1e-300
        return out

    assert status(roots(), affine, det).tolist() == [LEAF_JACOBIAN_PARITY_UNRESOLVED, 0]


def test_the_jacobian_flags_add_to_a_deviation_failure_and_never_clear_it():
    def curved(p):
        return np.stack((p[..., 0] + 50.0 * p[..., 0] ** 2, p[..., 1]), axis=-1)

    def det(p):
        out = 1.0 + 100.0 * p[..., 0]
        out[1, 2] *= -1
        return out

    assert status(roots(), curved, det).tolist() == [
        LEAF_CONVERGENCE_FAILED,
        LEAF_CONVERGENCE_FAILED | LEAF_JACOBIAN_PARITY_UNRESOLVED,
    ]


def test_the_level_scales_the_deviation_threshold_with_the_triangle():
    def curved(p):
        return np.stack((p[..., 0] + 5.0 * p[..., 0] ** 2, p[..., 1]), axis=-1)

    def det(p):
        return 1.0 + 10.0 * p[..., 0]

    coarse = status(roots(h0=H0), curved, det, level=0, h0=H0)
    fine = status(roots(h0=H0), curved, det, level=1, h0=2 * H0)
    assert np.array_equal(coarse, fine)


def test_the_approximate_parity_bit_is_child_parity():
    rng = np.random.default_rng(5)
    beta6 = f64(rng.normal(size=(64, 6, 2)))
    det6 = f64(np.ones((64, 6)))
    n = 64
    got = to_np(
        lens_status(beta6, det6, i64([0] * n), i64([0] * n), H0, 0.01)
        & LEAF_APPROX_PARITY_UNRESOLVED
    )
    child_ok = to_np(
        parity_from_children(child_shape_matrices(beta6[:, :3], beta6[:, 3:]))
    )
    assert np.array_equal(got != 0, ~child_ok)


def test_the_leaf_flags_are_distinct_single_bits():
    flags = [
        LEAF_CONVERGENCE_FAILED,
        LEAF_APPROX_PARITY_UNRESOLVED,
        LEAF_JACOBIAN_PARITY_UNRESOLVED,
        LEAF_RAYTRACE_NONFINITE,
        LEAF_JACOBIAN_NONFINITE,
    ]
    assert LEAF_CONVERGED == 0 and len(set(flags)) == len(flags)
    assert all(f > 0 and f & (f - 1) == 0 for f in flags)


def test_a_band_leaf_has_finite_dets_of_both_classes_zero_counting_positive():
    det6 = f64(
        [
            [1, 2, 3, -1, 1, 1],
            [1, 2, 3, 4, 5, 6],
            [0, 0, 0, 1, 1, 1],
            [0, -1, 1, 1, 1, 1],
            [np.nan, -1, 1, 1, 1, 1],
        ]
    )
    assert to_np(in_band(det6)).tolist() == [True, False, False, True, False]


def test_build_band_numbers_samples_by_key_and_reads_their_values():
    lat = make_lattice(4.0, 0.0, 0.0, 4, 2)
    keys6 = i64([[30, 10, 20, 11, 21, 31], [10, 40, 20, 41, 21, 11]])
    keys = np.array([10, 11, 20, 21, 30, 31, 40, 41])
    table = f64([[k, -k, k / 100.0] for k in keys])
    band = build_band(lat, i64([3, 7]), keys6, i64(keys), table)
    assert to_np(band.leaves).tolist() == [3, 7]
    assert np.array_equal(keys[to_np(band.samples)], to_np(keys6))
    assert np.array_equal(to_np(band.source)[:, 0], keys)
    assert np.array_equal(to_np(band.det), keys / 100.0)
    ij = np.stack((keys // (lat.n + 1), keys % (lat.n + 1)), axis=-1)
    assert np.array_equal(to_np(band.lens), to_np(lattice_xy(lat, i64(ij))))


@pytest.fixture(scope="module")
def sie_mesh():
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
    return build_lens_mesh(lens.raytrace, lens.jacobian_lens_equation, 5.0, 32, 1e-2)


def test_the_sampler_hands_both_callbacks_the_same_float64_points_on_the_device(device):
    seen = []

    def raytrace(x, y):
        seen.append((x.dtype, backend.device(x), to_np(x), to_np(y)))
        return 0.5 * x, 0.5 * y

    def jacobian(x, y):
        seen.append((x.dtype, backend.device(x), to_np(x), to_np(y)))
        half, zero = 0.5 + 0.0 * x, 0.0 * x
        return stack_2x2(half, zero, zero, half)

    xy = f64([[0.0, 1.0], [2.0, -3.0]])
    out = make_sampler(raytrace, jacobian, device)(xy)
    want = backend.device(backend.as_array([0.0], device=device))
    (rd, rdev, rx, ry), (jd, jdev, jx, jy) = seen
    assert rd == jd == backend.float64 and rdev == jdev == want
    assert np.array_equal(rx, jx) and np.array_equal(ry, jy)
    assert out.dtype == backend.float64 and backend.device(out) == backend.device(xy)
    assert to_np(out).tolist() == [[0.0, 0.5, 0.25], [1.0, -1.5, 0.25]]


def test_float32_callbacks_give_float64_values_with_det_a_formed_in_float32():
    A32 = np.array([[0.7, 0.1], [-0.2, 0.9]], dtype=np.float32)

    def raytrace(x, y):
        x32 = backend.to(x, dtype=backend.float32)
        y32 = backend.to(y, dtype=backend.float32)
        return 0.7 * x32 + 0.1 * y32, -0.2 * x32 + 0.9 * y32

    def jacobian(x, y):
        return backend.as_array(np.tile(A32, (x.shape[0], 1, 1)))

    out = make_sampler(raytrace, jacobian, None)(f64([[1.0, 2.0]]))
    J = backend.as_array(A32[None])
    det32 = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]
    assert out.dtype == backend.float64
    assert to_np(out)[0, 2] == to_np(backend.to(det32, dtype=backend.float64))[0]


def test_an_affine_lens_converges_at_level_zero():
    mesh, _ = build(affine, affine_jacobian, init_res=2, min_img_sep=1.0, max_depth=3)
    assert mesh.origin_leaves.shape[0] == 8
    assert (to_np(mesh.origin_level) == 0).all()
    assert (to_np(mesh.origin_status) == LEAF_CONVERGED).all()


def test_the_build_samples_no_point_twice_and_calls_both_callbacks_alike():
    _, calls = build(localised_fold, localised_fold_jacobian, min_img_sep=0.1)
    points = np.concatenate(calls.raytrace)
    assert len({tuple(p) for p in points}) == len(points)
    assert [len(c) for c in calls.raytrace] == [len(c) for c in calls.jacobian]


def test_statuses_are_set_only_at_the_finest_level_and_index_exactly_the_converged_leaves(
    sie_mesh,
):
    mesh = sie_mesh
    status, level = to_np(mesh.origin_status), to_np(mesh.origin_level)
    assert (status[level < mesh.lattice.level - 1] == LEAF_CONVERGED).all()
    assert (status != LEAF_CONVERGED).any()
    indexed = np.unique(to_np(mesh.index.cell_leaves))
    want = np.flatnonzero(status[to_np(mesh.leaf_origin)] == LEAF_CONVERGED)
    assert np.array_equal(indexed, want)


def test_finest_leaves_are_no_longer_than_the_mesh_s_min_img_sep(sie_mesh):
    mesh = sie_mesh
    tri = to_np(mesh.vertices_lens)[to_np(mesh.leaves)]
    longest = np.linalg.norm(tri - tri[:, [1, 2, 0]], axis=-1).max(axis=1)
    level = to_np(mesh.origin_level)[to_np(mesh.leaf_origin)]
    finest = level == mesh.lattice.level - 1
    assert finest.any()
    assert longest[finest].max() <= mesh.min_img_sep * (1 + 1e-12)


def test_finest_level_midpoints_are_band_samples_but_never_vertices():
    mesh, _ = build(row_fold, row_fold_jacobian, init_res=8, min_img_sep=2e-2)
    lat, band = mesh.lattice, mesh.critical_band
    ij = np.rint((to_np(band.lens) - to_np(lat.lo)) / lat.scale).astype(np.int64)
    ij = ij + lat.origin
    odd = (ij % 2 == 1).any(axis=1)
    vertices = {tuple(p) for p in to_np(mesh.vertices_ij)}
    assert odd.any() and (~odd).any()
    assert all((tuple(p) in vertices) != o for p, o in zip(ij, odd))


@pytest.mark.parametrize("batch_size", [1, 7])
def test_the_build_does_not_depend_on_the_batch_size(batch_size):
    want, _ = build(localised_fold, localised_fold_jacobian, min_img_sep=0.1)
    got, calls = build(
        localised_fold, localised_fold_jacobian, min_img_sep=0.1, batch_size=batch_size
    )
    assert max(len(c) for c in calls.raytrace + calls.jacobian) <= batch_size
    assert_same(want, got)


def identity(p):
    return p * 1.0


def identity_jacobian(p):
    return np.tile(np.eye(2), (p.shape[0], 1, 1))


def _sie_like(x, y):
    r = (x * x + y * y + 0.05) ** 0.5
    return x - 1.2 * x / r, y - 1.2 * y / r


def _sie_like_jacobian(x, y):
    r = (x * x + y * y + 0.05) ** 0.5
    k = 1.2 / r**3
    return stack_2x2(
        1.0 - 1.2 / r + k * x * x, k * x * y, k * x * y, 1.0 - 1.2 / r + k * y * y
    )


def _affine(x, y):
    return 2.0 * x + 0.5 * y, -0.25 * x + 1.5 * y


def _affine_jacobian(x, y):
    one = backend.ones_like(x)
    return stack_2x2(2.0 * one, 0.5 * one, -0.25 * one, 1.5 * one)


SIE_LIKE = lens(_sie_like, _sie_like_jacobian)
AFFINE_LENS = lens(_affine, _affine_jacobian)
BUILD = dict(fov=4.0, init_res=3, min_img_sep=0.5, max_depth=3)


def leaf_status(mesh):
    return to_np(mesh.origin_status)[to_np(mesh.leaf_origin)]


def leaf_level(mesh):
    return to_np(mesh.origin_level)[to_np(mesh.leaf_origin)]


def test_every_indexed_leaf_has_finite_source_vertices():
    fn, jac = broken_where(
        localised_fold, localised_fold_jacobian, lambda p: p[:, 0] > 1.0
    )
    mesh, _ = build(fn, jac, min_img_sep=0.05)
    vs = to_np(mesh.vertices_source)
    leaves = to_np(mesh.leaves)
    for leaf in np.unique(to_np(mesh.index.cell_leaves)):
        assert np.isfinite(vs[leaves[leaf]]).all()


def test_index_registers_every_leaf_in_the_cell_of_each_of_its_vertices():
    mesh, _ = build(localised_fold, localised_fold_jacobian, min_img_sep=0.05)
    vs = to_np(mesh.vertices_source)
    leaves = to_np(mesh.leaves)
    offs = to_np(mesh.index.cell_offsets)
    cells = to_np(mesh.index.cell_leaves)
    lo, cell = to_np(mesh.index.lo), to_np(mesh.index.cell)
    nx, ny = mesh.index.nx, mesh.index.ny
    status = leaf_status(mesh)
    rng = np.random.default_rng(20260904)
    for leaf in rng.choice(len(leaves), size=50, replace=False):
        if status[leaf] != LEAF_CONVERGED:
            continue
        for q in vs[leaves[leaf]]:
            # `build_index`'s own arithmetic: divide, truncate, clamp.
            ix = int(np.clip(np.trunc((q[0] - lo[0]) / cell[0]), 0, nx - 1))
            iy = int(np.clip(np.trunc((q[1] - lo[1]) / cell[1]), 0, ny - 1))
            c = ix * ny + iy
            assert leaf in cells[offs[c] : offs[c + 1]]


def test_build_halves_the_requested_min_img_sep():
    mesh = build_lens_mesh(
        AFFINE_LENS.raytrace,
        AFFINE_LENS.jacobian_lens_equation,
        fov=4.0,
        init_res=2,
        min_img_sep=0.4,
        max_depth=3,
    )
    assert mesh.min_img_sep == pytest.approx(0.2)


def test_unconverged_leaves_are_kept_but_excluded_from_the_index():
    mesh = build_lens_mesh(SIE_LIKE.raytrace, SIE_LIKE.jacobian_lens_equation, **BUILD)
    status = leaf_status(mesh)
    indexed = set(to_np(mesh.index.cell_leaves).tolist())
    assert (status != LEAF_CONVERGED).any(), "fixture must flag some leaf"
    assert indexed == set(np.flatnonzero(status == LEAF_CONVERGED).tolist())


def test_mesh_arrays_are_float64():
    mesh = build_lens_mesh(
        AFFINE_LENS.raytrace, AFFINE_LENS.jacobian_lens_equation, **BUILD
    )
    for name in ("vertices_lens", "vertices_source", "vertices_det"):
        assert getattr(mesh, name).dtype == backend.float64, name


def test_build_returns_a_consistent_mesh_for_an_affine_map():
    mesh, _ = build(affine, affine_jacobian)
    L = to_np(mesh.leaves).shape[0]
    assert L == 2 * 4**2
    assert (leaf_status(mesh) == LEAF_CONVERGED).sum() == L
    assert to_np(mesh.origin_leaves).shape[0] == L
    assert np.array_equal(to_np(mesh.leaf_origin), np.arange(L))
    src, lens_xy = to_np(mesh.vertices_source), to_np(mesh.vertices_lens)
    assert np.allclose(src, lens_xy @ AFFINE.T, rtol=1e-10, atol=1e-12)


def test_vertices_are_compacted_and_ordered_by_lattice_key():
    mesh, _ = build(localised_fold, localised_fold_jacobian, min_img_sep=0.05)
    used = np.unique(to_np(mesh.leaves))
    assert used.tolist() == list(range(mesh.vertices_lens.shape[0]))
    xy = to_np(mesh.vertices_lens)
    key = np.lexsort((xy[:, 1], xy[:, 0]))
    assert np.array_equal(key, np.arange(len(xy)))


def test_build_is_deterministic_on_a_refined_mesh():
    a, _ = build(localised_fold, localised_fold_jacobian, min_img_sep=0.05)
    b, _ = build(localised_fold, localised_fold_jacobian, min_img_sep=0.05)
    assert_same(a, b)


def test_depth_limit_warns_and_names_the_required_max_depth():
    with pytest.warns(
        UserWarning, match=r"Lens mesh is depth-limited.*max_depth >= \d+"
    ):
        mesh, _ = build(
            localised_fold, localised_fold_jacobian, min_img_sep=1e-4, max_depth=2
        )
    assert mesh.lattice.level - 1 == 2


def test_depth_limited_warning_never_bare_quotes_the_halved_value():
    """The warning quotes what the caller passed, and the halved value only labelled."""
    with pytest.warns(UserWarning, match=r"min_img_sep=0\.0002 arcsec") as record:
        build(identity, identity_jacobian, min_img_sep=2e-4, max_depth=1)
    assert not any("min_img_sep=0.0001" in str(w.message) for w in record)


def test_nonfinite_leaves_from_a_nonfinite_region_are_excluded_from_the_index():
    fn, jac = broken_where(
        localised_fold, localised_fold_jacobian, lambda p: p[:, 0] > 1.0, np.inf
    )
    mesh, _ = build(fn, jac, min_img_sep=0.05)
    status = leaf_status(mesh)
    nonfinite = (status & LEAF_RAYTRACE_NONFINITE) != 0
    assert nonfinite.any()
    assert not np.isfinite(to_np(mesh.vertices_source)).all()
    indexed = set(to_np(mesh.index.cell_leaves).tolist())
    assert not indexed & set(np.flatnonzero(nonfinite).tolist())


def test_parity_invalid_partitions_the_max_level_leaves(sie_mesh):
    """Finest-level leaves converge or carry only criterion flags; coarser ones all converge.

    The cored SIE is finite everywhere and its Jacobian is never exactly
    singular at a sample, so no non-finite flag can occur.
    """
    mesh = sie_mesh
    assert np.isfinite(to_np(mesh.vertices_source)).all()
    status, level = leaf_status(mesh), leaf_level(mesh)
    at_max = level == mesh.lattice.level - 1
    assert at_max.any()
    criterion_flags = (
        LEAF_CONVERGENCE_FAILED
        | LEAF_APPROX_PARITY_UNRESOLVED
        | LEAF_JACOBIAN_PARITY_UNRESOLVED
    )
    assert not (status[at_max] & ~criterion_flags).any()
    assert (status[at_max] != LEAF_CONVERGED).any()
    assert (status[at_max] == LEAF_CONVERGED).any()
    assert (status[~at_max] == LEAF_CONVERGED).all()


def test_parity_band_is_bounded_by_a_small_multiple_of_min_img_sep(sie_mesh):
    """Halving min_img_sep keeps the condemned band near the center within 2.5x the request."""
    requested_min_img_sep = 1e-2
    mesh = sie_mesh
    status = leaf_status(mesh)
    leaves = to_np(mesh.leaves)
    vl = to_np(mesh.vertices_lens)
    invalid = np.flatnonzero(status != LEAF_CONVERGED)
    assert invalid.size > 0
    radius = np.linalg.norm(vl[leaves[invalid]].mean(axis=1), axis=-1)
    near_center = radius < 0.05
    assert near_center.any()
    core = radius[near_center]
    assert core.max() - core.min() <= 2.5 * requested_min_img_sep


def lattice_samples(mesh):
    """Lattice-exact positions of every finite finest-level origin's six samples."""
    lat = mesh.lattice
    lo, scale = to_np(lat.lo), lat.scale
    level, status = to_np(mesh.origin_level), to_np(mesh.origin_status)
    rows = np.flatnonzero(
        (level == lat.level - 1) & ((status & LEAF_RAYTRACE_NONFINITE) == 0)
    )
    vij = to_np(mesh.vertices_ij)[to_np(mesh.origin_leaves)[rows]] - lat.origin
    mij = np.stack(
        [
            (vij[:, 1] + vij[:, 2]) // 2,
            (vij[:, 2] + vij[:, 0]) // 2,
            (vij[:, 0] + vij[:, 1]) // 2,
        ],
        axis=1,
    )
    return rows, lo + np.concatenate([vij, mij], axis=1).astype(np.float64) * scale


def independent_band(mesh, jac):
    """Band origins recomputed from lattice-exact samples, without the build."""
    rows, xy = lattice_samples(mesh)
    J = jac(xy.reshape(-1, 2)).reshape(-1, 6, 2, 2)
    det = J[..., 0, 0] * J[..., 1, 1] - J[..., 0, 1] * J[..., 1, 0]
    positive = det >= 0
    band = np.isfinite(det).all(axis=1) & positive.any(axis=1) & (~positive).any(axis=1)
    return rows[band]


@pytest.mark.parametrize(
    "fn, jac, kw",
    [
        (localised_fold, localised_fold_jacobian, dict(init_res=4, min_img_sep=0.05)),
        (row_fold, row_fold_jacobian, dict(init_res=8, min_img_sep=2e-2)),
    ],
    ids=["localised_fold", "row_fold"],
)
def test_band_is_the_sign_change_leaves_recomputed_independently(fn, jac, kw):
    """The band is exactly the finest origins whose six dets change class.

    Every ``LEAF_JACOBIAN_PARITY_UNRESOLVED`` origin is in it; every other band
    origin is ``LEAF_JACOBIAN_NONFINITE`` with an exactly zero sample.
    `localised_fold`'s curve is never a lattice row, `row_fold`'s is one.
    """
    mesh, _ = build(fn, jac, fov=4.0, **kw)
    band = mesh.critical_band
    got = to_np(band.leaves)
    want = independent_band(mesh, jac)
    assert want.size > 0
    assert sorted(got.tolist()) == sorted(want.tolist())
    status = to_np(mesh.origin_status)
    flagged = np.flatnonzero((status & LEAF_JACOBIAN_PARITY_UNRESOLVED) != 0)
    assert set(flagged.tolist()) <= set(got.tolist())
    extra = ~np.isin(got, flagged)
    assert ((status[got[extra]] & LEAF_JACOBIAN_NONFINITE) != 0).all()
    det = to_np(band.det)[to_np(band.samples)]
    assert (det[extra] == 0).any(axis=1).all()
    if fn is row_fold:
        assert flagged.size == 0 and extra.all()
    else:
        assert not extra.any()


def test_band_samples_are_lattice_exact_and_carry_the_lens_values():
    mesh, _ = build(row_fold, row_fold_jacobian, fov=4.0, init_res=8, min_img_sep=2e-2)
    band = mesh.critical_band
    leaves, samples = to_np(band.leaves), to_np(band.samples)
    xy, source, det = to_np(band.lens), to_np(band.source), to_np(band.det)
    assert leaves.size > 0
    vertices = to_np(mesh.vertices_lens)[to_np(mesh.origin_leaves)[leaves]]
    assert np.array_equal(xy[samples[:, :3]], vertices)
    rows, want = lattice_samples(mesh)
    position = {r: i for i, r in enumerate(rows.tolist())}
    assert np.array_equal(xy[samples], want[[position[r] for r in leaves.tolist()]])
    assert np.unique(xy, axis=0).shape[0] == xy.shape[0]
    assert np.array_equal(source, row_fold(xy))
    J = row_fold_jacobian(xy)
    assert np.array_equal(det, J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0])


def test_band_never_holds_a_leaf_with_a_nonfinite_sample():
    fn, jac = broken_where(
        localised_fold, localised_fold_jacobian, lambda p: p[:, 0] > 1.0
    )
    mesh, _ = build(fn, jac, init_res=4, min_img_sep=0.05)
    band = mesh.critical_band
    got = to_np(band.leaves)
    assert got.size > 0
    assert sorted(got.tolist()) == sorted(independent_band(mesh, jac).tolist())
    status = to_np(mesh.origin_status)
    assert not ((status[got] & LEAF_RAYTRACE_NONFINITE) != 0).any()
    assert np.isfinite(to_np(band.det)).all()
    assert (to_np(band.lens)[:, 0] <= 1.0).all()


@pytest.mark.parametrize(
    "fn, jac, kw",
    [
        (affine, affine_jacobian, dict()),
        (collapse, collapse_jacobian, dict(min_img_sep=0.5)),
    ],
    ids=["affine", "kappa_one_sheet"],
)
def test_band_is_empty_without_a_sign_change(fn, jac, kw):
    mesh, _ = build(fn, jac, **kw)
    band = mesh.critical_band
    assert tuple(band.leaves.shape) == (0,)
    assert tuple(band.samples.shape) == (0, 6)
    assert tuple(band.lens.shape) == (0, 2)
    assert tuple(band.source.shape) == (0, 2)
    assert tuple(band.det.shape) == (0,)


def test_a_mesh_survives_a_pickle_round_trip():
    mesh = build_lens_mesh(
        AFFINE_LENS.raytrace, AFFINE_LENS.jacobian_lens_equation, **BUILD
    )
    again = pickle.loads(pickle.dumps(mesh))
    for name in ("index", "critical_band", "holes", "lattice"):
        assert type(getattr(again, name)) is type(getattr(mesh, name)), name
    for name in ("vertices_lens", "vertices_source", "leaves", "origin_status"):
        assert np.array_equal(to_np(getattr(again, name)), to_np(getattr(mesh, name)))


def test_every_vertex_carries_det_a_as_float64():
    mesh, _ = build(localised_fold, localised_fold_jacobian, min_img_sep=0.05)
    assert mesh.vertices_det.dtype == backend.float64
    J = localised_fold_jacobian(to_np(mesh.vertices_lens))
    want = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]
    assert np.array_equal(to_np(mesh.vertices_det), want)


def test_a_critical_band_sample_at_a_vertex_carries_that_vertex_s_det():
    mesh, _ = build(row_fold, row_fold_jacobian, init_res=8, min_img_sep=2e-2)
    det = dict(
        zip(
            map(tuple, to_np(mesh.vertices_ij).tolist()),
            to_np(mesh.vertices_det).tolist(),
        )
    )
    lat, band = mesh.lattice, mesh.critical_band
    ij = np.rint((to_np(band.lens) - to_np(lat.lo)) / lat.scale).astype(np.int64)
    at = [k for k, p in enumerate(map(tuple, ij.tolist())) if p in det]
    assert len(at) > 0
    assert [to_np(band.det)[k] for k in at] == [det[tuple(ij[k])] for k in at]


def test_a_float32_jacobian_still_gives_float64_det_a():
    """Formed from the float32 entries, then cast, as the band's det is."""
    lens_, _ = numpy_lens(localised_fold, localised_fold_jacobian)

    def jacobian32(x, y):
        return backend.to(lens_.jacobian_lens_equation(x, y), dtype=backend.float32)

    mesh = build_lens_mesh(lens_.raytrace, jacobian32, 4.0, 4, 0.05)
    assert mesh.vertices_det.dtype == backend.float64
    J = localised_fold_jacobian(to_np(mesh.vertices_lens)).astype(np.float32)
    want = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]
    assert np.array_equal(to_np(mesh.vertices_det), want.astype(np.float64))


def origin_samples(mesh, rows):
    """Lattice-exact lens-plane positions of origins ``rows``' six samples, ``(n, 6, 2)``."""
    lat = mesh.lattice
    vij = to_np(mesh.vertices_ij)[to_np(mesh.origin_leaves)[rows]] - lat.origin
    mij = np.stack(
        [
            (vij[:, 1] + vij[:, 2]) // 2,
            (vij[:, 2] + vij[:, 0]) // 2,
            (vij[:, 0] + vij[:, 1]) // 2,
        ],
        axis=1,
    )
    six = np.concatenate([vij, mij], axis=1).astype(np.float64)
    return to_np(lat.lo) + six * lat.scale


def numpy_det(jac, xy):
    J = jac(xy.reshape(-1, 2)).reshape(*xy.shape[:-1], 2, 2)
    return J[..., 0, 0] * J[..., 1, 1] - J[..., 0, 1] * J[..., 1, 0]


def _gaussian_bump(p, w=0.08, amp=1.0, c=(0.13, 0.07)):
    """``0.5 p`` plus a narrow Gaussian bump along ``(1, 0.3)``."""
    r2 = ((p - np.asarray(c)) ** 2).sum(axis=-1)
    return p * 0.5 + (amp * np.exp(-r2 / (2 * w**2)))[:, None] * np.array([1.0, 0.3])


def _gaussian_bump_jacobian(p, w=0.08, amp=1.0, c=(0.13, 0.07)):
    d = p - np.asarray(c)
    g = amp * np.exp(-(d**2).sum(axis=-1) / (2 * w**2))
    grad = -(g / w**2)[:, None] * d
    return 0.5 * np.eye(2) + np.array([1.0, 0.3])[None, :, None] * grad[:, None, :]


def test_a_kappa_one_sheet_samples_every_lattice_point_once_and_converges_nothing():
    """``A == 0`` everywhere: every triangle fails the deviation test and has zero dets."""
    mesh, calls = build(collapse, collapse_jacobian, min_img_sep=1.0)
    level, status = to_np(mesh.origin_level), to_np(mesh.origin_status)
    assert (level == mesh.lattice.level - 1).all()
    assert (status == (LEAF_CONVERGENCE_FAILED | LEAF_JACOBIAN_NONFINITE)).all()
    assert np.isfinite(to_np(mesh.vertices_source)).all()
    assert sum(len(c) for c in calls.raytrace) == (mesh.lattice.n + 1) ** 2
    assert to_np(mesh.index.cell_leaves).size == 0


def test_a_nonfinite_subregion_is_split_down_to_the_finest_level():
    fn, jac = broken_where(identity, identity_jacobian, lambda p: p[:, 0] > 0.5)
    mesh, _ = build(fn, jac, min_img_sep=1.0)
    max_level = mesh.lattice.level - 1
    assert max_level > 0
    status, level = to_np(mesh.origin_status), to_np(mesh.origin_level)
    vs = to_np(mesh.vertices_source)[to_np(mesh.origin_leaves)]
    nonfinite = status == LEAF_RAYTRACE_NONFINITE
    good = status == LEAF_CONVERGED
    assert nonfinite.any() and good.any()
    assert (level[nonfinite] == max_level).all()
    assert not np.isfinite(vs[nonfinite]).all()
    assert np.isfinite(vs[good]).all()
    assert (good | nonfinite).all()


def test_only_the_six_triangles_at_a_point_singularity_stay_nonfinite():
    """A red split hands the bad vertex to one child, so six triangles share it at every level."""
    mesh, _ = build(sis_raytrace, sis_jacobian, min_img_sep=0.1)
    assert mesh.lattice.level - 1 == 5
    status, level = to_np(mesh.origin_status), to_np(mesh.origin_level)
    nonfinite = (status & LEAF_RAYTRACE_NONFINITE) != 0
    assert nonfinite.sum() == 6
    assert (status[nonfinite] == LEAF_RAYTRACE_NONFINITE).all()
    assert (level[nonfinite] == 5).all()


def test_nonfinite_vertices_are_flagged_even_when_the_finest_level_is_zero():
    fn, jac = broken_where(identity, identity_jacobian, lambda p: p[:, 0] > 0.5)
    mesh, _ = build(fn, jac, min_img_sep=4.0)
    assert mesh.lattice.level - 1 == 0
    status = to_np(mesh.origin_status)
    vs = to_np(mesh.vertices_source)[to_np(mesh.origin_leaves)]
    assert (status == LEAF_RAYTRACE_NONFINITE).any()
    assert (status == LEAF_CONVERGED).any()
    assert set(np.unique(status).tolist()) <= {LEAF_CONVERGED, LEAF_RAYTRACE_NONFINITE}
    assert np.isfinite(vs[status == LEAF_CONVERGED]).all()
    for row in np.flatnonzero(status == LEAF_RAYTRACE_NONFINITE):
        assert not np.isfinite(vs[row]).all()


def test_a_nonfinite_midpoint_with_finite_vertices_is_condemned_alone():
    """A NaN band around ``x = 0.5`` hits the eight triangles' midpoints only."""
    fn, jac = broken_where(
        identity, identity_jacobian, lambda p: np.abs(p[:, 0] - 0.5) < 0.1
    )
    mesh, _ = build(fn, jac, min_img_sep=4.0)
    assert mesh.lattice.level - 1 == 0
    status = to_np(mesh.origin_status)
    vs = to_np(mesh.vertices_source)[to_np(mesh.origin_leaves)]
    nonfinite = np.flatnonzero(status == LEAF_RAYTRACE_NONFINITE)
    assert nonfinite.size == 8
    assert np.isfinite(vs[nonfinite]).all()
    assert (status == LEAF_CONVERGED).any()


def test_finest_level_flags_match_both_parity_tests_row_for_row():
    """Child parity and Jacobian parity recomputed from lattice-exact samples."""
    mesh, _ = build(localised_fold, localised_fold_jacobian, min_img_sep=0.1)
    max_level = mesh.lattice.level - 1
    level, status = to_np(mesh.origin_level), to_np(mesh.origin_status)
    rows = np.flatnonzero(level == max_level)
    assert rows.size > 0
    xy = origin_samples(mesh, rows)
    beta6 = localised_fold(xy.reshape(-1, 2)).reshape(-1, 6, 2)
    child_ok = to_np(
        parity_from_children(child_shape_matrices(f64(beta6[:, :3]), f64(beta6[:, 3:])))
    )
    det = numpy_det(localised_fold_jacobian, xy)
    usable = (np.isfinite(det) & (det != 0)).all(axis=1)
    mixed = usable & ~((det > 0).all(axis=1) | (det < 0).all(axis=1))
    st = status[rows]
    approx = (st & LEAF_APPROX_PARITY_UNRESOLVED) != 0
    jacobian = (st & LEAF_JACOBIAN_PARITY_UNRESOLVED) != 0
    for flag in (approx, jacobian):
        assert flag.any() and not flag.all()
    assert approx.tolist() == (~child_ok).tolist()
    assert jacobian.tolist() == mixed.tolist()
    assert (level[status != LEAF_CONVERGED] == max_level).all()


@pytest.mark.parametrize(
    "fn,jac,kw",
    [
        (localised_fold, localised_fold_jacobian, dict(min_img_sep=0.1)),
        (sis_raytrace, sis_jacobian, dict(min_img_sep=0.1)),
        (_gaussian_bump, _gaussian_bump_jacobian, dict(init_res=8, min_img_sep=0.04)),
    ],
    ids=["localised_fold", "sis", "gaussian_bump"],
)
def test_converged_leaves_pass_jacobian_parity_at_their_own_samples(fn, jac, kw):
    """No origin converges with a critical curve between its samples, balance-forced ones included."""
    mesh, _ = build(fn, jac, **kw)
    level, status = to_np(mesh.origin_level), to_np(mesh.origin_status)
    rows = np.flatnonzero(status == LEAF_CONVERGED)
    assert (level[rows] < mesh.lattice.level - 1).any()
    det = numpy_det(jac, origin_samples(mesh, rows))
    assert np.isfinite(det).all() and (det != 0).all()
    assert ((det > 0).all(axis=1) | (det < 0).all(axis=1)).all()


def test_failure_flags_are_mutually_consistent():
    fn, jac = broken_where(
        localised_fold, localised_fold_jacobian, lambda p: p[:, 0] > 1.0
    )
    mesh, _ = build(fn, jac, min_img_sep=0.1)
    status, level = to_np(mesh.origin_status), to_np(mesh.origin_level)
    raytrace_bad = (status & LEAF_RAYTRACE_NONFINITE) != 0
    parity_bad = (
        status & (LEAF_APPROX_PARITY_UNRESOLVED | LEAF_JACOBIAN_PARITY_UNRESOLVED)
    ) != 0
    assert raytrace_bad.any() and parity_bad.any()
    finite = np.isfinite(to_np(mesh.vertices_source)[to_np(mesh.origin_leaves)]).all(
        axis=(1, 2)
    )
    assert (status[raytrace_bad] == LEAF_RAYTRACE_NONFINITE).all()
    assert not finite[raytrace_bad].any()
    assert finite[~raytrace_bad].all()
    assert (level[status != LEAF_CONVERGED] == mesh.lattice.level - 1).all()


def test_a_kappa_one_sheet_indexes_no_leaf():
    mesh, _ = build(collapse, collapse_jacobian, min_img_sep=1.0)
    status = to_np(mesh.origin_status)
    assert not (status == LEAF_CONVERGED).any()
    assert ((status & LEAF_JACOBIAN_NONFINITE) != 0).all()


def test_coverage_does_not_drop_at_level_transitions():
    """Every point a uniform finest-level mesh covers, the adaptive mesh covers too."""
    fov, init_res, sep = 4.0, 4, 0.1
    mesh, _ = build(
        localised_fold,
        localised_fold_jacobian,
        fov=fov,
        init_res=init_res,
        min_img_sep=sep,
    )
    ml = mesh.lattice.level - 1
    assert ml >= 3 and len(set(leaf_level(mesh).tolist())) > 1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        uniform, _ = build(
            localised_fold,
            localised_fold_jacobian,
            fov=fov,
            init_res=init_res * 2**ml,
            min_img_sep=sep,
            max_depth=0,
        )
    grid = np.linspace(-0.9, 0.9, 120)
    beta = f64(np.stack(np.meshgrid(grid, grid, indexing="ij"), axis=-1).reshape(-1, 2))

    def covered(m):
        qidx, _, _ = index_hits(m.index, m.vertices_source, m.leaves, beta)
        return np.bincount(to_np(qidx), minlength=beta.shape[0]) > 0

    hit_a, hit_u = covered(mesh), covered(uniform)
    assert int((hit_u & ~hit_a).sum()) == 0


def test_the_criterion_is_blind_to_structure_below_the_sampling_scale():
    """A bump inside every sample's blind radius is invisible at init_res 2, and seen at 32."""
    center = np.array([[-2.0, -2.0], [0.0, 0.0], [-2.0, 0.0]]).mean(axis=0)

    def bumped(p):
        r2 = ((p - center) ** 2).sum(axis=-1)
        bump = (2.0 * np.exp(-r2 / (2 * 0.08**2)))[:, None] * np.array([1.0, 0.0])
        return p * 0.5 + bump

    def bumped_jacobian(p):
        d = p - center
        g = 2.0 * np.exp(-(d**2).sum(axis=-1) / (2 * 0.08**2))
        grad = -(g / 0.08**2)[:, None] * d
        return 0.5 * np.eye(2) + np.array([1.0, 0.0])[None, :, None] * grad[:, None, :]

    coarse, _ = build(bumped, bumped_jacobian, init_res=2, min_img_sep=0.05)
    fine, _ = build(bumped, bumped_jacobian, init_res=32, min_img_sep=0.05)
    assert fine.lattice.level - 1 >= 1
    assert (to_np(coarse.origin_level) == 0).all()
    assert (to_np(coarse.origin_status) == LEAF_CONVERGED).all()
    assert (to_np(fine.origin_level) > 0).any()
