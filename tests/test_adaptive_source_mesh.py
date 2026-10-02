import warnings

import numpy as np
import pytest

from caustics.backend_obj import backend
from caustics.lenses.func.adaptive import build_adaptive_mesh, child_matrix_tables
from caustics.lenses.func.adaptive.geometry import _CHILD_VERTEX_INDEX_TABLE
from caustics.lenses.func.adaptive.lattice import (
    initial_triangles,
    lattice_on_boundary,
    lattice_xy,
    make_lattice,
    midpoint_ij,
)
from caustics.lenses.func.adaptive.magnification import (
    mesh_total_magnification,
    sheet_edges,
)
from caustics.lenses.func.adaptive.source_mesh import (
    build_magnification_mesh,
    level_cells,
    segment_cells,
    triangle_cells,
)


def to_np(x):
    return backend.to_numpy(x)


def _arr(x):
    return backend.as_array(np.asarray(x, dtype=np.float64), dtype=backend.float64)


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


def _stack_2x2(a, b, c, d):
    return backend.stack(
        (backend.stack((a, b), dim=-1), backend.stack((c, d), dim=-1)), dim=-2
    )


def _row_fold(x, y):
    """``det A = 1 - 2y``: a fold on ``y = 0.5``, its caustic ``beta_y = 1/4``.

    Below the caustic ``mu_tot = 2 / sqrt(1 - 4 beta_y)``, so ``mu_tot >= 4``
    on the strip ``3/16 <= beta_y <= 1/4``.
    """
    return x * 1.0, y - y * y


def _row_fold_jacobian(x, y):
    one, zero = backend.ones_like(x), backend.zeros_like(x)
    return _stack_2x2(one, zero, zero, 1.0 - 2.0 * y)


@pytest.fixture(scope="module")
def lens_mesh():
    # Lens plane x in [-1, 1], y in [-0.5, 1.5]: source x in [-1, 1],
    # beta_y in [-0.75, 0.25].
    return build_adaptive_mesh(_row_fold, _row_fold_jacobian, 2.0, 8, 0.02, y0=0.5)


@pytest.fixture(scope="module")
def mag(lens_mesh):
    # A window clear of the fov's image, which runs along x = +-1 and
    # beta_y = -0.75.
    return build_magnification_mesh(
        lens_mesh, 8, 0.02, mu_min=4.0, fov=1.0, x0=0.0, y0=0.0
    )


def test_vertex_values_are_the_sampler_at_the_vertices(lens_mesh, mag):
    mu, n = mesh_total_magnification(lens_mesh, mag.vertices)
    assert np.array_equal(to_np(mag.mu), to_np(mu))
    assert np.array_equal(to_np(mag.n), to_np(n))


def test_the_closed_leaves_are_conforming_and_positively_oriented(mag):
    leaves, xy = to_np(mag.leaves), to_np(mag.vertices)
    e1 = xy[leaves[:, 1]] - xy[leaves[:, 0]]
    e2 = xy[leaves[:, 2]] - xy[leaves[:, 0]]
    assert (e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0] > 0).all()
    a, b = leaves.reshape(-1), leaves[:, [1, 2, 0]].reshape(-1)
    directed = set(zip(a.tolist(), b.tolist()))
    assert len(directed) == a.shape[0]
    on_edge = to_np(lattice_on_boundary(mag.lattice, mag.vertices_ij))
    for p, q in directed:
        if (q, p) not in directed:
            assert on_edge[p] and on_edge[q]


def test_every_leaf_straddling_a_target_is_at_the_floor(mag):
    inside = to_np(mag.mu)[to_np(mag.leaves)] >= 4.0
    mixed = inside.any(axis=1) & ~inside.all(axis=1)
    assert mixed.any()
    assert (to_np(mag.leaf_level)[mixed] == mag.max_level).all()


def test_no_leaf_below_the_floor_is_touched_by_a_band_edge(lens_mesh, mag):
    edges = sheet_edges(lens_mesh)
    inner = _arr(to_np(edges.source)[~to_np(edges.fov)])
    level = to_np(mag.leaf_level)
    ij = mag.vertices_ij[mag.leaves]
    for d in sorted(set(level.tolist()) - {mag.max_level}):
        at = backend.as_array(np.flatnonzero(level == d), dtype=backend.int64)
        cells = to_np(level_cells(mag.lattice, inner, 0.0, d))
        own = to_np(triangle_cells(mag.lattice, ij[at], d))
        assert not np.isin(own, cells).any()


def test_only_leaves_the_fov_image_touches_are_incomplete(lens_mesh, mag):
    assert not to_np(mag.incomplete).any()
    wide = build_magnification_mesh(lens_mesh, 8, 0.05, mu_min=4.0)
    inc = to_np(wide.incomplete)
    assert inc.any()
    edges = sheet_edges(lens_mesh)
    outer = _arr(to_np(edges.source)[to_np(edges.fov)])
    level = to_np(wide.leaf_level)
    ij = wide.vertices_ij[wide.leaves]
    for d in sorted(set(level[inc].tolist())):
        at = backend.as_array(np.flatnonzero(inc & (level == d)), dtype=backend.int64)
        cells = to_np(level_cells(wide.lattice, outer, 1e-6, d))
        assert np.isin(to_np(triangle_cells(wide.lattice, ij[at], d)), cells).all()


def test_the_build_does_not_depend_on_the_batch_size(lens_mesh, mag):
    other = build_magnification_mesh(
        lens_mesh, 8, 0.02, mu_min=4.0, fov=1.0, x0=0.0, y0=0.0, batch_size=37
    )
    for name in (
        "vertices",
        "vertices_ij",
        "mu",
        "n",
        "leaves",
        "leaf_level",
        "incomplete",
    ):
        assert np.array_equal(
            to_np(getattr(other, name)), to_np(getattr(mag, name))
        ), name


def test_sweep_mode_leaves_meet_the_log1p_tolerance(lens_mesh):
    rtol = 0.05
    sweep = build_magnification_mesh(
        lens_mesh, 8, 0.02, rtol=rtol, fov=1.0, x0=0.0, y0=0.0
    )
    lat = sweep.lattice
    leaves, level = to_np(sweep.leaves), to_np(sweep.leaf_level)
    ij = to_np(sweep.vertices_ij)[leaves]
    e1, e2 = ij[:, 1] - ij[:, 0], ij[:, 2] - ij[:, 0]
    side = 1 << (lat.level - level)
    # Whole lattice triangles below the floor: origins closure left as they were.
    whole = (e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0] == side * side) & (
        level < sweep.max_level
    )
    assert whole.any()
    mid_ij = (ij[whole][:, [1, 2, 0]] + ij[whole][:, [2, 0, 1]]) // 2
    xy = lattice_xy(lat, backend.as_array(mid_ij.reshape(-1, 2), dtype=backend.int64))
    mu_m, _ = mesh_total_magnification(lens_mesh, xy)
    # The backend's own `log`, as the build's, so no ulp separates the two.
    L_m = to_np(backend.log(1.0 + mu_m)).reshape(-1, 3)
    L_v = to_np(backend.log(1.0 + sweep.mu))[leaves[whole]]
    dev = np.abs(L_m - 0.5 * (L_v[:, [1, 2, 0]] + L_v[:, [2, 0, 1]]))
    assert (dev < rtol).all()


def test_a_capped_depth_warns(lens_mesh):
    with pytest.warns(UserWarning, match="depth-limited"):
        build_magnification_mesh(
            lens_mesh, 8, 0.02, max_depth=1, mu_min=4.0, fov=1.0, x0=0.0, y0=0.0
        )


def test_a_threshold_above_the_band_limit_warns(lens_mesh):
    with pytest.warns(UserWarning, match="exceeds mu_band"):
        build_magnification_mesh(
            lens_mesh, 8, 0.05, mu_min=1e6, fov=1.0, x0=0.0, y0=0.0
        )


def test_a_resolved_build_raises_neither_warning(lens_mesh):
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        build_magnification_mesh(
            lens_mesh, 8, 0.05, mu_min=4.0, fov=1.0, x0=0.0, y0=0.0
        )
    messages = [str(w.message) for w in record]
    assert not [m for m in messages if "depth-limited" in m or "mu_band" in m]


@pytest.mark.parametrize(
    "mu_min, expected",
    [(4, (4.0,)), ([4.0, 8.0], (4.0, 8.0)), (np.float64(4.0), (4.0,)), (None, ())],
)
def test_mu_min_is_stored_as_a_tuple_of_floats(lens_mesh, mu_min, expected):
    mag = build_magnification_mesh(
        lens_mesh, 4, 0.1, mu_min=mu_min, fov=1.0, x0=0.0, y0=0.0
    )
    assert mag.mu_min == expected


def test_a_non_positive_src_tol_is_rejected(lens_mesh):
    with pytest.raises(ValueError, match="src_tol must be positive"):
        build_magnification_mesh(lens_mesh, 8, 0.0)
