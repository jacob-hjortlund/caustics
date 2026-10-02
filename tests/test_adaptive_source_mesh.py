import numpy as np

from caustics.backend_obj import backend
from caustics.lenses.func.adaptive import child_matrix_tables
from caustics.lenses.func.adaptive.geometry import _CHILD_VERTEX_INDEX_TABLE
from caustics.lenses.func.adaptive.lattice import (
    initial_triangles,
    make_lattice,
    midpoint_ij,
)
from caustics.lenses.func.adaptive.source_mesh import (
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
