"""The multi-level source-plane index and its lookups."""

import numpy as np
import pytest

from caustics.backend_obj import backend
from caustics.lenses.func.adaptive.mesh_backend import mesh_backend, to_mesh
from caustics.lenses.func.adaptive.criterion import LEAF_CONVERGED
from caustics.lenses.func.adaptive.geometry import triangle_weights
from caustics.lenses.func.adaptive.index import (
    CELLS_PER_BOX,
    KEY_BITS,
    as_points,
    build_index,
    index_cells,
    index_hits,
)

from adaptive_maps import (
    assert_hits_equal,
    assert_same,
    box_hits,
    brute_hits,
    build,
    cell_meets_triangle,
    collapse,
    collapse_jacobian,
    f64,
    finite_rows,
    i64,
    index_cell,
    index_cell_ranges,
    index_point_cells,
    SIE_LIKE_LENS,
    to_np,
)
from caustics.cosmology import FlatLambdaCDM
from caustics.lenses import Point
from caustics.lenses.func.adaptive.lens_mesh import build_lens_mesh, leaf_grow
from caustics.lenses.func.adaptive import index as index_module


def test_build_index_of_an_empty_leaf_set_has_no_levels_and_finds_nothing():
    vs, leaves = f64(np.zeros((3, 2))), i64(np.zeros((0, 3)))
    idx = build_index(vs, leaves, i64(np.zeros(0)))
    assert tuple(idx.levels.shape) == (0,) and tuple(idx.keys.shape) == (0,)
    assert to_np(idx.offsets).tolist() == [0] and tuple(idx.leaves.shape) == (0,)
    start, count = index_cells(idx, f64([[0.5, 0.5]]))
    assert tuple(start.shape) == tuple(count.shape) == (1, 0)
    hits = index_hits(idx, vs, leaves, f64([[0.5, 0.5]]))
    assert [tuple(a.shape) for a in hits] == [(0,), (0,), (0, 3)]


@pytest.mark.skipif(
    backend.backend != "torch", reason="needs torch's meta default device"
)
@pytest.mark.parametrize(
    ("n_rows", "grown"),
    [(0, False), (60, False), (60, True)],
    ids=["empty", "filled", "grown"],
)
def test_build_index_allocates_on_its_inputs_device_not_the_default_one(n_rows, grown):
    """The index is built where its inputs live, which need not be the default device.

    With the default device set to ``meta``, anything ``build_index``
    allocates without its inputs' device lands there and fails to mix.
    """
    import torch

    rng = np.random.default_rng(13)
    vs = f64(rng.normal(size=(50, 2)))
    leaves = i64(rng.integers(0, 50, (60, 3)))
    rows = i64(np.arange(n_rows))
    grow = f64(rng.uniform(0, 0.1, n_rows)) if grown else None
    with torch.device("meta"):
        idx = build_index(vs, leaves, rows, grow)
    for name in ("lo", "hi", "fine", "levels", "keys", "offsets", "leaves"):
        assert getattr(idx, name).device == vs.device, name


def test_as_points_flattens_any_shape_into_float64_rows():
    got = as_points(2.0, 3.0, None)
    assert got.dtype == mesh_backend.float64 and to_np(got).tolist() == [[2.0, 3.0]]
    got = as_points(f64([[0.0, 1.0]]), f64([[2.0, 3.0]]), None)
    assert to_np(got).tolist() == [[0.0, 2.0], [1.0, 3.0]]
    assert as_points(f64([]), f64([]), None).shape == (0, 2)


def test_build_index_with_a_zero_grow_is_the_plain_index():
    rng = np.random.default_rng(11)
    vs = f64(rng.uniform(-1, 1, (90, 2)))
    leaves = i64(np.arange(90).reshape(30, 3))
    rows = i64(np.arange(30))
    assert_same(
        build_index(vs, leaves, rows, f64(np.zeros(30))), build_index(vs, leaves, rows)
    )


def test_index_hits_with_grow_returns_a_point_within_grow_of_a_triangle_and_not_beyond():
    vs = f64([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [5.0, 5.0], [6.0, 5.0], [5.0, 6.0]])
    leaves = i64([[0, 1, 2], [3, 4, 5]])
    grow = f64([0.2, 0.0])
    idx = build_index(vs, leaves, i64([0, 1]), grow)
    beta = f64([[0.5, -0.1], [0.5, -0.3], [0.25, 0.25], [5.5, 4.9]])
    qidx, tri, w = index_hits(idx, vs, leaves, beta, grow)
    assert list(zip(to_np(qidx).tolist(), to_np(tri).tolist())) == [(0, 0), (2, 0)]
    expected = triangle_weights(vs[leaves[tri]], beta[qidx])
    assert np.array_equal(to_np(w), to_np(expected))
    plain_q, plain_t, _ = index_hits(idx, vs, leaves, beta)
    assert list(zip(to_np(plain_q).tolist(), to_np(plain_t).tolist())) == [(2, 0)]


def _scattered_triangles(seed, n=400):
    """``n`` triangles with sides from 1e-9 to 1e2 arcsec, the first five moved out to about 1e6."""
    rng = np.random.default_rng(seed)
    size = 10.0 ** rng.uniform(-9, 2, n)
    corner = rng.uniform(-1, 1, (n, 2))
    corner[:5] *= 1e6
    return corner[:, None, :] + size[:, None, None] * rng.uniform(-1, 1, (n, 3, 2))


def _as_mesh(tri):
    """``tri`` ``(n, 3, 2)`` as vertices, triangles and all their rows."""
    n = tri.shape[0]
    return (
        f64(tri.reshape(-1, 2)),
        i64(np.arange(3 * n).reshape(n, 3)),
        i64(np.arange(n)),
    )


def test_index_hits_equal_brute_force_across_eleven_orders_of_magnitude():
    tri = _scattered_triangles(9)
    vs, leaves, rows = _as_mesh(tri)
    rng = np.random.default_rng(10)
    n = tri.shape[0]
    grow = f64(np.where(rng.uniform(size=n) < 0.2, 10.0 ** rng.uniform(-9, 0, n), 0.0))
    inner = tri[:, 0] + 0.3 * (tri[:, 1] - tri[:, 0]) + 0.3 * (tri[:, 2] - tri[:, 0])
    beta = f64(np.concatenate([inner, tri[:, 1], rng.uniform(-1, 1, (200, 2))]))
    idx = build_index(vs, leaves, rows, grow)
    assert_hits_equal(
        index_hits(idx, vs, leaves, beta, grow),
        brute_hits(vs, leaves, rows, beta, grow),
    )


def test_index_hits_skip_nan_infinite_and_far_points_and_take_empty_input():
    rng = np.random.default_rng(14)
    tri = rng.uniform(-1, 1, (40, 1, 2)) + 0.2 * rng.uniform(-1, 1, (40, 3, 2))
    vs, leaves, rows = _as_mesh(tri)
    idx = build_index(vs, leaves, rows)
    beta = f64(
        [[np.nan, 0.0], [np.inf, 0.0], [0.0, -np.inf], [1e9, 1e9], tri[3].mean(axis=0)]
    )
    qidx, hit, w = index_hits(idx, vs, leaves, beta)
    assert set(to_np(qidx).tolist()) == {4}
    want_q, want_t, want_w = brute_hits(vs, leaves, rows, beta[4:])
    assert want_t.size > 0
    assert_hits_equal((qidx, hit, w), (want_q + 4, want_t, want_w))
    empty = index_hits(idx, vs, leaves, f64(np.zeros((0, 2))))
    assert [tuple(a.shape) for a in empty] == [(0,), (0,), (0, 3)]


def test_build_index_registers_each_row_at_one_level_in_cells_of_its_box_only():
    tri = _scattered_triangles(21)
    n = tri.shape[0]
    rng = np.random.default_rng(22)
    grow = np.where(rng.uniform(size=n) < 0.2, 10.0 ** rng.uniform(-9, 0, n), 0.0)
    vs, leaves, rows = _as_mesh(tri)
    idx = build_index(vs, leaves, rows, f64(grow))
    level, i0, i1 = index_cell_ranges(idx, tri, grow)
    assert np.unique(level).size > 5, "fixture must spread over many levels"
    assert ((i1 - i0 + 1) <= CELLS_PER_BOX + 2).all()
    keys, offsets = to_np(idx.keys), to_np(idx.offsets)
    key_of_entry = np.repeat(keys, np.diff(offsets))
    row_of_entry = to_np(idx.leaves)
    for r in range(n):
        ix, iy = np.meshgrid(
            np.arange(i0[r, 0], i1[r, 0] + 1),
            np.arange(i0[r, 1], i1[r, 1] + 1),
            indexing="ij",
        )
        want = (level[r] << (2 * KEY_BITS)) | (ix.ravel() << KEY_BITS) | iy.ravel()
        got = key_of_entry[row_of_entry == r]
        assert got.size > 0, r
        assert np.unique(got).size == got.size, r
        assert np.isin(got, want).all(), r


def test_build_index_keys_ascend_and_give_its_levels_and_fine_side():
    vs, leaves, rows = _as_mesh(_scattered_triangles(23))
    idx = build_index(vs, leaves, rows)
    keys, offsets = to_np(idx.keys), to_np(idx.offsets)
    assert (np.diff(keys) > 0).all()
    assert offsets[0] == 0 and offsets[-1] == to_np(idx.leaves).size
    assert (np.diff(offsets) > 0).all()
    assert to_np(idx.levels).tolist() == np.unique(keys >> (2 * KEY_BITS)).tolist()
    entries = to_np(idx.leaves)
    assert all(
        (np.diff(entries[a:b]) > 0).all() for a, b in zip(offsets[:-1], offsets[1:])
    )
    assert (
        np.diff(offsets) > 1
    ).any(), "fixture must put several triangles in one cell"
    extent = (to_np(idx.hi) - to_np(idx.lo)).max()
    assert float(to_np(idx.fine)) == extent / 2**KEY_BITS


def test_index_cells_gives_the_cell_each_point_falls_in_at_every_level():
    tri = _scattered_triangles(27)
    vs, leaves, rows = _as_mesh(tri)
    idx = build_index(vs, leaves, rows)
    rng = np.random.default_rng(28)
    beta = np.concatenate(
        [tri[:, 0], rng.uniform(-1, 1, (100, 2)), [[np.nan, 0.0], [1e9, 1e9]]]
    )
    start, count = index_cells(idx, f64(beta))
    start, count = to_np(start), to_np(count)
    levels, entries = to_np(idx.levels), to_np(idx.leaves)
    assert start.shape == count.shape == (beta.shape[0], levels.size)
    lo, hi = to_np(idx.lo), to_np(idx.hi)
    for b, point in enumerate(beta):
        inside = bool(np.all((point >= lo) & (point <= hi)))
        for j, level in enumerate(levels):
            want = np.zeros(0, dtype=np.int64)
            if inside:
                ix, iy = index_point_cells(idx, [level], point[None])[0]
                want = index_cell(idx, level, ix, iy)
            got = entries[start[b, j] : start[b, j] + count[b, j]]
            assert np.array_equal(got, want), (b, level)


def test_index_hits_find_points_on_the_cell_lines_of_every_level():
    """Points snapped onto each level's cell lines, and ``hi`` itself, are found as brute force finds them."""
    tri = _scattered_triangles(25)[5:]
    vs, leaves, rows = _as_mesh(tri)
    idx = build_index(vs, leaves, rows)
    lo, fine = to_np(idx.lo), float(to_np(idx.fine))
    points = [to_np(idx.hi)[None]]
    for level in to_np(idx.levels):
        side = fine * 2.0**level
        snapped = lo + np.floor((tri[::20, 0] - lo) / side) * side
        points += [snapped, np.stack([snapped[:, 0], tri[::20, 0, 1]], axis=-1)]
    beta = f64(np.concatenate(points))
    assert_hits_equal(
        index_hits(idx, vs, leaves, beta), brute_hits(vs, leaves, rows, beta)
    )


def _grown(seed, n, share=0.2):
    rng = np.random.default_rng(seed)
    return np.where(rng.uniform(size=n) < share, 10.0 ** rng.uniform(-9, 0, n), 0.0)


def _edge_and_cell_line_points(idx, tri, grow):
    """The triangles' vertices and the points where their edges cross their level's cell lines, each also moved one float every way."""
    level, i0, i1 = index_cell_ranges(idx, tri, grow)
    lo, fine = to_np(idx.lo), float(to_np(idx.fine))
    points = [tri.reshape(-1, 2)]
    for t in range(tri.shape[0]):
        side = fine * 2.0 ** level[t]
        for axis in (0, 1):
            lines = lo[axis] + side * np.arange(i0[t, axis], i1[t, axis] + 2)
            for e in range(3):
                p, q = tri[t, e], tri[t, (e + 1) % 3]
                if p[axis] == q[axis]:
                    continue
                s = (lines - p[axis]) / (q[axis] - p[axis])
                on = (s >= 0) & (s <= 1)
                cross = p + s[on, None] * (q - p)
                cross[:, axis] = lines[on]
                points.append(cross)
    points = np.concatenate(points)
    moved = [points]
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1)):
        p = points.copy()
        if dx:
            p[:, 0] = np.nextafter(p[:, 0], dx * np.inf)
        if dy:
            p[:, 1] = np.nextafter(p[:, 1], dy * np.inf)
        moved.append(p)
    return np.concatenate(moved)


@pytest.mark.parametrize("grown", [False, True], ids=["plain", "grown"])
def test_index_hits_equal_brute_force_on_edges_and_cell_lines(grown):
    tri = _scattered_triangles(31, n=120)
    grow = _grown(32, 120) if grown else None
    vs, leaves, rows = _as_mesh(tri)
    idx = build_index(vs, leaves, rows, None if grow is None else f64(grow))
    beta = f64(_edge_and_cell_line_points(idx, tri, grow))
    g = None if grow is None else f64(grow)
    assert_hits_equal(
        index_hits(idx, vs, leaves, beta, g), brute_hits(vs, leaves, rows, beta, g)
    )


def _slivers(seed, n=60):
    """Thin triangles along the diagonal of their boxes, counter-clockwise."""
    rng = np.random.default_rng(seed)
    a = rng.uniform(-1, 1, (n, 2))
    b = a + rng.uniform(0.2, 1.0, (n, 1)) * np.array([1.0, 1.0])
    c = b + rng.uniform(0.01, 0.02, (n, 1)) * np.array([0.0, 1.0])
    return np.stack([a, b, c], axis=1)


def test_a_leaf_is_registered_only_in_cells_of_its_box_that_it_reaches():
    tri = _slivers(33)
    n = tri.shape[0]
    vs, leaves, rows = _as_mesh(tri)
    idx = build_index(vs, leaves, rows)
    level, i0, i1 = index_cell_ranges(idx, tri)
    keys, offsets = to_np(idx.keys), to_np(idx.offsets)
    key_of_entry = np.repeat(keys, np.diff(offsets))
    row_of_entry = to_np(idx.leaves)
    box_cells = np.prod(i1 - i0 + 1, axis=1)
    assert row_of_entry.size < 0.5 * box_cells.sum()
    lo, fine = to_np(idx.lo), float(to_np(idx.fine))
    mask = (1 << KEY_BITS) - 1
    for r in range(n):
        got = key_of_entry[row_of_entry == r]
        assert got.size > 0 and (got >> (2 * KEY_BITS) == level[r]).all()
        cell = np.stack([(got >> KEY_BITS) & mask, got & mask], axis=1)
        assert ((cell >= i0[r]) & (cell <= i1[r])).all()
        side = fine * 2.0 ** level[r]
        margin = 2 * index_module.PAD * side
        for c in cell:
            assert cell_meets_triangle(
                lo + c * side - margin, lo + (c + 1) * side + margin, tri[r]
            )


def _degenerate():
    """A vertical and a horizontal segment, a diagonal one, a point, and two proper triangles beside them."""
    return np.array(
        [
            [[0.0, 0.0], [0.0, 1.0], [0.0, 0.4]],
            [[0.2, 0.3], [1.1, 0.3], [0.6, 0.3]],
            [[-1.0, -1.0], [0.0, 0.0], [-0.5, -0.5]],
            [[0.7, -0.6], [0.7, -0.6], [0.7, -0.6]],
            [[-0.9, 0.1], [-0.2, 0.2], [-0.6, 0.9]],
            [[0.3, -0.9], [0.9, -0.8], [0.5, -0.2]],
        ]
    )


@pytest.mark.parametrize(
    "offset", [0.0, 1e7], ids=["at the origin", "1e6 extents away"]
)
@pytest.mark.parametrize("grown", [False, True], ids=["plain", "grown"])
def test_index_hits_on_degenerate_triangles_are_those_of_every_cell_of_their_boxes(
    grown, offset
):
    """Brute force finds more here than any index can (:func:`box_hits`): an index of every box cell finds these."""
    tri = _degenerate() + np.array([offset, -3 * offset])
    grow = np.array([0.0, 0.05, 0.0, 0.1, 0.02, 0.0]) if grown else None
    vs, leaves, rows = _as_mesh(tri)
    g = None if grow is None else f64(grow)
    idx = build_index(vs, leaves, rows, g)
    beta = f64(_edge_and_cell_line_points(idx, tri, grow))
    want = box_hits(idx, vs, leaves, rows, beta, g)
    assert np.isin(np.arange(4), want[1]).all(), "every degenerate triangle has a hit"
    assert_hits_equal(index_hits(idx, vs, leaves, beta, g), want)


def test_index_hits_find_sources_on_the_index_s_outer_edge():
    tri = _scattered_triangles(34, n=80)
    vs, leaves, rows = _as_mesh(tri)
    idx = build_index(vs, leaves, rows)
    lo, hi = to_np(idx.lo), to_np(idx.hi)
    flat = tri.reshape(-1, 2)
    on_edge = flat[((flat == lo) | (flat == hi)).any(axis=1)]
    assert on_edge.shape[0] >= 2
    beta = f64(np.concatenate([on_edge, [lo, hi, [lo[0], hi[1]], [hi[0], lo[1]]]]))
    want = box_hits(idx, vs, leaves, rows, beta)
    assert (want[0] < on_edge.shape[0]).any(), "a source on the edge has a hit"
    assert_hits_equal(index_hits(idx, vs, leaves, beta), want)


def test_build_index_of_float32_vertices_is_that_of_their_float64_values():
    vs, leaves, rows = _as_mesh(_scattered_triangles(35, n=80).astype(np.float32))
    vs32 = mesh_backend.to(vs, dtype=mesh_backend.float32)
    assert_same(build_index(vs32, leaves, rows), build_index(vs, leaves, rows))


def test_build_index_of_vertices_that_carry_gradients_is_that_of_the_detached_ones():
    vs, leaves, rows = _as_mesh(_scattered_triangles(36, n=80))
    grow = f64(_grown(37, 80))
    tracked = build_index(vs.clone().requires_grad_(True), leaves, rows, grow)
    assert not tracked.keys.requires_grad and not tracked.leaves.requires_grad
    plain = build_index(vs, leaves, rows, grow)
    for name in ("levels", "keys", "offsets", "leaves"):
        assert np.array_equal(
            to_np(getattr(tracked, name)), to_np(getattr(plain, name))
        )


def test_a_kappa_one_sheet_s_single_source_point_hits_every_leaf():
    """Every source vertex is one point, so the index has no extent; that point lies in every leaf."""
    mesh, _ = build(collapse, collapse_jacobian, min_img_sep=1.0)
    vs = mesh.vertices_source
    assert (
        np.unique(to_np(vs), axis=0).shape[0] == 1
    ), "fixture must collapse every vertex"
    _, hit, _ = index_hits(mesh.index, vs, mesh.leaves, vs[:1])
    assert to_np(hit).tolist() == list(range(mesh.leaves.shape[0]))


BUILD = dict(fov=6.0, init_res=4, min_img_sep=0.1, max_depth=6)


@pytest.fixture(scope="module")
def mesh():
    return to_mesh(build_lens_mesh(SIE_LIKE_LENS.raytrace, **BUILD))


@pytest.fixture(scope="module")
def beta():
    rng = np.random.default_rng(21)
    return mesh_backend.as_array(
        rng.uniform(-1.5, 1.5, (64, 2)), dtype=mesh_backend.float64
    )


@pytest.fixture(scope="module")
def point_mesh():
    """A point mass of Einstein radius 1 with its center a hole: source triangles from about 1e-12 to 1e4 arcsec^2."""
    lens = Point(
        name="pt",
        cosmology=FlatLambdaCDM(name="cosmo"),
        z_l=0.5,
        z_s=1.5,
        x0=0.0,
        y0=0.0,
        Rein=1.0,
        s=0.0,
    )
    mesh = build_lens_mesh(
        lens.raytrace,
        fov=5.0,
        init_res=32,
        min_img_sep=1e-2,
        centers=[[0.0, 0.0]],
    )
    return to_mesh(mesh)


def _hits_against_brute_force(mesh, points):
    """``index_hits`` on ``mesh`` at ``points``, asserted equal to brute force over every finite leaf."""
    grow = leaf_grow(mesh)
    rows = i64(finite_rows(mesh))
    got = index_hits(mesh.index, mesh.vertices_source, mesh.leaves, points, grow)
    assert_hits_equal(
        got, brute_hits(mesh.vertices_source, mesh.leaves, rows, points, grow)
    )
    return got


def test_index_hits_equal_brute_force_on_the_sie_like_mesh(mesh, beta):
    vs = to_np(mesh.vertices_source)
    finite = vs[np.isfinite(vs).all(axis=1)]
    hi = to_np(mesh.index.hi)
    upper = np.array([hi, [hi[0], 0.0], [0.0, hi[1]]])
    points = f64(np.concatenate([to_np(beta), finite[::7][:40], upper]))
    qidx, _, _ = _hits_against_brute_force(mesh, points)
    assert to_np(qidx).size > 0


def test_index_hits_equal_brute_force_on_a_point_mass_mesh(point_mesh):
    rng = np.random.default_rng(31)
    vs = to_np(point_mesh.vertices_source)
    finite = vs[np.isfinite(vs).all(axis=1)]
    points = np.concatenate(
        [
            rng.uniform(-0.3, 0.3, (60, 2)),
            rng.uniform(-5.0, 5.0, (30, 2)),
            finite[rng.choice(len(finite), 20, replace=False)],
        ]
    )
    qidx, _, _ = _hits_against_brute_force(point_mesh, f64(points))
    assert to_np(qidx).size > 0


def test_index_hits_equal_brute_force_beside_the_leaves_left_out_of_a_point_mass_mesh(
    point_mesh,
):
    """Just beyond the source triangles of the leaves sharing a vertex with a non-finite one, which reach beyond them though they converge.

    Each vertex of such a triangle is pushed half its reach away from the
    triangle's centroid: outside the triangle, within its reach.
    """
    rows = finite_rows(point_mesh)
    leaves = to_np(point_mesh.leaves)
    out = np.setdiff1d(np.arange(leaves.shape[0]), rows)
    beside = rows[np.isin(leaves[rows], leaves[out]).any(axis=1)]
    status = to_np(point_mesh.origin_status)[to_np(point_mesh.leaf_origin)]
    assert (
        status[beside] == LEAF_CONVERGED
    ).any(), "fixture must converge next to the center"
    reach = to_np(leaf_grow(point_mesh))[beside]
    tri = to_np(point_mesh.vertices_source)[leaves[beside]]
    away = tri - tri.mean(axis=1, keepdims=True)
    away /= np.linalg.norm(away, axis=-1, keepdims=True)
    points = (tri + 0.5 * reach[:, None, None] * away).reshape(-1, 2)
    qidx, hit, _ = _hits_against_brute_force(point_mesh, f64(points))
    qidx, hit = to_np(qidx), to_np(hit)
    owner = np.repeat(beside, 3)
    assert all(owner[k] in hit[qidx == k] for k in range(owner.size))


def test_point_mass_candidates_average_at_most_three_times_the_box_floor(point_mesh):
    """Near a point mass's caustic, candidates per point average at most three times the floor.

    The floor is the number of grown leaf boxes containing the point, which
    no bounding-box index can go below. The single uniform grid this index
    replaced read 700 to 2,700 times the floor here, and the multi-level
    grid measured 1.3 to 2.0 times on 13 lens meshes. The bound is on the
    mean: next to the point-like caustic the floor itself reaches the
    thousands, and where the floor is small, candidates have exceeded four
    times it by up to 219.
    """
    rows = finite_rows(point_mesh)
    vs, leaves = to_np(point_mesh.vertices_source), to_np(point_mesh.leaves)
    grow = to_np(leaf_grow(point_mesh))[rows]
    tri = vs[leaves[rows]]
    box_lo, box_hi = tri.min(axis=1) - grow[:, None], tri.max(axis=1) + grow[:, None]
    beta = np.random.default_rng(41).uniform(-0.3, 0.3, (300, 2))
    floor = np.array(
        [np.all((box_lo <= p) & (p <= box_hi), axis=1).sum() for p in beta]
    )
    _, count = index_cells(point_mesh.index, f64(beta))
    candidates = to_np(count).sum(axis=1)
    assert candidates.mean() <= 3 * floor.mean()


def test_index_hits_hold_at_most_pair_cap_pairs_per_step(mesh, beta, monkeypatch):
    """Each step of a query holds at most ``PAIR_CAP`` pairs, or one point holding more, and the hits do not change."""
    grow = leaf_grow(mesh)
    whole = index_hits(mesh.index, mesh.vertices_source, mesh.leaves, beta, grow)
    steps = []
    chunk_hits = index_module._chunk_hits

    def spy(index, vertices, triangles, points, start, count, total, grow):
        steps.append((points.shape[0], total))
        return chunk_hits(index, vertices, triangles, points, start, count, total, grow)

    monkeypatch.setattr(index_module, "_chunk_hits", spy)
    over_cap = shared = False
    for cap in (7, 64):
        monkeypatch.setattr(index_module, "PAIR_CAP", cap)
        steps.clear()
        got = index_hits(mesh.index, mesh.vertices_source, mesh.leaves, beta, grow)
        for name, a, b in zip(("qidx", "tri", "w"), got, whole):
            np.testing.assert_array_equal(
                to_np(a), to_np(b), err_msg=f"{name}, cap {cap}"
            )
        assert all(total <= cap or n == 1 for n, total in steps), steps
        over_cap |= any(n == 1 and total > cap for n, total in steps)
        shared |= any(n > 1 for n, _ in steps)
    assert over_cap, "no point held more than the cap"
    assert shared, "no step held several points"


def test_build_index_holds_at_most_entry_cap_entries_per_step(mesh, monkeypatch):
    """Each step of a build holds at most ``ENTRY_CAP`` entries, or one column of cells -- one level and x -- holding more, and the index does not change."""
    rows = i64(finite_rows(mesh))
    grow = leaf_grow(mesh)[rows]
    steps = []
    step_cells = index_module._step_cells

    def spy(code, first, height, column_rows, total):
        steps.append((np.unique(to_np(code)).size, total))
        return step_cells(code, first, height, column_rows, total)

    monkeypatch.setattr(index_module, "_step_cells", spy)
    whole = build_index(mesh.vertices_source, mesh.leaves, rows, grow)
    assert len(steps) == 1, "the default cap must hold the fixture in one step"
    over_cap = shared = False
    for cap in (7, 64):
        monkeypatch.setattr(index_module, "ENTRY_CAP", cap)
        steps.clear()
        assert_same(build_index(mesh.vertices_source, mesh.leaves, rows, grow), whole)
        assert all(total <= cap or columns == 1 for columns, total in steps), steps
        over_cap |= any(columns == 1 and total > cap for columns, total in steps)
        shared |= any(columns > 1 for columns, _ in steps)
    assert over_cap, "no column held more than the cap"
    assert shared, "no step held several columns"


def test_build_index_sorts_int64_local_keys_to_the_same_index(mesh, monkeypatch):
    """With ``INT32_KEYS = 0`` every sort of a build is on int64 keys, and the index does not change."""
    rows = i64(finite_rows(mesh))
    args = (mesh.vertices_source, mesh.leaves, rows, leaf_grow(mesh)[rows])
    sorted_dtypes = []
    real_sort = mesh_backend.module.sort

    def spy(keys, *a, **kw):
        sorted_dtypes.append(keys.dtype)
        return real_sort(keys, *a, **kw)

    monkeypatch.setattr(mesh_backend.module, "sort", spy)
    whole = build_index(*args)
    assert mesh_backend.int32 in sorted_dtypes, "the default build sorts int32 keys"
    monkeypatch.setattr(index_module, "INT32_KEYS", 0)
    for cap in (index_module.ENTRY_CAP, 7):
        monkeypatch.setattr(index_module, "ENTRY_CAP", cap)
        sorted_dtypes.clear()
        assert_same(build_index(*args), whole)
        assert sorted_dtypes and set(sorted_dtypes) == {mesh_backend.int64}, cap


def test_build_index_refuses_more_rows_than_a_packed_column_holds(monkeypatch):
    vs, leaves, rows = _as_mesh(_scattered_triangles(5, n=9))
    monkeypatch.setattr(index_module, "ROW_BITS", 3)
    with pytest.raises(ValueError, match="fewer than 8 rows; got 9"):
        build_index(vs, leaves, rows)
