"""Growing a built adaptive mesh to a larger fov.

The extension's central claim is exactness: a mesh extended from fov ``A`` to
``B`` is bit for bit the mesh a fresh build of ``B`` produces on the same
lattice. The fixtures use dyadic fovs, centres and cell sizes, where a fresh
build's own lattice coincides with the extended one exactly.
"""

import warnings
from types import SimpleNamespace

import numpy as np
import pytest

from caustics.backend_obj import backend
from caustics.lenses.func import adaptive as new
from caustics.lenses.func import adaptive_critical as crit


def to_np(x):
    return backend.to_numpy(x)


def i64(x):
    return backend.as_array(np.asarray(x, dtype=np.int64), dtype=backend.int64)


def f64(x):
    return backend.as_array(np.asarray(x, dtype=np.float64), dtype=backend.float64)


def _grid(n):
    """Every point of an ``(n + 1) x (n + 1)`` lattice, shape ``((n + 1)**2, 2)``."""
    axis = np.arange(n + 1)
    return np.stack(np.meshgrid(axis, axis, indexing="ij"), axis=-1).reshape(-1, 2)


def _same_lattice(a, b):
    return (a.level, a.n, a.stride, a.scale, a.origin) == (
        b.level,
        b.n,
        b.stride,
        b.scale,
        b.origin,
    ) and np.array_equal(to_np(a.lo), to_np(b.lo))


# ---------------------------------------------------------------------------
# The lattice anchor
# ---------------------------------------------------------------------------


def test_a_fresh_lattice_places_points_exactly_as_before():
    lat = new.make_lattice(4.0, 0.3, -0.1, 3, 4)
    ij = _grid(lat.n)
    assert lat.origin == 0
    want = to_np(lat.lo) + ij.astype(np.float64) * lat.scale
    assert np.array_equal(to_np(new.lattice_xy(lat, i64(ij))), want)


def test_extend_lattice_keeps_every_old_position_bit_for_bit():
    """A non-dyadic centre on purpose: nothing may be recomputed from a new fov."""
    lat = new.make_lattice(4.0, 0.3, -0.1, 3, 4)
    ext = new.extend_lattice(lat, 2)
    pad = 2 << lat.level
    ij = _grid(lat.n)
    assert np.array_equal(
        to_np(new.lattice_xy(ext, i64(ij + pad))),
        to_np(new.lattice_xy(lat, i64(ij))),
    )


def test_extend_lattice_grows_the_extent_and_keeps_key_order():
    lat = new.make_lattice(4.0, 0.3, -0.1, 3, 4)
    ext = new.extend_lattice(lat, 2)
    pad = 2 << lat.level
    assert (ext.level, ext.scale) == (lat.level, lat.scale)
    assert ext.n == lat.n + 2 * pad
    assert ext.stride == ext.n + 1
    assert ext.origin == pad
    assert np.array_equal(to_np(ext.lo), to_np(lat.lo))
    ij = _grid(lat.n)
    old = to_np(new.lattice_key(lat, i64(ij)))
    got = to_np(new.lattice_key(ext, i64(ij + pad)))
    assert np.array_equal(np.argsort(old), np.argsort(got))


def test_an_extended_dyadic_lattice_is_the_fresh_lattice_of_the_larger_fov():
    """The premise of every equivalence test below.

    With a dyadic fov, centre and cell size, a fresh lattice over the larger
    fov places every point exactly where the extended one does.
    """
    lat = new.make_lattice(4.0, 0.5, -0.25, 8, 3)
    ext = new.extend_lattice(lat, 2)
    fresh = new.make_lattice(6.0, 0.5, -0.25, 12, 3)
    assert (ext.n, ext.stride, ext.scale) == (fresh.n, fresh.stride, fresh.scale)
    ij = i64(_grid(ext.n))
    assert np.array_equal(
        to_np(new.lattice_xy(ext, ij)), to_np(new.lattice_xy(fresh, ij))
    )


def test_extend_lattice_chains_and_accepts_zero():
    lat = new.make_lattice(4.0, 0.0, 0.0, 2, 3)
    assert _same_lattice(new.extend_lattice(lat, 0), lat)
    assert _same_lattice(
        new.extend_lattice(new.extend_lattice(lat, 1), 2), new.extend_lattice(lat, 3)
    )


def test_extend_lattice_rejects_a_negative_k():
    with pytest.raises(ValueError, match="non-negative"):
        new.extend_lattice(new.make_lattice(4.0, 0.0, 0.0, 2, 3), -1)


def test_check_lattice_keys_rejects_exactly_the_lattices_int64_cannot_key():
    """``45 * 2**26 + 1`` points per axis still key in int64; ``46 * 2**26 + 1`` do not."""
    new.check_lattice_keys(45, 25, "unused")
    with pytest.raises(ValueError, match="lattice too fine.*Rebuild coarser"):
        new.check_lattice_keys(46, 25, "Rebuild coarser.")


# ---------------------------------------------------------------------------
# Lenses and maps
# ---------------------------------------------------------------------------


def recording_lens(fn, jac, out_dtype=np.float64):
    """A lens over numpy maps that records every point each method is called on.

    ``fn`` maps ``(N, 2) -> (N, 2)`` and ``jac`` gives its ``(N, 2, 2)``
    Jacobian. ``raytrace`` returns ``out_dtype``, so a float32 lens can be
    simulated; the Jacobian is always float64.
    """
    calls = {"raytrace": [], "jacobian": []}

    def raytrace(x, y):
        xy = np.stack([to_np(x), to_np(y)], axis=-1)
        calls["raytrace"].append(xy)
        out = fn(xy).astype(out_dtype)
        return backend.as_array(out[:, 0]), backend.as_array(out[:, 1])

    def jacobian_lens_equation(x, y):
        xy = np.stack([to_np(x), to_np(y)], axis=-1)
        calls["jacobian"].append(xy)
        return backend.as_array(jac(xy), dtype=backend.float64)

    lens = SimpleNamespace(
        raytrace=raytrace, jacobian_lens_equation=jacobian_lens_equation
    )
    return lens, calls


def called_at(calls, method):
    """Every point ``method`` was called on, stacked, shape ``(K, 2)``."""
    return np.concatenate(calls[method], axis=0) if calls[method] else np.zeros((0, 2))


def build(fn, jac, fov, init_res, min_img_sep, **kw):
    lens, calls = recording_lens(fn, jac, kw.pop("out_dtype", np.float64))
    mesh = new.build_adaptive_mesh(lens, fov, init_res, min_img_sep, **kw)
    return mesh, lens, calls


def localised_fold(p):
    """Affine outside ``|y| < 0.5``; curved inside, with a fold at ``y = -0.3``."""
    y = p[:, 1]
    bend = np.where(np.abs(y) < 0.5, y**2 - 0.25, 0.0)
    return np.stack([p[:, 0], 0.6 * y + bend], axis=-1)


def localised_fold_jacobian(p):
    y = p[:, 1]
    J = np.zeros((p.shape[0], 2, 2))
    J[:, 0, 0] = 1.0
    J[:, 1, 1] = 0.6 + np.where(np.abs(y) < 0.5, 2.0 * y, 0.0)
    return J


def row_fold(p):
    """``(x, y) -> (x, y - y**2)``: ``det A = 1 - 2y`` is exactly zero on the lattice row ``y = 0.5``."""
    return np.stack([p[:, 0], p[:, 1] - p[:, 1] ** 2], axis=-1)


def row_fold_jacobian(p):
    J = np.zeros((p.shape[0], 2, 2))
    J[:, 0, 0] = 1.0
    J[:, 1, 1] = 1.0 - 2.0 * p[:, 1]
    return J


def sie_like(p):
    """A cored isothermal sphere: tangential curve at radius ~1.18, radial at ~0.32."""
    r = np.sqrt(p[:, 0] ** 2 + p[:, 1] ** 2 + 0.05)
    return p - 1.2 * p / r[:, None]


def sie_like_jacobian(p):
    x, y = p[:, 0], p[:, 1]
    r = np.sqrt(x * x + y * y + 0.05)
    k = 1.2 / r**3
    J = np.empty((p.shape[0], 2, 2))
    J[:, 0, 0] = 1.0 - 1.2 / r + k * x * x
    J[:, 0, 1] = k * x * y
    J[:, 1, 0] = k * x * y
    J[:, 1, 1] = 1.0 - 1.2 / r + k * y * y
    return J


AFFINE = np.array([[0.7, 0.1], [-0.2, 0.9]])


def affine(p):
    return p @ AFFINE.T


def affine_jacobian(p):
    return np.tile(AFFINE, (p.shape[0], 1, 1))


def broken_where(fn, jac, where, value=np.nan):
    """``fn`` and ``jac`` with ``value`` wherever ``where(p)`` holds."""

    def broken(p):
        out = fn(p)
        out[where(p)] = value
        return out

    def broken_jacobian(p):
        J = jac(p)
        J[where(p)] = value
        return J

    return broken, broken_jacobian


def seam_fold(seam):
    """Curved, with a fold, just left of ``x = seam``; affine from ``seam`` on.

    ``x' = 0.6 x - 5 u**3 - 5.4 max(x - seam, 0)`` with
    ``u = clip(x - seam + 0.6, 0, 0.6)``: affine for ``x <= seam - 0.6``,
    curved on ``[seam - 0.6, seam]`` with ``det A = 0.6 - 15 u**2`` vanishing
    at ``x = seam - 0.4``, and continued affinely (``C1`` at ``seam``) with
    slope ``-4.8`` beyond. Triangles right of ``seam`` converge at level 0
    beside deep leaves left of it -- the configuration only `balance` can
    settle when ``seam`` is where a refinement pass stops.
    """

    def fn(p):
        x = p[:, 0]
        u = np.clip(x - (seam - 0.6), 0.0, 0.6)
        tail = np.maximum(x - seam, 0.0)
        return np.stack([0.6 * x - 5.0 * u**3 - 5.4 * tail, p[:, 1]], axis=-1)

    def jac(p):
        x = p[:, 0]
        u = np.clip(x - (seam - 0.6), 0.0, 0.6)
        J = np.zeros((p.shape[0], 2, 2))
        J[:, 0, 0] = np.where(x > seam, -4.8, 0.6 - 15.0 * u**2)
        J[:, 1, 1] = 1.0
        return J

    return fn, jac


def mirrored(fn, jac, seam):
    """``fn`` reflected in the line ``x = seam``."""

    def reflect(p):
        q = p.copy()
        q[:, 0] = 2.0 * seam - q[:, 0]
        return q

    def fn_r(p):
        return fn(reflect(p))

    def jac_r(p):
        J = jac(reflect(p))
        J[:, :, 0] = -J[:, :, 0]
        return J

    return fn_r, jac_r


def ring_fold(seam):
    """`seam_fold` mirrored: affine up to ``x = seam``, curved and folded just past it.

    A mesh ending at ``x = seam`` is level 0 along that edge, and the ring an
    extension adds refines deep right beside it, so the extension must
    force-split old leaves -- the other direction of the seam from
    `seam_fold`, whose old leaves are the deep ones.
    """
    return mirrored(*seam_fold(seam), seam)


def edge_bump(p):
    """A fold confined to the right edge of ``[-2, 2]**2``, exactly affine outside it.

    ``x' = 0.6 x + g(x) h(y)`` with ``g = 60 (x - 1.4)**2 (2 - x)**2`` on
    ``[1.4, 2]`` and ``h = (1 - y**2 / 4)**2`` on ``|y| < 2``, each zero
    elsewhere and C1 where it stops. ``det A = 0.6 + g'(x) h(y)`` vanishes
    within a few hundredths of ``x = 2``, so a fov-4 mesh is ``max_level``
    deep along that edge, while every ring cell a fov-6 extension adds is
    exactly affine and converges at level 0. The ring's own cascade ends with
    it, so only `balance` can grade the ring down to the old leaves -- unlike
    `seam_fold`, whose curved strip runs on into the ring and keeps the
    ring's cascade going.
    """
    x, y = p[:, 0], p[:, 1]
    g = np.where((x > 1.4) & (x < 2.0), 60.0 * (x - 1.4) ** 2 * (2.0 - x) ** 2, 0.0)
    h = np.where(np.abs(y) < 2.0, (1.0 - y**2 / 4.0) ** 2, 0.0)
    return np.stack([0.6 * x + g * h, y], axis=-1)


def edge_bump_jacobian(p):
    x, y = p[:, 0], p[:, 1]
    inx, iny = (x > 1.4) & (x < 2.0), np.abs(y) < 2.0
    g = np.where(inx, 60.0 * (x - 1.4) ** 2 * (2.0 - x) ** 2, 0.0)
    h = np.where(iny, (1.0 - y**2 / 4.0) ** 2, 0.0)
    dg = np.where(inx, 120.0 * (x - 1.4) * (2.0 - x) * (3.4 - 2.0 * x), 0.0)
    dh = np.where(iny, -y * (1.0 - y**2 / 4.0), 0.0)
    J = np.zeros((p.shape[0], 2, 2))
    J[:, 0, 0] = 0.6 + dg * h
    J[:, 0, 1] = g * dh
    J[:, 1, 1] = 1.0
    return J


# ---------------------------------------------------------------------------
# The state a mesh keeps for its own extension
# ---------------------------------------------------------------------------


def test_a_mesh_records_its_lattice_and_the_lattice_coordinates_of_its_vertices():
    mesh, _, _ = build(localised_fold, localised_fold_jacobian, 4.0, 4, 0.05, x0=0.25)
    lat = mesh.lattice
    assert (lat.level, lat.origin) == (mesh.max_level + 1, 0)
    assert lat.n == 4 << lat.level
    assert (mesh.fov, mesh.init_res) == (4.0, 4)
    assert to_np(mesh.vertices_ij).dtype == np.int64
    assert np.array_equal(
        to_np(mesh.vertices_lens), to_np(new.lattice_xy(lat, mesh.vertices_ij))
    )
    keys = to_np(new.lattice_key(lat, mesh.vertices_ij))
    assert (np.diff(keys) > 0).all()


def test_origin_cls_is_the_orientation_class_of_each_origin_leaf():
    """``P = 2**(L - d) * R @ G[c]`` in lattice units, for every pre-closure leaf.

    `child_matrix_tables` defines the class ``c`` of a level-``d`` triangle by
    ``P = 2**-d * h0 * R @ G[c]``, and ``h0`` is ``2**L`` lattice units.
    """
    mesh, _, _ = build(localised_fold, localised_fold_jacobian, 4.0, 4, 0.05)
    _, G, _, _, _ = new.child_matrix_tables()
    ij = to_np(mesh.vertices_ij)[to_np(mesh.origin_leaves)]
    P = np.stack((ij[:, 1] - ij[:, 0], ij[:, 2] - ij[:, 0]), axis=-1)
    first = np.searchsorted(to_np(mesh.leaf_origin), np.arange(ij.shape[0]))
    level = to_np(mesh.leaf_level)[first]
    R = np.array([[1, 0], [1, 1]])
    scale = (1 << (mesh.lattice.level - level))[:, None, None]
    assert len(set(level.tolist())) > 1, "fixture must have several levels"
    assert np.array_equal(P, scale * (R @ to_np(G)[to_np(mesh.origin_cls)]))


def test_band_rows_are_in_leaf_order():
    mesh, _, _ = build(row_fold, row_fold_jacobian, 4.0, 8, 2e-2)
    leaves = to_np(mesh.critical_band.leaves)
    assert leaves.size > 1
    assert (np.diff(leaves) > 0).all()


def test_band_samples_are_in_ascending_lattice_key_order():
    mesh, _, _ = build(row_fold, row_fold_jacobian, 4.0, 8, 2e-2)
    lat = mesh.lattice
    lens = to_np(mesh.critical_band.lens)
    ij = np.rint((lens - to_np(lat.lo)) / lat.scale).astype(np.int64) + lat.origin
    keys = ij[:, 0] * lat.stride + ij[:, 1]
    assert (np.diff(keys) > 0).all()


def _two_triangle_state():
    """Two level-0 triangles sharing a cell's diagonal, on a level-2 lattice.

    ``A = (0,0),(4,4),(0,4)`` and ``B = (0,0),(4,0),(4,4)``: their six samples
    share ``(0,0)``, ``(4,4)`` and the diagonal's midpoint ``(2,2)``.
    """
    lat = new.make_lattice(4.0, 0.0, 0.0, 1, 2)
    vertices = np.array([[0, 0], [0, 4], [4, 0], [4, 4]])  # ascending key order
    cache = new.VertexCache(
        keys=i64(vertices[:, 0] * lat.stride + vertices[:, 1]),
        slots=i64(np.arange(4)),
        ij=i64(vertices),
        beta=f64(vertices * 1.0),
    )
    store, _ = new.store_add(
        new.empty_store(), i64([[0, 3, 1], [0, 2, 3]]), 0, 0, new.LEAF_CONVERGED
    )
    return lat, cache, store


def _band_over(lat, cache, store, rows, value):
    """A band on store ``rows`` whose sample with key ``k`` carries ``value(k)``."""
    keys = to_np(new.band_sample_keys(lat, cache, store, i64(rows)))
    own, samples = np.unique(keys.reshape(-1), return_inverse=True)
    return new.CriticalBand(
        leaves=i64(rows),
        samples=i64(samples.reshape(-1, 6)),
        lens=f64(np.zeros((own.size, 2))),
        source=f64(np.stack([value(own), -value(own)], axis=-1)),
        det=f64(value(own)),
    )


def test_merge_bands_keeps_one_sample_per_key_and_the_first_band_s_value():
    lat, cache, store = _two_triangle_state()
    first = _band_over(lat, cache, store, [0], lambda k: k * 1.0)
    second = _band_over(lat, cache, store, [1], lambda k: k * 10.0 + 0.5)
    merged = new.merge_bands(lat, cache, store, first, second)

    row_keys = to_np(new.band_sample_keys(lat, cache, store, i64([0, 1])))
    keys = np.unique(row_keys)
    first_keys = set(row_keys[0].tolist())
    det = np.array([k * 1.0 if k in first_keys else k * 10.0 + 0.5 for k in keys])
    assert keys.size == 9
    assert to_np(merged.leaves).tolist() == [0, 1]
    assert np.array_equal(keys[to_np(merged.samples)], row_keys)
    assert np.array_equal(to_np(merged.det), det)
    assert np.array_equal(to_np(merged.source), np.stack([det, -det], axis=-1))
    ij = np.stack((keys // lat.stride, keys % lat.stride), axis=-1)
    assert np.array_equal(to_np(merged.lens), to_np(lat.lo) + ij * lat.scale)


def test_merge_bands_of_two_empty_bands_is_empty():
    lat, cache, store = _two_triangle_state()
    merged = new.merge_bands(lat, cache, store, new.empty_band(), new.empty_band())
    assert tuple(merged.leaves.shape) == (0,)
    assert tuple(merged.samples.shape) == (0, 6)
    assert tuple(merged.det.shape) == (0,)


def test_merge_bands_rejects_a_band_whose_rows_do_not_hold_its_samples():
    lat, cache, store = _two_triangle_state()
    good = _band_over(lat, cache, store, [0], lambda k: k * 1.0)
    bad = good._replace(det=good.det[:-1], source=good.source[:-1])
    with pytest.raises(AssertionError, match="rows do not hold exactly its samples"):
        new.merge_bands(lat, cache, store, bad, new.empty_band())


# ---------------------------------------------------------------------------
# Seeded refinement and balance
# ---------------------------------------------------------------------------


def refine_setup(fn, jac, fov, init_res, min_img_sep):
    """Everything `refine` takes for a fresh build of these parameters."""
    sep = min_img_sep / 2
    max_level = new.depth_floor(fov, init_res, sep)
    lens, calls = recording_lens(fn, jac)
    return SimpleNamespace(
        lat=new.make_lattice(fov, 0.0, 0.0, init_res, max_level + 1),
        max_level=max_level,
        sep=sep,
        h0=fov / init_res,
        init_res=init_res,
        tables=new.child_matrix_tables(),
        raytrace_fn=new.make_raytrace(lens.raytrace, None),
        jacobian=lens.jacobian_lens_equation,
        calls=calls,
    )


def refine_with(ctx, roots=None, seed=None):
    cache, active, store, _, band = new.refine(
        ctx.raytrace_fn,
        ctx.jacobian,
        ctx.lat,
        ctx.init_res,
        ctx.h0,
        ctx.sep,
        ctx.max_level,
        ctx.tables,
        None,
        roots=roots,
        seed=seed,
    )
    return cache, active, store, band


def balance_with(ctx, cache, active, store):
    return new.balance(
        store,
        cache,
        ctx.lat,
        active,
        ctx.max_level,
        ctx.tables[2],
        ctx.raytrace_fn,
        None,
    )


def leaf_set(lat, cache, store):
    """Every valid leaf as ``(sorted vertex keys, level, status)``, as a set."""
    v, level, _, status = new.store_compact(store)
    keys = np.sort(to_np(new.lattice_key(lat, cache.ij[v])), axis=1)
    return set(
        zip(map(tuple, keys.tolist()), to_np(level).tolist(), to_np(status).tolist())
    )


def assert_balanced(lat, cache, active, store, max_level):
    """No leaf at ``level <= max_level - 2`` has an active quarter point.

    Reads `edge_quarter_keys`, the reference `find_unbalanced` is itself
    tested against, so `balance` is not judged by its own scan.
    """
    v, level, _, _ = new.store_compact(store)
    rows = backend.flatnonzero(level <= max_level - 2)
    keys = new.edge_quarter_keys(lat, cache.ij[v[rows]])
    assert not bool(backend.any(new.active_contains(active, cache, keys)))


def test_ring_triangles_are_the_level0_triangles_outside_the_central_block():
    level = 3
    root_class = new.child_matrix_tables()[4]
    every_ij, every_cls = new.initial_triangles(6, level, root_class)
    ring_ij, ring_cls = new.ring_triangles(6, 1, level, root_class)

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


def test_a_fresh_refinement_is_already_balanced_so_balance_forces_nothing():
    ctx = refine_setup(localised_fold, localised_fold_jacobian, 4.0, 4, 0.05)
    cache, active, store, _ = refine_with(ctx)
    before = leaf_set(ctx.lat, cache, store)
    store, cache, active, forced = balance_with(ctx, cache, active, store)
    assert forced == 0
    assert leaf_set(ctx.lat, cache, store) == before


def test_refining_in_two_seeded_passes_then_balancing_matches_one_pass():
    """The extension's mechanics, without the mesh around them.

    The left half of the cells is refined first; the right half is refined
    seeded with that result and given only its own cells as roots.
    ``seam_fold(0.0)`` folds just left of ``x = 0`` and is affine right of
    it, so the second pass converges at level 0 beside deep first-pass leaves
    and its own cascade never sees them: only `balance` can make the two
    passes agree with one.
    """
    fn, jac = seam_fold(0.0)
    ctx = refine_setup(fn, jac, 4.0, 4, 0.05)
    ij, cls = new.initial_triangles(4, ctx.lat.level, ctx.tables[4])
    half = 2 << ctx.lat.level  # cells with i < 2 lie at x < 0
    left = backend.flatnonzero(ij[:, 0, 0] < half)
    right = backend.flatnonzero(ij[:, 0, 0] >= half)
    cache, active, store, _ = refine_with(ctx, roots=(ij[left], cls[left]))
    cache, active, store, _ = refine_with(
        ctx, roots=(ij[right], cls[right]), seed=(cache, active, store)
    )
    store, cache, active, forced = balance_with(ctx, cache, active, store)

    one = refine_setup(fn, jac, 4.0, 4, 0.05)
    cache1, active1, store1, _ = refine_with(one)

    assert forced > 0
    assert_balanced(ctx.lat, cache, active, store, ctx.max_level)
    assert leaf_set(ctx.lat, cache, store) == leaf_set(one.lat, cache1, store1)


# ---------------------------------------------------------------------------
# Seeding from a frozen mesh
# ---------------------------------------------------------------------------


def _assert_same(a, b, name):
    if hasattr(a, "shape"):
        a, b = to_np(a), to_np(b)
        assert (a.dtype, a.shape) == (b.dtype, b.shape), name
        assert np.array_equal(a, b, equal_nan=a.dtype.kind == "f"), name
    else:
        assert a == b, name


def assert_meshes_equal(got, want):
    """Every field equal, arrays bit for bit, NaN matching NaN.

    ``lattice`` is compared as the map it defines -- level, extent, spacing
    and the positions of its corners -- because an extension keeps its
    build's anchor (``origin > 0``) where a fresh lattice anchors at its own
    corner. Every vertex and band position is compared anyway, through
    ``vertices_lens`` and ``critical_band.lens``.
    """
    for name in new.AdaptiveMesh._fields:
        a, b = getattr(got, name), getattr(want, name)
        if name == "lattice":
            assert (a.level, a.n, a.stride, a.scale) == (
                b.level,
                b.n,
                b.stride,
                b.scale,
            )
            corners = backend.to(i64([[0, 0], [a.n, a.n]]), device=backend.device(a.lo))
            _assert_same(new.lattice_xy(a, corners), new.lattice_xy(b, corners), name)
        elif name in ("index", "critical_band"):
            for field in type(a)._fields:
                _assert_same(getattr(a, field), getattr(b, field), f"{name}.{field}")
        else:
            _assert_same(a, b, name)


def seed_of(mesh, lens, k):
    lat = new.extend_lattice(mesh.lattice, k)
    raytrace_fn = new.make_raytrace(lens.raytrace, None)
    return (lat, *new.seed_from_mesh(mesh, lat, k, raytrace_fn, None))


def test_seeding_a_mesh_and_freezing_it_again_reproduces_it():
    """With nothing added, seed then freeze is the identity -- non-finite
    leaves and the band included -- and a float64 mesh costs no lens call."""
    fn, jac = broken_where(
        localised_fold, localised_fold_jacobian, lambda p: p[:, 0] > 1.5
    )
    mesh, lens, calls = build(fn, jac, 4.0, 4, 0.05)
    assert mesh.critical_band.leaves.shape[0] > 0
    assert ((to_np(mesh.leaf_status) & new.LEAF_RAYTRACE_NONFINITE) != 0).any()
    calls["raytrace"].clear()
    lat, cache, active, store, band = seed_of(mesh, lens, 0)
    again = new.freeze(
        lat,
        cache,
        active,
        store,
        new.empty_band(),
        band,
        fov=mesh.fov,
        init_res=mesh.init_res,
        min_img_sep=mesh.min_img_sep,
        d_floor=mesh.d_floor,
        max_level=mesh.max_level,
        dtype=mesh.dtype,
        device=mesh.device,
        index_cells=None,
    )
    assert_meshes_equal(again, mesh)
    assert not calls["raytrace"]


def test_seeding_drops_the_freeze_time_flag_below_max_level():
    """Below ``max_level`` a stored flag can only be freeze's
    ``LEAF_RAYTRACE_NONFINITE``, which freeze derives again and a forced child
    must not inherit; ``max_level`` rows keep their whole record."""
    mesh, lens, _ = build(localised_fold, localised_fold_jacobian, 4.0, 4, 0.05)
    status = to_np(mesh.leaf_status).copy()
    status[to_np(mesh.leaf_level) < mesh.max_level] |= new.LEAF_RAYTRACE_NONFINITE
    flagged = mesh._replace(leaf_status=i64(status))
    _, _, _, store, _ = seed_of(flagged, lens, 0)
    level, got = to_np(store.level), to_np(store.status)
    low, top = level < mesh.max_level, level == mesh.max_level
    assert low.any() and top.any()
    assert (got[low] == new.LEAF_CONVERGED).all()
    first = np.searchsorted(to_np(mesh.leaf_origin), np.arange(level.size))
    assert np.array_equal(got[top], status[first][top])


def test_seeding_a_float32_mesh_raytraces_its_outer_boundary_again():
    """Only the old boundary is re-traced -- the only old points the ring's
    criterion reads -- and those values come straight from raytrace."""
    mesh, lens, calls = build(
        localised_fold, localised_fold_jacobian, 4.0, 4, 0.05, dtype=backend.float32
    )
    calls["raytrace"].clear()
    lat, cache, _, _, _ = seed_of(mesh, lens, 1)
    old = to_np(mesh.vertices_ij)
    edge = ((old == 0) | (old == mesh.lattice.n)).any(axis=1)
    pad = 1 << lat.level
    xy = to_np(lat.lo) + (old[edge] + pad - lat.origin) * lat.scale
    assert edge.any() and (~edge).any()
    assert np.array_equal(called_at(calls, "raytrace"), xy)
    beta = to_np(cache.beta)
    assert np.array_equal(beta[edge], localised_fold(xy))
    assert np.array_equal(
        beta[~edge], to_np(mesh.vertices_source)[~edge].astype(np.float64)
    )


# ---------------------------------------------------------------------------
# extend_adaptive_mesh
# ---------------------------------------------------------------------------


def fresh_equivalent(kw, target):
    """``build`` arguments for a fresh build of ``target`` on the same cells."""
    out = dict(kw)
    out["init_res"] = round(kw["init_res"] * target / kw["fov"])
    out["fov"] = target
    return out


EQUIVALENCE = {
    "sie_like": (
        sie_like,
        sie_like_jacobian,
        dict(fov=2.0, init_res=4, min_img_sep=0.05),
        3.0,
    ),
    "localised_fold": (
        localised_fold,
        localised_fold_jacobian,
        dict(fov=4.0, init_res=4, min_img_sep=0.05),
        6.0,
    ),
    "row_fold": (
        row_fold,
        row_fold_jacobian,
        dict(fov=4.0, init_res=8, min_img_sep=2e-2),
        6.0,
    ),
    "nan_across_seam": (
        *broken_where(localised_fold, localised_fold_jacobian, lambda p: p[:, 0] > 1.5),
        dict(fov=4.0, init_res=4, min_img_sep=0.05),
        6.0,
    ),
    "affine": (
        affine,
        affine_jacobian,
        dict(fov=4.0, init_res=4, min_img_sep=0.05),
        6.0,
    ),
    "seam_fold": (*seam_fold(2.0), dict(fov=4.0, init_res=4, min_img_sep=0.05), 6.0),
    "ring_fold": (*ring_fold(2.0), dict(fov=4.0, init_res=4, min_img_sep=0.05), 6.0),
    "edge_bump": (
        edge_bump,
        edge_bump_jacobian,
        dict(fov=4.0, init_res=4, min_img_sep=0.05),
        6.0,
    ),
    "off_centre": (
        localised_fold,
        localised_fold_jacobian,
        dict(fov=4.0, init_res=8, min_img_sep=0.05, x0=0.5, y0=-0.25),
        6.0,
    ),
    "depth_limited": (
        localised_fold,
        localised_fold_jacobian,
        dict(fov=4.0, init_res=4, min_img_sep=0.05, max_depth=3),
        6.0,
    ),
}


@pytest.mark.filterwarnings("ignore:Adaptive mesh is depth-limited")
@pytest.mark.parametrize("case", list(EQUIVALENCE))
def test_an_extension_is_the_fresh_build_of_the_larger_fov(case):
    fn, jac, kw, target = EQUIVALENCE[case]
    mesh, lens, _ = build(fn, jac, **kw)
    got = new.extend_adaptive_mesh(mesh, lens, target)
    want, _, _ = build(fn, jac, **fresh_equivalent(kw, target))
    assert got.init_res > mesh.init_res
    assert_meshes_equal(got, want)


def test_extensions_chain():
    kw = dict(fov=4.0, init_res=8, min_img_sep=0.05)
    mesh, lens, _ = build(localised_fold, localised_fold_jacobian, **kw)
    twice = new.extend_adaptive_mesh(
        new.extend_adaptive_mesh(mesh, lens, 6.0), lens, 8.0
    )
    once = new.extend_adaptive_mesh(mesh, lens, 8.0)
    fresh, _, _ = build(
        localised_fold, localised_fold_jacobian, **fresh_equivalent(kw, 8.0)
    )
    assert_meshes_equal(twice, fresh)
    assert_meshes_equal(once, fresh)


def test_extending_a_float32_mesh_matches_a_fresh_float32_build():
    kw = dict(fov=4.0, init_res=4, min_img_sep=0.05, dtype=backend.float32)
    mesh, lens, _ = build(localised_fold, localised_fold_jacobian, **kw)
    got = new.extend_adaptive_mesh(mesh, lens, 6.0)
    want, _, _ = build(
        localised_fold, localised_fold_jacobian, **fresh_equivalent(kw, 6.0)
    )
    assert got.dtype == backend.float32
    assert_meshes_equal(got, want)


def _in_any_triangle(points, triangles):
    """True where an integer point lies in or on some integer triangle."""
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    p = points[:, None, :]

    def cross(u, v, w):
        return (v[..., 0] - u[..., 0]) * (w[..., 1] - u[..., 1]) - (
            v[..., 1] - u[..., 1]
        ) * (w[..., 0] - u[..., 0])

    d = np.stack((cross(a, b, p), cross(b, c, p), cross(c, a, p)), axis=-1)
    return ((d >= 0).all(axis=-1) | (d <= 0).all(axis=-1)).any(axis=1)


def test_an_extension_calls_the_lens_only_in_the_ring_and_in_old_leaves_it_splits():
    """No Jacobian call strictly inside the old domain; no old vertex
    raytraced again; and every raytraced point strictly inside lies in an old
    leaf the extension split, which ``ring_fold(2.0)`` makes it do."""
    fn, jac = ring_fold(2.0)
    mesh, lens, calls = build(fn, jac, 4.0, 4, 0.05)
    calls["raytrace"].clear()
    calls["jacobian"].clear()
    ext = new.extend_adaptive_mesh(mesh, lens, 6.0)
    lat = ext.lattice
    pad = lat.origin
    lo, hi = pad, pad + mesh.lattice.n

    def to_ij(xy):
        return np.rint((xy - to_np(lat.lo)) / lat.scale).astype(np.int64) + lat.origin

    def strictly_inside(ij):
        return ((ij > lo) & (ij < hi)).all(axis=1)

    jacobian_ij = to_ij(called_at(calls, "jacobian"))
    assert jacobian_ij.shape[0] > 0
    assert not strictly_inside(jacobian_ij).any()

    raytrace_ij = to_ij(called_at(calls, "raytrace"))
    old_vertices = to_np(mesh.vertices_ij) + pad
    assert not set(map(tuple, raytrace_ij.tolist())) & set(
        map(tuple, old_vertices.tolist())
    )

    inner = raytrace_ij[strictly_inside(raytrace_ij)]
    kept = {
        tuple(sorted(map(tuple, t)))
        for t in to_np(ext.vertices_ij)[to_np(ext.origin_leaves)].tolist()
    }
    split = np.array(
        [
            t
            for t in old_vertices[to_np(mesh.origin_leaves)].tolist()
            if tuple(sorted(map(tuple, t))) not in kept
        ]
    )
    assert inner.shape[0] > 0 and split.shape[0] > 0, "ring_fold must split old leaves"
    assert _in_any_triangle(inner, split).all()


@pytest.fixture
def affine_mesh():
    """``h0 = 0.5``: each ring adds ``1.0`` to the fov."""
    return build(affine, affine_jacobian, 4.0, 8, 0.05)


@pytest.mark.parametrize(
    "fov, want",
    [
        (5.9, 6.0),
        (6.0, 6.0),
        (6, 6.0),
        (float(np.nextafter(6.0, np.inf)), 6.0),
        (4.2, 5.0),
    ],
)
def test_the_requested_fov_rounds_up_to_whole_cells(affine_mesh, fov, want):
    mesh, lens, _ = affine_mesh
    ext = new.extend_adaptive_mesh(mesh, lens, fov)
    assert ext.fov == want
    assert ext.init_res == round(want / 0.5)


@pytest.mark.parametrize("fov", [4.0, 4.0 - 1e-12])
def test_a_fov_the_mesh_already_covers_returns_the_mesh_itself(affine_mesh, fov):
    mesh, lens, calls = affine_mesh
    calls["raytrace"].clear()
    calls["jacobian"].clear()
    assert new.extend_adaptive_mesh(mesh, lens, fov) is mesh
    assert not calls["raytrace"] and not calls["jacobian"]


def test_a_smaller_fov_raises(affine_mesh):
    mesh, lens, _ = affine_mesh
    with pytest.raises(ValueError, match="can only grow"):
        new.extend_adaptive_mesh(mesh, lens, 3.9)


@pytest.mark.parametrize("fov", [np.nan, np.inf])
def test_a_non_finite_fov_raises(affine_mesh, fov):
    mesh, lens, _ = affine_mesh
    with pytest.raises(ValueError, match="finite"):
        new.extend_adaptive_mesh(mesh, lens, fov)


@pytest.mark.filterwarnings("ignore:raytrace returned")
def test_an_extension_int64_keys_cannot_hold_raises():
    """At ``max_level = 25`` the lattice keys in int64 up to ``init_res = 45``."""
    mesh, lens, _ = build(affine, affine_jacobian, 1.0, 1, 1e-7)
    assert mesh.max_level == 25
    assert new.extend_adaptive_mesh(mesh, lens, 45.0).init_res == 45
    with pytest.raises(ValueError, match="lattice too fine.*Extend by less"):
        new.extend_adaptive_mesh(mesh, lens, 46.0)


def _messages(record):
    """The ``UserWarning`` messages recorded, in order."""
    return [str(w.message) for w in record if issubclass(w.category, UserWarning)]


def test_an_extension_warns_depth_limited_as_the_fresh_build_would():
    kw = dict(fov=4.0, init_res=4, min_img_sep=0.05, max_depth=2)
    with pytest.warns(UserWarning, match="depth-limited"):
        mesh, lens, _ = build(localised_fold, localised_fold_jacobian, **kw)
    with warnings.catch_warnings(record=True) as got:
        warnings.simplefilter("always")
        new.extend_adaptive_mesh(mesh, lens, 6.0)
    with warnings.catch_warnings(record=True) as want:
        warnings.simplefilter("always")
        build(localised_fold, localised_fold_jacobian, **fresh_equivalent(kw, 6.0))
    assert any("depth-limited" in m for m in _messages(got))
    assert _messages(got) == _messages(want)


def test_an_extension_checks_the_cancellation_floor_at_the_larger_fov():
    """float32 resolves ``min_img_sep / 2 = 2e-3`` at fov 4 (floor 1.95e-3)
    but not at fov 6 (floor 2.39e-3)."""
    kw = dict(fov=4.0, init_res=4, min_img_sep=4e-3, out_dtype=np.float32)
    with warnings.catch_warnings(record=True) as base:
        warnings.simplefilter("always")
        mesh, lens, _ = build(affine, affine_jacobian, **kw)
    assert not any("cancellation floor" in m for m in _messages(base))
    with warnings.catch_warnings(record=True) as got:
        warnings.simplefilter("always")
        new.extend_adaptive_mesh(mesh, lens, 6.0)
    with warnings.catch_warnings(record=True) as want:
        warnings.simplefilter("always")
        build(affine, affine_jacobian, **fresh_equivalent(kw, 6.0))
    assert any("cancellation floor" in m for m in _messages(got))
    assert _messages(got) == _messages(want)


def test_an_extension_closes_the_critical_curves_the_fov_cut():
    """At fov 2 the tangential curve (radius ~1.18) crosses the domain's
    edge; at fov 3 it lies inside."""
    kw = dict(fov=2.0, init_res=4, min_img_sep=0.05)
    mesh, lens, _ = build(sie_like, sie_like_jacobian, **kw)
    assert not to_np(crit.mesh_critical_curves(mesh).closed).all()
    ext = new.extend_adaptive_mesh(mesh, lens, 3.0)
    got = crit.mesh_critical_curves(ext)
    fresh, _, _ = build(sie_like, sie_like_jacobian, **fresh_equivalent(kw, 3.0))
    assert got.closed.shape[0] > 0 and to_np(got.closed).all()
    for field in crit.CriticalCurves._fields:
        _assert_same(
            getattr(got, field),
            getattr(crit.mesh_critical_curves(fresh), field),
            field,
        )


def test_an_extension_of_a_mesh_on_a_device_stays_on_it(device):
    kw = dict(fov=4.0, init_res=4, min_img_sep=0.05, device=device)
    mesh, lens, _ = build(localised_fold, localised_fold_jacobian, **kw)
    got = new.extend_adaptive_mesh(mesh, lens, 6.0)
    want, _, _ = build(
        localised_fold, localised_fold_jacobian, **fresh_equivalent(kw, 6.0)
    )
    assert got.device == device
    assert backend.device(got.leaves) == backend.device(want.leaves)
    assert_meshes_equal(got, want)


def test_batch_size_and_index_cells_reach_every_path_an_extension_traces():
    """``raytrace_batch_size`` caps every raytrace an extension makes: the
    ring's own, `seed_from_mesh`'s re-trace of a float32 mesh's boundary, and
    the midpoints `force_split` fetches for the leaves `balance` splits --
    `edge_bump` at float32 needs all three. The mesh still equals the fresh
    build given the same ``index_cells``."""
    kw = dict(
        fov=4.0,
        init_res=4,
        min_img_sep=0.05,
        dtype=backend.float32,
        raytrace_batch_size=5,
        index_cells=16,
    )
    mesh, lens, calls = build(edge_bump, edge_bump_jacobian, **kw)
    calls["raytrace"].clear()
    got = new.extend_adaptive_mesh(
        mesh, lens, 6.0, raytrace_batch_size=5, index_cells=16
    )
    assert calls["raytrace"] and max(len(xy) for xy in calls["raytrace"]) <= 5
    want, _, _ = build(edge_bump, edge_bump_jacobian, **fresh_equivalent(kw, 6.0))
    assert_meshes_equal(got, want)
