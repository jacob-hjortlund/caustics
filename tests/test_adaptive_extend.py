"""Growing a built adaptive mesh to a larger fov.

The extension's central claim is exactness: a mesh extended from fov ``A`` to
``B`` is bit for bit the mesh a fresh build of ``B`` produces on the same
lattice. The fixtures use dyadic fovs, centres and cell sizes, where a fresh
build's own lattice coincides with the extended one exactly.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from caustics.backend_obj import backend
from caustics.lenses.func import adaptive as new


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
