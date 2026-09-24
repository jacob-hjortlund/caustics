"""Growing a built adaptive mesh to a larger fov.

The extension's central claim is exactness: a mesh extended from fov ``A`` to
``B`` is bit for bit the mesh a fresh build of ``B`` produces on the same
lattice. The fixtures use dyadic fovs, centres and cell sizes, where a fresh
build's own lattice coincides with the extended one exactly.
"""

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
