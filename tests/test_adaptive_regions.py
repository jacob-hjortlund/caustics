import numpy as np
import pytest

from caustics.backend_obj import backend
from caustics.lenses.func.adaptive import build_adaptive_mesh
from caustics.lenses.func.adaptive.regions import magnified_regions
from caustics.lenses.func.adaptive.source_mesh import build_magnification_mesh


def to_np(x):
    return backend.to_numpy(x)


def _arr(x):
    return backend.as_array(np.asarray(x, dtype=np.float64), dtype=backend.float64)


def _stack_2x2(a, b, c, d):
    return backend.stack(
        (backend.stack((a, b), dim=-1), backend.stack((c, d), dim=-1)), dim=-2
    )


def _row_fold(x, y):
    """``mu_tot = 2 / sqrt(1 - 4 beta_y)`` below the caustic ``beta_y = 1/4``:
    ``mu_tot >= 4`` on the strip ``3/16 <= beta_y <= 1/4``."""
    return x * 1.0, y - y * y


def _row_fold_jacobian(x, y):
    one, zero = backend.ones_like(x), backend.zeros_like(x)
    return _stack_2x2(one, zero, zero, 1.0 - 2.0 * y)


def _sis(x, y):
    """An SIS of Einstein radius 1: ``mu_tot = 2 / |beta|`` inside it."""
    r = backend.sqrt(x * x + y * y)
    return x - x / r, y - y / r


def _sis_jacobian(x, y):
    r = backend.sqrt(x * x + y * y)
    k = 1.0 / r**3
    return _stack_2x2(
        1.0 - 1.0 / r + k * x * x, k * x * y, k * x * y, 1.0 - 1.0 / r + k * y * y
    )


@pytest.fixture(scope="module")
def fold_lens():
    return build_adaptive_mesh(_row_fold, _row_fold_jacobian, 2.0, 8, 0.02, y0=0.5)


@pytest.fixture(scope="module")
def fold_mag(fold_lens):
    return build_magnification_mesh(
        fold_lens, 8, 0.02, mu_min=4.0, fov=1.0, x0=0.0, y0=0.0
    )


@pytest.fixture(scope="module")
def wide_mag(fold_lens):
    # The default window takes in the image of the lens fov's boundary.
    return build_magnification_mesh(fold_lens, 8, 0.05, mu_min=1.5)


def _curves(regions):
    pts, off = to_np(regions.source), to_np(regions.offsets)
    return [pts[off[c] : off[c + 1]] for c in range(len(off) - 1)], to_np(
        regions.closed
    )


def test_the_fold_strip_is_bounded_by_its_level_line_and_its_caustic(fold_mag):
    b = to_np(magnified_regions(fold_mag, 4.0).source)[:, 1]
    near_level = np.abs(b - 0.1875) < 0.04
    # A crossing on an edge from the band strip (mu = inf) to beyond the
    # caustic (mu = 0) lies between the two samples: within one floor edge,
    # at most src_tol, of the caustic, on either side.
    near_caustic = np.abs(b - 0.25) <= fold_mag.src_tol
    assert (near_level | near_caustic).all()
    assert near_level.any() and near_caustic.any()


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


def _point_segment_distance(p, a, b):
    d = b - a
    t = np.clip(
        ((p[:, None] - a[None]) * d[None]).sum(-1) / (d * d).sum(-1)[None], 0, 1
    )
    return np.linalg.norm(p[:, None] - (a[None] + t[..., None] * d[None]), axis=-1)


def test_curves_stop_where_the_data_is_incomplete(wide_mag):
    curves, closed = _curves(magnified_regions(wide_mag, 1.5))
    assert not closed.all()
    ends = np.array([p for c, k in zip(curves, closed) if not k for p in (c[0], c[-1])])
    lo = to_np(wide_mag.lattice.lo)
    hi = lo + wide_mag.fov
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
    assert (np.abs(b - 0.25) <= fold_mag.src_tol).all()
