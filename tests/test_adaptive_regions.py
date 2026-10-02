import numpy as np
import pytest

from caustics.backend_obj import backend
from caustics.lenses.func.adaptive import build_adaptive_mesh
from caustics.lenses.func.adaptive.regions import (
    in_magnified_region,
    magnified_area,
    magnified_regions,
)
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


@pytest.fixture(scope="module")
def sis_lens():
    return build_adaptive_mesh(_sis, _sis_jacobian, 4.0, 8, 0.01, centres=[(0.0, 0.0)])


@pytest.fixture(scope="module")
def sis_mag(sis_lens):
    # mu_tot >= 4 on the disk |beta| <= 0.5; the window stays inside the
    # image of the lens fov's boundary.
    return build_magnification_mesh(
        sis_lens, 8, 0.02, mu_min=4.0, fov=1.5, x0=0.0, y0=0.0
    )


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


def test_the_area_is_the_signed_area_the_closed_loops_bound(sis_mag):
    curves, closed = _curves(magnified_regions(sis_mag, 4.0))
    assert closed.all()
    area, complete = magnified_area(sis_mag, 4.0)
    assert bool(to_np(complete))
    np.testing.assert_allclose(
        float(to_np(area)), sum(_shoelace(c) for c in curves), rtol=1e-9
    )


def test_membership_is_the_side_of_the_loops_a_point_is_on(sis_mag):
    curves, _ = _curves(magnified_regions(sis_mag, 4.0))
    points = np.random.default_rng(11).uniform(-0.74, 0.74, (2000, 2))
    inside, complete = in_magnified_region(sis_mag, 4.0, _arr(points))
    assert to_np(complete).all()
    assert np.array_equal(to_np(inside), _winding(points, curves) > 0.5)


def test_points_outside_the_window_are_not_known(sis_mag):
    inside, complete = in_magnified_region(sis_mag, 4.0, _arr([[5.0, 5.0], [0.0, 0.0]]))
    assert to_np(complete).tolist() == [False, True]
    assert to_np(inside).tolist() == [False, True]


def test_the_area_takes_the_shape_of_mu_min(sis_mag):
    area, complete = magnified_area(sis_mag, [4.0, 8.0])
    assert tuple(area.shape) == (2,) and tuple(complete.shape) == (2,)
    single, _ = magnified_area(sis_mag, 4.0)
    assert tuple(single.shape) == ()
    assert float(to_np(area)[0]) == float(to_np(single))
    assert to_np(area)[1] < to_np(area)[0]


def test_a_zero_threshold_takes_the_whole_window(sis_mag):
    area, complete = magnified_area(sis_mag, 0.0)
    np.testing.assert_allclose(float(to_np(area)), sis_mag.fov**2, rtol=1e-12)
    assert not bool(to_np(complete))
    assert to_np(magnified_regions(sis_mag, 0.0).offsets).tolist() == [0]


def test_a_window_that_misses_every_image_has_no_region(sis_lens):
    far = build_magnification_mesh(
        sis_lens, 4, 0.1, mu_min=4.0, fov=1.0, x0=50.0, y0=0.0
    )
    assert (to_np(far.mu) == 0).all()
    assert not to_np(far.incomplete).any()
    assert to_np(magnified_regions(far, 4.0).offsets).tolist() == [0]
    area, complete = magnified_area(far, 4.0)
    assert float(to_np(area)) == 0.0 and bool(to_np(complete))
    inside, known = in_magnified_region(far, 4.0, _arr([[50.0, 0.0]]))
    assert to_np(inside).tolist() == [False] and to_np(known).tolist() == [True]


def test_the_fold_strip_area_is_open_and_about_its_width(fold_mag):
    area, complete = magnified_area(fold_mag, 4.0)
    assert not bool(to_np(complete))
    # The strip is 1/16 tall and 1 wide; the area ratio's few-per-cent error
    # moves its level line by about 0.01.
    assert abs(float(to_np(area)) - 0.0625) < 0.025
