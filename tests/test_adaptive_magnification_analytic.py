"""
Magnified regions of three analytic lenses against their exact regions.

The tolerances come from the measured accuracy of the area-ratio
magnification (spec, section 8): it errs like sqrt(min_img_sep), a few per
cent at the lens tolerances used here, which moves a boundary by about that
relative change in mu over |d mu / d beta|, plus src_tol.
"""

import numpy as np
import pytest

from caustics import (
    build_adaptive_mesh,
    build_magnification_mesh,
    magnified_area,
    magnified_regions,
)
from caustics.backend_obj import backend


def to_np(x):
    return backend.to_numpy(x)


def _stack_2x2(a, b, c, d):
    return backend.stack(
        (backend.stack((a, b), dim=-1), backend.stack((c, d), dim=-1)), dim=-2
    )


def _row_fold(x, y):
    return x * 1.0, y - y * y


def _row_fold_jacobian(x, y):
    one, zero = backend.ones_like(x), backend.zeros_like(x)
    return _stack_2x2(one, zero, zero, 1.0 - 2.0 * y)


def _sis(x, y):
    r = backend.sqrt(x * x + y * y)
    return x - x / r, y - y / r


def _sis_jacobian(x, y):
    r = backend.sqrt(x * x + y * y)
    k = 1.0 / r**3
    return _stack_2x2(
        1.0 - 1.0 / r + k * x * x, k * x * y, k * x * y, 1.0 - 1.0 / r + k * y * y
    )


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


def test_the_fold_strip_has_its_exact_edges_and_area():
    lens = build_adaptive_mesh(_row_fold, _row_fold_jacobian, 2.0, 8, 0.002, y0=0.5)
    mag = build_magnification_mesh(lens, 8, 0.005, mu_min=4.0, fov=1.0, x0=0.0, y0=0.0)
    b = to_np(magnified_regions(mag, 4.0).source)[:, 1]
    lower = b < 0.22
    assert lower.any() and (~lower).any()
    assert np.abs(b[lower] - 0.1875).max() < 0.02
    # A crossing on an edge across the caustic lies between its two samples.
    assert (np.abs(b[~lower] - 0.25) <= mag.src_tol).all()
    area, _ = magnified_area(mag, 4.0)
    assert abs(float(to_np(area)) - 0.0625) < 0.008


@pytest.fixture(scope="module")
def sis_lens():
    return build_adaptive_mesh(_sis, _sis_jacobian, 4.0, 8, 0.005, centres=[(0.0, 0.0)])


@pytest.fixture(scope="module")
def sis_mag(sis_lens):
    return build_magnification_mesh(
        sis_lens, 8, 0.005, mu_min=4.0, fov=1.5, x0=0.0, y0=0.0
    )


def test_the_sis_region_is_the_disk_of_radius_two_over_mu_min(sis_mag):
    regions = magnified_regions(sis_mag, 4.0)
    assert to_np(regions.closed).all()
    r = np.linalg.norm(to_np(regions.source), axis=1)
    assert np.abs(r - 0.5).max() < 0.06
    area, complete = magnified_area(sis_mag, 4.0)
    assert bool(to_np(complete))
    assert abs(float(to_np(area)) / (np.pi * 0.25) - 1.0) < 0.08


def test_sweep_and_threshold_modes_agree_on_the_sis_disk(sis_lens, sis_mag):
    # Sweep mode resolves log(1 + mu) to rtol rather than a threshold to the
    # floor, so its boundary sits about rtol / |d log(1 + mu) / d beta| =
    # 0.02 / 1.6 from the threshold mesh's: some 5% of the disk's area.
    sweep = build_magnification_mesh(
        sis_lens, 8, 0.01, rtol=0.02, fov=1.5, x0=0.0, y0=0.0
    )
    a_threshold, _ = magnified_area(sis_mag, 4.0)
    a_sweep, _ = magnified_area(sweep, 4.0)
    assert abs(float(to_np(a_sweep)) / float(to_np(a_threshold)) - 1.0) < 0.06


def test_the_cored_region_is_a_disk_and_an_annulus_inside_the_radial_caustic():
    lens = build_adaptive_mesh(_cored, _cored_jacobian, 4.5, 9, 0.005)
    mag = build_magnification_mesh(lens, 8, 0.005, mu_min=6.0, fov=1.6, x0=0.0, y0=0.0)
    r_disk, r_annulus, r_caustic = _cored_crossings(6.0)
    r = np.linalg.norm(to_np(magnified_regions(mag, 6.0).source), axis=1)
    radii = np.array([r_disk, r_annulus, r_caustic])
    assert np.abs(r[:, None] - radii[None, :]).min(axis=1).max() < 0.06
    area, _ = magnified_area(mag, 6.0)
    exact = np.pi * (r_disk**2 + r_caustic**2 - r_annulus**2)
    assert abs(float(to_np(area)) / exact - 1.0) < 0.12
