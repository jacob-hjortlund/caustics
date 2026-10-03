"""The lens mesh: its criterion and band, its build, extension and closed build."""

import numpy as np
import pytest

from caustics.lenses.func.adaptive.band import build_band, in_band
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
from caustics.lenses.func.adaptive.geometry import ROOT_CLASS, ROOT_SHAPES
from caustics.lenses.func.adaptive.lattice import lattice_xy, make_lattice

from adaptive_maps import f64, i64, to_np

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


A = np.array([[0.7, 0.1], [-0.2, 0.9]])


def affine(p):
    return p @ A.T


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
