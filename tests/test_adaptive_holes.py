"""Holes around lens centres: merging centres, sampling hole curves, storing them on the mesh."""

import numpy as np
import pytest

from caustics.backend_obj import backend
from caustics.lenses.func import adaptive as new


def to_np(x):
    return backend.to_numpy(x)


def f64(x):
    return backend.as_array(np.asarray(x, dtype=np.float64), dtype=backend.float64)


# ---------------------------------------------------------------------------
# merge_centres
# ---------------------------------------------------------------------------


def test_empty_holes_has_no_hole():
    holes = new.empty_holes()
    assert tuple(holes.centres.shape) == (0, 2)
    assert tuple(holes.lens.shape) == (0, 2) and tuple(holes.source.shape) == (0, 2)
    for field in ("radius", "angle", "growth"):
        assert tuple(getattr(holes, field).shape) == (0,), field
    assert to_np(holes.offsets).tolist() == [0]


@pytest.mark.parametrize(
    "centres", [None, [], np.zeros((0, 2))], ids=["None", "empty list", "(0, 2)"]
)
def test_merge_centres_of_nothing_is_empty(centres):
    got, radius = new.merge_centres(centres, 0.01)
    assert tuple(got.shape) == (0, 2) and tuple(radius.shape) == (0,)


@pytest.mark.parametrize(
    "centres",
    [[(0.3, -0.2)], ((0.3, -0.2),), np.array([[0.3, -0.2]]), f64([[0.3, -0.2]])],
    ids=["list", "tuple", "numpy", "backend"],
)
def test_a_lone_centre_of_any_array_like_keeps_its_exact_position(centres):
    got, radius = new.merge_centres(centres, 0.01)
    assert to_np(got).tolist() == [[0.3, -0.2]]
    assert to_np(radius).tolist() == [0.01]


def test_coincident_centres_share_one_hole_at_their_common_position():
    got, radius = new.merge_centres([(2.0, 2.0), (0.3, 0.1), (0.3, 0.1)], 0.01)
    assert to_np(got).tolist() == [[0.3, 0.1], [2.0, 2.0]]
    assert to_np(radius).tolist() == [0.01, 0.01]


def test_centres_closer_than_twice_min_img_sep_merge_at_their_mean():
    got, radius = new.merge_centres([(0.0, 0.0), (0.015, 0.0)], 0.01)
    assert np.allclose(to_np(got), [[0.0075, 0.0]], rtol=0, atol=1e-15)
    assert np.allclose(to_np(radius), [0.0175], rtol=0, atol=1e-15)


def test_centres_exactly_twice_min_img_sep_apart_keep_their_own_holes():
    got, _ = new.merge_centres([(0.0, 0.0), (0.02, 0.0)], 0.01)
    assert got.shape[0] == 2


def test_merging_repeats_until_no_two_disks_overlap():
    """``a`` and ``b`` link; ``c`` is farther than 2 * min_img_sep from both,
    yet its disk overlaps theirs once they merge, so all three share a hole."""
    pts = np.array([(0.0, 0.0), (0.019, 0.0), (0.0095, 0.025)])
    got, radius = new.merge_centres(pts, 0.01)
    mean = pts.mean(axis=0)
    assert got.shape[0] == 1
    assert np.allclose(to_np(got)[0], mean, rtol=0, atol=1e-15)
    want = 0.01 + np.hypot(*(pts - mean).T).max()
    assert np.isclose(to_np(radius)[0], want, rtol=0, atol=1e-15)


def test_merged_holes_are_disjoint_hold_their_centres_and_ignore_input_order():
    rng = np.random.default_rng(3)
    pts = np.concatenate(
        [rng.uniform(-1.0, 1.0, (20, 2)), rng.uniform(0.0, 0.03, (10, 2))]
    )
    got, radius = new.merge_centres(pts, 0.01)
    g, r = to_np(got), to_np(radius)
    gap = np.hypot(*(g[:, None, :] - g[None, :, :]).transpose(2, 0, 1))
    np.fill_diagonal(gap, np.inf)
    assert (gap >= r[:, None] + r[None, :]).all()
    held = np.hypot(*(pts[:, None, :] - g[None, :, :]).transpose(2, 0, 1)) < r[None, :]
    assert (held.sum(axis=1) == 1).all()
    for seed in range(3):
        perm = np.random.default_rng(seed).permutation(len(pts))
        again, again_radius = new.merge_centres(pts[perm], 0.01)
        assert np.array_equal(to_np(again), g)
        assert np.array_equal(to_np(again_radius), r)


@pytest.mark.parametrize(
    "centres",
    [np.zeros(3), np.zeros((2, 3)), [(0.0, np.nan)], [(np.inf, 0.0)]],
    ids=["flat", "three columns", "nan", "inf"],
)
def test_merge_centres_rejects_malformed_centres(centres):
    with pytest.raises(ValueError, match="centres must"):
        new.merge_centres(centres, 0.01)
