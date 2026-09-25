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


# ---------------------------------------------------------------------------
# sample_holes
# ---------------------------------------------------------------------------


def sis_raytrace(c, b):
    """A singular isothermal sphere at ``c``: ``f(c + r u) = c + (r - b) u``."""

    def raytrace(x, y):
        dx, dy = x - c[0], y - c[1]
        r = backend.sqrt(dx * dx + dy * dy)
        return x - b * dx / r, y - b * dy / r

    return raytrace


def point_mass_raytrace(c, theta_e):
    """A point mass at ``c``: ``f(c + r u) = c + (r - theta_e**2 / r) u``."""

    def raytrace(x, y):
        dx, dy = x - c[0], y - c[1]
        r2 = dx * dx + dy * dy
        return x - theta_e**2 * dx / r2, y - theta_e**2 * dy / r2

    return raytrace


def affine_raytrace(x, y):
    return 0.7 * x + 0.1 * y, -0.2 * x + 0.9 * y


def nan_where(raytrace, where):
    """``raytrace`` with NaN wherever ``where(x, y)`` holds."""

    def broken(x, y):
        bx, by = raytrace(x, y)
        nan = backend.where(
            where(x, y), backend.zeros_like(x) + float("nan"), backend.zeros_like(x)
        )
        return bx + nan, by + nan

    return broken


def sample(raytrace, centres, radius, min_img_sep, batch_size=None):
    """``sample_holes`` through ``make_raytrace``, and every point it traced."""
    calls = []

    def recorded(x, y):
        calls.append(np.stack([to_np(x), to_np(y)], axis=-1))
        return raytrace(x, y)

    holes = new.sample_holes(
        new.make_raytrace(recorded, None),
        f64(centres),
        f64(radius),
        min_img_sep,
        batch_size,
    )
    return holes, calls


def test_an_sis_hole_curve_is_its_analytic_circle_sampled_to_min_img_sep():
    c, b, r = (0.3, -0.2), 1.0, 0.01
    holes, _ = sample(sis_raytrace(c, b), [c], [r], r)
    angle, lens, source = to_np(holes.angle), to_np(holes.lens), to_np(holes.source)
    u = np.stack([np.cos(angle), np.sin(angle)], axis=-1)
    assert to_np(holes.offsets).tolist() == [0, angle.size]
    assert angle[0] >= 0 and angle[-1] < 2 * np.pi and (np.diff(angle) > 0).all()
    assert np.allclose(lens, np.array(c) + r * u, rtol=0, atol=1e-14)
    assert np.allclose(source, np.array(c) + (r - b) * u, rtol=0, atol=1e-12)
    chords = np.hypot(*(np.roll(source, -1, axis=0) - source).T)
    assert chords.max() <= r
    assert abs(to_np(holes.growth)[0]) < 0.01


def test_a_point_mass_hole_curve_grows_as_the_hole_shrinks():
    c = (0.1, 0.2)
    holes, _ = sample(point_mass_raytrace(c, 0.1), [c], [0.01], 0.01)
    assert abs(to_np(holes.growth)[0] + 1.0) < 0.01


def test_a_regular_point_hole_curve_shrinks_with_the_hole():
    holes, _ = sample(affine_raytrace, [(0.4, -0.3)], [0.01], 0.01)
    assert abs(to_np(holes.growth)[0] - 1.0) < 1e-9


def test_a_hole_curve_too_large_to_resolve_stops_at_the_cap_and_warns():
    """A 1" point mass maps a 0.005" circle to a loop of radius about 200":
    ``2**16`` samples cannot bring its chords down to 0.005"."""
    c = (0.0, 0.0)
    with pytest.warns(UserWarning, match="reached 65536 samples"):
        holes, _ = sample(point_mass_raytrace(c, 1.0), [c], [0.005], 0.005)
    assert to_np(holes.offsets)[-1] <= new.HOLE_MAX_SAMPLES


def test_a_lens_not_finite_on_a_hole_circle_raises_naming_the_centre():
    c = (0.4, -0.3)
    broken = nan_where(affine_raytrace, lambda x, y: x > c[0])
    with pytest.raises(
        ValueError,
        match=r"not finite on the hole circle of radius 0\.01 around \(0\.4, -0\.3\)",
    ):
        sample(broken, [c], [0.01], 0.01)


def test_the_growth_circle_at_a_quarter_radius_must_be_finite_too():
    c = (0.4, -0.3)
    broken = nan_where(
        affine_raytrace,
        lambda x, y: (x - c[0]) ** 2 + (y - c[1]) ** 2 < 0.005**2,
    )
    with pytest.raises(ValueError, match=r"radius 0\.0025"):
        sample(broken, [c], [0.01], 0.01)


def test_holes_raytrace_only_their_circles_and_batching_changes_nothing():
    c, r = (0.3, -0.2), 0.01
    holes, calls = sample(sis_raytrace(c, 1.0), [c], [r], r)
    d = np.hypot(*(np.concatenate(calls) - np.array(c)).T)
    assert (
        np.isclose(d, r, rtol=0, atol=1e-14) | np.isclose(d, r / 4, rtol=0, atol=1e-14)
    ).all()
    batched, batched_calls = sample(sis_raytrace(c, 1.0), [c], [r], r, batch_size=100)
    assert max(len(x) for x in batched_calls) <= 100
    for field in new.CentreHoles._fields:
        assert np.array_equal(
            to_np(getattr(batched, field)), to_np(getattr(holes, field))
        ), field


def test_holes_sampled_together_match_holes_sampled_alone():
    """One call per round covers every hole; no hole's samples leak into another's."""
    lens = sis_raytrace((0.3, -0.2), 1.0)
    pairs = [((0.3, -0.2), 0.01), ((1.5, 1.0), 0.02)]
    both, _ = sample(lens, [c for c, _ in pairs], [r for _, r in pairs], 0.01)
    off = to_np(both.offsets)
    for h, (c, r) in enumerate(pairs):
        alone, _ = sample(lens, [c], [r], 0.01)
        for field in ("angle", "lens", "source"):
            assert np.array_equal(
                to_np(getattr(both, field))[off[h] : off[h + 1]],
                to_np(getattr(alone, field)),
            ), field
        assert np.array_equal(to_np(both.growth)[h], to_np(alone.growth)[0])
