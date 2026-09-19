import numpy as np
import pytest

from caustics.backend_obj import backend
from caustics.cosmology import FlatLambdaCDM
from caustics.lenses import SIE, Point
from caustics.lenses.adaptive import (
    LeafStatus,
    BuildStats,
    Mesh,
    _dedup_representatives,
    build_adaptive_mesh,
)
from caustics.lenses.func import forward_raytrace_rootfind
from caustics.lenses.func.old_adaptive import shape_matrix
from caustics.utils import meshgrid

RNG = np.random.default_rng(20260904)


def to_np(x):
    return backend.to_numpy(x)


def as_arr(x):
    return backend.as_array(np.asarray(x, dtype=np.float64))


def signed_area(tri):
    """Twice the signed area of a (..., 3, 2) triangle."""
    P = shape_matrix(tri)
    return P[..., 0, 0] * P[..., 1, 1] - P[..., 0, 1] * P[..., 1, 0]


def make_counting_raytrace(fn):
    """Wrap a numpy (N,2)->(N,2) map as a backend raytrace, counting evaluations."""
    calls = {"points": 0, "batches": 0}

    def raytrace(x, y):
        xy = np.stack([backend.to_numpy(x), backend.to_numpy(y)], axis=-1)
        calls["points"] += xy.shape[0]
        calls["batches"] += 1
        out = fn(xy)
        return backend.as_array(out[:, 0]), backend.as_array(out[:, 1])

    return raytrace, calls


AFFINE = np.array([[0.7, 0.1], [-0.2, 0.9]])


def localised_fold(p):
    """Affine away from a narrow band, curved and fold-bearing inside it.

    Outside |y| < 0.5 the map is exactly affine with sigma_min = 0.6, so those
    triangles converge at level 0. Inside, beta2 = 0.6y + y^2 - 0.25 is curved and
    its Jacobian 0.6 + 2y changes sign at y = -0.3, so both the deviation test and
    the parity test fire. The result is a mesh with real level transitions for the
    cascade, closure, and crack tests to work on.

    Continuous at |y| = 0.5, where y^2 - 0.25 vanishes.

    Do NOT replace the bend with a constant outside the band (e.g. 0.25*sign(y)):
    that makes the map degenerate in y everywhere outside, so sigma_min == 0, every
    triangle splits, and the mesh refines uniformly to max_level with no level
    transitions at all -- silently voiding every test that depends on them.
    """
    y = p[:, 1]
    bend = np.where(np.abs(y) < 0.5, y**2 - 0.25, 0.0)
    return np.stack([p[:, 0], 0.6 * y + bend], axis=-1)


def build(fn, fov=4.0, init_res=4, min_img_sep=0.25, **kw):
    raytrace, calls = make_counting_raytrace(fn)
    mesh = build_adaptive_mesh(raytrace, fov, init_res, min_img_sep, **kw)
    return mesh, calls


def test_geometry_wrappers_match_a_manual_gather():
    mesh, _ = build(localised_fold, min_img_sep=0.05)
    beta = RNG.uniform(-1.5, 1.5, size=(30, 2))
    idx, off, bary = mesh.query(beta)
    for name, verts in (
        ("triangles_lens", mesh.vertices_lens),
        ("triangles_source", mesh.vertices_source),
    ):
        via_beta, offsets = getattr(mesh, name)(beta)
        via_idx = getattr(mesh, name)(leaf_indices=idx)
        manual = verts[mesh.leaves[idx]]
        assert np.array_equal(backend.to_numpy(via_beta), backend.to_numpy(manual))
        assert np.array_equal(backend.to_numpy(via_idx), backend.to_numpy(manual))
        assert np.array_equal(backend.to_numpy(offsets), backend.to_numpy(off))


def cross2(u, v):
    """Scalar cross product of 2-D vectors.

    ``np.cross`` on 2-vectors is deprecated in NumPy 2.0 and emits a
    ``DeprecationWarning`` per call -- 141 of them across this file's runs, and
    a hard failure under ``-W error::DeprecationWarning``. This is the same
    value, computed the way :func:`signed_area` above already does it, and is
    bit-identical to the ``np.cross`` result.
    """
    return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]


def test_seeds_lie_inside_their_lens_triangle():
    mesh, _ = build(localised_fold, min_img_sep=0.05)
    beta = RNG.uniform(-1.5, 1.5, size=(50, 2))
    idx, off, bary = mesh.query(beta)
    seed, offsets = mesh.seeds(beta)
    direct = mesh.seeds(leaf_indices=idx, bary=bary)
    assert np.allclose(backend.to_numpy(seed), backend.to_numpy(direct))
    assert np.array_equal(backend.to_numpy(offsets), backend.to_numpy(off))
    tri = backend.to_numpy(mesh.triangles_lens(leaf_indices=idx))
    s = backend.to_numpy(seed)
    for k in range(len(s)):
        w = np.array(
            [
                cross2(tri[k, (i + 1) % 3] - s[k], tri[k, (i + 2) % 3] - s[k])
                for i in range(3)
            ]
        )
        assert (w >= -1e-9).all() or (w <= 1e-9).all()


@pytest.mark.parametrize("name", ["triangles_lens", "triangles_source", "seeds"])
def test_wrappers_reject_ambiguous_arguments(name):
    mesh, _ = build(lambda p: p @ AFFINE.T)
    fn = getattr(mesh, name)
    with pytest.raises(ValueError):
        fn()
    with pytest.raises(ValueError):
        fn(np.zeros((1, 2)), leaf_indices=backend.as_array([0]))


def test_seeds_requires_bary_with_leaf_indices():
    mesh, _ = build(lambda p: p @ AFFINE.T)
    with pytest.raises(ValueError, match="bary"):
        mesh.seeds(leaf_indices=backend.as_array([0]))


def test_public_symbols_are_re_exported():
    import caustics

    assert caustics.build_adaptive_mesh is build_adaptive_mesh
    assert caustics.Mesh is Mesh
    assert caustics.LeafStatus is LeafStatus
    assert caustics.BuildStats is BuildStats
    assert caustics.func.sigma_min_2x2 is not None
    assert caustics.func.triangle_weights is not None


def dedup(points, tol):
    """Greedy clustering; returns one representative per cluster, sorted."""
    keep = []
    for p in points:
        if all(np.linalg.norm(p - q) >= tol for q in keep):
            keep.append(p)
    return np.array(sorted(keep, key=tuple)) if keep else np.zeros((0, 2))


def test_sie_candidates_recover_forward_raytrace_images(device):
    """Spec test 26, split into the two contracts this module actually owns.

    1. **Coverage** -- every image ``forward_raytrace`` finds has a candidate
       seed within ``min_img_sep``. That is exactly what :meth:`Mesh.seeds`
       promises: the hit leaf's own affine map is the one step 7 bounds, so the
       seed is accurate to ``min_img_sep`` by construction. Measured across
       these three source points, the worst distance is 2.5e-3 against a 1e-2
       tolerance -- four times better than the guarantee.
    2. **No spurious images** -- every candidate the root-finder converges on is
       a genuine image.

    This deliberately does **not** assert that the refined set has the same
    cardinality as ``forward_raytrace``'s. For ``sp = [0.2, 0.2]`` this SIE has
    a central image at radius ~7e-4, *inside* its own softening radius
    ``s = 1e-3``, where the Jacobian is nearly degenerate.
    ``forward_raytrace_rootfind`` diverges there (residual 0.31 against a 1e-3
    filter) even though the mesh does supply a seed 2.5e-3 away from it. A
    cardinality assertion would therefore be reporting the downstream
    root-finder's convergence on a softened singularity, not this module's
    coverage -- conflating two systems in one number. Contract 1 fails loudly if
    coverage is ever genuinely lost, which is the property worth pinning.
    """
    lens = SIE(
        name="sie",
        cosmology=FlatLambdaCDM(name="cosmo"),
        z_l=0.5,
        z_s=1.5,
        x0=0.0,
        y0=0.0,
        q=0.4,
        phi=np.pi / 5,
        Rein=1.0,
        s=1e-3,
    ).to(device)
    mesh = build_adaptive_mesh(
        lens.raytrace, fov=5.0, init_res=32, min_img_sep=1e-2, device=device
    )
    for sp in ([0.2, 0.2], [0.05, -0.05], [1.4, 1.1]):
        sx = backend.as_array(sp[0], device=device)
        sy = backend.as_array(sp[1], device=device)
        ex, ey = lens.forward_raytrace(sx, sy)
        expected = dedup(
            np.stack([backend.to_numpy(ex), backend.to_numpy(ey)], axis=-1), 1e-2
        )
        # The coverage contract below goes vacuous if `dedup` ever returned an
        # empty `expected`: `.all()` over an empty array is True.
        assert expected.shape[0] > 0, f"{sp}: forward_raytrace found no images"
        seed, offsets = mesh.seeds(backend.as_array(np.asarray([sp])))
        seed = backend.to_numpy(seed)
        assert seed.shape[0] >= expected.shape[0], "candidates must cover the images"
        refined = forward_raytrace_rootfind(
            backend.as_array(seed[:, 0], device=device),
            backend.as_array(seed[:, 1], device=device),
            sx,
            sy,
            lens.raytrace,
        )
        refined = backend.to_numpy(refined)
        bx, by = lens.raytrace(
            backend.as_array(refined[:, 0], device=device),
            backend.as_array(refined[:, 1], device=device),
        )
        residual = np.linalg.norm(
            np.stack([backend.to_numpy(bx), backend.to_numpy(by)], -1) - np.asarray(sp),
            axis=-1,
        )
        got = dedup(refined[residual < 1e-3], 1e-2)
        # Contract 1: coverage. Every image has a seed within min_img_sep.
        nearest = np.linalg.norm(expected[:, None, :] - seed[None, :, :], axis=-1).min(
            axis=1
        )
        assert (
            nearest < 1e-2
        ).all(), f"{sp}: uncovered image, worst seed distance {nearest.max():.3e}"
        # Contract 2: no spurious images among those the root-finder converged on.
        assert got.shape[0] > 0, f"{sp}: nothing converged"
        for p in got:
            assert (
                np.linalg.norm(expected - p, axis=-1).min() < 1e-2
            ), f"{sp}: converged to {p}, which is not a forward_raytrace image"


def test_point_mass_recovers_the_analytic_image_pair():
    lens = Point(
        name="pt",
        cosmology=FlatLambdaCDM(name="cosmo"),
        z_l=0.5,
        z_s=1.5,
        x0=0.0,
        y0=0.0,
        Rein=1.0,
        s=1e-6,
    )
    mesh = build_adaptive_mesh(lens.raytrace, fov=8.0, init_res=64, min_img_sep=1e-2)
    b = 0.4
    seed, offsets = mesh.seeds(backend.as_array(np.array([[b, 0.0]])))
    seed = backend.to_numpy(seed)
    # theta_pm = (b +- sqrt(b^2 + 4 Rein^2)) / 2, both on the x axis
    expected = np.array([(b + np.sqrt(b**2 + 4)) / 2, (b - np.sqrt(b**2 + 4)) / 2])
    for theta in expected:
        assert np.abs(seed[:, 0] - theta).min() < 5e-2
        assert np.abs(seed[np.argmin(np.abs(seed[:, 0] - theta)), 1]) < 5e-2


def test_build_and_query_run_on_the_configured_device(device):
    """Build and query complete on the configured device and return sane CSR.

    Note what this does **not** verify: every array is converted through
    ``backend.to_numpy`` before inspection, so this test cannot distinguish
    "computed on the requested device" from "computed elsewhere and converted
    back". It pins that the pipeline runs end to end under the ``device``
    fixture and returns coherent results, not placement.
    """
    lens = SIE(
        name="sie",
        cosmology=FlatLambdaCDM(name="cosmo"),
        z_l=0.5,
        z_s=1.5,
        x0=0.0,
        y0=0.0,
        q=0.7,
        phi=0.0,
        Rein=1.0,
        s=1e-3,
    ).to(device)
    mesh = build_adaptive_mesh(
        lens.raytrace, fov=4.0, init_res=8, min_img_sep=0.1, device=device
    )
    idx, off, bary = mesh.query(backend.as_array(np.array([[0.1, 0.1], [3.0, 3.0]])))
    off_np = backend.to_numpy(off)
    bary_np = backend.to_numpy(bary)
    assert off_np.shape == (3,)
    # The hit/miss pair is what makes this falsifiable. `off.shape` is
    # `(B + 1,)` for any B by the CSR contract, and `np.isfinite` is vacuously
    # True on an empty array, so shape-plus-finiteness alone would pass even if
    # the query silently returned nothing for both points. Measured: [0, 3, 3].
    assert off_np[1] > off_np[0], "the interior source point must hit a leaf"
    assert off_np[2] == off_np[1], "the far exterior point must hit nothing"
    assert bary_np.shape[0] == off_np[-1], "bary rows must match the CSR total"
    assert np.isfinite(bary_np).all()


def test_dedup_collapses_points_closer_than_the_tolerance():
    points = as_arr([[0.0, 0.0], [0.0, 0.001], [1.0, 0.0]])
    keep = to_np(_dedup_representatives(points, np.array([3]), 0.01))
    assert keep.sum() == 2, "the coincident pair must collapse to one image"
    assert keep[2], "the distant point must survive"


def test_dedup_keeps_points_separated_by_the_tolerance():
    """Separation *of* min_img_sep means distinct, matching the build contract.

    The size floor is ``l_max <= min_img_sep`` and step 7 hides no pair separated
    by more than ``min_img_sep / 4``, so the boundary belongs to the distinct
    side. Adjacency is ``d < tol``, not ``<=``.
    """
    points = as_arr([[0.0, 0.0], [0.01, 0.0]])
    keep = to_np(_dedup_representatives(points, np.array([2]), 0.01))
    assert keep.sum() == 2


def test_dedup_counts_connected_components_not_greedy_clusters():
    """Order independence, which greedy clustering does not have.

    Three collinear points spaced ``0.9 * tol`` apart form one connected
    component. Greedy returns 2 for the order below and 1 for ``[1, 0, 2]`` --
    the answer would depend on the order ``query`` happened to emit candidates
    in, which is not something a multiplicity map may depend on.
    """
    p = np.array([[0.0, 0.0], [0.009, 0.0], [0.018, 0.0]])
    counts = np.array([3])
    base = to_np(_dedup_representatives(as_arr(p), counts, 0.01)).sum()
    assert base == 1, f"one chained component expected, got {base}"
    for order in ([1, 0, 2], [2, 1, 0], [0, 2, 1], [2, 0, 1]):
        got = to_np(_dedup_representatives(as_arr(p[order]), counts, 0.01)).sum()
        assert got == base, f"order {order} gave {got}, not {base}"


def test_dedup_never_merges_across_blocks():
    """Two source points whose images coincide must not collapse into one.

    The padded ``(B, M, M)`` formulation makes cross-block bleed the natural bug
    here, and it would silently halve a multiplicity map.
    """
    points = as_arr([[0.0, 0.0], [0.0, 0.0]])
    keep = to_np(_dedup_representatives(points, np.array([1, 1]), 0.01))
    assert keep.sum() == 2, "identical points in different blocks are distinct"


def test_dedup_handles_ragged_blocks_and_empty_blocks():
    """Padding must not invent images in a block that found none."""
    points = as_arr([[0.0, 0.0], [5.0, 5.0], [5.0, 5.0005]])
    keep = to_np(_dedup_representatives(points, np.array([1, 0, 2]), 0.01))
    assert keep.tolist() == [True, True, False]


def test_dedup_is_unchanged_by_bucketing_on_randomised_blocks():
    """The bucketed kernel must agree with a per-block reference exactly.

    Blocks are grouped by count and run at their own M rather than padded to
    the global maximum, so the risk is a scatter that puts one block's answer
    on another block's rows. Running each block *alone* through the same
    function is the independent reference: a single-block call has nothing to
    mis-scatter.

    What this does NOT check: both sides call the same clustering kernel
    (:func:`_dedup_block_group` via :func:`_dedup_representatives`), so a bug
    inside that kernel itself -- a wrong sentinel, a wrong iteration bound --
    reproduces identically on both sides and is invisible to this comparison.
    """
    rng = np.random.default_rng(20260910)
    # The all-empty vector is explicit: 20 random draws from this seed never
    # produce one, and it is the case that reaches the `total == 0` early exit.
    cases = [np.zeros(4, dtype=np.int64)]
    cases += [rng.integers(0, 6, size=rng.integers(1, 12)) for _ in range(20)]
    for counts in cases:
        pts = rng.normal(scale=0.01, size=(int(counts.sum()), 2)).reshape(-1, 2)
        got = to_np(_dedup_representatives(as_arr(pts), counts, 0.01))
        starts = np.cumsum(counts) - counts
        want = np.concatenate(
            [
                to_np(
                    _dedup_representatives(as_arr(pts[s : s + c]), np.array([c]), 0.01)
                )
                for s, c in zip(starts, counts)
            ]
            + [np.zeros(0, dtype=bool)]
        )
        assert got.tolist() == want.tolist(), f"counts={counts.tolist()}"


def test_dedup_keeps_exactly_one_point_per_singleton_block():
    """Blocks of one bypass the clustering kernel; they must still be kept."""
    points = as_arr([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    keep = to_np(_dedup_representatives(points, np.array([1, 1, 1]), 0.01))
    assert keep.tolist() == [True, True, True]


def test_dedup_mixes_singleton_and_clustered_blocks_in_order():
    """The bypass and the kernel write into one output; order must survive.

    Block 0 is a singleton, block 1 collapses to one image, block 2 is a
    singleton again. A scatter that appends the bypassed blocks after the
    clustered ones would pass every count-based assertion and still return the
    representatives in the wrong rows.
    """
    points = as_arr([[9.0, 9.0], [0.0, 0.0], [0.0, 0.001], [5.0, 5.0]])
    keep = to_np(_dedup_representatives(points, np.array([1, 2, 1]), 0.01))
    assert keep.tolist() == [True, True, False, True]


def sie_fixture(device=None):
    lens = SIE(
        name="sie",
        cosmology=FlatLambdaCDM(name="cosmo"),
        z_l=0.5,
        z_s=1.5,
        x0=0.0,
        y0=0.0,
        q=0.4,
        phi=np.pi / 5,
        Rein=1.0,
        s=1e-3,
    )
    if device is not None:
        lens = lens.to(device)
    mesh = build_adaptive_mesh(
        lens.raytrace, fov=5.0, init_res=32, min_img_sep=1e-2, device=device
    )
    return lens, mesh


def test_forward_raytrace_finds_no_spurious_sie_images():
    """Every returned image is an image `lens.forward_raytrace` also finds.

    The residual and leaf-or-ball filters exist for this: a stalled
    Levenberg-Marquardt solve leaves a point that is not an image, and dedup will
    not absorb it when it sits further than `min_img_sep` from a real one.
    """
    lens, mesh = sie_fixture()
    for sp in ([0.2, 0.2], [0.05, -0.05]):
        images, counts = mesh.forward_raytrace(as_arr([sp]), lens.raytrace)
        images = to_np(images)
        assert to_np(counts).tolist() == [images.shape[0]]
        assert images.shape[0] > 0, f"{sp}: no images found"
        # The reference path is float32-only: `LensBase.forward_raytrace` raises
        # "expected scalar type Float but found Double" on float64 input. The mesh
        # itself is float64, so only this comparison call is narrowed.
        ex, ey = lens.forward_raytrace(backend.as_array(sp[0]), backend.as_array(sp[1]))
        expected = dedup(np.stack([to_np(ex), to_np(ey)], axis=-1), 1e-2)
        for p in images:
            assert (
                np.linalg.norm(expected - p, axis=-1).min() < 1e-2
            ), f"{sp}: returned {p}, which is not a forward_raytrace image"


def test_forward_raytrace_covers_every_sie_image():
    """The converse contract: no image is dropped by the filters or the dedup."""
    lens, mesh = sie_fixture()
    for sp in ([0.2, 0.2], [0.05, -0.05]):
        images, _ = mesh.forward_raytrace(as_arr([sp]), lens.raytrace)
        images = to_np(images)
        # The reference path is float32-only: `LensBase.forward_raytrace` raises
        # "expected scalar type Float but found Double" on float64 input. The mesh
        # itself is float64, so only this comparison call is narrowed.
        ex, ey = lens.forward_raytrace(backend.as_array(sp[0]), backend.as_array(sp[1]))
        expected = dedup(np.stack([to_np(ex), to_np(ey)], axis=-1), 1e-2)
        assert expected.shape[0] > 0, f"{sp}: reference found no images"
        nearest = np.linalg.norm(expected[:, None, :] - images[None, :, :], axis=-1)
        worst = nearest.min(axis=1).max()
        assert worst < 1e-2, f"{sp}: uncovered image, worst distance {worst:.3e}"


def test_forward_raytrace_recovers_the_analytic_point_mass_pair():
    """The one case with a closed form: exactly two images, at known positions.

    ``theta = (b +- sqrt(b^2 + 4 Rein^2)) / 2``, both on the x axis. Asserting
    the cardinality is only defensible here, where it is a theorem rather than a
    measurement.
    """
    lens = Point(
        name="pt",
        cosmology=FlatLambdaCDM(name="cosmo"),
        z_l=0.5,
        z_s=1.5,
        x0=0.0,
        y0=0.0,
        Rein=1.0,
        s=1e-6,
    )
    mesh = build_adaptive_mesh(lens.raytrace, fov=8.0, init_res=64, min_img_sep=1e-2)
    b = 0.4
    images, counts = mesh.forward_raytrace(as_arr([[b, 0.0]]), lens.raytrace)
    images = to_np(images)
    assert to_np(counts).tolist() == [2], f"expected 2 images, got {images}"
    expected = np.sort([(b + np.sqrt(b**2 + 4)) / 2, (b - np.sqrt(b**2 + 4)) / 2])
    assert np.allclose(np.sort(images[:, 0]), expected, atol=1e-4)
    assert np.abs(images[:, 1]).max() < 1e-4


def test_forward_raytrace_batches_independently():
    """A batched call equals looping one source at a time.

    Falsifiable against the two bugs the ragged layout invites: targets paired
    with the wrong seeds, and dedup merging images of different sources.
    """
    lens, mesh = sie_fixture()
    points = [[0.2, 0.2], [0.05, -0.05], [0.4, -0.3]]
    images, counts = mesh.forward_raytrace(as_arr(points), lens.raytrace)
    counts = to_np(counts)
    assert counts.shape == (3,)
    assert counts.sum() == to_np(images).shape[0]
    offsets = np.concatenate(([0], np.cumsum(counts)))
    for i, sp in enumerate(points):
        one, one_counts = mesh.forward_raytrace(as_arr([sp]), lens.raytrace)
        assert to_np(one_counts).tolist() == [counts[i]], f"{sp}: count differs"
        block = to_np(images)[offsets[i] : offsets[i + 1]]
        assert np.allclose(block, to_np(one), atol=1e-8), f"{sp}: images differ"


def test_forward_raytrace_batch_size_does_not_change_the_answer():
    lens, mesh = sie_fixture()
    beta = as_arr([[0.2, 0.2], [0.05, -0.05], [0.4, -0.3], [0.0, 0.3]])
    full, full_counts = mesh.forward_raytrace(beta, lens.raytrace)
    for size in (1, 2, 3):
        part, part_counts = mesh.forward_raytrace(beta, lens.raytrace, batch_size=size)
        assert to_np(part_counts).tolist() == to_np(full_counts).tolist()
        assert np.allclose(to_np(part), to_np(full), atol=1e-8)


def test_forward_raytrace_returns_an_empty_block_outside_the_source_plane():
    """A source the mesh never maps to has zero images, not a raised error."""
    lens, mesh = sie_fixture()
    images, counts = mesh.forward_raytrace(as_arr([[50.0, 50.0]]), lens.raytrace)
    assert to_np(counts).tolist() == [0]
    assert to_np(images).shape == (0, 2)


def test_forward_raytrace_rejects_an_unknown_method():
    lens, mesh = sie_fixture()
    with pytest.raises(ValueError, match="rootfind"):
        mesh.forward_raytrace(as_arr([[0.05, 0.02]]), lens.raytrace, method="nope")


def test_multiplicity_map_rejects_an_unknown_method():
    lens, mesh = sie_fixture()
    with pytest.raises(ValueError, match="dedup"):
        mesh.multiplicity_map(lens.raytrace, pixelscale=0.2, method="nope")


def test_dedup_method_never_calls_raytrace():
    """The whole point of `method="dedup"` is that the lens is not evaluated.

    A mesh seed is the preimage of beta under its own leaf's affine map, so it
    is already an approximate image; there is nothing left to solve. If this
    fails, the method is doing the work it exists to skip.
    """
    lens, mesh = sie_fixture()

    def exploding_raytrace(x, y):
        raise AssertionError("raytrace must not be called for method='dedup'")

    images, counts = mesh.forward_raytrace(
        as_arr([[0.05, 0.02], [0.4, 0.3]]), exploding_raytrace, method="dedup"
    )
    assert int(to_np(counts).sum()) == images.shape[0]


def test_dedup_method_matches_rootfind_layout():
    lens, mesh = sie_fixture()
    beta = as_arr([[0.05, 0.02], [3.0, 3.0], [0.0, 0.0]])
    images, counts = mesh.forward_raytrace(beta, lens.raytrace, method="dedup")
    counts_np = to_np(counts)
    assert images.shape[1] == 2
    assert counts_np.shape == (3,)
    assert int(counts_np.sum()) == images.shape[0]
    assert counts_np[1] == 0, "a point outside the source-plane mesh has no images"


def test_dedup_method_is_invariant_to_batch_size():
    lens, mesh = sie_fixture()
    beta = as_arr(RNG.uniform(-0.3, 0.3, size=(40, 2)))
    ref_i, ref_c = mesh.forward_raytrace(beta, lens.raytrace, method="dedup")
    for step in (1, 7, 40, 1000):
        got_i, got_c = mesh.forward_raytrace(
            beta, lens.raytrace, batch_size=step, method="dedup"
        )
        assert to_np(got_c).tolist() == to_np(ref_c).tolist(), f"batch_size={step}"
        assert np.allclose(to_np(got_i), to_np(ref_i)), f"batch_size={step}"


def test_dedup_method_agrees_with_rootfind_away_from_the_caustic():
    """Counts must match where the answer is unambiguous.

    Inside the tangential caustic an SIE has four images, outside it two, and
    the two methods may legitimately disagree only within about min_img_sep of
    the caustic itself (spec section 4.3). Sampling well inside and well
    outside keeps the assertion on the part of the contract that is exact.
    """
    lens, mesh = sie_fixture()
    beta = as_arr([[0.01, 0.0], [0.0, 0.01], [-0.015, 0.008], [0.8, 0.8], [-0.9, 0.7]])
    _, rootfind = mesh.forward_raytrace(beta, lens.raytrace, method="rootfind")
    _, dedup = mesh.forward_raytrace(beta, lens.raytrace, method="dedup")
    assert to_np(dedup).tolist() == to_np(rootfind).tolist()


def test_dedup_positions_are_within_min_img_sep_of_the_refined_roots():
    """Positions are accurate to min_img_sep, the build's *lens-plane* tolerance.

    Not to a source-plane residual: `min_img_sep` bounds the seed's distance
    from the image in the lens plane, and the source-plane residual is that
    distance times the local Jacobian. On a SIZE_FLOOR leaf, which stopped
    because it hit the floor rather than because the deviation test passed,
    there is no source-plane bound at all -- measured residuals there reach
    7e-2 against a min_img_sep of 1e-2. Comparing against the root finder's
    own answer is what the documented contract actually claims.

    Measured worst case on this fixture: 3.7e-3 against min_img_sep = 1e-2.
    """
    lens, mesh = sie_fixture()
    beta = as_arr([[0.02, 0.01], [0.05, 0.02], [-0.03, 0.04], [0.3, 0.2]])
    dedup_i, dedup_c = mesh.forward_raytrace(beta, lens.raytrace, method="dedup")
    root_i, root_c = mesh.forward_raytrace(beta, lens.raytrace, method="rootfind")
    dedup_c, root_c = to_np(dedup_c), to_np(root_c)
    assert dedup_c.tolist() == root_c.tolist(), "fixture must not straddle a caustic"

    do = np.concatenate(([0], np.cumsum(dedup_c)))
    ro = np.concatenate(([0], np.cumsum(root_c)))
    di, ri = to_np(dedup_i), to_np(root_i)
    for b in range(dedup_c.size):
        D, R = di[do[b] : do[b + 1]], ri[ro[b] : ro[b + 1]]
        nearest = np.linalg.norm(D[:, None, :] - R[None, :, :], axis=-1).min(axis=1)
        assert (nearest <= mesh.min_img_sep).all(), f"source {b}: {nearest}"


def test_multiplicity_map_has_the_requested_shape_and_extent():
    lens, mesh = sie_fixture()
    m, extent = mesh.multiplicity_map(lens.raytrace, 0.2, nx=7, ny=5, x0=0.0, y0=0.0)
    assert to_np(m).shape == (5, 7), "shape is (ny, nx), imshow-ready"
    # Outer pixel edges, not first/last centres: `extent` is what `imshow` wants.
    assert np.allclose(extent, (-0.7, 0.7, -0.5, 0.5))


def test_multiplicity_map_handles_a_degenerate_axis_under_both_methods():
    """``nx=0`` or ``ny=0`` must return an empty ``(ny, nx)`` map, not raise.

    Regression test for a `_image_chunks` bug: its batch step was
    ``n if batch_size is None else max(1, int(batch_size))``, so with ``n ==
    0`` (an empty pixel grid) and the default ``batch_size=None``, ``step``
    came out ``0`` and ``range(0, 0, 0)`` raised ``ValueError: range() arg 3
    must not be zero``. ``forward_raytrace`` never hits this because of its
    own ``n == 0`` early return before it ever calls `_image_chunks`;
    `multiplicity_map` calls `_image_chunks` directly and had no such guard.
    At the pre-dedup baseline this could not happen because `multiplicity_map`
    went through `forward_raytrace` and inherited its guard; the direct
    `_image_chunks` call is what reintroduced the gap.
    """
    lens, mesh = sie_fixture()
    for nx, ny in ((0, 3), (3, 0)):
        for method in ("rootfind", "dedup"):
            m, extent = mesh.multiplicity_map(
                lens.raytrace, 0.2, nx=nx, ny=ny, method=method
            )
            assert to_np(m).shape == (ny, nx), f"nx={nx} ny={ny} method={method}"


def test_multiplicity_map_agrees_with_forward_raytrace_pixel_by_pixel():
    """Pins the grid orientation, which a transposed reshape would silently flip.

    The map must be the per-pixel image count of the very same source points
    ``forward_raytrace`` would be given, laid out so that ``m[j, i]`` is the pixel
    at ``(x_i, y_j)``.
    """
    lens, mesh = sie_fixture()
    pixelscale, nx, ny = 0.25, 4, 3
    m, _ = mesh.multiplicity_map(lens.raytrace, pixelscale, nx=nx, ny=ny)
    m = to_np(m)
    xs = (np.arange(nx) - (nx - 1) / 2) * pixelscale
    ys = (np.arange(ny) - (ny - 1) / 2) * pixelscale
    lo, hi = to_np(mesh._index_lo), to_np(mesh._index_hi)
    x0, y0 = (lo + hi) / 2
    # A non-square grid makes the transpose check falsifiable.
    assert m.shape == (ny, nx)
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            _, counts = mesh.forward_raytrace(as_arr([[x + x0, y + y0]]), lens.raytrace)
            assert m[j, i] == to_np(counts)[0], f"pixel ({i}, {j}) at ({x}, {y})"


def test_multiplicity_map_defaults_its_field_of_view_to_the_source_plane_mesh():
    """With no x0/y0/nx/ny, the grid covers the indexed leaves' source-plane bbox."""
    lens, mesh = sie_fixture()
    lo, hi = to_np(mesh._index_lo), to_np(mesh._index_hi)
    pixelscale = 0.5
    m, extent = mesh.multiplicity_map(lens.raytrace, pixelscale)
    ny, nx = to_np(m).shape
    assert nx == int(np.ceil((hi[0] - lo[0]) / pixelscale))
    assert ny == int(np.ceil((hi[1] - lo[1]) / pixelscale))
    # Centred on the bbox, and covering it -- square pixels mean slight overhang.
    assert extent[0] <= lo[0] and extent[1] >= hi[0]
    assert extent[2] <= lo[1] and extent[3] >= hi[1]
    assert np.isclose((extent[0] + extent[1]) / 2, (lo[0] + hi[0]) / 2)
    assert np.isclose((extent[2] + extent[3]) / 2, (lo[1] + hi[1]) / 2)


def test_multiplicity_map_of_a_point_lens_is_two_away_from_the_centre():
    """A point lens has exactly two images for every source but the origin.

    Sampled away from the centre, where the softening core's demagnified third
    image lives at a scale ``min_img_sep`` cannot resolve.
    """
    lens = Point(
        name="pt",
        cosmology=FlatLambdaCDM(name="cosmo"),
        z_l=0.5,
        z_s=1.5,
        x0=0.0,
        y0=0.0,
        Rein=1.0,
        s=1e-6,
    )
    mesh = build_adaptive_mesh(lens.raytrace, fov=8.0, init_res=64, min_img_sep=1e-2)
    m, _ = mesh.multiplicity_map(lens.raytrace, 0.1, nx=5, ny=5, x0=1.5, y0=0.0)
    assert (to_np(m) == 2).all(), f"expected all 2, got\n{to_np(m)}"


def test_multiplicity_map_of_a_centred_sie_is_symmetric_under_point_reflection():
    """`beta -> -beta` must not change the image count.

    A centred SIE has an even convergence, so ``alpha(-theta) = -alpha(theta)``
    and the images of ``-beta`` are exactly the negatives of those of ``beta``.
    `utils.meshgrid` centres its samples on zero, so the reflection is a pixel
    permutation and the comparison is exact rather than interpolated.
    """
    lens, mesh = sie_fixture()
    m, _ = mesh.multiplicity_map(lens.raytrace, 0.15, nx=9, ny=9, x0=0.0, y0=0.0)
    m = to_np(m)
    assert (m == m[::-1, ::-1]).all(), f"not point-symmetric:\n{m}"
    assert m.max() > m.min(), "a caustic must show up as a change in multiplicity"


def test_multiplicity_map_of_a_cored_sie_obeys_the_odd_image_theorem():
    """A non-singular lens produces an odd number of images.

    The SIE's ``s = 1e-3`` core makes it non-singular, so every source off a
    caustic has odd multiplicity -- 1 outside the radial caustic, 3 between the
    two, 5 inside the tangential caustic. This is the sharpest available check on
    the whole pipeline, because any single dropped or spurious image flips the
    parity of the pixel it lands in. It caught nothing less than the
    ``batch_lm`` stopping bug: before that fix the central image was abandoned
    whenever its faster siblings converged, and those pixels read 4.
    """
    lens, mesh = sie_fixture()
    m = to_np(
        mesh.multiplicity_map(lens.raytrace, 0.08, nx=25, ny=25, x0=0.0, y0=0.0)[0]
    )
    assert set(np.unique(m).tolist()) <= {
        1,
        3,
        5,
    }, f"even counts present: {np.unique(m)}"
    assert (
        (m == 5).any() and (m == 3).any() and (m == 1).any()
    ), "the grid must span both caustics for this to be a real test"


def test_the_odd_image_theorem_test_can_actually_fail():
    """Falsifiability guard for the test above.

    Starve the root finder of iterations and images go missing; the parity check
    must notice. Without this, a pipeline that silently returned the same count
    everywhere would pass the theorem vacuously.
    """
    lens, mesh = sie_fixture()
    m = to_np(
        mesh.multiplicity_map(
            lens.raytrace,
            0.08,
            nx=25,
            ny=25,
            x0=0.0,
            y0=0.0,
            lm_kwargs={"max_iter": 2},
        )[0]
    )
    assert not set(np.unique(m).tolist()) <= {
        1,
        3,
        5,
    }, f"under-convergence still gave odd counts everywhere: {np.unique(m)}"


def test_multiplicity_map_dedup_method_never_calls_raytrace():
    lens, mesh = sie_fixture()

    def exploding_raytrace(x, y):
        raise AssertionError("raytrace must not be called for method='dedup'")

    mult, extent = mesh.multiplicity_map(
        exploding_raytrace, pixelscale=0.1, nx=9, ny=7, method="dedup"
    )
    assert tuple(mult.shape) == (7, 9)
    assert len(extent) == 4


def test_multiplicity_map_dedup_agrees_with_forward_raytrace_pixel_by_pixel():
    """The map must be exactly its own per-pixel `forward_raytrace`.

    The map consumes counts from a generator that does not build the image
    array, so this is the check that dropping the positions did not drop or
    reorder a count with them.
    """
    lens, mesh = sie_fixture()
    nx, ny, pixelscale = 11, 9, 0.08
    mult, extent = mesh.multiplicity_map(
        lens.raytrace,
        pixelscale=pixelscale,
        nx=nx,
        ny=ny,
        x0=0.0,
        y0=0.0,
        method="dedup",
    )
    gx, gy = meshgrid(
        pixelscale, nx, ny, device=mesh.device, dtype=mesh.vertices_source.dtype
    )
    beta = backend.stack((gx, gy), dim=-1).reshape(-1, 2)
    _, counts = mesh.forward_raytrace(beta, lens.raytrace, method="dedup")
    assert to_np(mult).reshape(-1).tolist() == to_np(counts).tolist()


def test_multiplicity_map_dedup_tracks_rootfind_to_within_a_caustic_sliver():
    """Dedup must reproduce the root-finding map except very near a caustic.

    Note what this deliberately does NOT assert: the odd-image theorem. That
    invariant holds for `method="rootfind"` and is tested above, but dedup
    counts distinct *seeds*, and a near-tangential pair within min_img_sep of
    the caustic can merge -- on this fixture two of 625 pixels read 2 instead
    of 3. Asserting parity here would be asserting something the method does
    not promise (spec section 4.3).

    What it does promise is that the disagreement is rare and never off by
    more than one image. Measured: 2 pixels (0.32%), all delta = -1. A broken
    bucket or a mis-scattered representative blows past 1% immediately.
    """
    lens, mesh = sie_fixture()
    kw = dict(pixelscale=0.08, nx=25, ny=25, x0=0.0, y0=0.0)
    dedup = to_np(mesh.multiplicity_map(lens.raytrace, method="dedup", **kw)[0])
    root = to_np(mesh.multiplicity_map(lens.raytrace, method="rootfind", **kw)[0])
    delta = dedup.astype(np.int64) - root.astype(np.int64)
    differing = int((delta != 0).sum())
    assert differing <= 0.01 * delta.size, f"{differing}/{delta.size} pixels differ"
    assert np.abs(delta).max() <= 1, f"off by {np.abs(delta).max()} images"
    assert set(np.unique(root).tolist()) <= {1, 3, 5}, "rootfind reference is wrong"


def test_forward_chunk_want_images_false_returns_none_but_same_counts():
    """`want_images` must gate only the image gather, never the counts.

    `_forward_chunk` computes `counts` before it ever looks at `want_images`
    (adaptive.py:1404-1415) -- the flag only decides whether the deduplicated
    representatives are also returned. This pins that contract directly on the
    helper: flip `want_images` and the images slot changes from an array to
    `None`, but the counts must not move by a single element.
    """
    lens, mesh = sie_fixture()
    beta = as_arr([[0.05, 0.02], [3.0, 3.0], [0.0, 0.0]])
    tol = mesh.min_img_sep
    images, counts_true = mesh._forward_chunk(
        beta, lens.raytrace, "dedup", tol, {}, True
    )
    none_images, counts_false = mesh._forward_chunk(
        beta, lens.raytrace, "dedup", tol, {}, False
    )
    assert images is not None, "sanity: this beta must produce at least one image"
    assert none_images is None, "want_images=False must not materialize positions"
    assert counts_false.tolist() == counts_true.tolist(), (
        f"counts must be identical regardless of want_images: "
        f"{counts_false.tolist()} != {counts_true.tolist()}"
    )


def test_multiplicity_map_calls_forward_chunk_with_want_images_false(monkeypatch):
    """`multiplicity_map` must request counts only, never the image gather.

    This is the wiring half of the counts-only path: Task 4's whole point was
    to stop `multiplicity_map` materializing the per-chunk image array it was
    always going to discard, via `want_images=False` at the `_image_chunks`
    call site. Nothing else in the suite calls through `Mesh._forward_chunk`
    with a spy, so a regression that silently flipped that `False` back to
    `True` -- reintroducing the discarded gather -- would otherwise pass every
    existing test, since they only check the final counts, not how they were
    obtained.
    """
    lens, mesh = sie_fixture()
    seen = []
    original = Mesh._forward_chunk

    def spy(self, chunk, raytrace, method, tol, lm_kwargs, want_images):
        seen.append(want_images)
        return original(self, chunk, raytrace, method, tol, lm_kwargs, want_images)

    monkeypatch.setattr(Mesh, "_forward_chunk", spy)
    mesh.multiplicity_map(lens.raytrace, pixelscale=0.1, nx=5, ny=5)

    assert seen, "no calls observed -- test is vacuous"
    assert all(w is False for w in seen), (
        f"multiplicity_map must call _forward_chunk with want_images=False "
        f"for every chunk, got {seen}"
    )
