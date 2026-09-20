import numpy as np
import pytest

from caustics.backend_obj import backend
from caustics.cosmology import FlatLambdaCDM
from caustics.lenses import SIE, Point
from caustics.lenses.func import adaptive as new
from caustics.lenses.func import forward_raytrace_rootfind

RNG = np.random.default_rng(20260904)


@pytest.fixture
def oracle_module():
    return pytest.importorskip(
        "caustics.lenses.old_adaptive", reason="optional frozen differential oracle"
    )


def _sie_like(x, y):
    r = (x * x + y * y + 0.05) ** 0.5
    return x - 1.2 * x / r, y - 1.2 * y / r


BUILD = dict(fov=6.0, init_res=4, min_img_sep=0.1, max_depth=6)


@pytest.fixture(scope="module")
def mesh():
    return new.build_adaptive_mesh(_sie_like, **BUILD)


def _beta(points):
    return backend.as_array(np.asarray(points, dtype=np.float64), dtype=backend.float64)


def test_forward_raytrace_matches_the_oracle(mesh, oracle_module):
    old_mesh = oracle_module.build_adaptive_mesh(_sie_like, **BUILD)
    beta = _beta([[0.05, 0.02], [0.4, -0.3], [1.9, 1.7]])

    img, counts = new.mesh_forward_raytrace(mesh, beta, _sie_like)
    img_o, counts_o = old_mesh.forward_raytrace(beta, _sie_like)

    assert backend.to_numpy(counts).tolist() == backend.to_numpy(counts_o).tolist()
    assert np.allclose(backend.to_numpy(img), backend.to_numpy(img_o), atol=1e-8)


def test_images_solve_the_lens_equation(mesh):
    beta = _beta([[0.05, 0.02], [0.4, -0.3]])
    img, counts = new.mesh_forward_raytrace(mesh, beta, _sie_like)
    _img_np = backend.to_numpy(img)
    bx, by = _sie_like(img[:, 0], img[:, 1])
    got = np.stack((backend.to_numpy(bx), backend.to_numpy(by)), axis=-1)
    want = np.repeat(backend.to_numpy(beta), backend.to_numpy(counts), axis=0)
    assert np.abs(got - want).max() < 1e-6


def test_forward_raytrace_is_invariant_to_batch_size(mesh):
    beta = _beta(np.random.default_rng(4).uniform(-1, 1, (12, 2)))
    whole = [
        backend.to_numpy(t) for t in new.mesh_forward_raytrace(mesh, beta, _sie_like)
    ]
    for size in (1, 3, 100):
        got = [
            backend.to_numpy(t)
            for t in new.mesh_forward_raytrace(mesh, beta, _sie_like, batch_size=size)
        ]
        assert got[1].tolist() == whole[1].tolist()
        assert np.allclose(got[0], whole[0])


def test_forward_raytrace_returns_empty_outside_the_source_plane(mesh):
    beta = _beta([[1e6, 1e6]])
    img, counts = new.mesh_forward_raytrace(mesh, beta, _sie_like)
    assert backend.to_numpy(counts).tolist() == [0]
    assert backend.to_numpy(img).shape == (0, 2)


def test_forward_raytrace_handles_empty_input(mesh):
    img, counts = new.mesh_forward_raytrace(
        mesh, backend.as_array(np.zeros((0, 2)), dtype=backend.float64), _sie_like
    )
    assert backend.to_numpy(counts).size == 0
    assert backend.to_numpy(img).shape == (0, 2)


def test_forward_raytrace_rejects_an_unknown_method(mesh):
    with pytest.raises(ValueError, match="method must be one of"):
        new.mesh_forward_raytrace(mesh, _beta([[0.1, 0.1]]), _sie_like, method="bogus")


def test_dedup_method_never_calls_raytrace(mesh):
    def exploding(x, y):
        raise AssertionError("raytrace must not be called under method='dedup'")

    img, counts = new.mesh_forward_raytrace(
        mesh, _beta([[0.05, 0.02]]), exploding, method="dedup"
    )
    assert backend.to_numpy(counts)[0] >= 1


def test_dedup_positions_are_within_min_img_sep_of_the_refined_roots(mesh):
    beta = _beta([[0.4, -0.3]])
    a, ca = new.mesh_forward_raytrace(mesh, beta, _sie_like, method="rootfind")
    b, cb = new.mesh_forward_raytrace(mesh, beta, _sie_like, method="dedup")
    assert backend.to_numpy(ca).tolist() == backend.to_numpy(cb).tolist()
    pa = np.sort(backend.to_numpy(a), axis=0)
    pb = np.sort(backend.to_numpy(b), axis=0)
    assert np.abs(pa - pb).max() <= mesh.min_img_sep


# ---------------------------------------------------------------------------
# Ported from tests/test_adaptive_mesh.py (Task 15) -- see the task-15 report
# for the oracle derivation behind the two `coverable`-filtered assertions
# below, and for the three renames forced by a name collision with the
# Step-1 tests above.
# ---------------------------------------------------------------------------


def to_np(x):
    return backend.to_numpy(x)


def dedup(points, tol):
    """Greedy clustering; returns one representative per cluster, sorted."""
    keep = []
    for p in points:
        if all(np.linalg.norm(p - q) >= tol for q in keep):
            keep.append(p)
    return np.array(sorted(keep, key=tuple)) if keep else np.zeros((0, 2))


def test_sie_candidates_recover_forward_raytrace_images(device):
    """Spec test 26, split into the two contracts this module actually owns.

    1. **Coverage** -- every image ``forward_raytrace`` finds outside the
       lens's own softening radius has a candidate seed within
       ``min_img_sep``. That is exactly what :func:`mesh_seeds` promises: the
       hit leaf's own affine map is the one step 7 bounds, so the seed is
       accurate to ``min_img_sep`` by construction. Measured across the
       covered images at these three source points, the worst distance is
       1.1e-3 against a 1e-2 tolerance -- roughly ten times better than the
       guarantee.
    2. **No spurious images** -- every candidate the root-finder converges on
       is a genuine image.

    This deliberately does **not** assert that the refined set has the same
    cardinality as ``forward_raytrace``'s. For ``sp = [0.2, 0.2]`` and
    ``sp = [0.05, -0.05]`` this SIE has a central image at radius ~7.0e-4 and
    ~6.7e-5 respectively, *inside* its own softening radius ``s = 1e-3``,
    where the Jacobian is nearly degenerate. Oracle-verified directly against
    ``old_adaptive.py`` (and independently reproduced by this module's own
    ``build_adaptive_mesh``): the single lens-plane leaf containing that
    image is ``LEAF_INVALID`` -- a fold the criterion cannot resolve there,
    exactly the coverage hole ``AdaptiveMesh`` documents as deliberate -- so
    the mesh supplies no seed anywhere near it (nearest candidate ~0.52 and
    ~0.78 away for the two points respectively) while every other image is
    covered. A cardinality assertion over every analytic image would
    therefore spuriously fail on the one image the mesh was never going to
    cover, not on a genuine regression -- so Contract 1 below is scoped to
    the images the mesh can promise to cover (outside the softening radius),
    and still fails loudly if coverage is ever lost among *those*.
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
    mesh = new.build_adaptive_mesh(
        lens.raytrace, fov=5.0, init_res=32, min_img_sep=1e-2, device=device
    )
    for sp in ([0.2, 0.2], [0.05, -0.05], [1.4, 1.1]):
        sx = backend.as_array(sp[0], device=device)
        sy = backend.as_array(sp[1], device=device)
        ex, ey = lens.forward_raytrace(sx, sy)
        expected = dedup(
            np.stack([backend.to_numpy(ex), backend.to_numpy(ey)], axis=-1), 1e-2
        )
        # Excludes any image inside the lens's own softening radius -- see
        # the docstring above; the mesh deliberately has no coverage there.
        coverable = expected[np.linalg.norm(expected, axis=-1) >= 1e-3]
        assert coverable.shape[0] > 0, f"{sp}: no coverable reference images"
        idx, offsets, bary = new.mesh_query(mesh, backend.as_array(np.asarray([sp])))
        seed = backend.to_numpy(new.mesh_seeds(mesh, idx, bary))
        assert seed.shape[0] >= coverable.shape[0], "candidates must cover the images"
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
        # Contract 1: coverage. Every coverable image has a seed within
        # min_img_sep.
        nearest = np.linalg.norm(coverable[:, None, :] - seed[None, :, :], axis=-1).min(
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
    mesh = new.build_adaptive_mesh(
        lens.raytrace, fov=8.0, init_res=64, min_img_sep=1e-2
    )
    b = 0.4
    idx, offsets, bary = new.mesh_query(mesh, backend.as_array(np.array([[b, 0.0]])))
    seed = backend.to_numpy(new.mesh_seeds(mesh, idx, bary))
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
    mesh = new.build_adaptive_mesh(
        lens.raytrace, fov=4.0, init_res=8, min_img_sep=0.1, device=device
    )
    idx, off, bary = new.mesh_query(
        mesh, backend.as_array(np.array([[0.1, 0.1], [3.0, 3.0]]))
    )
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
    mesh = new.build_adaptive_mesh(
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
        images, counts = new.mesh_forward_raytrace(mesh, _beta([sp]), lens.raytrace)
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
    """The converse contract: no image is dropped by the filters or the dedup.

    Scoped to images outside the lens's own softening radius ``s = 1e-3``:
    for both source points below, the SIE's central image sits inside that
    core (radius 7.0e-4 and 6.7e-5 respectively), on a lens-plane leaf the
    build marks ``LEAF_INVALID`` -- oracle-verified, and independently
    reproduced by this module's own ``build_adaptive_mesh`` -- so
    ``mesh_query`` never returns a candidate for it and no downstream filter
    or dedup step is responsible for its absence (see
    ``test_sie_candidates_recover_forward_raytrace_images`` for the full
    derivation). Every other image is covered to within 1.1e-3.
    """
    lens, mesh = sie_fixture()
    for sp in ([0.2, 0.2], [0.05, -0.05]):
        images, _ = new.mesh_forward_raytrace(mesh, _beta([sp]), lens.raytrace)
        images = to_np(images)
        # The reference path is float32-only: `LensBase.forward_raytrace` raises
        # "expected scalar type Float but found Double" on float64 input. The mesh
        # itself is float64, so only this comparison call is narrowed.
        ex, ey = lens.forward_raytrace(backend.as_array(sp[0]), backend.as_array(sp[1]))
        expected = dedup(np.stack([to_np(ex), to_np(ey)], axis=-1), 1e-2)
        assert expected.shape[0] > 0, f"{sp}: reference found no images"
        # Excludes the one image inside the softening core; see the docstring.
        coverable = expected[np.linalg.norm(expected, axis=-1) >= 1e-3]
        nearest = np.linalg.norm(coverable[:, None, :] - images[None, :, :], axis=-1)
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
    mesh = new.build_adaptive_mesh(
        lens.raytrace, fov=8.0, init_res=64, min_img_sep=1e-2
    )
    b = 0.4
    images, counts = new.mesh_forward_raytrace(mesh, _beta([[b, 0.0]]), lens.raytrace)
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
    images, counts = new.mesh_forward_raytrace(mesh, _beta(points), lens.raytrace)
    counts = to_np(counts)
    assert counts.shape == (3,)
    assert counts.sum() == to_np(images).shape[0]
    offsets = np.concatenate(([0], np.cumsum(counts)))
    for i, sp in enumerate(points):
        one, one_counts = new.mesh_forward_raytrace(mesh, _beta([sp]), lens.raytrace)
        assert to_np(one_counts).tolist() == [counts[i]], f"{sp}: count differs"
        block = to_np(images)[offsets[i] : offsets[i + 1]]
        assert np.allclose(block, to_np(one), atol=1e-8), f"{sp}: images differ"


def test_forward_raytrace_batch_size_does_not_change_the_answer():
    lens, mesh = sie_fixture()
    beta = _beta([[0.2, 0.2], [0.05, -0.05], [0.4, -0.3], [0.0, 0.3]])
    full, full_counts = new.mesh_forward_raytrace(mesh, beta, lens.raytrace)
    for size in (1, 2, 3):
        part, part_counts = new.mesh_forward_raytrace(
            mesh, beta, lens.raytrace, batch_size=size
        )
        assert to_np(part_counts).tolist() == to_np(full_counts).tolist()
        assert np.allclose(to_np(part), to_np(full), atol=1e-8)


def test_forward_raytrace_returns_an_empty_block_outside_the_source_plane():
    """A source the mesh never maps to has zero images, not a raised error."""
    lens, mesh = sie_fixture()
    images, counts = new.mesh_forward_raytrace(
        mesh, _beta([[50.0, 50.0]]), lens.raytrace
    )
    assert to_np(counts).tolist() == [0]
    assert to_np(images).shape == (0, 2)


def test_forward_raytrace_rejects_an_unknown_method_on_the_sie_fixture():
    lens, mesh = sie_fixture()
    with pytest.raises(ValueError, match="rootfind"):
        new.mesh_forward_raytrace(
            mesh, _beta([[0.05, 0.02]]), lens.raytrace, method="nope"
        )


def test_dedup_method_never_calls_raytrace_across_a_batch():
    """The whole point of `method="dedup"` is that the lens is not evaluated.

    A mesh seed is the preimage of beta under its own leaf's affine map, so it
    is already an approximate image; there is nothing left to solve. If this
    fails, the method is doing the work it exists to skip.
    """
    lens, mesh = sie_fixture()

    def exploding_raytrace(x, y):
        raise AssertionError("raytrace must not be called for method='dedup'")

    images, counts = new.mesh_forward_raytrace(
        mesh, _beta([[0.05, 0.02], [0.4, 0.3]]), exploding_raytrace, method="dedup"
    )
    assert int(to_np(counts).sum()) == images.shape[0]


def test_dedup_method_matches_rootfind_layout():
    lens, mesh = sie_fixture()
    beta = _beta([[0.05, 0.02], [3.0, 3.0], [0.0, 0.0]])
    images, counts = new.mesh_forward_raytrace(
        mesh, beta, lens.raytrace, method="dedup"
    )
    counts_np = to_np(counts)
    assert images.shape[1] == 2
    assert counts_np.shape == (3,)
    assert int(counts_np.sum()) == images.shape[0]
    assert counts_np[1] == 0, "a point outside the source-plane mesh has no images"


def test_dedup_method_is_invariant_to_batch_size():
    lens, mesh = sie_fixture()
    beta = _beta(RNG.uniform(-0.3, 0.3, size=(40, 2)))
    ref_i, ref_c = new.mesh_forward_raytrace(mesh, beta, lens.raytrace, method="dedup")
    for step in (1, 7, 40, 1000):
        got_i, got_c = new.mesh_forward_raytrace(
            mesh, beta, lens.raytrace, batch_size=step, method="dedup"
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
    beta = _beta([[0.01, 0.0], [0.0, 0.01], [-0.015, 0.008], [0.8, 0.8], [-0.9, 0.7]])
    _, rootfind_counts = new.mesh_forward_raytrace(
        mesh, beta, lens.raytrace, method="rootfind"
    )
    _, dedup_counts = new.mesh_forward_raytrace(
        mesh, beta, lens.raytrace, method="dedup"
    )
    assert to_np(dedup_counts).tolist() == to_np(rootfind_counts).tolist()


def test_dedup_positions_are_within_min_img_sep_of_the_refined_roots_across_a_batch():
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
    beta = _beta([[0.02, 0.01], [0.05, 0.02], [-0.03, 0.04], [0.3, 0.2]])
    dedup_i, dedup_c = new.mesh_forward_raytrace(
        mesh, beta, lens.raytrace, method="dedup"
    )
    root_i, root_c = new.mesh_forward_raytrace(
        mesh, beta, lens.raytrace, method="rootfind"
    )
    dedup_c, root_c = to_np(dedup_c), to_np(root_c)
    assert dedup_c.tolist() == root_c.tolist(), "fixture must not straddle a caustic"

    do = np.concatenate(([0], np.cumsum(dedup_c)))
    ro = np.concatenate(([0], np.cumsum(root_c)))
    di, ri = to_np(dedup_i), to_np(root_i)
    for b in range(dedup_c.size):
        D, R = di[do[b] : do[b + 1]], ri[ro[b] : ro[b + 1]]
        nearest = np.linalg.norm(D[:, None, :] - R[None, :, :], axis=-1).min(axis=1)
        assert (nearest <= mesh.min_img_sep).all(), f"source {b}: {nearest}"
