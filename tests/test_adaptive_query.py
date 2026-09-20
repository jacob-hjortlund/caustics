import warnings

import numpy as np
import pytest

from caustics.backend_obj import backend
from caustics.lenses.func import adaptive as new


@pytest.fixture
def oracle_module():
    return pytest.importorskip(
        "caustics.lenses.old_adaptive", reason="optional frozen differential oracle"
    )


def _sie_like(x, y):
    r = (x * x + y * y + 0.05) ** 0.5
    return x - 1.2 * x / r, y - 1.2 * y / r


BUILD = dict(fov=4.0, init_res=3, min_img_sep=0.5, max_depth=3)


@pytest.fixture(scope="module")
def mesh():
    return new.build_adaptive_mesh(_sie_like, **BUILD)


@pytest.fixture(scope="module")
def beta():
    rng = np.random.default_rng(21)
    return backend.as_array(rng.uniform(-1.5, 1.5, (64, 2)), dtype=backend.float64)


def test_query_matches_the_oracle(mesh, beta, oracle_module):
    old_mesh = oracle_module.build_adaptive_mesh(_sie_like, **BUILD)
    idx, offsets, bary = new.mesh_query(mesh, beta)
    idx_o, off_o, bary_o = old_mesh.query(beta)

    assert backend.to_numpy(offsets).tolist() == backend.to_numpy(off_o).tolist()
    assert backend.to_numpy(idx).tolist() == backend.to_numpy(idx_o).tolist()
    assert np.allclose(backend.to_numpy(bary), backend.to_numpy(bary_o))


def test_query_csr_is_well_formed(mesh, beta):
    idx, offsets, bary = new.mesh_query(mesh, beta)
    off = backend.to_numpy(offsets)
    assert off[0] == 0
    assert off[-1] == backend.to_numpy(idx).size
    assert (np.diff(off) >= 0).all()
    assert off.size == backend.to_numpy(beta).shape[0] + 1
    assert backend.to_numpy(bary).shape == (off[-1], 3)


def test_query_blocks_are_strictly_ascending(mesh, beta):
    idx, offsets, _ = new.mesh_query(mesh, beta)
    idx_np, off = backend.to_numpy(idx), backend.to_numpy(offsets)
    for a, b in zip(off[:-1], off[1:]):
        block = idx_np[a:b]
        if block.size > 1:
            assert (np.diff(block) > 0).all()


def test_query_is_invariant_to_batch_size(mesh, beta):
    whole = [backend.to_numpy(t) for t in new.mesh_query(mesh, beta)]
    for size in (1, 5, 1000):
        got = [backend.to_numpy(t) for t in new.mesh_query(mesh, beta, batch_size=size)]
        for a, b in zip(whole, got):
            assert np.array_equal(a, b)


def test_query_handles_empty_input(mesh):
    empty = backend.as_array(np.zeros((0, 2)), dtype=backend.float64)
    idx, offsets, bary = new.mesh_query(mesh, empty)
    assert backend.to_numpy(idx).size == 0
    assert backend.to_numpy(offsets).tolist() == [0]
    assert backend.to_numpy(bary).shape == (0, 3)


def test_query_rejects_wrong_shapes(mesh):
    with pytest.raises(ValueError, match=r"shape \(B, 2\)"):
        new.mesh_query(mesh, backend.as_array(np.zeros(2), dtype=backend.float64))
    with pytest.raises(ValueError, match=r"shape \(B, 2\)"):
        new.mesh_query(mesh, backend.as_array(np.zeros((3, 3)), dtype=backend.float64))


def test_query_misses_return_empty_blocks(mesh):
    far = backend.as_array(np.array([[1e6, 1e6], [-1e6, 0.0]]), dtype=backend.float64)
    idx, offsets, _ = new.mesh_query(mesh, far)
    assert backend.to_numpy(idx).size == 0
    assert backend.to_numpy(offsets).tolist() == [0, 0, 0]


def test_bary_is_in_the_simplex(mesh, beta):
    _, _, bary = new.mesh_query(mesh, beta)
    b = backend.to_numpy(bary)
    assert (b >= 0).all() and (b <= 1).all()
    assert np.allclose(b.sum(axis=1), 1.0)


# ---------------------------------------------------------------------------
# Ported from tests/test_adaptive_mesh.py (Task 13, step 5). These exercise
# `mesh_query` through fixtures the module-level `_sie_like` mesh above does
# not reach -- multi-cell leaf AABBs, a curved mesh with real level
# transitions, an SIS singularity, and totally degenerate maps. Every fixture
# function and `RNG`'s seed are copied verbatim from the legacy suite so the
# numeric commentary in their docstrings still applies unless noted below.
#
# Two names collide with the module-level tests above by coincidence (the
# legacy suite's `test_query_csr_is_well_formed` and
# `test_query_rejects_wrong_shapes` predate this file's own tests of the same
# name from Step 1): both ported versions are suffixed to name the fixture
# that distinguishes them, rather than silently shadowing the Step-1 test of
# the same name.
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(20260904)

AFFINE = np.array([[0.7, 0.1], [-0.2, 0.9]])


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


def sis_raytrace(p, b=1.0):
    """SIS deflection ``beta = theta (1 - b/|theta|)``, non-finite at ``theta = 0``.

    A *point* non-finite set, unlike the half-plane fixtures: the origin is a
    lattice vertex for even ``init_res``, so exactly one sample point in the
    whole build is non-finite and the six level-0 triangles sharing it are the
    ones the old terminate-on-non-finite policy condemned wholesale.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.linalg.norm(p, axis=-1, keepdims=True)
        return p * (1.0 - b / r)


def localised_fold(p):
    """Affine away from a narrow band, curved and fold-bearing inside it.

    Outside |y| < 0.5 the map is exactly affine with sigma_min = 0.6, so those
    triangles converge at level 0. Inside, beta2 = 0.6y + y^2 - 0.25 is curved and
    its Jacobian 0.6 + 2y changes sign at y = -0.3, so both the deviation test and
    the parity test fire.
    """
    y = p[:, 1]
    bend = np.where(np.abs(y) < 0.5, y**2 - 0.25, 0.0)
    return np.stack([p[:, 0], 0.6 * y + bend], axis=-1)


def build(fn, fov=4.0, init_res=4, min_img_sep=0.25, **kw):
    raytrace, calls = make_counting_raytrace(fn)
    mesh = new.build_adaptive_mesh(raytrace, fov, init_res, min_img_sep, **kw)
    return mesh, calls


def query_np(mesh, beta, batch_size=None):
    idx, off, bary = new.mesh_query(mesh, beta, batch_size=batch_size)
    return backend.to_numpy(idx), backend.to_numpy(off), backend.to_numpy(bary)


def test_query_seeds_an_inner_image_that_runs_into_the_lens_centre():
    """The coverage the old terminate-on-non-finite policy destroyed.

    For the SIS the inner image runs continuously into the lens centre as the
    source approaches the cut: ``|theta_minus| = b - beta``. Terminating the six
    level-0 triangles that share the origin therefore removed a hexagon of
    half-width ``fov / init_res`` from the spatial index -- 1.0 arcsec on this
    fixture -- and with it the seed for every inner image inside it, exactly
    where a grid-and-Newton forward_raytrace is already weakest.

    Hand-derived, not read off the mesh: at ``beta = 0.8`` and ``b = 1`` the two
    images are ``theta = 1.8`` and ``theta = -0.2``, since
    ``1.8 * (1 - 1/1.8) = 0.8`` and ``-0.2 * (1 - 1/0.2) = 0.8``. The inner one
    sits 5x deeper inside the old hexagon than its half-width, so the old policy
    returns only the outer seed, 2.0 arcsec away.

    Compose the public ``mesh_query`` and ``mesh_seeds`` interfaces so this
    regression exercises the same seeding path used by forward raytracing.
    """
    mesh, _ = build(sis_raytrace, min_img_sep=0.05)
    idx, offsets, bary = query_np(mesh, np.array([[0.8, 0.0]]))
    seed = backend.to_numpy(
        new.mesh_seeds(mesh, backend.as_array(idx), backend.as_array(bary))
    )
    assert offsets.shape[0] == 2 and seed.shape[0] > 0
    for image in ([-0.2, 0.0], [1.8, 0.0]):
        gap = np.linalg.norm(seed - np.asarray(image), axis=1).min()
        assert gap <= 0.05, f"no seed within min_img_sep of {image}, closest {gap:.3g}"


def test_query_covers_points_on_the_source_bbox_upper_edge():
    """Regression: the upper bbox edge used to return zero candidates.

    `cell = span / [nx, ny]`, so a point at `x == hi_x` yields `u_x == nx`. The
    old cell-index containment test rejected it, while `build_index` clips leaf
    registration to `nx - 1` -- so leaves whose AABB reaches `hi` were indexed
    but unreachable. Measured before the fix: 18 of 18 upper-edge vertices
    returned nothing where brute-force containment found candidates.
    """
    mesh, _ = build(localised_fold, min_img_sep=0.05)
    vs = backend.to_numpy(mesh.vertices_source)
    leaves = backend.to_numpy(mesh.leaves)
    status = backend.to_numpy(mesh.leaf_status)
    hi = backend.to_numpy(mesh.index.hi)
    on_edge = np.flatnonzero((vs[:, 0] == hi[0]) | (vs[:, 1] == hi[1]))
    assert on_edge.size > 0, "fixture must have vertices on the upper bbox edge"
    for v in on_edge:
        beta = vs[v]
        tri = backend.as_array(vs[leaves])
        pts = backend.as_array(np.repeat(beta[None], leaves.shape[0], axis=0))
        truth = backend.to_numpy(new.contains(new.triangle_weights(tri, pts)))
        # UPDATED (b6dc3eb): the single legacy INVALID status split into
        # finite-only LEAF_INVALID and non-finite LEAF_NONFINITE, and
        # `build_adaptive_mesh` excludes BOTH from the spatial index (see
        # `AdaptiveMesh`'s docstring, and old_adaptive.py's own
        # `valid_rows = np.flatnonzero((leaf_status != LeafStatus.INVALID) &
        # (leaf_status != LeafStatus.NONFINITE))`). The original test filtered
        # on INVALID alone, which predates that split.
        expected = set(
            np.flatnonzero(
                truth & (status != new.LEAF_INVALID) & (status != new.LEAF_NONFINITE)
            ).tolist()
        )
        idx, off, _ = query_np(mesh, beta[None])
        assert (
            set(idx[off[0] : off[1]].tolist()) >= expected
        ), f"upper-edge point {beta} lost candidates"


def test_query_matches_brute_force_containment_on_multi_cell_leaves():
    """The one-cell-lookup completeness claim, on a mesh with wide leaf AABBs.

    `build_index` registers each leaf across its full cell rectangle, not just
    its three vertex cells -- and no other test distinguishes those, since the
    vertex-cell test checks only vertices and the crack test's uniform reference
    shares `build_index` so a common bug cancels. Measured on this fixture:
    263 of 1350 leaves span three or more index cells on an axis.
    """
    mesh, _ = build(localised_fold, min_img_sep=0.05)
    vs = backend.to_numpy(mesh.vertices_source)
    leaves = backend.to_numpy(mesh.leaves)
    status = backend.to_numpy(mesh.leaf_status)
    lo = backend.to_numpy(mesh.index.lo)
    cell = backend.to_numpy(mesh.index.cell)
    tri = vs[leaves]
    i0 = np.floor((tri.min(axis=1) - lo) / cell).astype(np.int64)
    i1 = np.floor((tri.max(axis=1) - lo) / cell).astype(np.int64)
    span = i1 - i0 + 1
    assert (span >= 3).any(), "fixture must contain multi-cell leaf AABBs"
    beta = RNG.uniform(-0.9, 0.9, size=(200, 2))
    idx, off, _ = query_np(mesh, beta)
    tri_b = backend.as_array(tri)
    for b in range(beta.shape[0]):
        pts = backend.as_array(np.repeat(beta[b][None], leaves.shape[0], axis=0))
        truth = backend.to_numpy(new.contains(new.triangle_weights(tri_b, pts)))
        # UPDATED (b6dc3eb): same derivation as
        # test_query_covers_points_on_the_source_bbox_upper_edge above.
        expected = set(
            np.flatnonzero(
                truth & (status != new.LEAF_INVALID) & (status != new.LEAF_NONFINITE)
            ).tolist()
        )
        assert set(idx[off[b] : off[b + 1]].tolist()) >= expected


def test_query_csr_is_well_formed_on_a_folded_mesh():
    mesh, _ = build(localised_fold, min_img_sep=0.05)
    beta = RNG.uniform(-2.5, 2.5, size=(64, 2))
    idx, off, bary = query_np(mesh, beta)
    assert off.shape == (65,) and off[0] == 0 and off[-1] == idx.shape[0]
    assert (np.diff(off) >= 0).all()
    assert bary.shape == (idx.shape[0], 3)
    for b in range(64):
        block = idx[off[b] : off[b + 1]]
        assert (np.diff(block) > 0).all(), "blocks must be strictly ascending"


def test_query_handles_empty_input_and_misses():
    mesh, _ = build(lambda p: p @ AFFINE.T)
    idx, off, bary = query_np(mesh, np.zeros((0, 2)))
    assert off.tolist() == [0] and idx.shape == (0,) and bary.shape == (0, 3)
    far = np.array([[1e6, 1e6], [-1e6, 0.0]])
    idx, off, bary = query_np(mesh, far)
    assert off.tolist() == [0, 0, 0]


def test_query_rejects_wrong_shapes_on_an_affine_mesh():
    mesh, _ = build(lambda p: p @ AFFINE.T)
    with pytest.raises(ValueError):
        new.mesh_query(mesh, np.array([0.0, 0.0]))
    with pytest.raises(ValueError):
        new.mesh_query(mesh, np.zeros((4, 3)))


def test_query_is_invariant_to_batch_size_and_point_order():
    mesh, _ = build(localised_fold, min_img_sep=0.05)
    beta = RNG.uniform(-1.5, 1.5, size=(97, 2))
    ref = query_np(mesh, beta)
    for bs in (1, 7, 96, 97, 1000):
        got = query_np(mesh, beta, batch_size=bs)
        for a, b in zip(ref, got):
            assert np.array_equal(a, b)
    perm = RNG.permutation(97)
    pidx, poff, pbary = query_np(mesh, beta[perm])
    for pos, old in enumerate(perm):
        assert np.array_equal(
            pidx[poff[pos] : poff[pos + 1]], ref[0][ref[1][old] : ref[1][old + 1]]
        )
        # `bary` too, not just the indices: a permutation-dependent bug that
        # scrambled coordinates while leaving leaf ids correct would otherwise
        # slip through this check.
        assert np.array_equal(
            pbary[poff[pos] : poff[pos + 1]], ref[2][ref[1][old] : ref[1][old + 1]]
        )


def test_query_finds_the_affine_preimage():
    mesh, _ = build(lambda p: p @ AFFINE.T, fov=4.0, init_res=4, min_img_sep=0.25)
    lens_pts = RNG.uniform(-1.8, 1.8, size=(200, 2))
    beta = lens_pts @ AFFINE.T
    idx, off, bary = query_np(mesh, beta)
    assert (np.diff(off) >= 1).all(), "every interior point must hit a leaf"
    leaves = backend.to_numpy(mesh.leaves)
    vl = backend.to_numpy(mesh.vertices_lens)
    seed = np.einsum("kj,kjd->kd", bary, vl[leaves[idx]])
    first = seed[off[:-1]]
    assert np.allclose(first, lens_pts, atol=1e-9)


def test_bary_is_in_the_simplex_on_every_leaf():
    """The simplex guarantee end-to-end, on a curved mesh and a degenerate one.

    The two fixtures need different query points. A ``kappa == 1`` sheet maps
    every leaf to the single point ``(0, 0)``, so its source-plane bounding box
    is degenerate and queries spread over a region land in empty index cells --
    returning zero candidates and leaving all three assertions vacuously true,
    since numpy's ``all`` and ``allclose`` are True over empty input. The
    ``idx.shape[0] > 0`` guard is what stops that passing silently for
    ``localised_fold`` -- measured, the spread-out draw yields 44 candidates
    there.

    UPDATED (oracle-derived, see task-13 report) for the degenerate branch:
    confirmed directly against ``old_adaptive.build_adaptive_mesh`` that this
    exact fixture has det J == 0 identically, so every leaf's four
    hypothetical children disagree on parity and all 2048 leaves land on
    ``LEAF_INVALID`` -- the mesh's own documented "contains a fold the mesh
    cannot resolve" rule. The spatial index is therefore empty and even an
    origin query now finds nothing, where an older build criterion found 4096
    candidates. That is the conservative, deliberate "no coverage" answer, not
    a regression, so the degenerate branch now asserts the empty result and
    the all-invalid status directly instead.
    """
    for fn, sep, degenerate in (
        (localised_fold, 0.05, False),
        (lambda p: np.zeros_like(p), 0.5, True),
    ):
        mesh, _ = build(fn, min_img_sep=sep)
        # Drawn in both branches so the shared module-level RNG sequence stays
        # unchanged for the tests that follow; discarded just below for the
        # degenerate case, which queries the origin instead.
        beta = RNG.uniform(-0.4, 0.4, size=(40, 2))
        if degenerate:
            beta = np.zeros((8, 2))
        idx, off, bary = query_np(mesh, beta)
        if degenerate:
            assert idx.shape[0] == 0
            status = backend.to_numpy(mesh.leaf_status)
            assert (status == new.LEAF_INVALID).all()
            continue
        assert idx.shape[0] > 0, "fixture returned no candidates"
        assert np.isfinite(bary).all()
        assert (bary >= 0).all() and (bary <= 1).all()
        assert np.allclose(bary.sum(axis=1), 1.0, atol=1e-12)


def test_bary_reconstructs_beta_on_every_hit_leaf():
    """Barycentric coordinates invert the source-plane map on this mesh.

    ``good`` is kept as a live invariant rather than used as a filter: no leaf
    in this fixture is anywhere near degenerate, so a ``~good`` branch would be
    dead code. ``assert good.all()`` fires if a future fixture change ever
    produces a sub-threshold leaf, at which point that branch needs writing;
    the centroid path is meanwhile covered directly at the kernel level by
    ``test_sanitize_bary_falls_back_to_the_centroid_on_total_degeneracy`` in
    tests/test_adaptive_kernels.py, and integration-level by
    ``test_centroid_fallback_on_a_totally_degenerate_leaf`` below.
    """
    mesh, _ = build(localised_fold, min_img_sep=0.05)
    beta = RNG.uniform(-1.5, 1.5, size=(200, 2))
    idx, off, bary = query_np(mesh, beta)
    area = np.abs(backend.to_numpy(mesh.leaf_area2))[idx]
    vs = backend.to_numpy(mesh.vertices_source)
    leaves = backend.to_numpy(mesh.leaves)
    owner = np.repeat(np.arange(len(beta)), np.diff(off))
    assert idx.shape[0] > 0, "fixture returned no candidates"
    good = area > 1e-10
    assert (
        good.all()
    ), "fixture produced a degenerate leaf; the ~good branch needs writing"
    recon = np.einsum("kj,kjd->kd", bary, vs[leaves[idx]])
    assert np.allclose(recon, beta[owner], atol=1e-8)


def test_centroid_fallback_on_a_totally_degenerate_leaf():
    """kappa == 1 maps every leaf to a point: w and d are exactly zero.

    UPDATED (oracle-derived, see task-13 report): confirmed directly against
    ``old_adaptive.build_adaptive_mesh`` that for this exact fixture all 2048
    leaves carry ``LEAF_INVALID`` status (``n_parity_invalid == 2048`` in its
    ``BuildStats``) -- det J == 0 identically, so no leaf's children can agree
    on a parity sign, and the mesh's "leaf whose four hypothetical children
    disagree on sign(det Q_k) contains a fold the mesh cannot resolve" rule
    marks every one of them invalid. The spatial index is therefore empty, so
    ``mesh_query`` returns nothing anywhere -- not just at the origin queried
    below. The centroid fallback this test originally exercised through a live
    query no longer arises this way; it stays covered directly at the kernel
    level by
    ``test_sanitize_bary_falls_back_to_the_centroid_on_total_degeneracy`` in
    tests/test_adaptive_kernels.py.
    """
    mesh, _ = build(lambda p: np.zeros_like(p), fov=4.0, init_res=4, min_img_sep=0.5)
    idx, off, bary = query_np(mesh, np.zeros((1, 2)))
    assert idx.shape[0] == 0
    assert off.tolist() == [0, 0]
    assert bary.shape == (0, 3)
    status = backend.to_numpy(mesh.leaf_status)
    assert (status == new.LEAF_INVALID).all()


def test_coverage_does_not_drop_at_level_transitions():
    """Spec test 28. Reports the gap rather than only thresholding it."""
    fov, init_res, sep = 4.0, 4, 0.05
    mesh, _ = build(localised_fold, fov=fov, init_res=init_res, min_img_sep=sep)
    ml = mesh.max_level
    assert ml >= 3 and len(set(backend.to_numpy(mesh.leaf_level).tolist())) > 1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        uniform, _ = build(
            localised_fold,
            fov=fov,
            init_res=init_res * 2**ml,
            min_img_sep=sep,
            max_depth=0,
        )
    grid = np.linspace(-0.9, 0.9, 120)
    beta = np.stack(np.meshgrid(grid, grid, indexing="ij"), axis=-1).reshape(-1, 2)
    _, off_a, _ = query_np(mesh, beta, batch_size=4096)
    _, off_u, _ = query_np(uniform, beta, batch_size=4096)
    hit_a = np.diff(off_a) > 0
    hit_u = np.diff(off_u) > 0
    gap = int((hit_u & ~hit_a).sum())
    print(f"coverage gap at level transitions: {gap} / {int(hit_u.sum())} covered")
    assert gap == 0


def _build_with_counters(fn, fov, init_res, min_img_sep, max_depth=25):
    """
    Recover the pre-closure termination counters the frozen oracle exposes as
    ``Mesh.stats`` (a ``BuildStats``) -- dropped from ``AdaptiveMesh``, which
    stores only the frozen result, not the build's own bookkeeping. Mirrors
    ``build_adaptive_mesh``'s prologue exactly, up to the point those numbers
    are available: the oracle computes
    ``n_converged_at_level_0=int(ref.counters["converged_level0"])``,
    ``n_deviation_splits=int(ref.counters["deviation_splits"])`` and
    ``n_leaves_pre_closure=int(pre_v.shape[0])`` (old_adaptive.py's
    ``build_adaptive_mesh``), and ``refine`` in ``func/adaptive.py`` returns
    the same-keyed ``counters`` dict this helper reads directly.
    """
    raytrace, _ = make_counting_raytrace(fn)
    requested_min_img_sep = min_img_sep
    min_img_sep = min_img_sep / 2
    new.validate_build_args(
        fov, init_res, min_img_sep, max_depth, requested_min_img_sep
    )
    d_floor = new.depth_floor(fov, init_res, min_img_sep)
    max_level = min(int(max_depth), d_floor)
    tables = new.child_matrix_tables()
    lat = new.make_lattice(fov, 0.0, 0.0, init_res, max_level + 1)
    raytrace_fn = new.make_raytrace(raytrace, None)
    _cache, _active, store, counters = new.refine(
        raytrace_fn,
        lat,
        init_res,
        fov / init_res,
        min_img_sep,
        max_level,
        tables,
        None,
    )
    pre_v, _pre_level, _pre_cls, _pre_status = new.store_compact(store)
    return counters, int(pre_v.shape[0]), max_level


def test_criterion_is_blind_to_structure_below_the_sampling_scale():
    """Spec section 2.6, asserted in both directions.

    The criterion reads six points per triangle and the centroid is 0.289*edge from
    the nearest of them, so a perturbation supported inside that radius is exactly
    invisible. Completeness is conditional on init_res resolving it.

    ADAPTED: ``AdaptiveMesh`` does not carry the oracle's ``Mesh.stats``, so
    the termination counters are recovered directly via
    ``refine``/``store_compact`` in ``_build_with_counters`` above -- the same
    internal calls ``build_adaptive_mesh`` itself makes to compute
    ``n_converged_at_level_0``/``n_deviation_splits``/``n_leaves_pre_closure``.
    """
    centre = np.array([[-2.0, -2.0], [0.0, 0.0], [-2.0, 0.0]]).mean(axis=0)

    def bumped(p):
        r2 = ((p - centre) ** 2).sum(axis=-1)
        bump = (2.0 * np.exp(-r2 / (2 * 0.08**2)))[:, None] * np.array([1.0, 0.0])
        return p * 0.5 + bump

    # At init_res=2 the nearest of the six sample points is 0.47 from the bump
    # centre, where the bump is 6e-8 -- far below the threshold. At init_res=32 the
    # cell is 0.125 and the deviation is ~0.6, well above it.
    sep = 0.05
    coarse_counters, coarse_pre_closure, _ = _build_with_counters(bumped, 4.0, 2, sep)
    fine_counters, fine_pre_closure, fine_max_level = _build_with_counters(
        bumped, 4.0, 32, sep
    )
    assert fine_max_level >= 1, "fine build must actually run the criterion"
    assert coarse_counters["converged_level0"] == coarse_pre_closure
    assert coarse_counters["deviation_splits"] == 0
    assert fine_counters["converged_level0"] < fine_pre_closure
    assert fine_counters["deviation_splits"] > 0


def test_seeds_lie_inside_their_lens_triangle(mesh, beta):
    idx, _, bary = new.mesh_query(mesh, beta)
    seeds = new.mesh_seeds(mesh, idx, bary)
    tri = backend.to_numpy(mesh.vertices_lens)[
        backend.to_numpy(mesh.leaves)[backend.to_numpy(idx)]
    ]
    s = backend.to_numpy(seeds)
    assert s.shape == (backend.to_numpy(idx).size, 2)
    lo, hi = tri.min(axis=1), tri.max(axis=1)
    assert (s >= lo - 1e-12).all() and (s <= hi + 1e-12).all()


def test_seeds_match_the_oracle(mesh, beta, oracle_module):
    old_mesh = oracle_module.build_adaptive_mesh(_sie_like, **BUILD)
    idx, _, bary = new.mesh_query(mesh, beta)
    idx_o, _, bary_o = old_mesh.query(beta)
    assert np.allclose(
        backend.to_numpy(new.mesh_seeds(mesh, idx, bary)),
        backend.to_numpy(old_mesh.seeds(leaf_indices=idx_o, bary=bary_o)),
    )


def _pts(x):
    return backend.as_array(np.asarray(x, dtype=np.float64), dtype=backend.float64)


def test_dedup_collapses_points_closer_than_the_tolerance():
    pts = _pts([[0.0, 0.0], [0.05, 0.0], [1.0, 0.0]])
    keep = backend.to_numpy(new.dedup_representatives(pts, np.array([3]), 0.1))
    assert keep.tolist() == [True, False, True]


def test_dedup_keeps_points_separated_by_exactly_the_tolerance():
    pts = _pts([[0.0, 0.0], [0.1, 0.0]])
    keep = backend.to_numpy(new.dedup_representatives(pts, np.array([2]), 0.1))
    assert keep.tolist() == [True, True]


def test_dedup_counts_connected_components_not_greedy_clusters():
    # three collinear points spaced 0.9 * tol: one component, one representative
    pts = _pts([[0.0, 0.0], [0.09, 0.0], [0.18, 0.0]])
    keep = backend.to_numpy(new.dedup_representatives(pts, np.array([3]), 0.1))
    assert keep.tolist() == [True, False, False]


def test_dedup_never_merges_across_blocks():
    pts = _pts([[0.0, 0.0], [0.0, 0.0]])
    keep = backend.to_numpy(new.dedup_representatives(pts, np.array([1, 1]), 0.1))
    assert keep.tolist() == [True, True]


def test_dedup_handles_empty_and_singleton_blocks():
    pts = _pts([[0.0, 0.0], [0.01, 0.0], [5.0, 5.0]])
    keep = backend.to_numpy(new.dedup_representatives(pts, np.array([0, 2, 0, 1]), 0.1))
    assert keep.tolist() == [True, False, True]


def test_dedup_matches_the_oracle_on_randomised_blocks(oracle_module):
    rng = np.random.default_rng(31)
    counts = rng.integers(0, 5, 20)
    pts = rng.normal(size=(int(counts.sum()), 2)) * 0.1
    p = _pts(pts)
    assert (
        backend.to_numpy(new.dedup_representatives(p, counts, 0.05))
        == backend.to_numpy(oracle_module._dedup_representatives(p, counts, 0.05))
    ).all()


# ---------------------------------------------------------------------------
# Ported from tests/test_adaptive_mesh.py (Task 14, step 5): mesh_seeds and
# dedup_representatives/dedup_block_group exercised through fixtures and
# regression cases the tests above (added directly from the Task 14 brief)
# do not reach -- a folded, multi-level mesh's own triangles (not just their
# bounding boxes), point-order invariance across explicit permutations, and
# the bucketed-vs-per-block equivalence check. Deleted from the legacy file
# after porting.
#
# Four names collide with the tests above by coincidence: the oracle's
# dual-mode `Mesh.seeds`/`Mesh.query` and the legacy suite both predate
# `mesh_seeds`'s single-mode contract, so a handful of test names describe
# the same idea independently on both sides. Each collision is RENAMED here
# to name what actually distinguishes it, rather than silently shadowing the
# same-named test above (a duplicate `def` means the first never runs):
#   test_seeds_lie_inside_their_lens_triangle
#       -> test_seeds_lie_inside_their_lens_triangle_on_a_folded_mesh
#   test_dedup_collapses_points_closer_than_the_tolerance
#       -> test_dedup_collapses_a_near_coincident_pair_and_keeps_the_lone_point
#   test_dedup_counts_connected_components_not_greedy_clusters
#       -> test_dedup_counts_connected_components_regardless_of_point_order
#   test_dedup_never_merges_across_blocks
#       -> test_dedup_keeps_identical_points_in_different_blocks_distinct
# ---------------------------------------------------------------------------


def _cross2(u, v):
    """Scalar cross product of 2-D vectors.

    ``np.cross`` on 2-vectors is deprecated in NumPy 2.0 and emits a
    ``DeprecationWarning`` per call. This is the same value, computed
    component-wise, and is bit-identical to the ``np.cross`` result.
    """
    return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]


def test_seeds_lie_inside_their_lens_triangle_on_a_folded_mesh():
    """RENAMED from ``test_seeds_lie_inside_their_lens_triangle``: this checks
    the stronger inside-the-triangle sign condition on a folded, multi-level
    mesh, distinct from the bounding-box check the same-named test above
    (added verbatim from the Task 14 brief) performs on the module-level
    ``mesh``/``beta`` fixtures.

    ADAPTED: the oracle's ``Mesh.seeds`` accepted ``beta`` (query-and-seed) or
    ``leaf_indices``/``bary`` (gather-only) behind a ``_check_call`` guard, so
    the legacy test compared the two modes against each other and also
    checked the CSR ``offsets`` the ``beta`` mode returned. ``mesh_seeds`` has
    only the gather-only form, so that comparison has no counterpart and is
    dropped; ``triangles_lens(leaf_indices=idx)`` (also dropped) is replaced
    by the same manual ``vertices_lens[leaves[idx]]`` gather ``mesh_seeds``
    itself performs.
    """
    mesh, _ = build(localised_fold, min_img_sep=0.05)
    beta = RNG.uniform(-1.5, 1.5, size=(50, 2))
    idx, _, bary = new.mesh_query(mesh, beta)
    seed = backend.to_numpy(new.mesh_seeds(mesh, idx, bary))
    assert seed.shape[0] > 0, "fixture returned no candidates"
    idx_np = backend.to_numpy(idx)
    leaves = backend.to_numpy(mesh.leaves)
    vl = backend.to_numpy(mesh.vertices_lens)
    tri = vl[leaves[idx_np]]
    for k in range(len(seed)):
        w = np.array(
            [
                _cross2(tri[k, (i + 1) % 3] - seed[k], tri[k, (i + 2) % 3] - seed[k])
                for i in range(3)
            ]
        )
        assert (w >= -1e-9).all() or (w <= 1e-9).all()


def test_dedup_collapses_a_near_coincident_pair_and_keeps_the_lone_point():
    """RENAMED from ``test_dedup_collapses_points_closer_than_the_tolerance``:
    a different magnitude (0.001 spacing at tol 0.01) and assertion style
    (explicit "the distant point must survive" check) from the same-named
    test above (added verbatim from the Task 14 brief)."""
    points = _pts([[0.0, 0.0], [0.0, 0.001], [1.0, 0.0]])
    keep = backend.to_numpy(new.dedup_representatives(points, np.array([3]), 0.01))
    assert keep.sum() == 2, "the coincident pair must collapse to one image"
    assert keep[2], "the distant point must survive"


def test_dedup_keeps_points_separated_by_the_tolerance():
    """Separation *of* min_img_sep means distinct, matching the build contract.

    The size floor is ``l_max <= min_img_sep`` and step 7 hides no pair separated
    by more than ``min_img_sep / 4``, so the boundary belongs to the distinct
    side. Adjacency is ``d < tol``, not ``<=``.
    """
    points = _pts([[0.0, 0.0], [0.01, 0.0]])
    keep = backend.to_numpy(new.dedup_representatives(points, np.array([2]), 0.01))
    assert keep.sum() == 2


def test_dedup_counts_connected_components_regardless_of_point_order():
    """RENAMED from ``test_dedup_counts_connected_components_not_greedy_clusters``:
    this checks invariance across four explicit permutations, distinct from
    the single-order check in the same-named test above (added verbatim from
    the Task 14 brief).

    Three collinear points spaced ``0.9 * tol`` apart form one connected
    component. Greedy returns 2 for the order below and 1 for ``[1, 0, 2]`` --
    the answer would depend on the order ``mesh_query`` happened to emit
    candidates in, which is not something a multiplicity map may depend on.
    """
    p = np.array([[0.0, 0.0], [0.009, 0.0], [0.018, 0.0]])
    counts = np.array([3])
    base = backend.to_numpy(new.dedup_representatives(_pts(p), counts, 0.01)).sum()
    assert base == 1, f"one chained component expected, got {base}"
    for order in ([1, 0, 2], [2, 1, 0], [0, 2, 1], [2, 0, 1]):
        got = backend.to_numpy(
            new.dedup_representatives(_pts(p[order]), counts, 0.01)
        ).sum()
        assert got == base, f"order {order} gave {got}, not {base}"


def test_dedup_keeps_identical_points_in_different_blocks_distinct():
    """RENAMED from ``test_dedup_never_merges_across_blocks`` to avoid
    shadowing the same-named test above (added verbatim from the Task 14
    brief); both exercise the same contract on equivalent input.

    Two source points whose images coincide must not collapse into one. The
    padded ``(B, M, M)`` formulation makes cross-block bleed the natural bug
    here, and it would silently halve a multiplicity map.
    """
    points = _pts([[0.0, 0.0], [0.0, 0.0]])
    keep = backend.to_numpy(new.dedup_representatives(points, np.array([1, 1]), 0.01))
    assert keep.sum() == 2, "identical points in different blocks are distinct"


def test_dedup_handles_ragged_blocks_and_empty_blocks():
    """Padding must not invent images in a block that found none."""
    points = _pts([[0.0, 0.0], [5.0, 5.0], [5.0, 5.0005]])
    keep = backend.to_numpy(
        new.dedup_representatives(points, np.array([1, 0, 2]), 0.01)
    )
    assert keep.tolist() == [True, True, False]


def test_dedup_is_unchanged_by_bucketing_on_randomised_blocks():
    """The bucketed kernel must agree with a per-block reference exactly.

    Blocks are grouped by count and run at their own M rather than padded to
    the global maximum, so the risk is a scatter that puts one block's answer
    on another block's rows. Running each block *alone* through the same
    function is the independent reference: a single-block call has nothing to
    mis-scatter.

    What this does NOT check: both sides call the same clustering kernel
    (:func:`~caustics.lenses.func.adaptive.dedup_block_group` via
    :func:`~caustics.lenses.func.adaptive.dedup_representatives`), so a bug
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
        got = backend.to_numpy(new.dedup_representatives(_pts(pts), counts, 0.01))
        starts = np.cumsum(counts) - counts
        want = np.concatenate(
            [
                backend.to_numpy(
                    new.dedup_representatives(_pts(pts[s : s + c]), np.array([c]), 0.01)
                )
                for s, c in zip(starts, counts)
            ]
            + [np.zeros(0, dtype=bool)]
        )
        assert got.tolist() == want.tolist(), f"counts={counts.tolist()}"


def test_dedup_keeps_exactly_one_point_per_singleton_block():
    """Blocks of one bypass the clustering kernel; they must still be kept."""
    points = _pts([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    keep = backend.to_numpy(
        new.dedup_representatives(points, np.array([1, 1, 1]), 0.01)
    )
    assert keep.tolist() == [True, True, True]


def test_dedup_mixes_singleton_and_clustered_blocks_in_order():
    """The bypass and the kernel write into one output; order must survive.

    Block 0 is a singleton, block 1 collapses to one image, block 2 is a
    singleton again. A scatter that appends the bypassed blocks after the
    clustered ones would pass every count-based assertion and still return the
    representatives in the wrong rows.
    """
    points = _pts([[9.0, 9.0], [0.0, 0.0], [0.0, 0.001], [5.0, 5.0]])
    keep = backend.to_numpy(
        new.dedup_representatives(points, np.array([1, 2, 1]), 0.01)
    )
    assert keep.tolist() == [True, True, False, True]
