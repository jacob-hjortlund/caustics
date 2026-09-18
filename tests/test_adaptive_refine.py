"""The refinement loop must reproduce the frozen oracle leaf-for-leaf."""

import numpy as np
import pytest

from caustics.backend_obj import backend
from caustics.lenses import old_adaptive as oracle
from caustics.lenses.func import adaptive as new


def _sie_like(x, y):
    r = (x * x + y * y + 0.05) ** 0.5
    return x - 1.2 * x / r, y - 1.2 * y / r


def _affine(x, y):
    return 2.0 * x + 0.5 * y, -0.25 * x + 1.5 * y


def _run_new(raytrace, fov, init_res, min_img_sep, max_level):
    tables = new.child_matrix_tables()
    lat = new.make_lattice(fov, 0.0, 0.0, init_res, max_level + 1)
    fn = new.make_raytrace(raytrace, None)
    return new.refine(
        fn, lat, init_res, fov / init_res, min_img_sep, max_level, tables, None
    )


def _run_old(raytrace, fov, init_res, min_img_sep, max_level):
    tables = oracle.child_matrix_tables()
    lat = oracle._Lattice(fov, 0.0, 0.0, init_res, max_level + 1)
    fn = oracle._make_raytrace_np(raytrace, None)
    return oracle._refine(
        fn, lat, init_res, fov / init_res, min_img_sep, max_level, tables, None
    )


def _leaf_key_set(v, ij_of_slot, key_of_ij):
    """Canonical, order-independent identity for a leaf set."""
    keys = np.sort(key_of_ij(ij_of_slot[v]), axis=1)
    return sorted(map(tuple, keys.tolist()))


@pytest.mark.parametrize(
    "raytrace,fov,init_res,min_img_sep,max_level",
    [
        (_affine, 4.0, 2, 0.5, 2),
        (_sie_like, 4.0, 4, 0.25, 3),
        (_sie_like, 5.0, 3, 0.1, 4),
    ],
)
def test_refine_reproduces_the_oracle_leaf_set(
    raytrace, fov, init_res, min_img_sep, max_level
):
    cache, active, store, counters = _run_new(
        raytrace, fov, init_res, min_img_sep, max_level
    )
    ref = _run_old(raytrace, fov, init_res, min_img_sep, max_level)

    v_new, lvl_new, _, st_new = new.store_compact(store)
    v_old, lvl_old, _, st_old = ref.store.compact()

    lat_old = oracle._Lattice(fov, 0.0, 0.0, init_res, max_level + 1)
    ij_new = backend.to_numpy(cache.ij)
    got = _leaf_key_set(backend.to_numpy(v_new), ij_new, lambda ij: lat_old.key(ij))
    want = _leaf_key_set(v_old, ref.cache.ij, lambda ij: lat_old.key(ij))
    assert got == want
    assert sorted(backend.to_numpy(lvl_new).tolist()) == sorted(lvl_old.tolist())
    assert sorted(backend.to_numpy(st_new).tolist()) == sorted(st_old.tolist())


@pytest.mark.parametrize(
    "raytrace,fov,init_res,min_img_sep,max_level",
    [(_sie_like, 4.0, 4, 0.25, 3), (_sie_like, 5.0, 3, 0.1, 4)],
)
def test_refine_counters_match_the_oracle(
    raytrace, fov, init_res, min_img_sep, max_level
):
    _, _, _, counters = _run_new(raytrace, fov, init_res, min_img_sep, max_level)
    ref = _run_old(raytrace, fov, init_res, min_img_sep, max_level)
    assert counters == ref.counters


def test_refine_converges_everywhere_at_level_zero_for_an_affine_map():
    _, _, store, counters = _run_new(_affine, 4.0, 2, 0.5, 3)
    v, level, _, status = new.store_compact(store)
    assert (backend.to_numpy(level) == 0).all()
    assert counters["converged_level0"] == 2 * 2**2
    assert bool(backend.all(status == new.LEAF_CONVERGED))
    assert counters["parity_splits"] == 0
    assert counters["deviation_splits"] == 0


def test_refine_evaluates_every_point_exactly_once():
    calls = []

    def counting(x, y):
        calls.append(backend.to_numpy(x).copy())
        return _sie_like(x, y)

    cache, _, _, _ = _run_new(counting, 4.0, 3, 0.25, 3)
    traced = np.concatenate(calls)
    # max_level midpoints are traced but never cached, so the cache is a subset
    assert len(np.unique(np.round(traced, 12))) <= traced.size
    keys = backend.to_numpy(cache.keys)
    assert len(np.unique(keys)) == keys.size


# ---------------------------------------------------------------------------
# Ported from tests/test_adaptive_mesh.py, which called `_refine` directly on
# `caustics.lenses.adaptive` (a module still byte-identical to the frozen
# oracle at the time of this port). `test_refine_converges_everywhere_at_
# level_zero_for_an_affine_map` is not re-added here: that exact name is
# already defined above, verbatim from the brief, covering the same property.
#
# Several of these were already failing at HEAD, and not only the
# kappa-one-sheet test the task brief calls out by name. Commit b6dc3eb
# ("updated leaf classifications for caustic extraction") reworked the
# max_level branch's status assignment -- splitting the old coarse INVALID
# into a finite-but-parity-failing INVALID and a has-a-non-finite-sample
# NONFINITE, retiring SIZE_FLOOR, and (in the balance cascade) replacing the
# dedicated FORCED tag with inheritance of the parent's status -- without
# updating every test that still encoded the pre-b6dc3eb classification. Each
# updated assertion below was checked against `_refine_with`, which calls the
# same frozen `_refine`/`_Lattice`/`_make_raytrace_np` the oracle module
# exposes, not guessed.
# ---------------------------------------------------------------------------


def _make_counting_raytrace(fn):
    """Wrap a numpy (N,2)->(N,2) map as a backend raytrace, counting evaluations."""
    calls = {"points": 0, "batches": 0}

    def raytrace(x, y):
        xy = np.stack([backend.to_numpy(x), backend.to_numpy(y)], axis=-1)
        calls["points"] += xy.shape[0]
        calls["batches"] += 1
        out = fn(xy)
        return backend.as_array(out[:, 0]), backend.as_array(out[:, 1])

    return raytrace, calls


def _refine_with(fn, fov=4.0, init_res=4, min_img_sep=0.5, max_depth=25):
    """Run `new.refine` on a numpy ``p -> out`` map, mirroring the legacy
    ``refine_with`` helper but built on the backend interfaces."""
    tables = new.child_matrix_tables()
    max_level = min(max_depth, new.depth_floor(fov, init_res, min_img_sep))
    lat = new.make_lattice(fov, 0.0, 0.0, init_res, max_level + 1)
    raytrace, calls = _make_counting_raytrace(fn)
    cache, active, store, counters = new.refine(
        new.make_raytrace(raytrace, None),
        lat,
        init_res,
        fov / init_res,
        min_img_sep,
        max_level,
        tables,
        None,
    )
    return cache, active, store, counters, lat, calls, max_level


def _signed_area(tri):
    """Twice the signed area of a (..., 3, 2) triangle."""
    P = new.shape_matrix(tri)
    return P[..., 0, 0] * P[..., 1, 1] - P[..., 0, 1] * P[..., 1, 0]


def _assert_balanced(cache, active, store, lat, max_level):
    """No leaf edge carries an active quarter point."""
    v, level, _, _ = new.store_compact(store)
    for lv in sorted(set(backend.to_numpy(level).tolist())):
        if lv > max_level - 2:
            continue
        sel = level == lv
        keys = new.edge_quarter_keys(lat, cache.ij[v[sel]])
        assert not bool(
            backend.any(new.active_contains(active, cache, keys))
        ), f"unbalanced at level {lv}"


def test_refine_never_evaluates_a_point_twice():
    cache, active, store, counters, lat, calls, max_level = _refine_with(
        lambda p: np.stack([p[:, 0], p[:, 1] ** 2], axis=-1), min_img_sep=0.05
    )
    assert calls["points"] == new.cache_size(cache) + counters["max_level_midpoints"]
    keys = backend.to_numpy(new.lattice_key(lat, cache.ij))
    assert len(np.unique(keys)) == new.cache_size(cache)


def test_refine_terminates_at_max_level_on_a_kappa_one_sheet():
    """kappa == 1 maps the whole lens plane to a point: A == 0 everywhere.

    STALE-TEST UPDATE (not the one named in the task brief, found by the same
    method): pre-b6dc3eb this fixture came to rest at ``SIZE_FLOOR``. With
    ``A == 0`` identically, every child edge matrix ``Q_k`` is also identically
    zero, so the quadratic-vertex parity check added after that commit
    (`quadratic_vertex_parity_ok`) sees an exactly-zero vertex determinant --
    neither strictly positive nor strictly negative -- and fails closed at
    every leaf. All six samples stay finite throughout, so `finite_samples` is
    True everywhere and the max_level branch marks every leaf ``INVALID``
    rather than the old ``SIZE_FLOOR``, and ``parity_invalid`` covers the whole
    leaf set rather than 0. Verified directly against `_refine_with` (which
    calls the frozen oracle's own `_refine`) on this exact fixture: 512 leaves,
    all level 2, all INVALID, parity_invalid == 512, sigma_zero == 672.
    """
    cache, active, store, counters, lat, calls, max_level = _refine_with(
        lambda p: np.zeros_like(p), min_img_sep=0.5
    )
    v, level, _, status = new.store_compact(store)
    assert bool(backend.all(level == max_level))
    assert bool(backend.all(status == new.LEAF_INVALID))
    assert counters["sigma_zero"] > 0
    assert bool(backend.all(backend.isfinite(cache.beta)))
    # This fixture genuinely reaches max_level, so it is the one that pins the
    # loop's batch structure on the full-descent path: one raytrace batch per
    # level that needs new points, and the max_level iteration needs one more
    # of its own for the midpoints -- hence max_level + 1 batches.
    assert calls["batches"] == max_level + 1
    # Every point of the widened lattice, exactly once. Even-even points are
    # triangle vertices; single-odd points are horizontal or vertical edge
    # midpoints; double-odd points are cell centres, which are the diagonal
    # edge's midpoint. The three cases are exhaustive and disjoint, so a fixture
    # that refines uniformly to max_level covers the lattice exactly.
    assert calls["points"] == (lat.n + 1) ** 2
    # kappa == 1 maps everything to a point, so all four child determinants are
    # exactly zero -- a constant sign under `parity_from_children` alone. But
    # `evaluate_criterion` also runs `quadratic_vertex_parity_ok`, which fails
    # closed on the exact-zero vertex determinant, so every leaf is condemned.
    assert counters["parity_invalid"] == v.shape[0]


def _sis_raytrace(p, b=1.0):
    """SIS deflection ``beta = theta (1 - b/|theta|)``, non-finite at ``theta = 0``.

    A *point* non-finite set, unlike the half-plane fixtures: the origin is a
    lattice vertex for even ``init_res``, so exactly one sample point in the
    whole build is non-finite and the six level-0 triangles sharing it are the
    ones the old terminate-on-non-finite policy condemned wholesale. An
    area-shaped fixture cannot distinguish a policy that refines into the bad
    set from one that stops at its boundary, because there the boundary is
    where all the leaves are anyway.

    The Jacobian is non-degenerate away from the critical curve ``|theta| = b``,
    so most of the domain converges early and the refinement that does happen is
    attributable.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.linalg.norm(p, axis=-1, keepdims=True)
        return p * (1.0 - b / r)


def test_refine_splits_a_nonfinite_subregion_down_to_max_level():
    """Non-finite is maximal ignorance, so it refines rather than terminating.

    The bad half-plane is condemned only at ``max_level``, where no split is
    available -- not at whatever level it was first sampled. Reinstating an
    early ``store_add(v[~good], ..., LEAF_NONFINITE)`` would put NONFINITE rows
    at level 0 and fail the level assertion.

    STALE-TEST UPDATE: pre-b6dc3eb the condemned status here was ``INVALID``.
    Post-b6dc3eb ``INVALID`` is reserved for a leaf whose six samples are all
    finite but whose children disagree on parity; a leaf with any non-finite
    sample -- this fixture's whole failure mode -- lands on ``NONFINITE``
    instead. Verified against `_refine_with` on this fixture: 224 leaves, 32
    CONVERGED (levels 0-1), 192 NONFINITE (all level 2 == max_level), 0
    INVALID.
    """

    def broken(p):
        out = p.copy()
        bad = p[:, 0] > 0.5
        out[bad] = np.nan
        return out

    cache, active, store, counters, lat, calls, max_level = _refine_with(
        broken, min_img_sep=0.5
    )
    assert max_level > 0, "fixture must allow at least one split"
    v, level, _, status = new.store_compact(store)
    nonfinite = status == new.LEAF_NONFINITE
    assert bool(backend.any(nonfinite))
    assert bool(backend.all(level[nonfinite] == max_level))
    assert not bool(backend.all(backend.isfinite(cache.beta[v[nonfinite]])))
    # a triangle wholly in the good half is untouched
    good = status == new.LEAF_CONVERGED
    assert bool(backend.any(good))
    assert bool(backend.all(backend.isfinite(cache.beta[v[good]])))


def test_refine_splits_only_the_triangles_that_touch_a_point_singularity():
    """The cost of refining on non-finite, counted exactly.

    Six level-0 triangles share the origin, and a red split hands the bad vertex
    to exactly one of the four children -- the corner child at that vertex -- so
    the non-finite frontier stays six triangles wide at every level rather than
    quadrupling. Hand-derived total: ``6 * max_level`` splits over levels
    ``0 .. max_level - 1``, with no non-finite triangle left to split at
    ``max_level``.

    This is the counter that would expose an area-shaped non-finite region
    driving an ``O(4**max_level)`` descent, which is the one real cost of
    inverting the policy.
    """
    cache, active, store, counters, lat, calls, max_level = _refine_with(
        _sis_raytrace, min_img_sep=0.05
    )
    assert max_level == 5, "hand-derived counts below assume this depth"
    assert counters["nonfinite_splits"] == 6 * max_level


def test_refine_marks_nonfinite_vertices_invalid_even_at_max_level():
    """The max_level short-circuit must not blanket-label everything one status.

    ``min_img_sep`` forces ``max_level == 0``, so the loop's first and only
    iteration *is* the max_level iteration, and the ``~finite_v`` branch is the
    whole of this build's non-finite handling -- there is no deeper level for a
    non-finite triangle to be pushed down to. That isolation is the point: with
    ``max_level > 0`` a failure here could equally be the split path
    misbehaving, whereas at ``max_level == 0`` only the short-circuit's own
    discrimination can be at fault.

    STALE-TEST UPDATE: pre-b6dc3eb the finite/non-finite pair here was
    SIZE_FLOOR/INVALID. Post-b6dc3eb it is CONVERGED/NONFINITE: INVALID no
    longer means "has a non-finite sample" at all, so this fixture -- which
    never disagrees on parity, only on finiteness -- produces no INVALID rows.
    Verified against `_refine_with`: 32 leaves, 16 CONVERGED (all finite), 16
    NONFINITE (none finite), parity_invalid == 0.
    """

    def broken(p):
        out = p.copy()
        out[p[:, 0] > 0.5] = np.nan
        return out

    cache, active, store, counters, lat, calls, max_level = _refine_with(
        broken, min_img_sep=2.0
    )
    assert max_level == 0
    v, level, _, status = new.store_compact(store)
    assert bool(backend.any(status == new.LEAF_NONFINITE))
    # Without this, a run producing zero CONVERGED rows would make the loop
    # below vacuously true.
    assert bool(backend.any(status == new.LEAF_CONVERGED))
    # Every vertex sits on an integer coordinate and the NaN half-plane starts
    # at x > 0.5, so no all-finite-vertex triangle here has a NaN midpoint, and
    # the map is the identity where it is finite, so parity never changes.
    # NONFINITE is therefore non-finiteness alone -- which is what the loop
    # below is entitled to assume.
    assert counters["parity_invalid"] == 0
    for row in backend.to_numpy(backend.flatnonzero(status == new.LEAF_CONVERGED)):
        assert bool(backend.all(backend.isfinite(cache.beta[v[row]])))
    for row in backend.to_numpy(backend.flatnonzero(status == new.LEAF_NONFINITE)):
        assert not bool(backend.all(backend.isfinite(cache.beta[v[row]])))


def _localised_fold(p):
    """Affine away from a narrow band, curved and fold-bearing inside it.

    Outside |y| < 0.5 the map is exactly affine with sigma_min = 0.6, so those
    triangles converge at level 0. Inside, beta2 = 0.6y + y^2 - 0.25 is curved and
    its Jacobian 0.6 + 2y changes sign at y = -0.3, so both the deviation test and
    the parity test fire. The result is a mesh with real level transitions for the
    cascade tests to work on.

    Continuous at |y| = 0.5, where y^2 - 0.25 vanishes.

    Do NOT replace the bend with a constant outside the band (e.g. 0.25*sign(y)):
    that makes the map degenerate in y everywhere outside, so sigma_min == 0, every
    triangle splits, and the mesh refines uniformly to max_level with no level
    transitions at all -- silently voiding every test that depends on them.
    """
    y = p[:, 1]
    bend = np.where(np.abs(y) < 0.5, y**2 - 0.25, 0.0)
    return np.stack([p[:, 0], 0.6 * y + bend], axis=-1)


def test_max_level_leaves_are_condemned_exactly_when_parity_changes():
    """Independent oracle: recompute parity from each max_level leaf's own
    geometry and require the assigned status to agree row for row.

    `_localised_fold`'s Jacobian 0.6 + 2y changes sign at y = -0.3, so the band of
    max_level leaves straddling that line is condemned and the rest are not --
    measured at 256 of 512 for this tolerance. Both arms are asserted non-empty,
    so the row-for-row equality cannot pass vacuously on an all-True or all-False
    mask.

    The oracle builds its midpoints with `lattice_xy(lat, midpoint_ij(...))`
    rather than averaging vertex positions. The two differ in the last ulp,
    which is enough to flip the sign of a near-zero child determinant right at
    the fold -- and then this test would be measuring float rounding rather
    than the branch it is aimed at.

    STALE-TEST UPDATE: pre-b6dc3eb the spared arm was ``SIZE_FLOOR``. The
    parity-vs-status agreement itself (`parity_from_children` alone, matching
    `got_invalid` row for row) is unaffected by that rename -- verified against
    `_refine_with` on this fixture: 512 max_level leaves, 256 INVALID (matching
    ``~want_ok`` exactly), 256 CONVERGED (not SIZE_FLOOR).
    """
    cache, active, store, counters, lat, calls, max_level = _refine_with(
        _localised_fold, min_img_sep=0.05
    )
    v, level, _, status = new.store_compact(store)
    sel = backend.flatnonzero(level == max_level)
    assert sel.shape[0] > 0, "fixture must reach max_level"

    ij = cache.ij[v[sel]]
    beta_v = cache.beta[v[sel]]
    mid_xy_np = backend.to_numpy(new.lattice_xy(lat, new.midpoint_ij(ij))).reshape(
        -1, 2
    )
    beta_m_np = _localised_fold(mid_xy_np).reshape(-1, 3, 2)
    beta_m = backend.as_array(beta_m_np, dtype=backend.float64)
    want_ok = new.parity_from_children(new.child_shape_matrices(beta_v, beta_m))

    got_invalid = status[sel] == new.LEAF_INVALID
    assert bool(backend.any(got_invalid)), "fixture must condemn something"
    assert bool(backend.any(~got_invalid)), "fixture must spare something"
    assert backend.to_numpy(got_invalid).tolist() == backend.to_numpy(~want_ok).tolist()
    assert bool(backend.all(status[sel][~got_invalid] == new.LEAF_CONVERGED))
    assert counters["parity_invalid"] == int(backend.to_numpy(backend.sum(got_invalid)))
    # Condemnation happens only at the size floor; nothing coarser is touched.
    assert bool(backend.all(level[status == new.LEAF_INVALID] == max_level))


def test_max_level_condemns_a_nonfinite_midpoint_with_finite_vertices():
    """A midpoint the criterion cannot evaluate is excluded before parity ever runs.

    `min_img_sep` forces max_level == 0, so the level-0 triangles *are* the
    max_level triangles. Vertices land on integer arcsec coordinates and
    midpoints on half-integers, so a NaN band of half-width 0.1 around x == 0.5
    hits midpoints only and leaves every vertex finite -- isolating the path
    where a non-finite midpoint, not the `finite_v` check, is what condemns.

    Exactly the eight triangles of the x in [0, 1] cell column are hit: both
    root shapes place a midpoint at x == 0.5, there are four cells in that
    column, and two triangles per cell.

    STALE-TEST UPDATE: pre-b6dc3eb this condemnation was ``INVALID`` and cost
    8 in ``parity_invalid`` -- a non-finite midpoint reached
    `parity_from_children`, whose NaN propagation happened to fail it closed.
    Post-b6dc3eb the max_level branch gates on ``finite_m`` (all six samples
    finite) *before* calling `evaluate_criterion` at all, so a triangle whose
    only problem is a non-finite midpoint is excluded up front and never
    reaches the parity test -- it is ``NONFINITE``, and ``parity_invalid`` is
    0, not 8. Verified against `_refine_with`: 32 leaves, 8 NONFINITE (all
    vertex-finite), 24 CONVERGED, parity_invalid == 0.
    """

    def broken(p):
        out = p.copy()
        out[np.abs(p[:, 0] - 0.5) < 0.1] = np.nan
        return out

    cache, active, store, counters, lat, calls, max_level = _refine_with(
        broken, min_img_sep=2.0
    )
    assert max_level == 0
    v, level, _, status = new.store_compact(store)
    nonfinite = backend.flatnonzero(status == new.LEAF_NONFINITE)
    assert nonfinite.shape[0] == 8
    assert counters["parity_invalid"] == 0
    for row in backend.to_numpy(nonfinite):
        assert bool(backend.all(backend.isfinite(cache.beta[v[row]])))
    assert bool(backend.any(status == new.LEAF_CONVERGED))


def test_refine_makes_one_batch_per_level_and_exits_early_when_affine():
    """One raytrace batch per level that needs new points, and no more.

    The identity map is affine everywhere, so every level-0 triangle converges
    and the loop exits through the empty-``active_ij`` break without ever
    reaching ``max_level``. This pins the level-synchronous one-batch-per-level
    structure on the early-exit path.
    """
    cache, active, store, counters, lat, calls, max_level = _refine_with(
        lambda p: p * 1.0, min_img_sep=0.5
    )
    v, level, _, status = new.store_compact(store)
    assert (
        max_level > 0
    ), "fixture must allow deeper levels for early exit to mean anything"
    assert bool(backend.all(level == 0))
    assert calls["batches"] == 1


def test_refine_all_leaves_are_positively_oriented():
    cache, active, store, counters, lat, calls, max_level = _refine_with(
        lambda p: np.stack([p[:, 0], p[:, 1] ** 2], axis=-1), min_img_sep=0.05
    )
    v, level, _, status = new.store_compact(store)
    tri = new.lattice_xy(lat, cache.ij[v])
    assert bool(backend.all(_signed_area(tri) > 0))


def test_find_unbalanced_matches_the_six_quarter_key_reference_during_the_cascade(
    monkeypatch,
):
    """The edge-at-a-time scan must select exactly the rows the batch form did,
    checked where a violation can actually occur.

    `find_unbalanced` does not build the whole `(cand, 6)` key array -- it tests
    the six quarter points one at a time to keep the temporary `(cand,)`-shaped.
    That is a reassociation of the same disjunction, so `edge_quarter_keys`,
    which is unchanged and separately tested, is the reference it must
    reproduce row for row.

    A completed refinement is balanced by construction -- that is exactly what
    `test_mesh_is_edge_balanced_after_refinement` asserts -- so every frontier
    level a *finished* result exposes has zero candidates and zero violators,
    and `got == want` would pass trivially as empty-to-empty even for a rewrite
    that silently *misses* violators, which is the dangerous direction: it
    yields an unbalanced mesh rather than an error. This test instead
    intercepts every call `refine`'s own balance cascade makes while it is
    actively running, which is where non-empty violator sets exist, and
    requires that at least one such non-empty comparison happened.
    """
    real_find_unbalanced = new.find_unbalanced
    seen_nonempty = False

    def shim(store, cache, lat, active, max_level, frontier_level):
        nonlocal seen_nonempty
        got = real_find_unbalanced(store, cache, lat, active, max_level, frontier_level)
        bound = min(frontier_level - 2, max_level - 2)
        cand = backend.flatnonzero(store.valid & (store.level <= bound))
        if cand.shape[0]:
            keys = new.edge_quarter_keys(lat, cache.ij[store.v[cand]])
            want = cand[backend.any(new.active_contains(active, cache, keys), dim=1)]
        else:
            want = cand
        assert (
            backend.to_numpy(got).tolist() == backend.to_numpy(want).tolist()
        ), f"frontier_level={frontier_level}"
        if want.shape[0] > 0:
            seen_nonempty = True
        return got

    monkeypatch.setattr(new, "find_unbalanced", shim)
    _refine_with(_localised_fold, min_img_sep=0.02)

    assert seen_nonempty, "shim never observed a non-empty violator set"


def test_mesh_is_edge_balanced_after_refinement():
    cache, active, store, counters, lat, calls, max_level = _refine_with(
        _localised_fold, fov=4.0, init_res=4, min_img_sep=0.05
    )
    assert max_level >= 4, "fixture must allow several levels"
    v, level, _, status = new.store_compact(store)
    assert (
        len(np.unique(backend.to_numpy(level))) > 1
    ), "fixture must produce level transitions"
    _assert_balanced(cache, active, store, lat, max_level)


def test_cascade_produces_forced_children():
    """The cascade must force-split at least one already-converged neighbour.

    STALE-TEST UPDATE: pre-b6dc3eb a forced child carried a dedicated FORCED
    status. Post-b6dc3eb (see the comment in `refine`'s cascade) a forced
    child inherits its parent's status instead -- always CONVERGED,
    transitively, since a violator is never INVALID or NONFINITE -- so
    `LEAF_FORCED` no longer appears in the store at all.
    ``counters["forced"]`` is the only remaining witness that the cascade
    forced anything, which is exactly what it counts; both counter checks
    below already passed before this update and are unchanged.
    """
    cache, active, store, counters, lat, calls, max_level = _refine_with(
        _localised_fold, fov=4.0, init_res=4, min_img_sep=0.05
    )
    assert counters["forced"] > 0
    assert counters["cascade_rounds"] > 0


def test_forced_and_invalid_statuses_are_mutually_consistent():
    """INVALID leaves are finite; NONFINITE leaves never are.

    STALE-TEST UPDATE: pre-b6dc3eb a single INVALID status covered both a
    parity-changing max_level leaf and a non-finite one, so this test's job
    was to show INVALID was a genuine mix of both, plus that a then-existing
    FORCED status was always finite. Post-b6dc3eb they are two disjoint
    statuses -- INVALID means all six samples were finite and the children
    still disagreed on parity; NONFINITE means some sample was not finite --
    and FORCED no longer appears (see `test_cascade_produces_forced_children`).
    So the two arms this test now checks are that neither status leaks into
    the other's territory, which is the "mutually consistent" the name refers
    to, rather than that INVALID alone contains both.

    `find_unbalanced` filters candidates on ``store.valid & (store.level <=
    min(frontier_level, max_level) - 2)`` and not on status, so an INVALID or
    NONFINITE leaf would be an ordinary violator candidate like any other --
    but both only ever land at ``max_level``, two levels above that bound, and
    the ``max_level`` branch breaks out of the level loop before any cascade
    runs. So no violator is ever INVALID or NONFINITE, and every forced child
    inherits CONVERGED, which is what makes the inheritance in `refine`'s
    cascade exact here, not a simplification that happens to hold.
    """

    def half_bad(p):
        out = _localised_fold(p)
        out[p[:, 0] > 1.0] = np.nan
        return out

    cache, active, store, counters, lat, calls, max_level = _refine_with(
        half_bad, fov=4.0, init_res=4, min_img_sep=0.05
    )
    v, level, _, status = new.store_compact(store)
    assert bool(backend.any(status == new.LEAF_INVALID))
    assert bool(backend.any(status == new.LEAF_NONFINITE))

    inv_beta = cache.beta[v[status == new.LEAF_INVALID]]
    assert bool(
        backend.all(backend.isfinite(inv_beta))
    ), "INVALID must imply every sample was finite"
    nonfinite_beta = cache.beta[v[status == new.LEAF_NONFINITE]]
    all_finite = backend.all(backend.isfinite(nonfinite_beta), dim=(1, 2))
    assert not bool(
        backend.any(all_finite)
    ), "NONFINITE must imply at least one non-finite sample"
    # INVALID is unreachable below max_level, which is what makes the
    # unconditional status-inheritance in `refine` exact here, not lucky.
    assert bool(backend.all(level[status == new.LEAF_INVALID] == max_level))


def test_cascade_still_evaluates_every_point_exactly_once():
    cache, active, store, counters, lat, calls, max_level = _refine_with(
        _localised_fold, fov=4.0, init_res=4, min_img_sep=0.05
    )
    assert calls["points"] == new.cache_size(cache) + counters["max_level_midpoints"]
