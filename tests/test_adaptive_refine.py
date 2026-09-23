"""The refinement loop: its Jacobian gate, the balance cascade, and closure.

Wherever the refinement criterion still coincides with the frozen oracle's --
on maps with no fold for either parity test to find -- the loop must reproduce
the oracle leaf for leaf. Across a fold the two criteria deliberately differ,
and the tests below pin this module's own behaviour instead.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from caustics.backend_obj import backend
from caustics.lenses.func import adaptive as new


@pytest.fixture
def oracle_module():
    return pytest.importorskip(
        "caustics.lenses.old_adaptive", reason="optional frozen differential oracle"
    )


def _stack_2x2(a, b, c, d):
    """``[[a, b], [c, d]]`` at every point, shape ``(N, 2, 2)``."""
    return backend.stack(
        (backend.stack((a, b), dim=-1), backend.stack((c, d), dim=-1)), dim=-2
    )


def _sie_like(x, y):
    r = (x * x + y * y + 0.05) ** 0.5
    return x - 1.2 * x / r, y - 1.2 * y / r


def _sie_like_jacobian(x, y):
    r = (x * x + y * y + 0.05) ** 0.5
    k = 1.2 / r**3
    return _stack_2x2(
        1.0 - 1.2 / r + k * x * x, k * x * y, k * x * y, 1.0 - 1.2 / r + k * y * y
    )


def _affine(x, y):
    return 2.0 * x + 0.5 * y, -0.25 * x + 1.5 * y


def _affine_jacobian(x, y):
    one = backend.ones_like(x)
    return _stack_2x2(2.0 * one, 0.5 * one, -0.25 * one, 1.5 * one)


def _fold_free(x, y):
    """Curved everywhere, folded nowhere on ``|x| <= 2.5``.

    ``det A = 1 + 0.2 x - 0.48 cos(1.5 x) cos(2 y) >= 0.02`` there, so neither
    parity test has a critical curve to find, while the curvature still drives
    the deviation test through several levels -- and, at ``init_res=3``, a
    balance cascade.
    """
    return x + 0.4 * backend.sin(2.0 * y) + 0.1 * x * x, y + 0.4 * backend.sin(1.5 * x)


def _fold_free_jacobian(x, y):
    return _stack_2x2(
        1.0 + 0.2 * x,
        0.8 * backend.cos(2.0 * y),
        0.6 * backend.cos(1.5 * x),
        backend.ones_like(x),
    )


def _run_new(raytrace, jacobian, fov, init_res, min_img_sep, max_level):
    tables = new.child_matrix_tables()
    lat = new.make_lattice(fov, 0.0, 0.0, init_res, max_level + 1)
    fn = new.make_raytrace(raytrace, None)
    return new.refine(
        fn,
        jacobian,
        lat,
        init_res,
        fov / init_res,
        min_img_sep,
        max_level,
        tables,
        None,
    )


def _run_old(oracle_module, raytrace, fov, init_res, min_img_sep, max_level):
    tables = oracle_module.child_matrix_tables()
    lat = oracle_module._Lattice(fov, 0.0, 0.0, init_res, max_level + 1)
    fn = oracle_module._make_raytrace_np(raytrace, None)
    return oracle_module._refine(
        fn, lat, init_res, fov / init_res, min_img_sep, max_level, tables, None
    )


def _leaf_records(v, level, cls, converged, ij_of_slot, key_of_ij):
    """Canonical joint geometry and convergence records for a leaf set."""
    keys = np.sort(key_of_ij(ij_of_slot[v]), axis=1)
    return sorted(
        (tuple(key_row), int(lvl), int(shape_cls), bool(ok))
        for key_row, lvl, shape_cls, ok in zip(keys, level, cls, converged)
    )


# Maps with no fold anywhere. The Jacobian test never fires on them, and the
# oracle's quadratic-vertex parity check fires only on triangles the deviation
# test splits anyway (see the counters test below), so the two criteria reach
# the same verdict on every triangle.
FOLD_FREE_CASES = [
    (_affine, _affine_jacobian, 4.0, 2, 0.5, 2),
    (_fold_free, _fold_free_jacobian, 4.0, 4, 0.25, 3),
    (_fold_free, _fold_free_jacobian, 5.0, 3, 0.1, 4),
    (_fold_free, _fold_free_jacobian, 4.0, 2, 0.02, 6),
]


@pytest.mark.parametrize(
    "raytrace,jacobian,fov,init_res,min_img_sep,max_level", FOLD_FREE_CASES
)
def test_refine_reproduces_the_oracle_leaf_set_where_no_fold_exists(
    oracle_module, raytrace, jacobian, fov, init_res, min_img_sep, max_level
):
    """Leaf for leaf, wherever the two criteria still coincide.

    The Jacobian test replaced the oracle's quadratic-vertex parity check, so
    across a fold the two refinements deliberately differ -- on `_sie_like` at
    ``(4.0, 4, 0.25, 3)`` they no longer even agree on the leaf count, 548
    against 536. Without a fold they reach the same verdict everywhere, and
    everything else in the loop -- the level-synchronous batches, the splits,
    the balance cascade and its deferred midpoints -- must still match the
    oracle exactly.
    """
    cache, active, store, counters = _run_new(
        raytrace, jacobian, fov, init_res, min_img_sep, max_level
    )
    ref = _run_old(oracle_module, raytrace, fov, init_res, min_img_sep, max_level)

    v_new, lvl_new, cls_new, st_new = new.store_compact(store)
    v_old, lvl_old, cls_old, st_old = ref.store.compact()

    lat_old = oracle_module._Lattice(fov, 0.0, 0.0, init_res, max_level + 1)
    ij_new = backend.to_numpy(cache.ij)
    got = _leaf_records(
        backend.to_numpy(v_new),
        backend.to_numpy(lvl_new),
        backend.to_numpy(cls_new),
        backend.to_numpy(st_new) == new.LEAF_CONVERGED,
        ij_new,
        lat_old.key,
    )
    want = _leaf_records(
        v_old,
        lvl_old,
        cls_old,
        st_old == oracle_module.LeafStatus.CONVERGED,
        ref.cache.ij,
        lat_old.key,
    )
    assert got == want


@pytest.mark.parametrize(
    "raytrace,jacobian,fov,init_res,min_img_sep,max_level", FOLD_FREE_CASES
)
def test_refine_counters_match_the_oracle_where_no_fold_exists(
    oracle_module, raytrace, jacobian, fov, init_res, min_img_sep, max_level
):
    """Every counter matches, up to how a split is attributed.

    The oracle's quadratic-vertex check fires on a few strongly curved but
    unfolded coarse triangles -- up to nine per fixture here -- and books them
    as parity splits, although the deviation test splits them anyway.
    With no fold to find, this module books every such split to deviation, so
    the total is unchanged.
    """
    _, _, _, got = _run_new(raytrace, jacobian, fov, init_res, min_img_sep, max_level)
    want = dict(
        _run_old(
            oracle_module, raytrace, fov, init_res, min_img_sep, max_level
        ).counters
    )
    got = dict(got)
    assert got.pop("parity_splits") == 0
    assert got.pop("deviation_splits") == want.pop("deviation_splits") + want.pop(
        "parity_splits"
    )
    assert got == want


def test_refine_converges_everywhere_at_level_zero_for_an_affine_map():
    _, _, store, counters = _run_new(_affine, _affine_jacobian, 4.0, 2, 0.5, 3)
    v, level, _, status = new.store_compact(store)
    assert (backend.to_numpy(level) == 0).all()
    assert counters["converged_level0"] == 2 * 2**2
    assert bool(backend.all(status == new.LEAF_CONVERGED))
    assert counters["parity_splits"] == 0
    assert counters["deviation_splits"] == 0


# ---------------------------------------------------------------------------
# Ported from tests/test_adaptive_mesh.py, which called `_refine` directly on
# `caustics.lenses.adaptive` (a module still byte-identical to the frozen
# oracle at the time of this port). `test_refine_converges_everywhere_at_
# level_zero_for_an_affine_map` is not re-added here: that exact name is
# already defined above, verbatim from the brief, covering the same property.
#
# Every fixture map now travels with its analytic Jacobian, since `refine`
# evaluates the Jacobian before any triangle may converge. A status is a
# bitmask of `LEAF_*` flags, so assertions test the flags they are about with
# `&`, and compare with `==` only where a leaf provably carries one flag. The
# counts quoted in docstrings were measured with `_refine_with` on each
# fixture exactly as written.
# ---------------------------------------------------------------------------


def _make_counting_lens(fn, jac):
    """Wrap numpy maps as a lens, counting evaluations.

    ``fn`` is a ``(N, 2) -> (N, 2)`` lens map and ``jac`` its Jacobian,
    ``(N, 2) -> (N, 2, 2)``. The stand-in exposes the two methods a build
    reads, ``raytrace`` and ``jacobian_lens_equation``, and ``calls`` counts
    the points and batches each one evaluated.
    """
    calls = {"points": 0, "batches": 0, "jacobian_points": 0, "jacobian_batches": 0}

    def raytrace(x, y):
        xy = np.stack([backend.to_numpy(x), backend.to_numpy(y)], axis=-1)
        calls["points"] += xy.shape[0]
        calls["batches"] += 1
        out = fn(xy)
        return backend.as_array(out[:, 0]), backend.as_array(out[:, 1])

    def jacobian_lens_equation(x, y):
        xy = np.stack([backend.to_numpy(x), backend.to_numpy(y)], axis=-1)
        calls["jacobian_points"] += xy.shape[0]
        calls["jacobian_batches"] += 1
        return backend.as_array(jac(xy), dtype=backend.float64)

    lens = SimpleNamespace(
        raytrace=raytrace, jacobian_lens_equation=jacobian_lens_equation
    )
    return lens, calls


def _refine_with(fn, jac, fov=4.0, init_res=4, min_img_sep=0.5, max_depth=25):
    """Run `new.refine` on a numpy ``p -> out`` map and its Jacobian, mirroring
    the legacy ``refine_with`` helper but built on the backend interfaces."""
    tables = new.child_matrix_tables()
    max_level = min(max_depth, new.depth_floor(fov, init_res, min_img_sep))
    lat = new.make_lattice(fov, 0.0, 0.0, init_res, max_level + 1)
    lens, calls = _make_counting_lens(fn, jac)
    cache, active, store, counters = new.refine(
        new.make_raytrace(lens.raytrace, None),
        lens.jacobian_lens_equation,
        lat,
        init_res,
        fov / init_res,
        min_img_sep,
        max_level,
        tables,
        None,
    )
    return cache, active, store, counters, lat, calls, max_level


def _broken(fn, jac, where, value=np.nan):
    """``fn`` and ``jac`` with ``value`` wherever ``where(p)`` holds.

    The raytrace and its Jacobian break over the same region, as a real lens's
    would, so a triangle that reaches the bad set is non-finite in both.
    """

    def broken(p):
        out = fn(p)
        out[where(p)] = value
        return out

    def broken_jacobian(p):
        J = jac(p)
        J[where(p)] = value
        return J

    return broken, broken_jacobian


def _identity(p):
    return p * 1.0


def _identity_jacobian(p):
    return np.tile(np.eye(2), (p.shape[0], 1, 1))


def _fold(p):
    """``(x, y) -> (x, y**2)``, folded along ``y == 0``."""
    return np.stack([p[:, 0], p[:, 1] ** 2], axis=-1)


def _fold_jacobian(p):
    J = np.zeros((p.shape[0], 2, 2))
    J[:, 0, 0] = 1.0
    J[:, 1, 1] = 2.0 * p[:, 1]
    return J


def _collapse(p):
    """A ``kappa == 1`` sheet: the whole lens plane maps to one point."""
    return np.zeros_like(p)


def _collapse_jacobian(p):
    return np.zeros((p.shape[0], 2, 2))


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
        _fold, _fold_jacobian, min_img_sep=0.05
    )
    assert calls["points"] == new.cache_size(cache) + counters["max_level_midpoints"]
    keys = backend.to_numpy(new.lattice_key(lat, cache.ij))
    assert len(np.unique(keys)) == new.cache_size(cache)


def test_refine_terminates_at_max_level_on_a_kappa_one_sheet():
    """kappa == 1 maps the whole lens plane to a point: A == 0 everywhere.

    Every child edge matrix ``Q_k`` is then exactly zero -- a constant sign,
    so child parity passes -- but ``s`` is exactly zero too, the deviation
    test's ``0 < 0`` fails, and no triangle converges, so the Jacobian is
    never evaluated below ``max_level``. There it is forced, finds
    ``det A == 0`` at every sample, and flags it as unusable. Every leaf ends
    ``LEAF_CONVERGENCE_FAILED | LEAF_JACOBIAN_NONFINITE``, and
    ``parity_invalid`` covers the whole leaf set on the Jacobian's verdict
    alone. Measured with `_refine_with` on this fixture: 512 leaves, all level
    2, ``sigma_zero == 672``.
    """
    cache, active, store, counters, lat, calls, max_level = _refine_with(
        _collapse, _collapse_jacobian, min_img_sep=0.5
    )
    v, level, _, status = new.store_compact(store)
    assert bool(backend.all(level == max_level))
    want = new.LEAF_CONVERGENCE_FAILED | new.LEAF_JACOBIAN_NONFINITE
    assert bool(backend.all(status == want))
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
    assert counters["parity_invalid"] == v.shape[0]


def test_refine_evaluates_the_jacobian_only_where_it_can_withhold_convergence():
    """Below ``max_level``, only a triangle about to converge is checked.

    Two fixtures pin both ends exactly. On the identity map every level-0
    triangle passes the other tests, so each is checked once and converges.
    On a ``kappa == 1`` sheet none ever passes the deviation test, so the
    Jacobian never runs below ``max_level`` and its one call is the forced
    pass there. Either way that is one batch, and exactly six points per leaf:
    unlike raytraced points, Jacobian samples are not deduplicated between
    triangles that share them.
    """
    for fn, jac in ((_identity, _identity_jacobian), (_collapse, _collapse_jacobian)):
        cache, active, store, counters, lat, calls, max_level = _refine_with(
            fn, jac, min_img_sep=0.5
        )
        v, _, _, _ = new.store_compact(store)
        assert calls["jacobian_batches"] == 1, fn.__name__
        assert calls["jacobian_points"] == 6 * v.shape[0], fn.__name__


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


def _sis_jacobian(p, b=1.0):
    """``(1 - b/r) I + b theta theta^T / r**3``, non-finite at the origin too."""
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.linalg.norm(p, axis=-1)[:, None, None]
        return (1.0 - b / r) * np.eye(2) + b * p[:, :, None] * p[:, None, :] / r**3


def test_refine_splits_a_nonfinite_subregion_down_to_max_level():
    """Non-finite is maximal ignorance, so it refines rather than terminating.

    The bad half-plane is flagged only at ``max_level``, where no split is
    available -- not at whatever level it was first sampled. Reinstating an
    early ``store_add(v[~good], ..., LEAF_RAYTRACE_NONFINITE)`` would put
    flagged rows at level 0 and fail the level assertion. A triangle with a
    non-finite sample never reaches the rest of the criterion, so
    ``LEAF_RAYTRACE_NONFINITE`` is its only flag. Measured with `_refine_with`
    on this fixture: 224 leaves, 32 ``LEAF_CONVERGED`` (levels 0-1) and 192
    ``LEAF_RAYTRACE_NONFINITE`` (all level 2 == max_level).
    """
    fn, jac = _broken(_identity, _identity_jacobian, lambda p: p[:, 0] > 0.5)
    cache, active, store, counters, lat, calls, max_level = _refine_with(
        fn, jac, min_img_sep=0.5
    )
    assert max_level > 0, "fixture must allow at least one split"
    v, level, _, status = new.store_compact(store)
    nonfinite = status == new.LEAF_RAYTRACE_NONFINITE
    assert bool(backend.any(nonfinite))
    assert bool(backend.all(level[nonfinite] == max_level))
    assert not bool(backend.all(backend.isfinite(cache.beta[v[nonfinite]])))
    # a triangle wholly in the good half is untouched
    good = status == new.LEAF_CONVERGED
    assert bool(backend.any(good))
    assert bool(backend.all(backend.isfinite(cache.beta[v[good]])))
    assert bool(backend.all(good | nonfinite)), "no other status can occur here"


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
        _sis_raytrace, _sis_jacobian, min_img_sep=0.05
    )
    assert max_level == 5, "hand-derived counts below assume this depth"
    assert counters["nonfinite_splits"] == 6 * max_level


def test_refine_flags_nonfinite_vertices_even_at_max_level():
    """The max_level short-circuit must not blanket-label everything one status.

    ``min_img_sep`` forces ``max_level == 0``, so the loop's first and only
    iteration *is* the max_level iteration, and the ``~finite_v`` branch is the
    whole of this build's non-finite handling -- there is no deeper level for a
    non-finite triangle to be pushed down to. That isolation is the point: with
    ``max_level > 0`` a failure here could equally be the split path
    misbehaving, whereas at ``max_level == 0`` only the short-circuit's own
    discrimination can be at fault.

    This fixture never disagrees on parity, only on finiteness, so every leaf
    is either ``LEAF_CONVERGED`` or ``LEAF_RAYTRACE_NONFINITE`` alone. Measured
    with `_refine_with`: 32 leaves, 16 converged (all finite), 16 non-finite
    (none finite), ``parity_invalid == 0``.
    """
    fn, jac = _broken(_identity, _identity_jacobian, lambda p: p[:, 0] > 0.5)
    cache, active, store, counters, lat, calls, max_level = _refine_with(
        fn, jac, min_img_sep=2.0
    )
    assert max_level == 0
    v, level, _, status = new.store_compact(store)
    assert bool(backend.any(status == new.LEAF_RAYTRACE_NONFINITE))
    # Without this, a run producing zero CONVERGED rows would make the loop
    # below vacuously true.
    assert bool(backend.any(status == new.LEAF_CONVERGED))
    # Every vertex sits on an integer coordinate and the NaN half-plane starts
    # at x > 0.5, so no all-finite-vertex triangle here has a NaN midpoint, and
    # the map is the identity where it is finite, so parity never changes.
    # LEAF_RAYTRACE_NONFINITE is therefore non-finiteness alone -- which is what
    # the loop below is entitled to assume.
    assert counters["parity_invalid"] == 0
    for row in backend.to_numpy(backend.flatnonzero(status == new.LEAF_CONVERGED)):
        assert bool(backend.all(backend.isfinite(cache.beta[v[row]])))
    for row in backend.to_numpy(
        backend.flatnonzero(status == new.LEAF_RAYTRACE_NONFINITE)
    ):
        assert not bool(backend.all(backend.isfinite(cache.beta[v[row]])))


def test_forced_jacobian_skips_every_triangle_with_a_nonfinite_sample():
    """Forcing the Jacobian at ``max_level`` still leaves non-finite rows out.

    `_broken` makes the Jacobian NaN over the same half-plane as the raytrace,
    so a non-finite triangle that reached the Jacobian would come back
    ``LEAF_JACOBIAN_NONFINITE`` as well. None may: ``max_level == 0`` here, so
    every finite level-0 triangle is checked exactly once and no other is.
    """
    fn, jac = _broken(_identity, _identity_jacobian, lambda p: p[:, 0] > 0.5)
    cache, active, store, counters, lat, calls, max_level = _refine_with(
        fn, jac, min_img_sep=2.0
    )
    assert max_level == 0
    v, level, _, status = new.store_compact(store)
    finite = int(backend.to_numpy(backend.sum(status != new.LEAF_RAYTRACE_NONFINITE)))
    assert 0 < finite < v.shape[0]
    assert calls["jacobian_points"] == 6 * finite
    assert not bool(backend.any((status & new.LEAF_JACOBIAN_NONFINITE) != 0))


def _localised_fold(p):
    """Affine away from a narrow band, curved and fold-bearing inside it.

    Outside |y| < 0.5 the map is exactly affine with sigma_min = 0.6, so those
    triangles converge at level 0. Inside, beta2 = 0.6y + y^2 - 0.25 is curved and
    its Jacobian 0.6 + 2y changes sign at y = -0.3, so the deviation test and
    both parity tests fire. The result is a mesh with real level transitions for
    the cascade tests to work on.

    Continuous at |y| = 0.5, where y^2 - 0.25 vanishes.

    Do NOT replace the bend with a constant outside the band (e.g. 0.25*sign(y)):
    that makes the map degenerate in y everywhere outside, so sigma_min == 0, every
    triangle splits, and the mesh refines uniformly to max_level with no level
    transitions at all -- silently voiding every test that depends on them.
    """
    y = p[:, 1]
    bend = np.where(np.abs(y) < 0.5, y**2 - 0.25, 0.0)
    return np.stack([p[:, 0], 0.6 * y + bend], axis=-1)


def _localised_fold_jacobian(p):
    """``diag(1, 0.6 + 2y)`` inside the band, ``diag(1, 0.6)`` outside it."""
    y = p[:, 1]
    J = np.zeros((p.shape[0], 2, 2))
    J[:, 0, 0] = 1.0
    J[:, 1, 1] = 0.6 + np.where(np.abs(y) < 0.5, 2.0 * y, 0.0)
    return J


def test_max_level_flags_match_both_parity_tests_row_for_row():
    """Independent oracle for both parity flags at ``max_level``.

    Recompute child parity from each ``max_level`` leaf's own mapped samples,
    and Jacobian parity from `_localised_fold_jacobian` at its six lens-plane
    samples, and require the stored flags to agree row for row. The Jacobian
    is forced at ``max_level``, so its flag is a complete record there, not a
    lazy one.

    `_localised_fold`'s Jacobian 0.6 + 2y changes sign at y = -0.3. Measured
    at this tolerance: 1024 ``max_level`` leaves, 512 of them flagged
    ``LEAF_JACOBIAN_PARITY_UNRESOLVED`` and 256 of those
    ``LEAF_APPROX_PARITY_UNRESOLVED`` as well. Both flags are asserted to both
    fire and spare, so neither row-for-row equality can pass vacuously on an
    all-True or all-False mask.

    The midpoints are built with `lattice_xy(lat, midpoint_ij(...))` rather
    than by averaging vertex positions. The two differ in the last ulp, which
    is enough to flip the sign of a near-zero determinant right at the fold --
    and then this test would be measuring float rounding rather than the
    branch it is aimed at.
    """
    cache, active, store, counters, lat, calls, max_level = _refine_with(
        _localised_fold, _localised_fold_jacobian, min_img_sep=0.05
    )
    v, level, _, status = new.store_compact(store)
    sel = backend.flatnonzero(level == max_level)
    assert sel.shape[0] > 0, "fixture must reach max_level"

    ij = cache.ij[v[sel]]
    theta_v = new.lattice_xy(lat, ij)
    theta_m = new.lattice_xy(lat, new.midpoint_ij(ij))
    beta_m_np = _localised_fold(backend.to_numpy(theta_m).reshape(-1, 2)).reshape(
        -1, 3, 2
    )
    beta_m = backend.as_array(beta_m_np, dtype=backend.float64)
    child_ok = backend.to_numpy(
        new.parity_from_children(new.child_shape_matrices(cache.beta[v[sel]], beta_m))
    )
    lens, _ = _make_counting_lens(_localised_fold, _localised_fold_jacobian)
    jac_ok, jac_bad = (
        backend.to_numpy(x)
        for x in new.jacobian_parity_ok(
            lens.jacobian_lens_equation, theta_v, theta_m, return_details=True
        )
    )

    st = backend.to_numpy(status[sel])
    approx = (st & new.LEAF_APPROX_PARITY_UNRESOLVED) != 0
    jacobian = (st & new.LEAF_JACOBIAN_PARITY_UNRESOLVED) != 0
    for name, flag in (("approximate", approx), ("Jacobian", jacobian)):
        assert flag.any(), f"fixture must raise the {name} parity flag somewhere"
        assert not flag.all(), f"fixture must spare something of {name} parity"
    assert approx.tolist() == (~child_ok).tolist()
    assert jacobian.tolist() == (~jac_ok & ~jac_bad).tolist()
    parity_flags = (
        new.LEAF_APPROX_PARITY_UNRESOLVED | new.LEAF_JACOBIAN_PARITY_UNRESOLVED
    )
    assert not (st & ~parity_flags).any(), "no other flag occurs on this fixture"
    assert counters["parity_invalid"] == int((approx | jacobian).sum())
    # Flags are stored only at the size floor; nothing coarser is touched.
    assert bool(backend.all(level[status != new.LEAF_CONVERGED] == max_level))


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

    The max_level branch requires all six samples finite before it calls
    `evaluate_criterion` at all, so such a triangle never reaches either parity
    test: it is ``LEAF_RAYTRACE_NONFINITE`` alone, and ``parity_invalid`` is 0.
    Measured with `_refine_with`: 32 leaves, 8 non-finite (all vertex-finite),
    24 ``LEAF_CONVERGED``.
    """
    fn, jac = _broken(
        _identity, _identity_jacobian, lambda p: np.abs(p[:, 0] - 0.5) < 0.1
    )
    cache, active, store, counters, lat, calls, max_level = _refine_with(
        fn, jac, min_img_sep=2.0
    )
    assert max_level == 0
    v, level, _, status = new.store_compact(store)
    nonfinite = backend.flatnonzero(status == new.LEAF_RAYTRACE_NONFINITE)
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
        _identity, _identity_jacobian, min_img_sep=0.5
    )
    v, level, _, status = new.store_compact(store)
    assert (
        max_level > 0
    ), "fixture must allow deeper levels for early exit to mean anything"
    assert bool(backend.all(level == 0))
    assert calls["batches"] == 1


def test_refine_all_leaves_are_positively_oriented():
    cache, active, store, counters, lat, calls, max_level = _refine_with(
        _fold, _fold_jacobian, min_img_sep=0.05
    )
    v, level, _, status = new.store_compact(store)
    tri = new.lattice_xy(lat, cache.ij[v])
    assert bool(backend.all(_signed_area(tri) > 0))


def _gaussian_bump(p, w=0.08, amp=1.0, c=(0.13, 0.07)):
    """Narrow Gaussian bump, curved enough to reach ``close``'s ``count == 3`` branch.

    An ``init_res=8`` grid over this map is coarse enough that most triangles
    converge quickly while a few interior ones split deep enough to leave a
    fully-hanging (3-node) origin behind for ``close`` to red-split.
    """
    centre = np.asarray(c)
    r2 = ((p - centre) ** 2).sum(axis=-1)
    return p * 0.5 + (amp * np.exp(-r2 / (2 * w**2)))[:, None] * np.array([1.0, 0.3])


def _gaussian_bump_jacobian(p, w=0.08, amp=1.0, c=(0.13, 0.07)):
    """``0.5 I + e grad(g)^T``, with ``e = (1, 0.3)`` and ``g`` the bump itself."""
    d = p - np.asarray(c)
    g = amp * np.exp(-(d**2).sum(axis=-1) / (2 * w**2))
    grad = -(g / w**2)[:, None] * d
    return 0.5 * np.eye(2) + np.array([1.0, 0.3])[None, :, None] * grad[:, None, :]


@pytest.mark.parametrize(
    "fn,jac,kw",
    [
        (_localised_fold, _localised_fold_jacobian, dict(min_img_sep=0.05)),
        (_sis_raytrace, _sis_jacobian, dict(min_img_sep=0.05)),
        (
            _gaussian_bump,
            _gaussian_bump_jacobian,
            dict(fov=4.0, init_res=8, min_img_sep=0.02),
        ),
    ],
    ids=["localised_fold", "sis", "gaussian_bump"],
)
def test_converged_leaves_pass_jacobian_parity_at_their_own_samples(fn, jac, kw):
    """No leaf converges while a critical curve runs between its samples.

    Recomputed independently from every converged leaf's own six lens-plane
    samples. Each fixture converges leaves well above ``max_level`` too, so this
    covers the lazy Jacobian path and not just the forced one.

    For a leaf the criterion converged itself this is guaranteed. A
    balance-cascade child is converged without a check of its own: it inherits
    ``LEAF_CONVERGED`` from a parent whose six samples the Jacobian cleared,
    while its three edge midpoints are new points no Jacobian ever saw. For
    those children the property is measured on these fixtures, not
    guaranteed -- a critical curve slipping between a parent's samples could in
    principle surface in a child.
    """
    cache, active, store, counters, lat, calls, max_level = _refine_with(fn, jac, **kw)
    v, level, _, status = new.store_compact(store)
    rows = backend.flatnonzero(status == new.LEAF_CONVERGED)
    assert bool(backend.any(level[rows] < max_level)), "fixture must converge early"
    assert counters["forced"] > 0, "fixture must exercise the cascade too"
    ij = cache.ij[v[rows]]
    lens, _ = _make_counting_lens(fn, jac)
    ok = new.jacobian_parity_ok(
        lens.jacobian_lens_equation,
        new.lattice_xy(lat, ij),
        new.lattice_xy(lat, new.midpoint_ij(ij)),
    )
    assert bool(backend.all(ok))


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
    _refine_with(_localised_fold, _localised_fold_jacobian, min_img_sep=0.02)

    assert seen_nonempty, "shim never observed a non-empty violator set"


def test_mesh_is_edge_balanced_after_refinement():
    cache, active, store, counters, lat, calls, max_level = _refine_with(
        _localised_fold, _localised_fold_jacobian, fov=4.0, init_res=4, min_img_sep=0.05
    )
    assert max_level >= 4, "fixture must allow several levels"
    v, level, _, status = new.store_compact(store)
    assert (
        len(np.unique(backend.to_numpy(level))) > 1
    ), "fixture must produce level transitions"
    _assert_balanced(cache, active, store, lat, max_level)


def test_cascade_produces_forced_children():
    """The cascade must force-split at least one already-converged neighbour.

    A forced child carries no flag of its own: it inherits its parent's status
    (see the comment in `refine`'s cascade), and that is always
    ``LEAF_CONVERGED``, transitively, since no flagged leaf is ever a violator.
    So nothing in the store marks a forced child, and ``counters["forced"]``
    is the only witness that the cascade forced anything, which is exactly
    what it counts.
    """
    cache, active, store, counters, lat, calls, max_level = _refine_with(
        _localised_fold, _localised_fold_jacobian, fov=4.0, init_res=4, min_img_sep=0.05
    )
    assert counters["forced"] > 0
    assert counters["cascade_rounds"] > 0


def test_failure_flags_are_mutually_consistent():
    """Each flag stays in its own territory.

    A leaf with a non-finite sample never reaches the rest of the criterion,
    so ``LEAF_RAYTRACE_NONFINITE`` is always alone and always has a
    non-finite vertex -- here the NaN region is a half-plane, so a non-finite
    midpoint implies a non-finite vertex. Every other flag needs all six
    samples finite to be computed at all. Measured on this fixture: 1221
    converged leaves, 8192 non-finite, 192 flagged for Jacobian parity alone
    and 192 for both parity tests.

    `find_unbalanced` filters candidates on ``store.valid & (store.level <=
    min(frontier_level, max_level) - 2)`` and not on status, so a flagged leaf
    would be an ordinary violator candidate like any other -- but flagged
    leaves only ever land at ``max_level``, two levels above that bound, and
    the ``max_level`` branch breaks out of the level loop before any cascade
    runs. So no violator is ever flagged, and every forced child inherits
    ``LEAF_CONVERGED``, which is what makes the inheritance in `refine`'s
    cascade exact here, not a simplification that happens to hold.
    """
    fn, jac = _broken(
        _localised_fold, _localised_fold_jacobian, lambda p: p[:, 0] > 1.0
    )
    cache, active, store, counters, lat, calls, max_level = _refine_with(
        fn, jac, fov=4.0, init_res=4, min_img_sep=0.05
    )
    v, level, _, status = new.store_compact(store)
    raytrace_bad = (status & new.LEAF_RAYTRACE_NONFINITE) != 0
    parity_bad = (
        status
        & (new.LEAF_APPROX_PARITY_UNRESOLVED | new.LEAF_JACOBIAN_PARITY_UNRESOLVED)
    ) != 0
    assert bool(backend.any(raytrace_bad))
    assert bool(backend.any(parity_bad))

    all_finite = backend.all(backend.isfinite(cache.beta[v]), dim=(1, 2))
    assert bool(
        backend.all(status[raytrace_bad] == new.LEAF_RAYTRACE_NONFINITE)
    ), "a non-finite leaf must carry no other flag"
    assert not bool(
        backend.any(all_finite[raytrace_bad])
    ), "LEAF_RAYTRACE_NONFINITE must imply at least one non-finite sample"
    assert bool(
        backend.all(all_finite[~raytrace_bad])
    ), "every other leaf's samples must be finite"
    # Flags are unreachable below max_level, which is what makes the
    # unconditional status inheritance in `refine` exact here, not lucky.
    assert bool(backend.all(level[status != new.LEAF_CONVERGED] == max_level))


def test_cascade_still_evaluates_every_point_exactly_once():
    cache, active, store, counters, lat, calls, max_level = _refine_with(
        _localised_fold, _localised_fold_jacobian, fov=4.0, init_res=4, min_img_sep=0.05
    )
    assert calls["points"] == new.cache_size(cache) + counters["max_level_midpoints"]


def test_min_angle_of_an_equilateral_triangle():
    tri = backend.as_array(
        np.array([[[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3) / 2]]]),
        dtype=backend.float64,
    )
    assert backend.to_numpy(new.min_angle(tri))[0] == pytest.approx(np.pi / 3)


def test_min_angle_of_a_degenerate_triangle_is_zero_not_nan():
    tri = backend.as_array(
        np.array([[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]]), dtype=backend.float64
    )
    assert backend.to_numpy(new.min_angle(tri))[0] == 0.0


def test_canonical_order_is_independent_of_input_order():
    cache, active, store, _ = _run_new(_sie_like, _sie_like_jacobian, 4.0, 3, 0.25, 3)
    lat = new.make_lattice(4.0, 0.0, 0.0, 3, 4)
    v, _, _, _ = new.store_compact(store)

    order = backend.to_numpy(new.canonical_order(lat, cache, v))
    v_np = backend.to_numpy(v)

    rng = np.random.default_rng(3)
    perm = rng.permutation(v_np.shape[0])
    v_shuf = backend.as_array(v_np[perm], dtype=backend.int64)
    order_shuf = backend.to_numpy(new.canonical_order(lat, cache, v_shuf))

    assert v_np[order].tolist() == v_np[perm][order_shuf].tolist()


def _oracle_cache_and_active(oracle_module, cache, active):
    """The oracle's cache and active set, holding exactly ``cache`` and ``active``.

    The oracle numbers its slots in insertion order, and the keys go in
    ascending, so its slots differ from this module's: ``remap[slot]`` is the
    oracle's slot for this module's ``slot``.
    """
    keys = backend.to_numpy(cache.keys)
    slots = backend.to_numpy(cache.slots)
    cache_o = oracle_module._VertexCache()
    slots_o = cache_o.insert(
        keys, backend.to_numpy(cache.ij)[slots], backend.to_numpy(cache.beta)[slots]
    )
    remap = np.empty_like(slots)
    remap[slots] = slots_o
    active_o = oracle_module._ActiveKeys(cache_o)
    active_o.add_slots(remap[np.flatnonzero(backend.to_numpy(active))])
    return cache_o, active_o, remap


def test_closure_matches_the_oracle(oracle_module):
    """`close` must reproduce the oracle's closure on identical input.

    Both sides close the same pre-closure mesh -- this module's refinement,
    mirrored into the oracle's own cache and active-set types -- so this
    compares closure alone. Refining each side separately would compare the
    two refinement criteria as well, and across the fold `_sie_like` has they
    deliberately differ.
    """
    cache, active, store, _ = _run_new(_sie_like, _sie_like_jacobian, 4.0, 3, 0.25, 3)
    lat = new.make_lattice(4.0, 0.0, 0.0, 3, 4)
    lat_old = oracle_module._Lattice(4.0, 0.0, 0.0, 3, 4)
    cache_o, active_o, remap = _oracle_cache_and_active(oracle_module, cache, active)

    v, level, _, status = new.store_compact(store)
    order = new.canonical_order(lat, cache, v)
    v, level, status = v[order], level[order], status[order]
    leaves, origin, out_level, out_status = new.close(
        lat, cache, active, v, level, status
    )
    leaves_o, origin_o, lvl_out_o, st_out_o = oracle_module._close(
        lat_old,
        cache_o,
        active_o,
        remap[backend.to_numpy(v)],
        backend.to_numpy(level),
        backend.to_numpy(status),
    )

    assert leaves.shape[0] > v.shape[0], "fixture must give closure work to do"
    assert remap[backend.to_numpy(leaves)].tolist() == leaves_o.tolist()
    assert backend.to_numpy(origin).tolist() == origin_o.tolist()
    assert backend.to_numpy(out_level).tolist() == lvl_out_o.tolist()
    assert backend.to_numpy(out_status).tolist() == st_out_o.tolist()


def test_closure_origin_is_non_decreasing():
    cache, active, store, _ = _run_new(_sie_like, _sie_like_jacobian, 4.0, 3, 0.25, 3)
    lat = new.make_lattice(4.0, 0.0, 0.0, 3, 4)
    v, level, _, status = new.store_compact(store)
    order = new.canonical_order(lat, cache, v)
    _, origin, _, _ = new.close(
        lat, cache, active, v[order], level[order], status[order]
    )
    o = backend.to_numpy(origin)
    assert (
        np.diff(o) >= 0
    ).all(), "origin must be non-decreasing for segment reduction"


# ---------------------------------------------------------------------------
# Ported from tests/test_adaptive_mesh.py's closure tests, which called
# `_canonical_order`/`_close` directly on `caustics.lenses.adaptive` (still
# byte-identical to the frozen oracle at the time of this port).
# `test_min_angle_of_an_equilateral_triangle` and
# `test_canonical_order_is_independent_of_input_order` are not re-added here:
# both names are already defined above, verbatim from the brief, covering the
# same properties -- re-porting the legacy bodies under the same names would
# just silently shadow the Step-1 versions rather than add coverage, since a
# second `def` of the same name in one module replaces the first in pytest's
# collection. Their legacy counterparts were deleted with no replacement body.
# ---------------------------------------------------------------------------


def _closed_mesh(fn, jac, **kw):
    """Backend port of the legacy ``closed_mesh`` helper."""
    cache, active, store, counters, lat, calls, max_level = _refine_with(fn, jac, **kw)
    v, level, _, status = new.store_compact(store)
    order = new.canonical_order(lat, cache, v)
    v, level, status = v[order], level[order], status[order]
    # Captured BEFORE closure. `close` must add no vertices, so a test checking
    # `leaves` against the cache size has to use a bound that predates the call;
    # reading `cache_size` afterwards would silently absorb any growth into the
    # bound and the check could never fail.
    n_cache_pre = new.cache_size(cache)
    leaves, origin, out_level, out_status = new.close(
        lat, cache, active, v, level, status
    )
    return (
        cache,
        active,
        lat,
        v,
        level,
        status,
        leaves,
        origin,
        out_level,
        out_status,
        n_cache_pre,
    )


def _undirected_edges(lat, cache, leaves):
    k = backend.to_numpy(new.lattice_key(lat, cache.ij[leaves]))  # (L, 3)
    e = np.stack([k[:, [0, 1]], k[:, [1, 2]], k[:, [2, 0]]], axis=1)
    return np.sort(e, axis=-1).reshape(-1, 2)


def _hanging_nodes(lat, cache, active, slots):
    """Hanging-node mask, the same predicate ``close`` itself applies.

    No exactness gate: the lattice is one level finer than max_level, so
    `midpoint_ij` is exact at every level and this names true midpoints
    everywhere. A max_level midpoint has an odd coordinate and is never
    cached, so `active_contains` reads it as absent -- which is correct, not
    an artifact.
    """
    mid_keys = new.lattice_key(lat, new.midpoint_ij(cache.ij[slots]))
    return new.active_contains(active, cache, mid_keys)


def test_closure_makes_every_edge_appear_once_or_twice():
    cache, active, lat, v, pre_lvl, pre_st, leaves, origin, lvl, st, n_pre = (
        _closed_mesh(_localised_fold, _localised_fold_jacobian, min_img_sep=0.05)
    )
    edges = _undirected_edges(lat, cache, leaves)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    assert set(np.unique(counts)) <= {1, 2}


def test_closure_leaves_no_hanging_node():
    """The property closure exists for, checked directly on the closed mesh.

    Edge multiplicity cannot substitute for this: a hanging node never raises
    any edge's count (the coarse triangle contributes ``(A, B)`` once, the finer
    neighbours contribute only ``(A, M)`` and ``(M, B)``), so a closure that
    left hanging nodes behind would still show multiplicities inside ``{1, 2}``.
    Multiplicity catches over-generation; this catches under-closure.
    """
    cache, active, lat, v, pre_lvl, pre_st, leaves, origin, lvl, st, n_pre = (
        _closed_mesh(_localised_fold, _localised_fold_jacobian, min_img_sep=0.05)
    )
    assert bool(backend.any(_hanging_nodes(lat, cache, active, v))), "nothing to close"
    assert not bool(backend.any(_hanging_nodes(lat, cache, active, leaves)))


def test_pre_closure_mesh_is_not_already_conforming():
    """Guard against a fixture where closure has nothing to do.

    Tested with the hanging-node predicate directly rather than via
    undirected-edge multiplicity, because a hanging node does not raise any
    edge's count: a coarse triangle contributes its edge ``(A, B)`` exactly once
    while the finer neighbours contribute the half-edges ``(A, M)`` and
    ``(M, B)`` -- never ``(A, B)``, since no fine triangle has both endpoints. So
    a mesh riddled with hanging nodes has the same ``{1, 2}`` multiplicity
    profile as a conforming one, and hanging nodes are indistinguishable from
    domain-boundary edges by counting alone.
    This is the same predicate :func:`close` itself uses to classify each leaf.
    On the widened lattice it needs no exactness gate: `midpoint_ij` is exact
    at every level, so a True here is a genuine hanging node and never the
    collapsed-pseudo-midpoint artifact the narrow lattice produced.
    """
    cache, active, store, counters, lat, calls, max_level = _refine_with(
        _localised_fold, _localised_fold_jacobian, min_img_sep=0.05
    )
    v, level, _, status = new.store_compact(store)
    assert bool(
        backend.any(_hanging_nodes(lat, cache, active, v))
    ), "fixture has no hanging nodes"


def test_closure_preserves_orientation_and_inherits_level_and_status():
    cache, active, lat, v, pre_lvl, pre_st, leaves, origin, lvl, st, n_pre = (
        _closed_mesh(_localised_fold, _localised_fold_jacobian, min_img_sep=0.05)
    )
    tri = new.lattice_xy(lat, cache.ij[leaves])
    assert bool(backend.all(_signed_area(tri) > 0))
    assert backend.to_numpy(lvl).tolist() == backend.to_numpy(pre_lvl[origin]).tolist()
    assert backend.to_numpy(st).tolist() == backend.to_numpy(pre_st[origin]).tolist()
    assert lvl.shape == (leaves.shape[0],) and st.shape == (leaves.shape[0],)


def test_leaf_origin_groups_are_contiguous_and_tile_their_origin():
    cache, active, lat, v, pre_lvl, pre_st, leaves, origin, lvl, st, n_pre = (
        _closed_mesh(_localised_fold, _localised_fold_jacobian, min_img_sep=0.05)
    )
    origin_np = backend.to_numpy(origin)
    assert (np.diff(origin_np) >= 0).all(), "origin must be non-decreasing"
    child_area = backend.to_numpy(_signed_area(new.lattice_xy(lat, cache.ij[leaves])))
    origin_area = backend.to_numpy(_signed_area(new.lattice_xy(lat, cache.ij[v])))
    summed = np.zeros_like(origin_area)
    np.add.at(summed, origin_np, child_area)
    assert np.allclose(summed, origin_area, rtol=1e-12)


def test_unclosed_leaf_is_its_own_origin_geometry():
    cache, active, lat, v, pre_lvl, pre_st, leaves, origin, lvl, st, n_pre = (
        _closed_mesh(_localised_fold, _localised_fold_jacobian, min_img_sep=0.05)
    )
    origin_np = backend.to_numpy(origin)
    _, counts = np.unique(origin_np, return_counts=True)
    solo = np.flatnonzero(counts == 1)
    assert solo.size > 0
    v_np = backend.to_numpy(v)
    leaves_np = backend.to_numpy(leaves)
    for o in solo[:20]:
        row = np.flatnonzero(origin_np == o)[0]
        assert np.array_equal(leaves_np[row], v_np[o])


def test_closure_adds_no_new_vertices():
    """Bound taken before closure, so the assertion can actually fail.

    Reading ``cache_size`` after ``close`` returns would fold any vertices it
    inserted into the bound itself, making the check unfalsifiable -- it
    passed against a known-buggy ``close`` for exactly that reason.
    """
    cache, active, lat, v, pre_lvl, pre_st, leaves, origin, lvl, st, n_pre = (
        _closed_mesh(_localised_fold, _localised_fold_jacobian, min_img_sep=0.05)
    )
    leaves_np = backend.to_numpy(leaves)
    assert leaves_np.max() < n_pre
    assert set(np.unique(leaves_np)) <= set(range(n_pre))


def test_closure_re_emits_every_origin_vertex():
    """Every closure pattern re-emits all three of its origin's vertices.

    `build_adaptive_mesh` unions the closed leaves' slots with the pre-closure
    leaves' slots to decide which vertices to keep. That second term is
    redundant *if* this holds for all four patterns -- count 0 emits `v`
    itself, count 1 emits `(v_i, v_j, m_i)` and `(v_i, m_i, v_k)`, count 2 the
    corner `(v_c, m_b, m_a)` plus two triangles spanning `v_a` and `v_b`, and
    count 3 is the red split, whose children include all three. Dropping the
    term without this test would be an unchecked proof.
    """
    for fn, jac, kw in (
        (_localised_fold, _localised_fold_jacobian, dict(min_img_sep=0.02)),
        (_fold, _fold_jacobian, dict(min_img_sep=0.05)),
        (
            _gaussian_bump,
            _gaussian_bump_jacobian,
            dict(fov=4.0, init_res=8, min_img_sep=0.02),
        ),
    ):
        cache, active, store, counters, lat, calls, max_level = _refine_with(
            fn, jac, **kw
        )
        v, level, _, status = new.store_compact(store)
        order = new.canonical_order(lat, cache, v)
        v, level, status = v[order], level[order], status[order]
        leaves, origin, _, _ = new.close(lat, cache, active, v, level, status)
        v_np = backend.to_numpy(v)
        leaves_np = backend.to_numpy(leaves)
        origin_np = backend.to_numpy(origin)
        # Not merely "the sets agree": each origin's own three slots must appear
        # among the leaves that origin produced, which is the property the
        # union relies on.
        for row in range(v_np.shape[0]):
            emitted = set(leaves_np[origin_np == row].reshape(-1).tolist())
            assert set(v_np[row].tolist()) <= emitted, f"origin {row} lost a vertex"


def test_close_reaches_the_count_equals_3_branch_via_a_gaussian_bump():
    """``close``'s ``count == 3`` branch, which no other fixture reaches.

    That branch re-derives the red split with a raw ``concatenate`` + fancy-index
    gather rather than calling ``red_split``, so ``red_split``'s own tests give
    it zero coverage. A narrow Gaussian bump gives an ``init_res=8`` grid coarse
    enough that most triangles converge quickly while a few interior ones split
    deep enough to leave a fully-hanging (3-node) origin behind for ``close`` to
    red-split.

    Measured with `_refine_with` on this exact fixture, and reproduced twice
    for determinism: 2408 pre-closure leaves, 2752 post-closure leaves, pattern
    ``(248, 39, 6)`` -- the Jacobian gate leaves this fixture's closure as it
    was.
    """
    cache, active, lat, v, pre_lvl, pre_st, leaves, origin, lvl, st, n_pre = (
        _closed_mesh(
            _gaussian_bump,
            _gaussian_bump_jacobian,
            fov=4.0,
            init_res=8,
            min_img_sep=0.02,
        )
    )
    origin_np = backend.to_numpy(origin)
    group_sizes = np.bincount(origin_np, minlength=v.shape[0])
    pattern = (
        int((group_sizes == 2).sum()),
        int((group_sizes == 3).sum()),
        int((group_sizes == 4).sum()),
    )
    assert pattern[2] > 0, f"fixture must reach the count == 3 branch, got {pattern}"
    assert pattern == (248, 39, 6), f"measured n_closure_by_pattern={pattern}"

    tri = new.lattice_xy(lat, cache.ij[leaves])
    assert bool(
        backend.all(_signed_area(tri) > 0)
    ), "every leaf must be positively oriented"

    origin_area = backend.to_numpy(_signed_area(new.lattice_xy(lat, cache.ij[v])))
    summed = np.zeros_like(origin_area)
    np.add.at(summed, origin_np, backend.to_numpy(_signed_area(tri)))
    assert np.allclose(
        summed, origin_area, rtol=1e-12
    ), "origin-group areas must tile their origin exactly"
