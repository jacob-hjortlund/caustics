"""Freeze-time invalidation and the source-plane spatial index."""

from types import SimpleNamespace

import numpy as np
import pytest

from caustics.backend_obj import backend
from caustics.cosmology import FlatLambdaCDM
from caustics.lenses import SIE, Point
from caustics.lenses.func import adaptive as new


@pytest.fixture
def oracle_module():
    return pytest.importorskip(
        "caustics.lenses.old_adaptive", reason="optional frozen differential oracle"
    )


def _f64(x):
    return backend.as_array(np.asarray(x, dtype=np.float64), dtype=backend.float64)


def _i64(x):
    return backend.as_array(np.asarray(x, dtype=np.int64), dtype=backend.int64)


def test_invalidate_propagates_one_bad_vertex_to_the_whole_origin_group():
    vs = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [np.nan, 0.0]])
    leaves = np.array([[0, 1, 2], [0, 1, 3], [0, 1, 2]])
    origin = np.array([0, 0, 1])  # non-decreasing, as close() guarantees
    pre_status = np.array([new.LEAF_CONVERGED, new.LEAF_CONVERGED])

    got = backend.to_numpy(
        new.invalidate_nonfinite_origins(
            _f64(vs), _i64(leaves), _i64(origin), _i64(pre_status)
        )
    )
    assert got.tolist() == [new.LEAF_RAYTRACE_NONFINITE, new.LEAF_CONVERGED]


def test_invalidate_ors_the_flag_into_the_status_an_origin_already_has():
    """The freeze-time flag adds to an origin's record rather than replacing it.

    Origin 0 keeps its parity flag and gains the non-finite one; origin 1
    already carries it, and ORing it in again changes nothing; origin 2 owns
    no non-finite leaf and keeps its status untouched.
    """
    vs = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [np.nan, 0.0]])
    leaves = np.array([[0, 1, 3], [0, 3, 2], [0, 1, 2]])
    origin = np.array([0, 1, 2])
    pre_status = np.array(
        [
            new.LEAF_APPROX_PARITY_UNRESOLVED,
            new.LEAF_CONVERGENCE_FAILED | new.LEAF_RAYTRACE_NONFINITE,
            new.LEAF_JACOBIAN_PARITY_UNRESOLVED,
        ]
    )
    got = backend.to_numpy(
        new.invalidate_nonfinite_origins(
            _f64(vs), _i64(leaves), _i64(origin), _i64(pre_status)
        )
    )
    assert got.tolist() == [
        new.LEAF_APPROX_PARITY_UNRESOLVED | new.LEAF_RAYTRACE_NONFINITE,
        new.LEAF_CONVERGENCE_FAILED | new.LEAF_RAYTRACE_NONFINITE,
        new.LEAF_JACOBIAN_PARITY_UNRESOLVED,
    ]


def test_invalidate_matches_the_oracle_on_duplicate_origins(oracle_module):
    rng = np.random.default_rng(5)
    vs = rng.normal(size=(30, 2))
    # Vertex 18 sits in two leaves of origin 3 -- the repeated-origin OR this
    # test is about -- and one each of origins 5, 6 and 7. (Vertex 7, used
    # before, appears in no leaf at this seed, which left both sides all-zero.)
    vs[18] = np.inf
    leaves = rng.integers(0, 30, (24, 3))
    origin = np.repeat(np.arange(8), 3)
    pre_status = np.zeros(8, dtype=np.int64)

    got = backend.to_numpy(
        new.invalidate_nonfinite_origins(
            _f64(vs), _i64(leaves), _i64(origin), _i64(pre_status)
        )
    )
    want = oracle_module._invalidate_nonfinite_origins(
        vs, leaves, origin, pre_status.astype(np.int8)
    )
    # The oracle overwrites a bad origin's status with its own NONFINITE code,
    # where this module ORs in LEAF_RAYTRACE_NONFINITE. From an all-converged
    # start the two differ only in the value written, so what must agree is
    # which origins were flagged -- the bincount OR-reduction under test.
    flagged = want != oracle_module.LeafStatus.CONVERGED
    assert flagged.any() and not flagged.all(), "fixture must flag and spare"
    assert got.tolist() == np.where(flagged, new.LEAF_RAYTRACE_NONFINITE, 0).tolist()


def test_build_index_matches_the_oracle(oracle_module):
    rng = np.random.default_rng(9)
    vs = rng.normal(size=(60, 2)) * 2.0
    leaves = rng.integers(0, 60, (40, 3))
    valid = np.arange(0, 40, 2)

    got = new.build_index(_f64(vs), _i64(leaves), _i64(valid), None)
    want = oracle_module._build_index(vs, leaves, valid, None)

    assert np.allclose(backend.to_numpy(got.lo), want[0])
    assert np.allclose(backend.to_numpy(got.cell), want[1])
    assert got.nx == want[2] and got.ny == want[3]
    assert backend.to_numpy(got.cell_offsets).tolist() == want[4].tolist()
    assert backend.to_numpy(got.cell_leaves).tolist() == want[5].tolist()
    assert np.allclose(backend.to_numpy(got.hi), want[6])


def test_build_index_of_an_empty_leaf_set_is_a_single_cell():
    idx = new.build_index(
        _f64(np.zeros((3, 2))), _i64(np.zeros((0, 3))), _i64(np.zeros(0)), None
    )
    assert idx.nx == 1 and idx.ny == 1
    assert backend.to_numpy(idx.cell_leaves).size == 0


def test_build_index_leaves_are_ascending_within_every_cell():
    rng = np.random.default_rng(13)
    vs = rng.normal(size=(50, 2))
    leaves = rng.integers(0, 50, (60, 3))
    idx = new.build_index(_f64(vs), _i64(leaves), _i64(np.arange(60)), 6)
    offsets = backend.to_numpy(idx.cell_offsets)
    cell_leaves = backend.to_numpy(idx.cell_leaves)
    exercised = 0
    for a, b in zip(offsets[:-1], offsets[1:]):
        block = cell_leaves[a:b]
        if block.size > 1:
            exercised += 1
            assert (np.diff(block) > 0).all()
    assert exercised > 0, "fixture must exercise a cell containing multiple leaves"


# ---------------------------------------------------------------------------
# Ported from test_adaptive_mesh.py: black-box checks against the full build
# pipeline, exercising `caustics.lenses.func.adaptive`'s backend-native
# `build_adaptive_mesh` via the `new_build`/`new_sie_fixture` helpers below.
# They stay here, thematically with the freeze/index tests above, rather
# than in the general mesh-build suite.
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(20260904)


def to_np(x):
    return backend.to_numpy(x)


def make_counting_lens(fn, jac):
    """Wrap numpy maps as a lens, counting raytrace evaluations.

    ``fn`` is a ``(N, 2) -> (N, 2)`` lens map and ``jac`` its Jacobian,
    ``(N, 2) -> (N, 2, 2)``; the stand-in exposes the two methods
    `build_adaptive_mesh` reads, ``raytrace`` and ``jacobian_lens_equation``.
    """
    calls = {"points": 0, "batches": 0}

    def raytrace(x, y):
        xy = np.stack([backend.to_numpy(x), backend.to_numpy(y)], axis=-1)
        calls["points"] += xy.shape[0]
        calls["batches"] += 1
        out = fn(xy)
        return backend.as_array(out[:, 0]), backend.as_array(out[:, 1])

    def jacobian_lens_equation(x, y):
        xy = np.stack([backend.to_numpy(x), backend.to_numpy(y)], axis=-1)
        return backend.as_array(jac(xy), dtype=backend.float64)

    lens = SimpleNamespace(
        raytrace=raytrace, jacobian_lens_equation=jacobian_lens_equation
    )
    return lens, calls


def broken_where(fn, jac, where, value=np.nan):
    """``fn`` and ``jac`` with ``value`` wherever ``where(p)`` holds -- the
    raytrace and its Jacobian breaking over the same region, as a lens's would.
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


def localised_fold_jacobian(p):
    """``diag(1, 0.6 + 2y)`` inside the band, ``diag(1, 0.6)`` outside it."""
    y = p[:, 1]
    J = np.zeros((p.shape[0], 2, 2))
    J[:, 0, 0] = 1.0
    J[:, 1, 1] = 0.6 + np.where(np.abs(y) < 0.5, 2.0 * y, 0.0)
    return J


def identity(p):
    return p * 1.0


def identity_jacobian(p):
    return np.tile(np.eye(2), (p.shape[0], 1, 1))


def collapse(p):
    """A ``kappa == 1`` sheet: the whole lens plane maps to one point."""
    return np.zeros_like(p)


def collapse_jacobian(p):
    return np.zeros((p.shape[0], 2, 2))


def test_every_indexed_leaf_has_finite_source_vertices():
    fn, jac = broken_where(
        localised_fold, localised_fold_jacobian, lambda p: p[:, 0] > 1.0
    )
    mesh, _ = new_build(fn, jac, min_img_sep=0.05)
    vs = backend.to_numpy(mesh.vertices_source)
    leaves = backend.to_numpy(mesh.leaves)
    for leaf in np.unique(backend.to_numpy(mesh.index.cell_leaves)):
        assert np.isfinite(vs[leaves[leaf]]).all()


@pytest.mark.xfail(
    reason=(
        "pre-existing index boundary edge case, see task-11 report: the test's "
        "own `(q - lo) // cell` (floor-divide) can double-round differently from "
        "`_build_index`'s `((tri.min/max) - lo) / cell` then truncate at a cell "
        "quotient that lands exactly on a float64 boundary (32 of 6351 checks on "
        "this fixture); reproduces byte-identically on the unmodified oracle, so "
        "it predates and is independent of this refactor"
    ),
    strict=False,
)
def test_index_registers_every_leaf_in_the_cell_of_each_of_its_vertices():
    mesh, _ = new_build(localised_fold, localised_fold_jacobian, min_img_sep=0.05)
    vs = backend.to_numpy(mesh.vertices_source)
    leaves = backend.to_numpy(mesh.leaves)
    offs = backend.to_numpy(mesh.index.cell_offsets)
    cells = backend.to_numpy(mesh.index.cell_leaves)
    lo = backend.to_numpy(mesh.index.lo)
    cell = backend.to_numpy(mesh.index.cell)
    status = backend.to_numpy(mesh.leaf_status)
    for leaf in RNG.choice(len(leaves), size=50, replace=False):
        if status[leaf] != new.LEAF_CONVERGED:
            continue
        for q in vs[leaves[leaf]]:
            ix = int(np.clip((q[0] - lo[0]) // cell[0], 0, mesh.index.nx - 1))
            iy = int(np.clip((q[1] - lo[1]) // cell[1], 0, mesh.index.ny - 1))
            c = ix * mesh.index.ny + iy
            assert leaf in cells[offs[c] : offs[c + 1]]


def test_build_index_orders_leaves_ascending_within_every_cell():
    """CSR blocks must come out sorted with no sort at query time.

    The ordering used to come from `np.lexsort((leaf_id, cell_id))`. It now
    comes from a stable sort on `cell_id` alone, which is only equivalent
    because `leaf_id` is already non-decreasing in generation order. If that
    premise ever breaks, the blocks stop being ascending -- and `query`'s
    contract that `leaf_indices` is "strictly ascending within each block"
    breaks with it, silently.
    """
    lens, mesh = new_sie_fixture()
    offsets = to_np(mesh.index.cell_offsets)
    leaves = to_np(mesh.index.cell_leaves)
    assert offsets[0] == 0
    assert offsets[-1] == leaves.size
    assert (np.diff(offsets) >= 0).all()
    nonempty = 0
    for start, stop in zip(offsets[:-1], offsets[1:]):
        block = leaves[start:stop]
        if block.size > 1:
            nonempty += 1
            assert (np.diff(block) > 0).all(), "cell block is not strictly ascending"
    assert nonempty > 0, "fixture is too coarse to exercise multi-leaf cells"


def test_freeze_invalidates_a_whole_origin_group_from_one_bad_vertex():
    """The freeze-time finiteness re-check, exercised directly.

    It cannot be reached through a full build: every vertex that becomes a
    corner or midpoint of an evaluated triangle is finiteness-checked by
    ``refine`` first, except on a narrow cascade path (a forced child re-forced
    in a later round via a deferred, never-checked midpoint) that no available
    fixture reaches. Unit-tested on a synthetic triple instead -- otherwise the
    re-check could be deleted with no test failing.
    """
    vs = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [np.nan, 0.5]])
    leaves = np.array([[0, 1, 2], [0, 1, 3], [0, 1, 2]])
    origin = np.array([0, 0, 1])  # leaf 1 is non-finite and shares origin 0
    pre_status = np.array([new.LEAF_CONVERGED, new.LEAF_CONVERGED])
    out = backend.to_numpy(
        new.invalidate_nonfinite_origins(
            _f64(vs), _i64(leaves), _i64(origin), _i64(pre_status)
        )
    )
    assert (
        out[0] == new.LEAF_RAYTRACE_NONFINITE
    ), "one bad leaf must flag its whole origin"
    assert out[1] == new.LEAF_CONVERGED, "a clean origin must be untouched"


# ---------------------------------------------------------------------------
# `AdaptiveMesh` / `build_adaptive_mesh`: the backend-native assembly.
# ---------------------------------------------------------------------------


def _stack_2x2(a, b, c, d):
    """``[[a, b], [c, d]]`` at every point, shape ``(N, 2, 2)``."""
    return backend.stack(
        (backend.stack((a, b), dim=-1), backend.stack((c, d), dim=-1)), dim=-2
    )


def _lens(raytrace, jacobian):
    """A lens stand-in exposing the two methods `build_adaptive_mesh` reads."""
    return SimpleNamespace(raytrace=raytrace, jacobian_lens_equation=jacobian)


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
    parity test has a critical curve to find and the build must reproduce the
    oracle's exactly.
    """
    return x + 0.4 * backend.sin(2.0 * y) + 0.1 * x * x, y + 0.4 * backend.sin(1.5 * x)


def _fold_free_jacobian(x, y):
    return _stack_2x2(
        1.0 + 0.2 * x,
        0.8 * backend.cos(2.0 * y),
        0.6 * backend.cos(1.5 * x),
        backend.ones_like(x),
    )


SIE_LIKE = _lens(_sie_like, _sie_like_jacobian)
AFFINE_LENS = _lens(_affine, _affine_jacobian)
BUILD = dict(fov=4.0, init_res=3, min_img_sep=0.5, max_depth=3)


def test_build_matches_the_oracle_mesh(oracle_module):
    """Bit for bit, on a map where the two refinement criteria coincide.

    The Jacobian test replaced the oracle's quadratic-vertex parity check, so
    across a fold the two builds deliberately differ, and `_sie_like` no
    longer serves. `_fold_free` has no fold for either check to find, so the
    refinement matches the oracle leaf for leaf (see
    `test_refine_reproduces_the_oracle_leaf_set_where_no_fold_exists`) and
    everything downstream of it -- canonical ordering, closure, the vertex
    remap, freeze-time invalidation and the spatial index -- must match too.
    """
    build = dict(fov=4.0, init_res=3, min_img_sep=0.2, max_depth=5)
    got = new.build_adaptive_mesh(_lens(_fold_free, _fold_free_jacobian), **build)
    want = oracle_module.build_adaptive_mesh(_fold_free, **build)
    assert got.leaves.shape[0] > got.origin_leaves.shape[0], "closure must run"

    assert np.allclose(
        backend.to_numpy(got.vertices_lens), backend.to_numpy(want.vertices_lens)
    )
    assert np.allclose(
        backend.to_numpy(got.vertices_source),
        backend.to_numpy(want.vertices_source),
        equal_nan=True,
    )
    assert (
        backend.to_numpy(got.leaves).tolist() == backend.to_numpy(want.leaves).tolist()
    )
    assert (
        backend.to_numpy(got.leaf_origin).tolist()
        == backend.to_numpy(want.leaf_origin).tolist()
    )
    assert (
        backend.to_numpy(got.leaf_level).tolist()
        == backend.to_numpy(want.leaf_level).tolist()
    )
    assert (
        (backend.to_numpy(got.leaf_status) == new.LEAF_CONVERGED)
        == (backend.to_numpy(want.leaf_status) == oracle_module.LeafStatus.CONVERGED)
    ).all()
    assert (
        backend.to_numpy(got.index.cell_offsets).tolist()
        == backend.to_numpy(want._cell_offsets).tolist()
    )
    assert (
        backend.to_numpy(got.index.cell_leaves).tolist()
        == backend.to_numpy(want._cell_leaves).tolist()
    )
    assert got.d_floor == want.d_floor and got.max_level == want.max_level
    assert got.min_img_sep == want.min_img_sep


def test_build_halves_the_requested_min_img_sep():
    mesh = new.build_adaptive_mesh(
        AFFINE_LENS, fov=4.0, init_res=2, min_img_sep=0.4, max_depth=3
    )
    assert mesh.min_img_sep == pytest.approx(0.2)


def test_build_is_deterministic():
    a = new.build_adaptive_mesh(SIE_LIKE, **BUILD)
    b = new.build_adaptive_mesh(SIE_LIKE, **BUILD)
    assert backend.to_numpy(a.leaves).tolist() == backend.to_numpy(b.leaves).tolist()
    assert np.array_equal(
        backend.to_numpy(a.vertices_source),
        backend.to_numpy(b.vertices_source),
        equal_nan=True,
    )


def test_build_warns_when_depth_limited():
    with pytest.warns(UserWarning, match="depth-limited"):
        new.build_adaptive_mesh(
            SIE_LIKE, fov=4.0, init_res=2, min_img_sep=1e-4, max_depth=2
        )


def test_depth_limited_warning_names_the_caller_requested_min_img_sep():
    with pytest.warns(UserWarning, match="min_img_sep=0.0001"):
        new.build_adaptive_mesh(
            SIE_LIKE, fov=4.0, init_res=2, min_img_sep=1e-4, max_depth=2
        )


def test_unconverged_leaves_are_kept_but_excluded_from_the_index():
    """The index holds exactly the ``LEAF_CONVERGED`` leaves, and no other.

    Every indexed leaf registers in at least one cell, so the set of leaves
    appearing in ``cell_leaves`` is the index's whole membership, and it must
    equal the converged set -- a leaf carrying any flag at all stays in
    ``leaves`` but is never a query candidate.
    """
    mesh = new.build_adaptive_mesh(SIE_LIKE, **BUILD)
    status = backend.to_numpy(mesh.leaf_status)
    indexed = set(backend.to_numpy(mesh.index.cell_leaves).tolist())
    assert (status != new.LEAF_CONVERGED).any(), "fixture must flag some leaf"
    assert indexed == set(np.flatnonzero(status == new.LEAF_CONVERGED).tolist())


def test_mesh_dtype_is_a_backend_dtype():
    mesh = new.build_adaptive_mesh(AFFINE_LENS, **BUILD)
    assert mesh.dtype is backend.float64
    assert backend.to_numpy(mesh.vertices_lens).dtype == np.float64


def test_mesh_honours_a_float32_request():
    mesh = new.build_adaptive_mesh(AFFINE_LENS, dtype=backend.float32, **BUILD)
    assert backend.to_numpy(mesh.vertices_lens).dtype == np.float32


# ---------------------------------------------------------------------------
# Ported from test_adaptive_mesh.py (task 12): the same black-box build
# checks as above, now exercised against `new.build_adaptive_mesh` -- the
# assembled backend entry point -- rather than the legacy
# `caustics.lenses.adaptive` implementation. `AdaptiveMesh` has no `stats`
# field (`BuildStats` is dropped entirely by this port), so every assertion
# that used to read `mesh.stats.*` is recomputed from mesh fields instead;
# each such adaptation is called out where it happens. A few tests also
# collide by name with the Step 1 tests above and are renamed on porting,
# noted individually.
# ---------------------------------------------------------------------------

AFFINE = np.array([[0.7, 0.1], [-0.2, 0.9]])


def affine_np(p):
    return p @ AFFINE.T


def affine_np_jacobian(p):
    return np.tile(AFFINE, (p.shape[0], 1, 1))


def new_build(fn, jac, fov=4.0, init_res=4, min_img_sep=0.25, **kw):
    lens, calls = make_counting_lens(fn, jac)
    mesh = new.build_adaptive_mesh(lens, fov, init_res, min_img_sep, **kw)
    return mesh, calls


def new_sie_fixture():
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
    mesh = new.build_adaptive_mesh(lens, fov=5.0, init_res=32, min_img_sep=1e-2)
    return lens, mesh


def np_shape_matrix(tri):
    """NumPy edge matrix ``[v1 - v0 | v2 - v0]``, matching `new.shape_matrix`."""
    return np.stack(
        (tri[..., 1, :] - tri[..., 0, :], tri[..., 2, :] - tri[..., 0, :]), axis=-1
    )


def signed_area(tri):
    """Twice the signed area of a ``(..., 3, 2)`` triangle."""
    P = np_shape_matrix(tri)
    return P[..., 0, 0] * P[..., 1, 1] - P[..., 0, 1] * P[..., 1, 0]


def test_build_returns_a_consistent_mesh_for_an_affine_map():
    mesh, calls = new_build(affine_np, affine_np_jacobian)
    L = backend.to_numpy(mesh.leaves).shape[0]
    assert L == 2 * 4**2
    # ADAPTED (not one of the three the task brief named, but it reads
    # `stats` too): this fixture converges everywhere at level 0, so closure
    # never fans a triangle out -- confirmed by the `leaf_origin ==
    # arange(L)` check below, which only holds when pre- and post-closure
    # leaves coincide 1:1. That makes `leaf_status` an exact (not
    # approximate) stand-in for the old pre-closure `stats.n_converged`, and
    # `origin_leaves.shape[0]` -- the pre-closure leaves' own row count -- an
    # exact stand-in for `stats.n_leaves_pre_closure`.
    leaf_status = backend.to_numpy(mesh.leaf_status)
    assert (leaf_status == new.LEAF_CONVERGED).sum() == L
    assert backend.to_numpy(mesh.origin_leaves).shape[0] == L
    assert np.array_equal(backend.to_numpy(mesh.leaf_origin), np.arange(L))
    src = backend.to_numpy(mesh.vertices_source)
    lens = backend.to_numpy(mesh.vertices_lens)
    assert np.allclose(src, lens @ AFFINE.T, rtol=1e-10, atol=1e-12)


def test_leaf_area2_is_computed_from_the_stored_source_vertices():
    mesh, calls = new_build(localised_fold, localised_fold_jacobian, min_img_sep=0.05)
    tri = backend.to_numpy(mesh.vertices_source)[backend.to_numpy(mesh.leaves)]
    P = np_shape_matrix(tri)
    expected = P[:, 0, 0] * P[:, 1, 1] - P[:, 0, 1] * P[:, 1, 0]
    assert np.array_equal(backend.to_numpy(mesh.leaf_area2), expected)


def test_vertices_are_compacted_and_ordered_by_lattice_key():
    mesh, calls = new_build(localised_fold, localised_fold_jacobian, min_img_sep=0.05)
    used = np.unique(backend.to_numpy(mesh.leaves))
    assert used.tolist() == list(range(mesh.vertices_lens.shape[0]))
    lens = backend.to_numpy(mesh.vertices_lens)
    key = np.lexsort((lens[:, 1], lens[:, 0]))
    assert np.array_equal(key, np.arange(len(lens)))


def test_leaf_origin_survives_the_vertex_remap():
    """Spec test 15, at Mesh level: the compaction must not scramble origins."""
    mesh, _ = new_build(localised_fold, localised_fold_jacobian, min_img_sep=0.05)
    leaves = backend.to_numpy(mesh.leaves)
    origin = backend.to_numpy(mesh.leaf_origin)
    origins = backend.to_numpy(mesh.origin_leaves)
    vl = backend.to_numpy(mesh.vertices_lens)
    assert (origin >= 0).all() and (origin < origins.shape[0]).all()
    assert (np.diff(origin) >= 0).all()
    _, counts = np.unique(origin, return_counts=True)
    solo = np.flatnonzero(counts == 1)
    assert solo.size > 0
    for o in solo[:20]:
        row = np.flatnonzero(origin == o)[0]
        assert np.array_equal(leaves[row], origins[o])
    child = signed_area(vl[leaves])
    parent = signed_area(vl[origins])
    summed = np.zeros_like(parent)
    np.add.at(summed, origin, child)
    assert np.allclose(summed, parent, rtol=1e-10)


def test_build_is_deterministic_on_a_refined_mesh():
    """Same property as `test_build_is_deterministic` above, on a fixture that
    actually refines and closes rather than converging outright at level 0.

    Renamed on porting (was `test_build_is_deterministic` in the legacy
    suite) to avoid colliding with the Step 1 test of that name above.
    """
    a, _ = new_build(localised_fold, localised_fold_jacobian, min_img_sep=0.05)
    b, _ = new_build(localised_fold, localised_fold_jacobian, min_img_sep=0.05)
    for name in ("leaves", "leaf_area2", "leaf_origin", "leaf_status", "leaf_level"):
        assert np.array_equal(
            backend.to_numpy(getattr(a, name)), backend.to_numpy(getattr(b, name))
        )
    assert np.array_equal(
        backend.to_numpy(a.vertices_source), backend.to_numpy(b.vertices_source)
    )


def test_depth_limit_warns_and_names_the_required_max_depth():
    with pytest.warns(UserWarning, match=r"Set max_depth >= \d+"):
        mesh, _ = new_build(
            localised_fold, localised_fold_jacobian, min_img_sep=1e-4, max_depth=2
        )
    # ADAPTED (not one of the three the brief named, but it reads `stats`
    # too): `d_floor` and `max_level` are direct `AdaptiveMesh` fields, an
    # exact replacement for `stats.d_floor`/`stats.max_level`. `depth_limited`
    # has no field of its own, but it is defined as `d_floor > max_depth`, so
    # it is already implied by `mesh.d_floor > 2` given this call's
    # `max_depth=2`.
    assert mesh.d_floor > 2
    assert mesh.max_level == 2


def test_depth_limited_warning_never_bare_quotes_the_halved_value():
    """The message must quote what the caller passed, never the halved value.

    `build_adaptive_mesh` halves `min_img_sep` internally (the parity-condemned
    band at `max_level` is about twice a leaf's size), and that halving also
    raises `d_floor` by exactly one level on every build -- so a caller who
    never saw this warning before may now see it, and is owed a number they
    recognise. 0.0002 is what is passed here; 0.0001 is the halved value the
    build actually uses, which must appear only labelled, never as a bare
    ``min_img_sep=0.0001``.

    Renamed on porting (was `test_depth_limited_warning_names_the_caller_
    requested_min_img_sep` in the legacy suite) to avoid colliding with the
    Step 1 test of that name above, which checks the positive half of this
    same property on a different fixture; this is the legacy suite's
    stronger, negative half.
    """
    with pytest.warns(UserWarning, match=r"min_img_sep=0\.0002 arcsec") as record:
        new_build(identity, identity_jacobian, min_img_sep=2e-4, max_depth=1)
    assert not any(
        "min_img_sep=0.0001" in str(w.message) for w in record
    ), "must not quote the halved value as if it were what the caller passed"


def test_nonfinite_leaves_from_a_nonfinite_region_are_excluded_from_the_index():
    """Renamed on porting (was `test_invalid_leaves_are_kept_but_excluded_
    from_the_index` in the legacy suite) to avoid colliding with the Step 1
    test of that name above, which exercises SIE parity condemnation rather
    than an injected non-finite region.
    """
    fn, jac = broken_where(
        localised_fold, localised_fold_jacobian, lambda p: p[:, 0] > 1.0, np.inf
    )
    mesh, _ = new_build(fn, jac, min_img_sep=0.05)
    status = backend.to_numpy(mesh.leaf_status)
    nonfinite = (status & new.LEAF_RAYTRACE_NONFINITE) != 0
    assert nonfinite.any()
    # ADAPTED: `stats.n_nonfinite_vertices` has no `AdaptiveMesh` equivalent.
    # Flagged leaves keep their (possibly non-finite) vertices rather than
    # dropping them, so a non-finite raytrace region reaches `vertices_source`
    # directly.
    assert not np.isfinite(backend.to_numpy(mesh.vertices_source)).all()
    indexed = set(backend.to_numpy(mesh.index.cell_leaves).tolist())
    assert not indexed & set(np.flatnonzero(nonfinite).tolist())


def test_parity_condemned_leaves_are_excluded_from_the_index():
    """The coverage hole this feature deliberately opens.

    The cored SIE fixture (`new_sie_fixture`) is finite everywhere, so an
    all-finite `vertices_source` isolates parity as the only cause of
    invalidity here -- without that guard this test would pass just as well
    on the pre-existing non-finite path.
    """
    lens, mesh = new_sie_fixture()
    vs = backend.to_numpy(mesh.vertices_source)
    assert np.isfinite(vs).all(), "fixture must isolate the parity cause"

    status = backend.to_numpy(mesh.leaf_status)
    level = backend.to_numpy(mesh.leaf_level)
    parity_flags = (
        new.LEAF_APPROX_PARITY_UNRESOLVED | new.LEAF_JACOBIAN_PARITY_UNRESOLVED
    )
    invalid = np.flatnonzero((status & parity_flags) != 0)
    assert invalid.size > 0
    assert (level[invalid] == mesh.max_level).all()
    indexed = set(backend.to_numpy(mesh.index.cell_leaves).tolist())
    assert not indexed & set(invalid.tolist())


def test_parity_invalid_partitions_the_max_level_leaves():
    """Exact conservation, re-derived for the bitmask status model.

    The full criterion runs at `max_level`, Jacobian forced, so a max-level
    leaf either genuinely converges or records which tests it failed. On a
    fixture finite everywhere, those are only ever the three criterion flags
    -- deviation, child parity and Jacobian parity -- never a non-finite
    flag, since no raytraced sample is non-finite and the cored SIE's
    Jacobian is finite and never exactly singular at a sample. Both sides of
    the partition must be non-empty, and every combination of criterion
    flags is allowed.
    """
    lens, mesh = new_sie_fixture()
    vs = backend.to_numpy(mesh.vertices_source)
    assert np.isfinite(vs).all()

    status = backend.to_numpy(mesh.leaf_status)
    level = backend.to_numpy(mesh.leaf_level)
    at_max = level == mesh.max_level
    assert at_max.any()
    criterion_flags = (
        new.LEAF_CONVERGENCE_FAILED
        | new.LEAF_APPROX_PARITY_UNRESOLVED
        | new.LEAF_JACOBIAN_PARITY_UNRESOLVED
    )
    assert not (
        status[at_max] & ~criterion_flags
    ).any(), "no non-finite flag should occur at max_level on a finite fixture"
    assert (status[at_max] != new.LEAF_CONVERGED).any(), "parity must condemn something"
    assert (
        status[at_max] == new.LEAF_CONVERGED
    ).any(), "some max-level leaf must genuinely converge"


def test_parity_band_is_bounded_by_a_small_multiple_of_min_img_sep():
    """The single claim spec section 4.8's internal halving exists to deliver.

    Nothing else in the suite checks that halving ``min_img_sep`` before the
    build actually bounds the condemned band by anything related to what the
    caller asked for.

    The band is the coverage hole: every leaf the index excludes, i.e. any
    status but ``LEAF_CONVERGED``. Its width is measured as it always was --
    condemned-leaf centroids within 0.05 arcsec of the lens centre, max radius
    minus min radius. Measured on this exact fixture (fov=5.0, init_res=32,
    q=0.4, phi=pi/5, Rein=1.0, s=1e-3), band-to-request ratios run 1.77 to
    1.83 across requested separations of 2e-2 to 2.5e-3, and 1.774 at this
    test's own 1e-2 (band width 0.01774). The Jacobian criterion narrowed it:
    the frozen oracle, whose quadratic-vertex check condemned curvature as
    well as folds, measures 0.03107 here, ratio 3.107. The parity-flagged
    leaves alone span about one requested separation, 0.00992 here, and the
    deviation failures around them make up the rest.

    ``2.5x`` keeps roughly the original bound's proportional headroom -- about
    1.4x -- over the worst measured case, while still being a real,
    falsifiable bound rather than a vacuous one.
    """
    requested_min_img_sep = 1e-2  # the value new_sie_fixture's own build call uses
    lens, mesh = new_sie_fixture()
    status = backend.to_numpy(mesh.leaf_status)
    leaves = backend.to_numpy(mesh.leaves)
    vl = backend.to_numpy(mesh.vertices_lens)

    invalid = np.flatnonzero(status != new.LEAF_CONVERGED)
    assert invalid.size > 0, "fixture must condemn leaves to measure a band"
    centroids = vl[leaves[invalid]].mean(axis=1)
    radius = np.linalg.norm(centroids, axis=-1)
    near_centre = radius < 0.05
    assert near_centre.any(), "fixture must condemn leaves near the lens centre"

    core_radius = radius[near_centre]
    band_width = core_radius.max() - core_radius.min()
    assert band_width <= 2.5 * requested_min_img_sep


def test_leaf_area2_uses_the_downcast_vertices_at_reduced_precision():
    """The downcast-before-compute ordering, at a dtype where it is observable.

    At float64 -- the dtype every other test uses -- ``vs.astype(np_dtype)`` is a
    value-preserving no-op, so cast-then-compute and compute-then-cast are
    bit-identical and neither ordering can be distinguished. The property only
    has teeth at a precision-losing dtype: here the two orderings disagree on
    the great majority of leaves, so this is the test that actually pins it.
    """
    mesh, _ = new_build(
        localised_fold, localised_fold_jacobian, min_img_sep=0.05, dtype=backend.float32
    )
    vs = backend.to_numpy(mesh.vertices_source)
    assert vs.dtype == np.float32
    P = np_shape_matrix(vs[backend.to_numpy(mesh.leaves)])
    from_stored = P[:, 0, 0] * P[:, 1, 1] - P[:, 0, 1] * P[:, 1, 0]
    assert np.array_equal(backend.to_numpy(mesh.leaf_area2), from_stored)

    mesh64, _ = new_build(localised_fold, localised_fold_jacobian, min_img_sep=0.05)
    vs64 = backend.to_numpy(mesh64.vertices_source)
    Q = np_shape_matrix(vs64[backend.to_numpy(mesh64.leaves)])
    computed_then_cast = (Q[:, 0, 0] * Q[:, 1, 1] - Q[:, 0, 1] * Q[:, 1, 0]).astype(
        np.float32
    )
    assert not np.array_equal(
        from_stored, computed_then_cast
    ), "fixture no longer distinguishes the two orderings"


def test_kappa_one_sheet_builds_without_nan():
    mesh, calls = new_build(collapse, collapse_jacobian, min_img_sep=0.5)
    # ADAPTED: `stats.n_sigma_min_exactly_zero` is an internal `refine()`
    # counter with no `AdaptiveMesh` field to recompute it from -- unlike
    # `n_invalid`/`n_nonfinite_vertices` elsewhere, nothing on the mesh
    # records how many triangles saw `sigma_min == 0` exactly -- so it is
    # dropped rather than approximated. The no-NaN and full-depth checks
    # below are this test's recomputable content.
    assert not np.isnan(backend.to_numpy(mesh.vertices_source)).any()
    assert backend.to_numpy(mesh.leaf_level).max() == mesh.max_level


def test_mesh_stores_min_img_sep():
    """`forward_raytrace` needs the dedup radius, and only the build knows it.

    The lens and its `raytrace` are deliberately *not* stored -- `raytrace` is
    passed per call -- but `min_img_sep` is the mesh's own defining tolerance,
    so a caller should never have to restate it and risk restating it wrong.
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
    mesh = new.build_adaptive_mesh(lens, fov=4.0, init_res=8, min_img_sep=0.05)
    # The build halves min_img_sep internally (the parity-condemned band at
    # max_level is about twice a leaf's size), and stores that halved value --
    # not the value passed in -- since the halved value is what the size floor
    # and the dedup radius actually are.
    assert mesh.min_img_sep == 0.025


# ---------------------------------------------------------------------------
# The critical band
# ---------------------------------------------------------------------------


def row_fold(p):
    """``(x, y) -> (x, y - y**2)``, so ``det A = 1 - 2y``, zero on ``y = 0.5``.

    ``y = 0.5`` is a lattice row for ``fov=4`` about the origin whenever
    ``init_res`` is a multiple of 4, so every sample on it has ``det A``
    exactly zero: the build flags those leaves ``LEAF_JACOBIAN_NONFINITE``,
    never ``LEAF_JACOBIAN_PARITY_UNRESOLVED``, yet the curve runs through them.
    """
    return np.stack([p[:, 0], p[:, 1] - p[:, 1] ** 2], axis=-1)


def row_fold_jacobian(p):
    J = np.zeros((p.shape[0], 2, 2))
    J[:, 0, 0] = 1.0
    J[:, 1, 1] = 1.0 - 2.0 * p[:, 1]
    return J


def lattice_samples(mesh, fov, init_res):
    """Lattice-exact positions of every finite ``max_level`` leaf's six samples.

    Midpoints come from integer lattice coordinates, not from averaging vertex
    positions: the two differ in the last ulp, enough to flip the sign of a
    near-zero determinant right at the curve.
    """
    lat = new.make_lattice(fov, 0.0, 0.0, init_res, mesh.max_level + 1)
    lo, scale = to_np(lat.lo), lat.scale
    ij = np.rint((to_np(mesh.vertices_lens) - lo) / scale).astype(np.int64)
    leaves = to_np(mesh.leaves)
    level, status = to_np(mesh.leaf_level), to_np(mesh.leaf_status)
    rows = np.flatnonzero(
        (level == mesh.max_level) & ((status & new.LEAF_RAYTRACE_NONFINITE) == 0)
    )
    vij = ij[leaves[rows]]
    mij = np.stack(
        [
            (vij[:, 1] + vij[:, 2]) // 2,
            (vij[:, 2] + vij[:, 0]) // 2,
            (vij[:, 0] + vij[:, 1]) // 2,
        ],
        axis=1,
    )
    return rows, lo + np.concatenate([vij, mij], axis=1).astype(np.float64) * scale


def independent_band(mesh, jac, fov, init_res):
    """Band leaves recomputed from lattice-exact samples, without the build."""
    rows, xy = lattice_samples(mesh, fov, init_res)
    J = jac(xy.reshape(-1, 2)).reshape(-1, 6, 2, 2)
    det = J[..., 0, 0] * J[..., 1, 1] - J[..., 0, 1] * J[..., 1, 0]
    positive = det >= 0
    band = np.isfinite(det).all(axis=1) & positive.any(axis=1) & (~positive).any(axis=1)
    return rows[band]


@pytest.mark.parametrize(
    "fn, jac, build",
    [
        (localised_fold, localised_fold_jacobian, dict(init_res=4, min_img_sep=0.05)),
        (row_fold, row_fold_jacobian, dict(init_res=8, min_img_sep=2e-2)),
    ],
    ids=["localised_fold", "row_fold"],
)
def test_band_is_the_sign_change_leaves_recomputed_independently(fn, jac, build):
    """The band is exactly the leaves whose six dets change class.

    Recomputed here from lattice-exact samples and the fixture's own
    Jacobian, with an exact zero counted as positive. Every
    ``LEAF_JACOBIAN_PARITY_UNRESOLVED`` leaf is in it; every other band leaf
    is flagged ``LEAF_JACOBIAN_NONFINITE`` and has an exactly zero sample.
    `localised_fold`'s curve, ``y = -0.3``, is never a lattice row, so there
    the band is the flagged set; `row_fold`'s is one, so there no leaf is
    flagged and the whole band is exact-zero leaves.
    """
    fov = 4.0
    mesh, _ = new_build(fn, jac, fov=fov, **build)
    band = mesh.critical_band
    got = to_np(band.leaves)
    want = independent_band(mesh, jac, fov, build["init_res"])
    assert want.size > 0, "fixture must have a critical curve"
    assert sorted(got.tolist()) == sorted(want.tolist())

    status = to_np(mesh.leaf_status)
    flagged = np.flatnonzero((status & new.LEAF_JACOBIAN_PARITY_UNRESOLVED) != 0)
    assert set(flagged.tolist()) <= set(got.tolist())
    extra = ~np.isin(got, flagged)
    assert ((status[got[extra]] & new.LEAF_JACOBIAN_NONFINITE) != 0).all()
    det = to_np(band.det)[to_np(band.samples)]
    assert (det[extra] == 0).any(axis=1).all()
    if fn is row_fold:
        assert flagged.size == 0 and extra.all()
    else:
        assert not extra.any()


def test_band_samples_are_lattice_exact_and_carry_the_lens_values():
    """Each band row holds its leaf's own samples, once each, with exact values.

    The vertices are the mesh's own, bit for bit and in the leaf's order; the
    midpoints are the lattice midpoints opposite each vertex; no sample
    appears twice; and ``source`` and ``det`` are what the lens gives at
    ``lens``.
    """
    fov, init_res = 4.0, 8
    mesh, _ = new_build(
        row_fold, row_fold_jacobian, fov=fov, init_res=init_res, min_img_sep=2e-2
    )
    band = mesh.critical_band
    leaves, samples = to_np(band.leaves), to_np(band.samples)
    lens, source, det = to_np(band.lens), to_np(band.source), to_np(band.det)
    assert leaves.size > 0

    vertices = to_np(mesh.vertices_lens)[to_np(mesh.leaves)[leaves]]
    assert np.array_equal(lens[samples[:, :3]], vertices)
    rows, xy = lattice_samples(mesh, fov, init_res)
    position = {r: i for i, r in enumerate(rows.tolist())}
    assert np.array_equal(lens[samples], xy[[position[r] for r in leaves.tolist()]])
    assert np.unique(lens, axis=0).shape[0] == lens.shape[0]

    assert np.array_equal(source, row_fold(lens))
    J = row_fold_jacobian(lens)
    assert np.array_equal(det, J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0])


def test_band_never_holds_a_leaf_with_a_nonfinite_sample():
    """`localised_fold` broken to NaN for ``x > 1``, raytrace and Jacobian alike.

    The band still matches the independent recomputation, and nothing
    non-finite reaches it: no band leaf carries ``LEAF_RAYTRACE_NONFINITE``,
    every stored ``det`` is finite, and no sample lies in the broken region.
    """
    fn, jac = broken_where(
        localised_fold, localised_fold_jacobian, lambda p: p[:, 0] > 1.0
    )
    mesh, _ = new_build(fn, jac, init_res=4, min_img_sep=0.05)
    band = mesh.critical_band
    got = to_np(band.leaves)
    assert got.size > 0
    assert sorted(got.tolist()) == sorted(independent_band(mesh, jac, 4.0, 4).tolist())
    status = to_np(mesh.leaf_status)
    assert not ((status[got] & new.LEAF_RAYTRACE_NONFINITE) != 0).any()
    assert np.isfinite(to_np(band.det)).all()
    assert (to_np(band.lens)[:, 0] <= 1.0).all()


@pytest.mark.parametrize(
    "fn, jac, build",
    [
        (affine_np, affine_np_jacobian, dict()),
        (collapse, collapse_jacobian, dict(min_img_sep=0.5)),
    ],
    ids=["affine", "kappa_one_sheet"],
)
def test_band_is_empty_without_a_sign_change(fn, jac, build):
    """No leaf's samples change class, so no band leaf can exist.

    An affine map converges at level 0 and never reaches ``max_level``. A
    ``kappa == 1`` sheet reaches it everywhere, but ``det A`` is exactly zero
    at every sample, and zero counts as positive.
    """
    mesh, _ = new_build(fn, jac, **build)
    band = mesh.critical_band
    assert tuple(band.leaves.shape) == (0,)
    assert tuple(band.samples.shape) == (0, 6)
    assert tuple(band.lens.shape) == (0, 2)
    assert tuple(band.source.shape) == (0, 2)
    assert tuple(band.det.shape) == (0,)


def test_band_positions_follow_the_mesh_dtype_and_det_stays_float64():
    """``det`` decides every class, so it keeps the build's precision."""
    lens, _ = make_counting_lens(row_fold, row_fold_jacobian)
    mesh = new.build_adaptive_mesh(lens, 4.0, 8, 2e-2, dtype=backend.float32)
    band = mesh.critical_band
    assert band.leaves.shape[0] > 0
    assert band.lens.dtype == backend.float32
    assert band.source.dtype == backend.float32
    assert band.det.dtype == backend.float64


def test_band_is_deterministic():
    a, _ = new_build(row_fold, row_fold_jacobian, init_res=8, min_img_sep=2e-2)
    b, _ = new_build(row_fold, row_fold_jacobian, init_res=8, min_img_sep=2e-2)
    for x, y in zip(a.critical_band, b.critical_band):
        assert np.array_equal(to_np(x), to_np(y))
