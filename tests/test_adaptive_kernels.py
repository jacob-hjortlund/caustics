"""The backend kernels: agreement with the frozen numpy oracle, and the
Jacobian parity test the oracle never had."""

import numpy as np
import pytest

from caustics.backend_obj import backend
from caustics.lenses.func import adaptive as new

RNG = np.random.default_rng(20260918)


@pytest.fixture
def oracle_module():
    return pytest.importorskip(
        "caustics.lenses.func.old_adaptive",
        reason="optional frozen differential oracle",
    )


def _arr(x):
    return backend.as_array(np.asarray(x, dtype=np.float64), dtype=backend.float64)


def _samples(n=64):
    beta_v = RNG.normal(size=(n, 3, 2))
    beta_m = 0.5 * (beta_v[:, [1, 2, 0]] + beta_v[:, [2, 0, 1]])
    beta_m = beta_m + 0.05 * RNG.normal(size=(n, 3, 2))
    return beta_v, beta_m


def signed_area(tri):
    """Twice the signed area of a (..., 3, 2) triangle.

    Written directly from the edge vectors rather than via ``new.shape_matrix``,
    so the fixture helpers in this file stay plain numpy; the formula is the
    bit-exact same cross product ``shape_matrix`` computes.
    """
    e1 = tri[..., 1, :] - tri[..., 0, :]
    e2 = tri[..., 2, :] - tri[..., 0, :]
    return e1[..., 0] * e2[..., 1] - e1[..., 1] * e2[..., 0]


def red_split(tri):
    """Split a (..., 3, 2) triangle into (..., 4, 3, 2) children, canonical order."""
    t1, t2, t3 = tri[..., 0, :], tri[..., 1, :], tri[..., 2, :]
    six = np.stack([t1, t2, t3, (t2 + t3) / 2, (t3 + t1) / 2, (t1 + t2) / 2], axis=-2)
    idx = np.asarray(new.CHILD_VERTEX_INDICES)
    return six[..., idx, :].reshape(*tri.shape[:-2], 4, 3, 2)


def _six_points(tri):
    t1, t2, t3 = tri[:, 0, :], tri[:, 1, :], tri[:, 2, :]
    mid = np.stack([(t2 + t3) / 2, (t3 + t1) / 2, (t1 + t2) / 2], axis=1)
    return tri, mid


def _constant_jacobian(J):
    """A ``jacobian_fn`` returning the same ``2 x 2`` matrix at every point."""
    J = _arr(J)

    def jacobian_fn(x, y):
        return J + backend.zeros((x.shape[0], 2, 2), dtype=backend.float64)

    return jacobian_fn


def _fixed_jacobians(J):
    """A ``jacobian_fn`` returning the ``(6 * N, 2, 2)`` array ``J`` as given."""

    def jacobian_fn(x, y):
        return _arr(J)

    return jacobian_fn


def _fold_jacobian(x, y):
    """Jacobian of the fold ``(x, y) -> (x, y**2)``: ``diag(1, 2y)``."""
    one, zero = backend.ones_like(x), backend.zeros_like(x)
    return backend.stack(
        (backend.stack((one, zero), dim=-1), backend.stack((zero, 2 * y), dim=-1)),
        dim=-2,
    )


def _recording(jacobian_fn):
    """Wrap ``jacobian_fn``, recording the ``(K, 2)`` points of every call."""
    calls = []

    def wrapped(x, y):
        calls.append(np.stack([backend.to_numpy(x), backend.to_numpy(y)], axis=-1))
        return jacobian_fn(x, y)

    return wrapped, calls


def _exploding_jacobian(x, y):
    raise AssertionError("jacobian_fn must not be called here")


def _flagged(status, flag):
    return (np.asarray(status) & flag) != 0


def test_shape_matrix_matches_the_oracle(oracle_module):
    tri = RNG.normal(size=(32, 3, 2))
    assert np.allclose(
        backend.to_numpy(new.shape_matrix(_arr(tri))), oracle_module.shape_matrix(tri)
    )


def test_affine_from_triangles_matches_the_oracle(oracle_module):
    p, q = RNG.normal(size=(16, 3, 2)), RNG.normal(size=(16, 3, 2))
    assert np.allclose(
        backend.to_numpy(new.affine_from_triangles(_arr(p), _arr(q))),
        oracle_module.affine_from_triangles(p, q),
    )


def test_child_matrix_tables_match_the_oracle(oracle_module):
    got = new.child_matrix_tables()
    want = oracle_module.child_matrix_tables()
    for g, w in zip(got, want):
        assert np.allclose(backend.to_numpy(g), w)


def test_sigma_min_matches_the_oracle(oracle_module):
    A = RNG.normal(size=(64, 2, 2))
    assert np.allclose(
        backend.to_numpy(new.sigma_min_2x2(_arr(A))), oracle_module.sigma_min_2x2(A)
    )


def test_sigma_min_of_zero_matrix_is_exactly_zero():
    A = np.zeros((4, 2, 2))
    got = backend.to_numpy(new.sigma_min_2x2(_arr(A)))
    assert (got == 0.0).all()
    assert not np.isnan(got).any()


def test_sigma_min_propagates_nan():
    A = np.zeros((1, 2, 2))
    A[0, 0, 0] = np.nan
    assert np.isnan(backend.to_numpy(new.sigma_min_2x2(_arr(A)))).all()


def test_converged_from_deviation_fails_closed_on_nan():
    r = _arr(np.zeros((1, 3)))
    s = _arr(np.array([np.nan]))
    assert not bool(backend.to_numpy(new.converged_from_deviation(r, s, 1.0))[0])


def test_midpoint_deviation_matches_the_oracle(oracle_module):
    beta_v, beta_m = _samples()
    assert np.allclose(
        backend.to_numpy(new.midpoint_deviation(_arr(beta_v), _arr(beta_m))),
        oracle_module.midpoint_deviation(beta_v, beta_m),
    )


def test_child_shape_matrices_match_the_oracle(oracle_module):
    beta_v, beta_m = _samples()
    assert np.allclose(
        backend.to_numpy(new.child_shape_matrices(_arr(beta_v), _arr(beta_m))),
        oracle_module.child_shape_matrices(beta_v, beta_m),
    )


def test_parity_from_children_matches_the_oracle(oracle_module):
    beta_v, beta_m = _samples()
    Q = oracle_module.child_shape_matrices(beta_v, beta_m)
    assert (
        backend.to_numpy(new.parity_from_children(_arr(Q)))
        == oracle_module.parity_from_children(Q)
    ).all()


def test_parity_from_children_fails_closed_on_a_nonfinite_child():
    """A NaN determinant stays NaN, so sign constancy fails closed.

    Which is why a triangle with a non-finite sample carries
    ``LEAF_APPROX_PARITY_UNRESOLVED`` alongside ``LEAF_RAYTRACE_NONFINITE``
    when `evaluate_criterion` sees one; see
    `test_criterion_reports_nonfinite_as_split`. See spec section 3.3.
    """
    Q = np.tile(np.eye(2), (1, 4, 1, 1)).reshape(1, 4, 2, 2)
    Q[0, 2, 0, 0] = np.nan
    assert not bool(backend.to_numpy(new.parity_from_children(_arr(Q)))[0])


def test_quadratic_vertex_parity_matches_the_oracle(oracle_module):
    beta_v, beta_m = _samples()
    assert (
        backend.to_numpy(new.quadratic_vertex_parity_ok(_arr(beta_v), _arr(beta_m)))
        == oracle_module.quadratic_vertex_parity_ok(beta_v, beta_m)
    ).all()


@pytest.mark.parametrize("level", [0, 2, 5])
def test_evaluate_criterion_matches_the_oracle(oracle_module, level):
    """The criterion's approximate half is still exactly the oracle's.

    The oracle's own ``evaluate_criterion`` also ran
    ``quadratic_vertex_parity_ok``, which the Jacobian test has replaced, so
    its ``keep`` and ``parity_ok`` are no longer a reference. Its parts still
    are. With a Jacobian that passes everywhere, ``s`` must be the oracle's,
    the child-parity flag exactly the oracle's ``parity_from_children``
    failures, the deviation flag exactly its ``converged_from_deviation``
    failures, and ``keep`` exactly where both pass.
    """
    beta_v, beta_m = _samples()
    classes = RNG.integers(0, 6, beta_v.shape[0])
    _, _, compose, pinv0, _ = oracle_module.child_matrix_tables()
    _, _, want_s = oracle_module.evaluate_criterion(
        beta_v, beta_m, classes, level, 0.5, 0.01, pinv0, compose
    )
    want_child = oracle_module.parity_from_children(
        oracle_module.child_shape_matrices(beta_v, beta_m)
    )
    want_deviation = oracle_module.converged_from_deviation(
        oracle_module.midpoint_deviation(beta_v, beta_m), want_s, 0.01
    )
    assert want_child.any() and not want_child.all(), "both parity arms must occur"

    keep, parity_ok, s, status = (
        backend.to_numpy(x)
        for x in new.evaluate_criterion(
            _constant_jacobian(np.eye(2)),
            _arr(np.zeros_like(beta_v)),
            _arr(np.zeros_like(beta_m)),
            _arr(beta_v),
            _arr(beta_m),
            backend.as_array(classes, dtype=backend.int64),
            level,
            0.5,
            0.01,
            _arr(pinv0),
            backend.as_array(compose, dtype=backend.int64),
        )
    )
    assert np.allclose(s, want_s, equal_nan=True)
    assert (_flagged(status, new.LEAF_APPROX_PARITY_UNRESOLVED) == ~want_child).all()
    assert (_flagged(status, new.LEAF_CONVERGENCE_FAILED) == ~want_deviation).all()
    assert (keep == (want_child & want_deviation)).all()
    assert (parity_ok == want_child).all()
    # Every sample is finite and the Jacobian passes, so no other flag can occur.
    expected_flags = new.LEAF_APPROX_PARITY_UNRESOLVED | new.LEAF_CONVERGENCE_FAILED
    assert not (status & ~expected_flags).any()


def test_group_tables_close_and_have_order_six():
    M, G, COMPOSE, PINV0, ROOT_CLASS = new.child_matrix_tables()
    assert M.dtype == backend.int64
    assert G.dtype == backend.int64
    assert COMPOSE.dtype == backend.int64
    assert PINV0.dtype == backend.float64
    assert ROOT_CLASS.dtype == backend.int64
    M, G, COMPOSE, ROOT_CLASS = (
        backend.to_numpy(M),
        backend.to_numpy(G),
        backend.to_numpy(COMPOSE),
        backend.to_numpy(ROOT_CLASS),
    )
    assert M.shape == (4, 2, 2) and G.shape == (6, 2, 2)
    dets = M[:, 0, 0] * M[:, 1, 1] - M[:, 0, 1] * M[:, 1, 0]
    assert (dets == 1).all(), "every child matrix must have det +1"
    # G is closed under right multiplication by every M, and COMPOSE indexes it
    for c in range(6):
        for k in range(4):
            assert (G[c] @ M[k] == G[COMPOSE[c, k]]).all()
    # exactly six distinct elements
    assert len({G[c].tobytes() for c in range(6)}) == 6
    # both root shapes lie in one orbit
    R0 = backend.to_numpy(new.shape_matrix(_arr(new.ROOT_SHAPES[0])))
    R1 = backend.to_numpy(new.shape_matrix(_arr(new.ROOT_SHAPES[1])))
    assert np.allclose(R0 @ G[ROOT_CLASS[1]], R1)


def test_root_shapes_are_positively_oriented():
    for shape in new.ROOT_SHAPES:
        assert signed_area(np.asarray(shape, dtype=np.float64)) > 0


@pytest.mark.parametrize("flip", [False, True])
def test_children_preserve_orientation_and_quarter_the_area(flip):
    tri = RNG.normal(size=(64, 3, 2))
    if flip:
        tri = tri[:, ::-1, :]
    parent = signed_area(tri)
    kids = signed_area(red_split(tri))
    assert np.allclose(kids, parent[:, None] / 4, rtol=0, atol=1e-12)


def test_pinv_table_scales_by_two_to_the_level():
    M, G, COMPOSE, PINV0, ROOT_CLASS = new.child_matrix_tables()
    COMPOSE, PINV0, ROOT_CLASS = (
        backend.to_numpy(COMPOSE),
        backend.to_numpy(PINV0),
        backend.to_numpy(ROOT_CLASS),
    )
    h0 = 0.05
    for shape_idx in (0, 1):
        root = np.asarray(new.ROOT_SHAPES[shape_idx], dtype=np.float64) * h0
        cls = ROOT_CLASS[shape_idx]
        tri, level = root, 0
        for _ in range(5):
            kids = red_split(tri)
            for k in range(4):
                direct = backend.to_numpy(
                    backend.linalg.inv(new.shape_matrix(_arr(kids[k])))
                )
                table = (2.0 ** (level + 1) / h0) * PINV0[COMPOSE[cls, k]]
                assert np.allclose(direct, table, rtol=1e-12, atol=1e-12)
            tri, cls, level = kids[0], COMPOSE[cls, 0], level + 1


def test_affine_from_table_matches_explicit_inversion():
    M, G, COMPOSE, PINV0, ROOT_CLASS = new.child_matrix_tables()
    COMPOSE, PINV0, ROOT_CLASS = (
        backend.to_numpy(COMPOSE),
        backend.to_numpy(PINV0),
        backend.to_numpy(ROOT_CLASS),
    )
    h0 = 0.05
    tri = np.asarray(new.ROOT_SHAPES[1], dtype=np.float64) * h0
    cls, level = ROOT_CLASS[1], 0
    kids = red_split(tri)
    lens_map = RNG.normal(size=(2, 2))
    for k in range(4):
        q = kids[k] @ lens_map.T
        explicit = backend.to_numpy(new.affine_from_triangles(_arr(kids[k]), _arr(q)))
        Q = backend.to_numpy(new.shape_matrix(_arr(q)))
        table = Q @ ((2.0 ** (level + 1) / h0) * PINV0[COMPOSE[cls, k]])
        assert np.allclose(explicit, table, rtol=1e-10, atol=1e-12)
        assert np.allclose(explicit, lens_map, rtol=1e-10, atol=1e-12)


def test_sigma_min_matches_svd():
    A = RNG.normal(size=(2000, 2, 2))
    expected = np.linalg.svd(A, compute_uv=False)[:, -1]
    assert np.allclose(
        backend.to_numpy(new.sigma_min_2x2(_arr(A))), expected, rtol=1e-9, atol=1e-12
    )


def test_sigma_min_near_and_exactly_singular():
    a = RNG.normal(size=(500, 2))
    perp = np.stack([-a[:, 1], a[:, 0]], axis=1)
    # Rows offset perpendicularly, so det == eps * |a|^2 is genuinely small.
    # Scaling one row instead would give a rank-1 matrix with det exactly zero,
    # which tests the degenerate branch, not the near-degenerate one.
    for eps in (1e-4, 1e-7):
        A = np.stack([a, a + eps * perp], axis=1)
        got = backend.to_numpy(new.sigma_min_2x2(_arr(A)))
        expected = np.linalg.svd(A, compute_uv=False)[:, -1]
        assert np.isfinite(got).all()
        assert np.allclose(got, expected, rtol=1e-5, atol=0.0)
    exact = np.stack([a, 2.0 * a], axis=1)  # rank 1: det is exactly 0 in IEEE
    assert (backend.to_numpy(new.sigma_min_2x2(_arr(exact))) == 0.0).all()


def test_sigma_min_handles_conformal_ulp_negativity():
    """F - 2D is exactly zero in R but goes negative by an ulp in float.

    Without the clip this returns NaN for a near-circularly-symmetric lens core,
    which fail-closed then refines to max_level. See spec section 2.2.
    """
    ab = RNG.normal(size=(200000, 2)).astype(np.float32)
    A = np.empty((ab.shape[0], 2, 2), dtype=np.float32)
    A[:, 0, 0] = ab[:, 0]
    A[:, 0, 1] = ab[:, 1]
    A[:, 1, 0] = -ab[:, 1]
    A[:, 1, 1] = ab[:, 0]
    F = (A.astype(np.float32) ** 2).sum(axis=(-2, -1))
    D = np.abs(A[:, 0, 0] * A[:, 1, 1] - A[:, 0, 1] * A[:, 1, 0])
    assert (F - 2 * D < 0).any(), "test fixture no longer exercises the guard"
    got = backend.to_numpy(new.sigma_min_2x2(_arr(A.astype(np.float64))))
    assert np.isfinite(got).all()


def test_sigma_min_propagates_nan_input():
    A = np.full((1, 2, 2), np.nan)
    assert np.isnan(backend.to_numpy(new.sigma_min_2x2(_arr(A)))).all()


def test_converged_from_deviation_fails_closed():
    r = np.zeros((3, 3))
    assert backend.to_numpy(
        new.converged_from_deviation(_arr(r), _arr([1.0, 1.0, 1.0]), 1.0)
    ).all()
    # NaN must take the split branch, not the converged branch
    assert not backend.to_numpy(
        new.converged_from_deviation(_arr(r), _arr(np.full(3, np.nan)), 1.0)
    ).any()
    # s == 0 with r == 0 gives 0 < 0, False, split -- the conservative direction
    assert not backend.to_numpy(
        new.converged_from_deviation(_arr(r), _arr(np.zeros(3)), 1.0)
    ).any()


def test_criterion_converges_on_an_affine_map():
    M, G, COMPOSE, PINV0, ROOT_CLASS = new.child_matrix_tables()
    h0 = 0.05
    tri = np.stack(
        [np.asarray(new.ROOT_SHAPES[s], dtype=np.float64) * h0 for s in (0, 1)]
    )
    classes = backend.copy(ROOT_CLASS)
    lens_map = np.array([[0.7, 0.1], [-0.2, 0.9]])
    v, m = _six_points(tri)
    keep, parity_ok, s, status = (
        backend.to_numpy(x)
        for x in new.evaluate_criterion(
            _constant_jacobian(lens_map),
            _arr(v),
            _arr(m),
            _arr(v @ lens_map.T),
            _arr(m @ lens_map.T),
            classes,
            0,
            h0,
            1e-3,
            PINV0,
            COMPOSE,
        )
    )
    assert keep.all() and parity_ok.all()
    assert (status == new.LEAF_CONVERGED).all()
    assert np.allclose(s, np.linalg.svd(lens_map, compute_uv=False)[-1])


def _fold(p):
    return np.stack([p[..., 0], p[..., 1] ** 2], axis=-1)


def test_criterion_parity_fires_across_a_fold():
    M, G, COMPOSE, PINV0, ROOT_CLASS = new.child_matrix_tables()
    h0 = 1.0
    tri = np.asarray(new.ROOT_SHAPES[1], dtype=np.float64)[None] * h0 - np.array(
        [0.0, 0.5]
    )
    classes = backend.copy(ROOT_CLASS[1:2])
    v, m = _six_points(tri)
    keep, parity_ok, s, status = (
        backend.to_numpy(x)
        for x in new.evaluate_criterion(
            _fold_jacobian,
            _arr(v),
            _arr(m),
            _arr(_fold(v)),
            _arr(_fold(m)),
            classes,
            0,
            h0,
            1e-3,
            PINV0,
            COMPOSE,
        )
    )
    assert not parity_ok[0] and not keep[0]
    assert _flagged(status, new.LEAF_APPROX_PARITY_UNRESOLVED)[0]


@pytest.mark.parametrize(
    "shift,jacobian_flag",
    [
        # Two midpoints sit exactly on the critical line y == 0, where
        # det A == 0: a sign that says nothing, which is the non-finite flag.
        (0.5, new.LEAF_JACOBIAN_NONFINITE),
        # Off the line, the six determinants 2y are nonzero but mixed in sign.
        (0.4, new.LEAF_JACOBIAN_PARITY_UNRESOLVED),
    ],
)
def test_forced_jacobian_joins_the_approximate_flags_across_a_fold(
    shift, jacobian_flag
):
    """Without ``force_jacobian`` a failing triangle never reaches the Jacobian.

    Both placements fail child parity and the deviation test, so the
    Jacobian is skipped and ``status`` holds exactly those two flags. Forcing
    it adds the Jacobian's own verdict on top, which is what ``max_level``
    relies on to record every reason a leaf failed.
    """
    M, G, COMPOSE, PINV0, ROOT_CLASS = new.child_matrix_tables()
    tri = np.asarray(new.ROOT_SHAPES[1], dtype=np.float64)[None] - np.array(
        [0.0, shift]
    )
    v, m = _six_points(tri)
    approx = new.LEAF_APPROX_PARITY_UNRESOLVED | new.LEAF_CONVERGENCE_FAILED
    for force, want in ((False, approx), (True, approx | jacobian_flag)):
        keep, parity_ok, s, status = (
            backend.to_numpy(x)
            for x in new.evaluate_criterion(
                _fold_jacobian,
                _arr(v),
                _arr(m),
                _arr(_fold(v)),
                _arr(_fold(m)),
                backend.copy(ROOT_CLASS[1:2]),
                0,
                1.0,
                1e-3,
                PINV0,
                COMPOSE,
                force_jacobian=force,
            )
        )
        assert status.tolist() == [want], f"force_jacobian={force}"
        assert not keep[0] and not parity_ok[0]


def test_affine_and_sigma_min_are_invariant_to_simultaneous_relabelling():
    M, G, COMPOSE, PINV0, ROOT_CLASS = new.child_matrix_tables()
    p = RNG.normal(size=(200, 3, 2))
    q = RNG.normal(size=(200, 3, 2))
    base = backend.to_numpy(new.affine_from_triangles(_arr(p), _arr(q)))
    perm = np.argsort(RNG.random((200, 3)), axis=1)
    pp = np.take_along_axis(p, perm[..., None], axis=1)
    qp = np.take_along_axis(q, perm[..., None], axis=1)
    permuted = backend.to_numpy(new.affine_from_triangles(_arr(pp), _arr(qp)))
    assert np.allclose(base, permuted, rtol=1e-9, atol=1e-11)
    assert np.array_equal(
        np.sign(np.linalg.det(base)), np.sign(np.linalg.det(permuted))
    )
    assert np.allclose(
        backend.to_numpy(new.sigma_min_2x2(_arr(base))),
        backend.to_numpy(new.sigma_min_2x2(_arr(permuted))),
        rtol=1e-9,
    )


@pytest.mark.parametrize("force", [False, True])
def test_criterion_reports_nonfinite_as_split(force):
    """A non-finite sample is flagged, never converges, and never reaches the
    Jacobian -- not even under ``force_jacobian``, since a triangle the
    raytrace could not evaluate has nothing for the Jacobian to confirm.

    The NaN also fails child parity, so that flag rides along; the deviation
    flag does not, since it is only set on a triangle whose samples are
    finite.
    """
    M, G, COMPOSE, PINV0, ROOT_CLASS = new.child_matrix_tables()
    h0 = 0.05
    tri = np.asarray(new.ROOT_SHAPES[0], dtype=np.float64)[None] * h0
    classes = backend.copy(ROOT_CLASS[0:1])
    v, m = _six_points(tri)
    bad = m.copy()
    bad[0, 0, 0] = np.nan
    keep, parity_ok, s, status = (
        backend.to_numpy(x)
        for x in new.evaluate_criterion(
            _exploding_jacobian,
            _arr(v),
            _arr(m),
            _arr(v),
            _arr(bad),
            classes,
            0,
            h0,
            1e-3,
            PINV0,
            COMPOSE,
            force_jacobian=force,
        )
    )
    assert not keep[0]
    assert status.tolist() == [
        new.LEAF_RAYTRACE_NONFINITE | new.LEAF_APPROX_PARITY_UNRESOLVED
    ]


def _affine_pair(lens_map, h0=0.05):
    """Both level-0 root triangles and their exact affine images."""
    tri = np.stack(
        [np.asarray(new.ROOT_SHAPES[s], dtype=np.float64) * h0 for s in (0, 1)]
    )
    v, m = _six_points(tri)
    return v, m, v @ lens_map.T, m @ lens_map.T


def test_criterion_withholds_convergence_when_the_jacobian_disagrees():
    """The gate this criterion exists for: approximate tests are not enough.

    The samples are exactly affine, so child parity and the deviation test
    both pass and the triangle would converge on them alone. Flipping
    ``sign(det A)`` at a single one of its six samples -- an ``A`` that
    contradicts the samples, which only a unit test can build -- must still
    withhold convergence, and blame the Jacobian alone.
    """
    M, G, COMPOSE, PINV0, ROOT_CLASS = new.child_matrix_tables()
    lens_map = np.array([[0.7, 0.1], [-0.2, 0.9]])
    v, m, bv, bm = _affine_pair(lens_map)
    J = np.tile(lens_map, (12, 1, 1))
    J[7] = np.diag([1.0, -1.0])  # the second triangle's second vertex
    keep, parity_ok, s, status = (
        backend.to_numpy(x)
        for x in new.evaluate_criterion(
            _fixed_jacobians(J),
            _arr(v),
            _arr(m),
            _arr(bv),
            _arr(bm),
            backend.copy(ROOT_CLASS),
            0,
            0.05,
            1e-3,
            PINV0,
            COMPOSE,
        )
    )
    assert keep.tolist() == [True, False]
    assert parity_ok.tolist() == [True, False]
    assert status.tolist() == [new.LEAF_CONVERGED, new.LEAF_JACOBIAN_PARITY_UNRESOLVED]


def test_criterion_withholds_convergence_on_a_nonfinite_jacobian():
    M, G, COMPOSE, PINV0, ROOT_CLASS = new.child_matrix_tables()
    lens_map = np.array([[0.7, 0.1], [-0.2, 0.9]])
    v, m, bv, bm = _affine_pair(lens_map)
    J = np.tile(lens_map, (12, 1, 1))
    J[3, 0, 1] = np.nan  # the first triangle's first midpoint
    keep, parity_ok, s, status = (
        backend.to_numpy(x)
        for x in new.evaluate_criterion(
            _fixed_jacobians(J),
            _arr(v),
            _arr(m),
            _arr(bv),
            _arr(bm),
            backend.copy(ROOT_CLASS),
            0,
            0.05,
            1e-3,
            PINV0,
            COMPOSE,
        )
    )
    assert keep.tolist() == [False, True]
    assert parity_ok.tolist() == [False, True]
    assert status.tolist() == [new.LEAF_JACOBIAN_NONFINITE, new.LEAF_CONVERGED]


def test_criterion_evaluates_the_jacobian_only_where_it_could_matter():
    """Only a triangle about to converge pays for a Jacobian evaluation.

    The second triangle's midpoint is displaced far past the deviation
    tolerance, so it splits whatever its Jacobian says, and ``jacobian_fn``
    sees only the first triangle's six samples. ``force_jacobian`` evaluates
    both. Either way the call is a single batch, and a triangle the Jacobian
    never saw keeps its child-parity verdict in ``parity_ok``.
    """
    M, G, COMPOSE, PINV0, ROOT_CLASS = new.child_matrix_tables()
    lens_map = np.array([[0.7, 0.1], [-0.2, 0.9]])
    v, m, bv, bm = _affine_pair(lens_map)
    bm[1, 0] += 2e-3
    for force, checked in ((False, [0]), (True, [0, 1])):
        jacobian_fn, calls = _recording(_constant_jacobian(lens_map))
        keep, parity_ok, s, status = (
            backend.to_numpy(x)
            for x in new.evaluate_criterion(
                jacobian_fn,
                _arr(v),
                _arr(m),
                _arr(bv),
                _arr(bm),
                backend.copy(ROOT_CLASS),
                0,
                0.05,
                1e-3,
                PINV0,
                COMPOSE,
                force_jacobian=force,
            )
        )
        assert len(calls) == 1
        want = np.concatenate([v[checked], m[checked]], axis=1).reshape(-1, 2)
        assert np.array_equal(calls[0], want), f"force_jacobian={force}"
        assert keep.tolist() == [True, False]
        assert parity_ok.tolist() == [True, True]
        assert status.tolist() == [new.LEAF_CONVERGED, new.LEAF_CONVERGENCE_FAILED]


def test_forced_jacobian_adds_flags_but_never_rescues():
    """A clean Jacobian cannot clear a failed test, and a bad one only adds.

    Both triangles fail the deviation test. Forced, the Jacobian passes on the
    first -- which must still not converge -- and flips sign on the second,
    whose status must then carry both reasons.
    """
    M, G, COMPOSE, PINV0, ROOT_CLASS = new.child_matrix_tables()
    lens_map = np.array([[0.7, 0.1], [-0.2, 0.9]])
    v, m, bv, bm = _affine_pair(lens_map)
    bm[:, 0] += 2e-3
    J = np.tile(lens_map, (12, 1, 1))
    J[11] = np.diag([1.0, -1.0])  # the second triangle's third midpoint
    keep, parity_ok, s, status = (
        backend.to_numpy(x)
        for x in new.evaluate_criterion(
            _fixed_jacobians(J),
            _arr(v),
            _arr(m),
            _arr(bv),
            _arr(bm),
            backend.copy(ROOT_CLASS),
            0,
            0.05,
            1e-3,
            PINV0,
            COMPOSE,
            force_jacobian=True,
        )
    )
    assert keep.tolist() == [False, False]
    assert parity_ok.tolist() == [True, False]
    assert status.tolist() == [
        new.LEAF_CONVERGENCE_FAILED,
        new.LEAF_CONVERGENCE_FAILED | new.LEAF_JACOBIAN_PARITY_UNRESOLVED,
    ]


def _unit_triangles(n):
    """``n`` lens-plane unit triangles; only their count and shape matter here."""
    tri = np.tile(np.asarray(new.ROOT_SHAPES[0], dtype=np.float64), (n, 1, 1))
    return _six_points(tri)


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_jacobian_parity_accepts_a_strict_sign_of_either_orientation(sign):
    v, m = _unit_triangles(2)
    ok, nonfinite = (
        backend.to_numpy(x)
        for x in new.jacobian_parity_ok(
            _constant_jacobian(np.diag([1.0, sign])),
            _arr(v),
            _arr(m),
            return_details=True,
        )
    )
    assert ok.tolist() == [True, True]
    assert nonfinite.tolist() == [False, False]


@pytest.mark.parametrize("sample", range(6))
def test_jacobian_parity_rejects_a_sign_change_at_any_sample(sample):
    """A single flipped determinant, at each of the six positions in turn,
    condemns its own triangle and only its own."""
    v, m = _unit_triangles(2)
    J = np.tile(np.eye(2), (12, 1, 1))
    J[6 + sample] = np.diag([1.0, -1.0])
    ok, nonfinite = (
        backend.to_numpy(x)
        for x in new.jacobian_parity_ok(
            _fixed_jacobians(J), _arr(v), _arr(m), return_details=True
        )
    )
    assert ok.tolist() == [True, False]
    assert nonfinite.tolist() == [False, False]


@pytest.mark.parametrize(
    "bad",
    [
        np.array([[np.nan, 0.0], [0.0, 1.0]]),
        np.array([[1.0, np.inf], [0.0, 1.0]]),
        np.zeros((2, 2)),
        np.array([[1.0, 2.0], [2.0, 4.0]]),  # rank 1: det is exactly 0
    ],
    ids=["nan", "inf", "zero", "rank_one"],
)
def test_jacobian_parity_flags_nonfinite_and_singular_samples(bad):
    """Neither a non-finite nor a singular ``A`` has a usable sign.

    Both are reported through ``jacobian_nonfinite``, which is what separates
    them from a genuine sign change, and both fail parity.
    """
    v, m = _unit_triangles(2)
    J = np.tile(np.eye(2), (12, 1, 1))
    J[4] = bad
    ok, nonfinite = (
        backend.to_numpy(x)
        for x in new.jacobian_parity_ok(
            _fixed_jacobians(J), _arr(v), _arr(m), return_details=True
        )
    )
    assert ok.tolist() == [False, True]
    assert nonfinite.tolist() == [True, False]


@pytest.mark.parametrize("scale", [1e200, 1e-200])
def test_jacobian_parity_survives_overflow_and_underflow_of_the_determinant(scale):
    """The row scaling, at the magnitudes that would defeat a plain ``det``.

    ``scale**2`` overflows to ``inf`` at ``1e200`` and underflows to exactly
    ``0.0`` at ``1e-200``, so an unscaled determinant would misreport both
    perfectly regular Jacobians as non-finite or singular.
    """
    with np.errstate(over="ignore", under="ignore"):
        naive = np.float64(scale) * np.float64(scale)
    assert naive in (np.inf, 0.0), "fixture no longer defeats a plain determinant"
    v, m = _unit_triangles(1)
    for J in (scale * np.eye(2), scale * np.diag([1.0, -1.0])):
        ok, nonfinite = (
            backend.to_numpy(x)
            for x in new.jacobian_parity_ok(
                _constant_jacobian(J), _arr(v), _arr(m), return_details=True
            )
        )
        assert ok.tolist() == [True]
        assert nonfinite.tolist() == [False]


def test_jacobian_parity_evaluates_all_six_samples_in_one_call():
    """Vertices then midpoints, triangle-major, in a single batch."""
    v, m = _six_points(RNG.normal(size=(5, 3, 2)))
    jacobian_fn, calls = _recording(_constant_jacobian(np.eye(2)))
    ok = backend.to_numpy(new.jacobian_parity_ok(jacobian_fn, _arr(v), _arr(m)))
    assert ok.tolist() == [True] * 5
    assert len(calls) == 1
    assert np.array_equal(calls[0], np.concatenate([v, m], axis=1).reshape(-1, 2))


def test_jacobian_parity_of_no_triangles_never_calls_the_jacobian():
    empty = _arr(np.zeros((0, 3, 2)))
    ok = new.jacobian_parity_ok(_exploding_jacobian, empty, empty)
    assert backend.to_numpy(ok).shape == (0,)
    ok, nonfinite = new.jacobian_parity_ok(
        _exploding_jacobian, empty, empty, return_details=True
    )
    assert backend.to_numpy(ok).shape == (0,)
    assert backend.to_numpy(nonfinite).shape == (0,)


def test_jacobian_parity_rejects_mismatched_shapes():
    v, m = _unit_triangles(2)
    with pytest.raises(ValueError, match=r"\(N, 3, 2\)"):
        new.jacobian_parity_ok(_exploding_jacobian, _arr(v), _arr(m[:1]))
    with pytest.raises(ValueError, match=r"\(N, 3, 2\)"):
        new.jacobian_parity_ok(_exploding_jacobian, _arr(v[:, :2]), _arr(m[:, :2]))
    with pytest.raises(ValueError, match=r"\(6 \* N, 2, 2\)"):
        new.jacobian_parity_ok(
            _fixed_jacobians(np.tile(np.eye(2), (2, 1, 1))), _arr(v), _arr(m)
        )


def test_child_shape_matrices_match_shape_matrix_of_each_child():
    """Q_k must be exactly the edge matrix of child k, bit for bit.

    Both forms subtract the same operands in the same order, so this is an
    equality check and not an allclose: a reassociation that changed rounding
    would change the sign of a near-degenerate det and silently move the parity
    verdict, which is the whole thing this kernel decides.
    """
    rng = np.random.default_rng(0)
    beta_v = rng.normal(size=(5, 3, 2))
    beta_m = rng.normal(size=(5, 3, 2))
    Q = backend.to_numpy(new.child_shape_matrices(_arr(beta_v), _arr(beta_m)))
    assert Q.shape == (5, 4, 2, 2)
    six = np.concatenate([beta_v, beta_m], axis=1)
    for k, idx in enumerate(new.CHILD_VERTEX_INDICES):
        expected = backend.to_numpy(new.shape_matrix(_arr(six[:, list(idx)])))
        assert np.array_equal(Q[:, k], expected)


def test_parity_from_children_accepts_a_constant_sign():
    Q = np.tile(np.eye(2), (2, 4, 1, 1))
    assert backend.to_numpy(new.parity_from_children(_arr(Q))).tolist() == [
        True,
        True,
    ]


def test_parity_from_children_rejects_a_sign_change():
    Q = np.tile(np.eye(2), (1, 4, 1, 1))
    Q[0, 2] = np.array([[0.0, 1.0], [1.0, 0.0]])  # det -1 among three det +1
    assert backend.to_numpy(new.parity_from_children(_arr(Q))).tolist() == [False]


def test_parity_from_children_condemns_a_lone_degenerate_child():
    Q = np.tile(np.eye(2), (1, 4, 1, 1))
    Q[0, 3] = 0.0  # sign 0 differs from +1
    assert backend.to_numpy(new.parity_from_children(_arr(Q))).tolist() == [False]


def test_parity_from_children_passes_an_entirely_degenerate_triangle():
    """All four dets exactly zero is a constant sign, so there is no parity
    *change* and the triangle passes. Spec section 4.6: condemning this case
    would be the deviation test in disguise, which is out of scope. A kappa == 1
    sheet is the fixture that reaches it.
    """
    Q = np.zeros((1, 4, 2, 2))
    assert backend.to_numpy(new.parity_from_children(_arr(Q))).tolist() == [True]


@pytest.mark.parametrize(
    "determinants",
    [
        [np.nan, np.nan, np.nan, np.nan],
        [np.nan, 0.0, 0.0, 0.0],
    ],
)
def test_parity_from_children_rejects_nan_determinants(determinants):
    Q = np.zeros((1, 4, 2, 2))
    Q[0, :, 0, 0] = determinants
    Q[0, :, 1, 1] = 1.0
    assert backend.to_numpy(new.parity_from_children(_arr(Q))).tolist() == [False]


def test_midpoint_deviation_pairs_midpoint_with_opposite_edge():
    v = np.array([[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]])
    m = np.array([[[0.5, 0.5], [0.0, 0.5], [0.5, 0.0]]])  # exact affine images
    assert np.allclose(backend.to_numpy(new.midpoint_deviation(_arr(v), _arr(m))), 0.0)
    shifted = m.copy()
    shifted[0, 1] += np.array([0.0, 0.25])
    got = backend.to_numpy(new.midpoint_deviation(_arr(v), _arr(shifted)))
    assert np.allclose(got, [[0.0, 0.25, 0.0]])


def test_weights_sum_to_twice_the_signed_area():
    tri = RNG.normal(size=(500, 3, 2))
    beta = RNG.normal(size=(500, 2))
    w = backend.to_numpy(new.triangle_weights(_arr(tri), _arr(beta)))
    P = backend.to_numpy(new.shape_matrix(_arr(tri)))
    d = P[:, 0, 0] * P[:, 1, 1] - P[:, 0, 1] * P[:, 1, 0]
    assert np.allclose(w.sum(axis=1), d, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("flip", [False, True])
def test_containment_agrees_with_barycentric_truth_on_both_parities(flip):
    tri = np.array([[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]])
    if flip:
        tri = tri[:, ::-1, :]
    pts = RNG.uniform(-0.5, 1.5, size=(4000, 2))
    tiled = np.repeat(tri, len(pts), axis=0)
    hit = backend.to_numpy(new.contains(new.triangle_weights(_arr(tiled), _arr(pts))))
    truth = (pts[:, 0] >= 0) & (pts[:, 1] >= 0) & (pts[:, 0] + pts[:, 1] <= 1)
    assert (hit == truth).all()


def test_shared_edge_weights_are_exactly_negated():
    """Two leaves meeting on an edge must not both reject a point on that edge."""
    verts = np.array([[0.0, 0.0], [1.0, 0.3], [0.4, 1.0], [1.3, 1.2]])
    left = verts[[0, 1, 2]][None]
    right = verts[[1, 3, 2]][None]  # shares the edge (1, 2), opposite traversal
    beta = np.array([[0.62, 0.71]])
    wl = backend.to_numpy(new.triangle_weights(_arr(left), _arr(beta)))
    wr = backend.to_numpy(new.triangle_weights(_arr(right), _arr(beta)))
    # left's weight opposite vertex 0 uses the (1, 2) edge; right's opposite
    # vertex 3 uses (2, 1). They must be bit-exact negatives.
    assert wl[0, 0] == -wr[0, 1]


def test_sanitize_bary_is_always_in_the_simplex():
    """The clip path keeps wide-but-finite w/d ratios inside the simplex.

    The 1e-300 scaling stresses the clamp with extreme ratios; it does not reach
    the centroid fallback, because 1e-300 is a normal double and w/d is invariant
    under a uniform scaling. The fallback is covered by the two tests below.
    """
    w = RNG.normal(size=(1000, 3)) * 1e-300
    d = w.sum(axis=1)
    bary = backend.to_numpy(new.sanitize_bary(_arr(w), _arr(d)))
    assert np.isfinite(bary).all()
    assert (bary >= 0).all() and (bary <= 1).all()
    assert np.allclose(bary.sum(axis=1), 1.0, rtol=0, atol=1e-12)


def test_sanitize_bary_falls_back_to_the_centroid_on_total_degeneracy():
    w = np.zeros((4, 3))
    d = np.zeros(4)
    bary = backend.to_numpy(new.sanitize_bary(_arr(w), _arr(d)))
    assert np.array_equal(bary, np.full((4, 3), 1.0 / 3.0))


def test_sanitize_bary_selects_per_row_between_normalized_and_centroid():
    """Degenerate and ordinary rows in one call must be resolved independently.

    Both other sanitize_bary tests are all-or-nothing -- every row ordinary, or
    every row degenerate -- so neither would catch an implementation that decided
    the fallback once for the whole batch instead of per row.
    """
    w = _arr([[3.0, 1.5, 1.5], [0.0, 0.0, 0.0], [2.0, 1.0, 1.0], [0.0, 0.0, 0.0]])
    d = _arr([6.0, 0.0, 4.0, 0.0])
    bary = backend.to_numpy(new.sanitize_bary(w, d))
    assert np.allclose(bary[[0, 2]], [[0.5, 0.25, 0.25], [0.5, 0.25, 0.25]])
    assert np.allclose(bary[[1, 3]], 1.0 / 3.0)
    assert np.allclose(bary.sum(axis=1), 1.0, rtol=0, atol=1e-12)


def test_sanitize_bary_recovers_ordinary_coordinates():
    tri = np.array([[[0.0, 0.0], [2.0, 0.0], [0.0, 3.0]]])
    beta = np.array([[0.5, 0.75]])
    w = new.triangle_weights(_arr(tri), _arr(beta))
    d = _arr([2.0 * 3.0])
    bary = backend.to_numpy(new.sanitize_bary(w, d))
    assert np.allclose(bary @ tri[0], beta, rtol=1e-12, atol=1e-14)


def test_no_numpy_import_in_the_module():
    source = open(new.__file__).read()
    assert "import numpy" not in source


def test_parity_from_jacobians_matches_the_call_row_for_row():
    """The split-out arithmetic gives `jacobian_parity_ok`'s verdict exactly.

    One triangle per case: a sign change, a non-finite ``A``, an exactly
    singular ``A``, and a clean one.
    """
    v, m = _unit_triangles(4)
    J = np.tile(np.eye(2), (24, 1, 1))
    J[3] = np.diag([1.0, -1.0])
    J[8] = np.array([[np.nan, 0.0], [0.0, 1.0]])
    J[14] = np.array([[1.0, 2.0], [2.0, 4.0]])
    want = [
        backend.to_numpy(x)
        for x in new.jacobian_parity_ok(
            _fixed_jacobians(J), _arr(v), _arr(m), return_details=True
        )
    ]
    got = [
        backend.to_numpy(x)
        for x in new.parity_from_jacobians(
            _arr(J.reshape(4, 6, 2, 2)), return_details=True
        )
    ]
    assert got[0].tolist() == want[0].tolist() == [False, False, False, True]
    assert got[1].tolist() == want[1].tolist() == [False, True, True, False]


def test_parity_from_jacobians_of_no_triangles_is_empty():
    ok, nonfinite = new.parity_from_jacobians(
        _arr(np.zeros((0, 6, 2, 2))), return_details=True
    )
    assert backend.to_numpy(ok).shape == (0,)
    assert backend.to_numpy(nonfinite).shape == (0,)


def test_parity_from_jacobians_rejects_a_wrong_shape():
    with pytest.raises(ValueError, match=r"\(N, 6, 2, 2\)"):
        new.parity_from_jacobians(_arr(np.zeros((2, 5, 2, 2))))


def test_criterion_on_precomputed_jacobians_matches_the_call_when_forced():
    """``jacobian=`` gives the same four outputs as the call it replaces.

    Forced, both triangles are tested: the first has a non-finite ``A`` at its
    third vertex, the second a sign change at its second vertex. The
    precomputed path must reach the same verdicts without ever calling
    ``jacobian_fn``.
    """
    M, G, COMPOSE, PINV0, ROOT_CLASS = new.child_matrix_tables()
    lens_map = np.array([[0.7, 0.1], [-0.2, 0.9]])
    v, m, bv, bm = _affine_pair(lens_map)
    J = np.tile(lens_map, (12, 1, 1))
    J[2, 0, 0] = np.nan
    J[7] = np.diag([1.0, -1.0])
    args = (_arr(v), _arr(m), _arr(bv), _arr(bm))
    rest = (0, 0.05, 1e-3, PINV0, COMPOSE)
    want = [
        backend.to_numpy(x)
        for x in new.evaluate_criterion(
            _fixed_jacobians(J),
            *args,
            backend.copy(ROOT_CLASS),
            *rest,
            force_jacobian=True,
        )
    ]
    got = [
        backend.to_numpy(x)
        for x in new.evaluate_criterion(
            _exploding_jacobian,
            *args,
            backend.copy(ROOT_CLASS),
            *rest,
            force_jacobian=True,
            jacobian=_arr(J.reshape(2, 6, 2, 2)),
        )
    ]
    for g, w in zip(got, want):
        assert g.tolist() == w.tolist()
    assert want[3].tolist() == [
        new.LEAF_JACOBIAN_NONFINITE,
        new.LEAF_JACOBIAN_PARITY_UNRESOLVED,
    ]


@pytest.mark.parametrize(
    "flipped, want",
    [
        (3, [new.LEAF_JACOBIAN_PARITY_UNRESOLVED, new.LEAF_CONVERGENCE_FAILED]),
        (9, [new.LEAF_CONVERGED, new.LEAF_CONVERGENCE_FAILED]),
    ],
    ids=["candidate", "splitting"],
)
def test_criterion_reads_precomputed_jacobians_only_on_the_rows_it_tests(flipped, want):
    """Unforced, only the candidate's six rows of ``jacobian`` are read.

    The second triangle fails the deviation test, so it is never a candidate:
    a sign change in its rows must change nothing, while one in the first
    triangle's rows must withhold its convergence.
    """
    M, G, COMPOSE, PINV0, ROOT_CLASS = new.child_matrix_tables()
    lens_map = np.array([[0.7, 0.1], [-0.2, 0.9]])
    v, m, bv, bm = _affine_pair(lens_map)
    bm[1, 0] += 2e-3
    J = np.tile(lens_map, (12, 1, 1))
    J[flipped] = np.diag([1.0, -1.0])
    keep, parity_ok, s, status = (
        backend.to_numpy(x)
        for x in new.evaluate_criterion(
            _exploding_jacobian,
            _arr(v),
            _arr(m),
            _arr(bv),
            _arr(bm),
            backend.copy(ROOT_CLASS),
            0,
            0.05,
            1e-3,
            PINV0,
            COMPOSE,
            jacobian=_arr(J.reshape(2, 6, 2, 2)),
        )
    )
    assert status.tolist() == want
