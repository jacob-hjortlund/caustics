"""
The refinement criterion, and the ``LEAF_*`` status flags it records.

:func:`evaluate_criterion` decides, vectorized over triangles, whether a
triangle's affine model is accurate to ``min_img_sep`` and whether a critical
curve runs between its samples: child parity from the six mapped samples,
then the lens Jacobian's parity on the triangles that would otherwise
converge.
"""

from ....backend_obj import backend
from .geometry import (
    CHILD_VERTEX_INDICES,
    COMPOSE,
    PINV0,
    jacobian_signs,
    sigma_min_2x2,
)

__all__ = (
    "LEAF_CONVERGED",
    "LEAF_CONVERGENCE_FAILED",
    "LEAF_APPROX_PARITY_UNRESOLVED",
    "LEAF_JACOBIAN_PARITY_UNRESOLVED",
    "LEAF_RAYTRACE_NONFINITE",
    "LEAF_JACOBIAN_NONFINITE",
    "midpoint_deviation",
    "converged_from_deviation",
    "child_shape_matrices",
    "parity_from_children",
    "jacobian_parity_ok",
    "parity_from_jacobians",
    "parity_from_signs",
    "approximate_criterion",
    "jacobian_rows",
    "apply_jacobian_signs",
    "evaluate_criterion",
    "lens_status",
)


# Why a leaf is not converged, as a bitmask. Each failed test sets its own bit,
# so one status records every failure rather than a single chosen reason, and
# `LEAF_CONVERGED` -- zero, no bit set -- is the only status that means the leaf
# can be trusted. Only such leaves enter the spatial index. Test a flag with
# `(status & FLAG) != 0`, never `status == FLAG`, which misses every leaf that
# carries a second flag as well.
#
# Plain ints, not an `IntFlag`: `status` lives in a `backend.int64` array and
# is compared, OR-ed, scattered and broadcast through backend ops the whole
# way, which a NumPy-flavoured enum would fight at every one of those call
# sites for no benefit.
#
# - `LEAF_CONVERGENCE_FAILED`: the midpoint-deviation test (step 7) failed, so
#   the leaf's affine model is not accurate to `min_img_sep`.
# - `LEAF_APPROX_PARITY_UNRESOLVED`: the four red-split children disagree on
#   `sign(det Q_k)` (`parity_from_children`) -- a fold, as seen through the
#   mapped samples.
# - `LEAF_JACOBIAN_PARITY_UNRESOLVED`: the lens Jacobian's `sign(det A)`
#   differs between the six lens-plane samples (`jacobian_parity_ok`) -- a
#   critical curve runs between them.
# - `LEAF_RAYTRACE_NONFINITE`: some raytraced sample is non-finite, so the rest
#   of the criterion never ran. Also OR-ed in at freeze, by
#   `invalidate_nonfinite_origins`, onto every origin whose closure triangles
#   picked up a non-finite vertex.
# - `LEAF_JACOBIAN_NONFINITE`: some sample's Jacobian is non-finite or exactly
#   singular, so its sign says nothing. Singular counts: a sample lying exactly
#   on a critical curve lands here, not in `LEAF_JACOBIAN_PARITY_UNRESOLVED`.
#   The critical band still traces such a leaf, from its stored determinants.
#
# Below `max_level` every failing triangle splits, so no flag is ever stored
# there -- a failure is a reason to refine, not a verdict. At `max_level`
# nothing can split: the criterion runs with the Jacobian forced on every
# finite triangle, and the whole bitmask is stored. Balance-cascade children
# and closure triangles inherit their origin's status, which for a cascade
# child is always `LEAF_CONVERGED` (see the cascade in `refine`). So apart from
# the freeze-time `LEAF_RAYTRACE_NONFINITE`, a nonzero status means a
# `max_level` leaf.
LEAF_CONVERGED = 0
LEAF_CONVERGENCE_FAILED = 1 << 0
LEAF_APPROX_PARITY_UNRESOLVED = 1 << 1
LEAF_JACOBIAN_PARITY_UNRESOLVED = 1 << 2
LEAF_RAYTRACE_NONFINITE = 1 << 3
LEAF_JACOBIAN_NONFINITE = 1 << 4


def midpoint_deviation(beta_v, beta_m):
    """
    Distance between each mapped edge midpoint and its affine prediction.

    With ``m_i`` opposite ``theta_i``, the predicted image of ``m_i`` under an
    affine map is the mean of the two ``beta`` values at the endpoints of the edge
    it bisects, ``(beta_j + beta_k) / 2`` for cyclic ``(i, j, k)``.

    Parameters
    ----------
    beta_v: ndarray
        Source-plane vertices, shape ``(n, 3, 2)``.

        *Unit: arcsec*

    beta_m: ndarray
        Source-plane edge midpoints ``m1, m2, m3``, shape ``(n, 3, 2)``.

        *Unit: arcsec*

    Returns
    -------
    ndarray
        Shape ``(n, 3)``, a source-plane length.

        *Unit: arcsec*
    """
    predicted = 0.5 * (beta_v[:, [1, 2, 0], :] + beta_v[:, [2, 0, 1], :])
    return backend.norm(predicted - beta_m, dim=-1)


def converged_from_deviation(r, s, min_img_sep):
    """
    Step 7 of the refinement criterion.

    Written as ``all(r < threshold)`` and never as ``not any(r >= threshold)``.
    The two are not equivalent when ``s`` is ``NaN``: under IEEE every comparison
    against ``NaN`` is False, so the ``>=`` form would mark a maximally degenerate
    triangle *converged*, silently inverting the intended behaviour. The ``<`` form
    puts ``NaN`` in the split branch. It also settles the exact-equality edge -- an
    affine-but-singular triangle (``r == 0``, ``s == 0``) gives ``0 < 0``, False,
    and splits, which is the conservative direction.

    Note the comparison is evaluated in the **source plane**: ``r`` is a
    source-plane length and ``s`` is dimensionless, so ``s * min_img_sep`` converts
    the lens-plane tolerance ``min_img_sep`` into a source-plane one. The
    equivalent lens-plane form ``r / s < min_img_sep`` must not be used, because
    dividing by ``s`` breaks on the legal and expected ``s == 0``.

    Parameters
    ----------
    r: ndarray
        Midpoint deviations, shape ``(n, 3)``.

        *Unit: arcsec*

    s: ndarray
        Smallest singular value over the four children, shape ``(n,)``.
    min_img_sep: float
        Lens-plane tolerance.

        *Unit: arcsec*

    Returns
    -------
    ndarray
        Shape ``(n,)`` bool.
    """
    return backend.all(r < (s * min_img_sep)[:, None], dim=1)


def child_shape_matrices(beta_v, beta_m):
    """
    Source-plane edge matrices ``Q_k`` of the four red-split children.

    Split out of :func:`evaluate_criterion` so the child-parity test can be
    applied on its own, apart from the deviation and Jacobian halves of the
    criterion.

    Parameters
    ----------
    beta_v: ndarray
        Source-plane vertices, shape ``(n, 3, 2)``.

        *Unit: arcsec*

    beta_m: ndarray
        Source-plane midpoints ``m1, m2, m3``, shape ``(n, 3, 2)``.

        *Unit: arcsec*

    Returns
    -------
    ndarray
        Shape ``(n, 4, 2, 2)``, child index in :data:`CHILD_VERTEX_INDICES`
        order.
    """
    stacked = backend.concatenate([beta_v, beta_m], dim=1)  # (n, 6, 2)
    idx0, idx1, idx2 = zip(*CHILD_VERTEX_INDICES)  # each length 4
    q1 = stacked[:, list(idx0), :]
    q2 = stacked[:, list(idx1), :]
    q3 = stacked[:, list(idx2), :]
    return backend.stack((q2 - q1, q3 - q1), dim=-1)  # (n, 4, 2, 2)


def parity_from_children(Q):
    """
    True where ``sign(det Q_k)`` is constant over the four children.

    The parity test needs no lens-plane ``P`` at all: ``sign(det A_k) =
    sign(det Q_k) * sign(det P_k)``, and all four children share the parent's
    ``sign(det P_k)`` because every ``det M_k == +1``, so constancy of
    ``sign(det A_k)`` over ``k`` is equivalent to constancy of
    ``sign(det Q_k)``. That is what makes this usable at ``max_level``, where
    no child is ever built and no ``P`` is ever formed.

    A ``NaN`` never equals itself, so a non-finite child lands in the failing
    branch. An exact zero gives sign 0, which differs from ``+-1`` and also
    fails -- unless *every* child is zero, which is a constant sign and passes.

    Parameters
    ----------
    Q: ndarray
        Child edge matrices from :func:`child_shape_matrices`, shape
        ``(n, 4, 2, 2)``.

    Returns
    -------
    ndarray
        Shape ``(n,)`` bool.
    """
    det_q = Q[..., 0, 0] * Q[..., 1, 1] - Q[..., 0, 1] * Q[..., 1, 0]
    sign_q = backend.sign(det_q)
    return backend.all(sign_q == sign_q[:, :1], dim=1)


def _jacobian_signs_at(jacobian_fn, theta_v, theta_m):
    """
    :func:`~caustics.lenses.func.adaptive.geometry.jacobian_signs` at each
    triangle's six samples, shape ``(N, 6)``: one ``jacobian_fn`` call on all
    ``6 * N`` points, triangle-major, and none when ``N == 0``.
    """
    if theta_v.shape != theta_m.shape or theta_v.shape[1:] != (3, 2):
        raise ValueError("theta_v and theta_m must both have shape (N, 3, 2)")
    n = theta_v.shape[0]
    if n == 0:
        return backend.zeros(
            (0, 6), dtype=backend.int64, device=backend.device(theta_v)
        )
    theta = backend.concatenate((theta_v, theta_m), dim=1).reshape(-1, 2)
    J = jacobian_fn(theta[:, 0], theta[:, 1])
    # `getattr`, not `J.shape`: a raytrace passed as the Jacobian returns a
    # tuple, which should fail here, naming the Jacobian, not as an
    # AttributeError.
    shape = getattr(J, "shape", None)
    if shape != (6 * n, 2, 2):
        got = type(J).__name__ if shape is None else tuple(shape)
        raise ValueError(
            f"jacobian_fn must return an array of shape (6 * N, 2, 2), got {got}"
        )
    return jacobian_signs(J).reshape(n, 6)


def jacobian_parity_ok(jacobian_fn, theta_v, theta_m, *, return_details=False):
    """
    True where ``sign(det A)`` is one strict sign at all six sample points.

    The exact counterpart of :func:`parity_from_children`. That test reads
    orientation off the six *mapped* samples, so it sees a fold only through
    the source-plane shapes of the four children. This one evaluates the lens
    Jacobian ``A = d(beta) / d(theta)`` itself at the same six lens-plane
    points -- the three vertices and the three edge midpoints -- so a critical
    curve with samples on both sides of it is caught whatever the mapped
    shapes look like. It is still a six-point sample: a critical curve that
    enters and leaves the triangle between samples is invisible to it too.

    ``jacobian_fn`` is called once, on all ``6 * N`` points, flattened
    triangle-major in the order ``theta_1, theta_2, theta_3, m_1, m_2, m_3`` --
    and not at all when ``N == 0``. The verdict itself is
    :func:`parity_from_signs` of the points'
    :func:`~caustics.lenses.func.adaptive.geometry.jacobian_signs`.

    Parameters
    ----------
    jacobian_fn: Callable[[ArrayLike, ArrayLike], ArrayLike]
        ``jacobian_fn(x, y) -> A`` on 1-D arrays of shape ``(K,)``, returning
        shape ``(K, 2, 2)``, e.g. a lens's ``jacobian_lens_equation``.
    theta_v: ArrayLike
        Lens-plane vertices, shape ``(N, 3, 2)``.

        *Unit: arcsec*

    theta_m: ArrayLike
        Lens-plane edge midpoints ``m1, m2, m3``, with ``m_i`` opposite
        ``theta_i``, shape ``(N, 3, 2)``.

        *Unit: arcsec*

    return_details: bool
        Also return the non-finite mask.

    Returns
    -------
    parity_ok: ArrayLike
        ``(N,)`` bool, True where all six determinants are finite, nonzero and
        of one sign.
    jacobian_nonfinite: ArrayLike
        ``(N,)`` bool, returned only with ``return_details``. True where some
        ``A`` is non-finite or has ``det A == 0`` exactly -- a sample whose sign
        carries no information, which includes one lying exactly on a
        critical curve. Always a subset of ``~parity_ok``.

    Raises
    ------
    ValueError
        If ``theta_v`` and ``theta_m`` are not both ``(N, 3, 2)``, or if
        ``jacobian_fn`` does not return ``(6 * N, 2, 2)``.
    """
    signs = _jacobian_signs_at(jacobian_fn, theta_v, theta_m)
    return parity_from_signs(signs, return_details=return_details)


def parity_from_jacobians(J, *, return_details=False):
    """
    True where ``sign(det A)`` is one strict sign at all six samples of a triangle.

    The arithmetic half of :func:`jacobian_parity_ok`, on Jacobians already
    evaluated: :func:`parity_from_signs` of their
    :func:`~caustics.lenses.func.adaptive.geometry.jacobian_signs`, which
    reads the sign off a row-scaled ``A`` so that neither overflow nor
    underflow can fake one.

    Parameters
    ----------
    J: ArrayLike
        Shape ``(N, 6, 2, 2)``: each triangle's Jacobians at ``theta_1,
        theta_2, theta_3, m_1, m_2, m_3``.
    return_details: bool
        Also return the non-finite mask.

    Returns
    -------
    parity_ok: ArrayLike
        ``(N,)`` bool, as :func:`jacobian_parity_ok`.
    jacobian_nonfinite: ArrayLike
        ``(N,)`` bool, returned only with ``return_details``, as
        :func:`jacobian_parity_ok`.

    Raises
    ------
    ValueError
        If ``J`` is not ``(N, 6, 2, 2)``.
    """
    if len(J.shape) != 4 or tuple(J.shape[1:]) != (6, 2, 2):
        raise ValueError("J must have shape (N, 6, 2, 2)")

    n = J.shape[0]
    signs = jacobian_signs(J.reshape(-1, 2, 2)).reshape(n, 6)
    return parity_from_signs(signs, return_details=return_details)


def parity_from_signs(signs, *, return_details=False):
    """
    True where the criterion's sign of ``det A`` is one strict sign at all six samples.

    ``signs`` are :func:`~caustics.lenses.func.adaptive.geometry.jacobian_signs`
    at each triangle's ``theta_1, theta_2, theta_3, m_1, m_2, m_3``. A 0 is a
    sample whose sign says nothing -- non-finite or exactly singular -- and
    fails its triangle, as ``LEAF_JACOBIAN_NONFINITE``. Keeping the signs
    rather than the Jacobians is what lets the build evaluate each lattice
    point once and judge every triangle sharing it from the stored sign
    (:func:`~caustics.lenses.func.adaptive.sampling.sample_jacobians`).

    Parameters
    ----------
    signs: ArrayLike
        Shape ``(N, 6)`` int64.
    return_details: bool
        Also return the non-finite mask.

    Returns
    -------
    parity_ok: ArrayLike
        ``(N,)`` bool, True where all six signs are equal and nonzero.
    jacobian_nonfinite: ArrayLike
        ``(N,)`` bool, returned only with ``return_details``. True where some
        sign is 0. Always a subset of ``~parity_ok``.
    """
    if signs.shape[0] == 0:
        empty = backend.zeros((0,), dtype=backend.bool, device=backend.device(signs))
        return (empty, empty) if return_details else empty
    jacobian_nonfinite = backend.any(signs == 0, dim=1)
    same_sign = backend.all(signs > 0, dim=1) | backend.all(signs < 0, dim=1)
    parity_ok = ~jacobian_nonfinite & same_sign
    if return_details:
        return parity_ok, jacobian_nonfinite
    return parity_ok


def approximate_criterion(
    beta_v, beta_m, classes, level, h0, min_img_sep, pinv0, compose
):
    """
    Every test of :func:`evaluate_criterion` but the Jacobian's.

    Child parity (:func:`parity_from_children`), the deviation test
    (:func:`converged_from_deviation`) and finiteness, from the six mapped
    samples alone. Split out so that a caller holding a cache of Jacobian
    values -- :func:`~caustics.lenses.func.adaptive.refinement.refine` -- can
    pick the rows the Jacobian test needs (:func:`jacobian_rows`), find their
    signs, and finish with :func:`apply_jacobian_signs`.

    Parameters
    ----------
    beta_v, beta_m, classes, level, h0, min_img_sep, pinv0, compose:
        As for :func:`evaluate_criterion`.

    Returns
    -------
    status: ArrayLike
        ``(n,)`` int64 bitmask of ``LEAF_APPROX_PARITY_UNRESOLVED``,
        ``LEAF_CONVERGENCE_FAILED`` and ``LEAF_RAYTRACE_NONFINITE``.
    child_ok: ArrayLike
        ``(n,)`` bool, child parity.
    s: ArrayLike
        ``(n,)`` float64, as :func:`evaluate_criterion` returns it.
    """
    Q = child_shape_matrices(beta_v, beta_m)
    child_ok = parity_from_children(Q)

    A = Q @ pinv0[compose[classes]]
    s = backend.min(sigma_min_2x2(A), dim=1) * (2.0 ** (level + 1)) / h0

    r = midpoint_deviation(beta_v, beta_m)
    deviation_ok = converged_from_deviation(r, s, min_img_sep)

    finite_samples = backend.all(backend.isfinite(beta_v), dim=(1, 2)) & backend.all(
        backend.isfinite(beta_m), dim=(1, 2)
    )

    status = (
        (backend.long(~child_ok) * LEAF_APPROX_PARITY_UNRESOLVED)
        | (backend.long(finite_samples & ~deviation_ok) * LEAF_CONVERGENCE_FAILED)
        | (backend.long(~finite_samples) * LEAF_RAYTRACE_NONFINITE)
    )
    return status, child_ok, s


def jacobian_rows(status, force_jacobian=False):
    """
    Rows the Jacobian parity test runs on, given :func:`approximate_criterion`'s ``status``.

    Lazily, only the rows that would otherwise converge, ``status ==
    LEAF_CONVERGED``, which keeps the Jacobian off the split path. Forced,
    every row whose samples are finite -- every row without
    ``LEAF_RAYTRACE_NONFINITE`` -- for ``max_level``, where ``status`` is the
    leaf's final record. A row with a non-finite sample is never tested,
    forced or not.

    Parameters
    ----------
    status: ArrayLike
        ``(n,)`` int64, from :func:`approximate_criterion`.
    force_jacobian: bool

    Returns
    -------
    ArrayLike
        ``(k,)`` int64 indices into ``status``, ascending.
    """
    if force_jacobian:
        return backend.flatnonzero((status & LEAF_RAYTRACE_NONFINITE) == 0)
    return backend.flatnonzero(status == LEAF_CONVERGED)


def apply_jacobian_signs(status, child_ok, rows, signs):
    """
    Join the Jacobian parity test's verdict on ``rows`` to the other tests'.

    The test only adds flags: ``LEAF_JACOBIAN_PARITY_UNRESOLVED`` where the
    six signs are nonzero but mixed, ``LEAF_JACOBIAN_NONFINITE`` where some
    sign is 0 (:func:`parity_from_signs`). So it never rescues a row another
    test failed. ``status`` and ``child_ok`` are copied, never written.

    Parameters
    ----------
    status: ArrayLike
        ``(n,)`` int64, from :func:`approximate_criterion`.
    child_ok: ArrayLike
        ``(n,)`` bool, from :func:`approximate_criterion`.
    rows: ArrayLike
        ``(k,)`` int64, from :func:`jacobian_rows`.
    signs: ArrayLike
        ``(k, 6)`` int64,
        :func:`~caustics.lenses.func.adaptive.geometry.jacobian_signs` at the
        six samples of each of ``rows``.

    Returns
    -------
    keep: ArrayLike
        ``(n,)`` bool, ``status == LEAF_CONVERGED``.
    parity_ok: ArrayLike
        ``(n,)`` bool: on ``rows``, child parity and the Jacobian's both; on
        any other row, child parity alone.
    status: ArrayLike
        ``(n,)`` int64, with the Jacobian's flags OR-ed in on ``rows``.
    """
    jacobian_ok, jacobian_nonfinite = parity_from_signs(signs, return_details=True)
    jacobian_status = (
        backend.long(~jacobian_ok & ~jacobian_nonfinite)
        * LEAF_JACOBIAN_PARITY_UNRESOLVED
    ) | (backend.long(jacobian_nonfinite) * LEAF_JACOBIAN_NONFINITE)
    status = backend.fill_at_indices(
        backend.copy(status), rows, status[rows] | jacobian_status
    )
    parity_ok = backend.fill_at_indices(
        backend.copy(child_ok), rows, child_ok[rows] & jacobian_ok
    )
    return status == LEAF_CONVERGED, parity_ok, status


def evaluate_criterion(
    jacobian_fn,
    theta_v,
    theta_m,
    beta_v,
    beta_m,
    classes,
    level,
    h0,
    min_img_sep,
    pinv0,
    compose,
    force_jacobian=False,
    jacobian=None,
):
    """
    Steps 3 to 7 of the refinement criterion, vectorized over triangles.

    Every failed test is recorded, not just the first: ``status`` is a bitmask
    of the ``LEAF_*`` flags, and a triangle converges (``keep``) only where it
    is exactly ``LEAF_CONVERGED``, i.e. where all of the following hold.

    - All six samples in ``beta_v`` and ``beta_m`` are finite; otherwise
      ``LEAF_RAYTRACE_NONFINITE``.
    - The four red-split children agree on ``sign(det Q_k)`` (step 4,
      :func:`parity_from_children`); otherwise
      ``LEAF_APPROX_PARITY_UNRESOLVED``.
    - Every mapped midpoint lies within ``s * min_img_sep`` of its affine
      prediction (step 7, :func:`converged_from_deviation`), which catches
      curvature; otherwise ``LEAF_CONVERGENCE_FAILED``. Only set on a triangle
      whose samples are finite.
    - The lens Jacobian is finite, non-singular and of one sign at all six
      lens-plane samples (the Jacobian parity test, :func:`jacobian_parity_ok`
      or :func:`parity_from_jacobians` on precomputed values); otherwise
      ``LEAF_JACOBIAN_PARITY_UNRESOLVED``, or ``LEAF_JACOBIAN_NONFINITE`` where
      some ``A`` is non-finite or singular.

    The two parity tests complement each other rather than duplicate. Child
    parity comes free with the six samples, but sees a fold only through the
    source-plane shapes of the children. The Jacobian test is exact at its six
    points, but costs a ``jacobian_fn`` call, so it runs lazily: only on the
    triangles that pass every other test and would otherwise converge. A
    triangle that has already failed splits whatever the Jacobian says, so
    this keeps ``jacobian_fn`` off the split path entirely. ``force_jacobian``
    runs it on every triangle whose samples are finite instead -- for
    ``max_level``, where nothing splits and ``status`` is the leaf's final
    record. A triangle with a non-finite sample is never passed to
    ``jacobian_fn``, forced or not.

    The Jacobian can only add flags. It never clears one the other tests set,
    so it cannot rescue a triangle into convergence.

    It is :func:`approximate_criterion`, :func:`jacobian_rows` and
    :func:`apply_jacobian_signs` in turn, with the signs read off
    ``jacobian_fn`` or ``jacobian``.

    Parameters
    ----------
    jacobian_fn: Callable[[ArrayLike, ArrayLike], ArrayLike]
        ``jacobian_fn(x, y) -> (K, 2, 2)``, see :func:`jacobian_parity_ok`.
    theta_v: ArrayLike
        Lens-plane vertices, shape ``(n, 3, 2)``.

        *Unit: arcsec*

    theta_m: ArrayLike
        Lens-plane midpoints ``m1, m2, m3``, shape ``(n, 3, 2)``.

        *Unit: arcsec*

    beta_v: ndarray
        Source-plane vertices, shape ``(n, 3, 2)``.

        *Unit: arcsec*

    beta_m: ndarray
        Source-plane midpoints ``m1, m2, m3``, shape ``(n, 3, 2)``.

        *Unit: arcsec*

    classes: ndarray
        Orientation class of each triangle, shape ``(n,)`` int64.
    level: int
        Refinement level of the triangles.
    h0: float
        Level-0 cell size ``fov / init_res``.

        *Unit: arcsec*

    min_img_sep: float
        Lens-plane tolerance.

        *Unit: arcsec*

    pinv0: ndarray
        ``PINV0`` from :func:`child_matrix_tables`.
    compose: ndarray
        ``COMPOSE`` from :func:`child_matrix_tables`.
    force_jacobian: bool
        Evaluate the Jacobian on every triangle with finite samples, not just
        on those that would otherwise converge.
    jacobian: Optional[ArrayLike]
        Precomputed Jacobians, shape ``(n, 6, 2, 2)``, at each triangle's
        ``theta_1, theta_2, theta_3, m_1, m_2, m_3``. When given, the
        triangles chosen for the Jacobian test -- the same ones as without it
        -- are tested on these values through :func:`parity_from_jacobians`,
        and ``jacobian_fn`` is never called. ``theta_v`` and ``theta_m`` are
        not read on this path.

    Returns
    -------
    keep: ndarray
        ``(n,)`` bool, ``status == LEAF_CONVERGED``: True where the triangle is
        converged and terminal.
    parity_ok: ndarray
        ``(n,)`` bool, the parity verdict the refinement counters read. On a
        triangle the Jacobian was evaluated on, True where child parity and
        the Jacobian parity test (:func:`jacobian_parity_ok`, or
        :func:`parity_from_jacobians` on precomputed values) both pass; on
        any other, child parity alone.
    s: ndarray
        ``(n,)`` float64, ``min_k sigma_min(A_k)``. Exactly ``0.0`` is legal and
        expected near a critical curve; it forces the split, and the size floor
        terminates the descent.
    status: ndarray
        ``(n,)`` int64 bitmask of the ``LEAF_*`` flags above, ``LEAF_CONVERGED``
        where no test failed. A Jacobian flag can appear only on a triangle the
        Jacobian was evaluated on.
    """

    status, child_ok, s = approximate_criterion(
        beta_v, beta_m, classes, level, h0, min_img_sep, pinv0, compose
    )
    rows = jacobian_rows(status, force_jacobian)
    if jacobian is None:
        signs = _jacobian_signs_at(jacobian_fn, theta_v[rows], theta_m[rows])
    else:
        signs = jacobian_signs(jacobian[rows].reshape(-1, 2, 2)).reshape(-1, 6)
    keep, parity_ok, status = apply_jacobian_signs(status, child_ok, rows, signs)
    return keep, parity_ok, s, status


def lens_status(beta6, det6, cls, level, h0, min_img_sep):
    """
    Why each triangle's affine model of the lens map is not trusted, as a ``LEAF_*`` bitmask.

    ``LEAF_CONVERGED`` -- zero -- only where every one of these holds:

    - every image is finite; otherwise ``LEAF_RAYTRACE_NONFINITE`` is the
      row's only flag;
    - the four red-split children's source-plane edge matrices share a
      sign of determinant (``LEAF_APPROX_PARITY_UNRESOLVED``);
    - every mapped midpoint lies within ``s * min_img_sep`` of its affine
      prediction, ``s`` the smallest singular value of the children's
      affine maps (``LEAF_CONVERGENCE_FAILED``);
    - ``det A`` is finite and nonzero at all six samples
      (``LEAF_JACOBIAN_NONFINITE``) and of one sign
      (``LEAF_JACOBIAN_PARITY_UNRESOLVED``).

    Parameters
    ----------
    beta6: ArrayLike
        ``(n, 6, 2)`` images at ``theta_1, theta_2, theta_3, m_1, m_2, m_3``,
        ``m_i`` opposite ``theta_i``.

        *Unit: arcsec*
    det6: ArrayLike
        ``(n, 6)`` ``det A`` at the same samples.
    cls: ArrayLike
        ``(n,)`` int64 orientation classes.
    level: ArrayLike
        ``(n,)`` int64 refinement levels.
    h0: float
        Level-0 cell size.

        *Unit: arcsec*
    min_img_sep: float
        Lens-plane tolerance.

        *Unit: arcsec*

    Returns
    -------
    ArrayLike
        ``(n,)`` int64.
    """
    beta_v, beta_m = beta6[:, :3], beta6[:, 3:]
    Q = child_shape_matrices(beta_v, beta_m)
    s = (
        backend.min(sigma_min_2x2(Q @ PINV0[COMPOSE[cls]]), dim=1)
        * backend.to(2 ** (level + 1), dtype=backend.float64)
        / h0
    )
    deviation_ok = converged_from_deviation(
        midpoint_deviation(beta_v, beta_m), s, min_img_sep
    )
    det_ok = backend.all(backend.isfinite(det6) & (det6 != 0), dim=1)
    one_sign = backend.all(det6 > 0, dim=1) | backend.all(det6 < 0, dim=1)
    status = (
        backend.long(~parity_from_children(Q)) * LEAF_APPROX_PARITY_UNRESOLVED
        | backend.long(~deviation_ok) * LEAF_CONVERGENCE_FAILED
        | backend.long(~det_ok) * LEAF_JACOBIAN_NONFINITE
        | backend.long(det_ok & ~one_sign) * LEAF_JACOBIAN_PARITY_UNRESOLVED
    )
    finite = backend.all(backend.isfinite(beta6), dim=(1, 2))
    return backend.where(finite, status, LEAF_RAYTRACE_NONFINITE)
