"""
The lens mesh's refinement criterion, and the ``LEAF_*`` status flags it records.

:func:`lens_status` decides, vectorized over triangles, whether a triangle's
affine model of the lens map is accurate to ``min_img_sep`` and whether a
critical curve runs between its samples.
"""

from ....backend_obj import backend
from .geometry import (
    CHILD_VERTEX_INDICES,
    COMPOSE,
    PINV0,
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
    "lens_status",
)


# Why a leaf's affine model is not trusted, as a bitmask with one bit per
# failed test; `LEAF_CONVERGED` (zero) is the only status that is trusted.
# Test a flag with `(status & FLAG) != 0`. Only `max_level` leaves store a
# nonzero status: below it a failure means a split, and a closure triangle
# reads its origin's status.
#
# - `LEAF_CONVERGENCE_FAILED`: a mapped midpoint strays from its affine
#   prediction by more than `min_img_sep` allows.
# - `LEAF_APPROX_PARITY_UNRESOLVED`: the four red-split children disagree on
#   their orientation in the source plane.
# - `LEAF_JACOBIAN_PARITY_UNRESOLVED`: `det A` changes sign between the six
#   samples.
# - `LEAF_RAYTRACE_NONFINITE`: some image is non-finite; then it is the only
#   flag.
# - `LEAF_JACOBIAN_NONFINITE`: some `det A` is non-finite or exactly zero.
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
