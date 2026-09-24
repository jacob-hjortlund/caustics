"""
Pure kernels for the adaptive lens-plane mesh.

This module has two clearly separated halves.

**Host-side build kernels** operate on ``backend`` float64 arrays during the
host-side mesh build. Keeping them in one numerical world is what makes the
``NaN`` semantics of :func:`sigma_min_2x2` and :func:`converged_from_deviation`
verifiable.
:func:`build_adaptive_mesh` runs them as four stages -- seed, refine,
balance, freeze -- and :func:`extend_adaptive_mesh` runs the same stages from
a built mesh, growing it to a larger fov without repeating its lens calls.

**Query kernels** operate on ``backend`` arrays. :func:`mesh_query` and
:func:`mesh_seeds` return candidate regions and Newton seeds;
:func:`mesh_forward_raytrace` goes on to return images, either refining
those seeds to machine precision or deduplicating them as they stand,
depending on its ``method`` argument. The critical curves and caustics
through the mesh's :class:`CriticalBand` are traced by
:mod:`~caustics.lenses.func.adaptive_critical`.
"""

import math
from typing import Any, Callable, NamedTuple, Optional, Tuple
from warnings import warn

from ...backend_obj import ArrayLike, backend
from ...utils import batch_lm

__all__ = (
    "CHILD_VERTEX_INDICES",
    "ROOT_SHAPES",
    "shape_matrix",
    "child_matrix_tables",
    "affine_from_triangles",
    "sigma_min_2x2",
    "midpoint_deviation",
    "converged_from_deviation",
    "child_shape_matrices",
    "parity_from_children",
    "parity_from_jacobians",
    "evaluate_criterion",
    "triangle_weights",
    "contains",
    "sanitize_bary",
    "MAX_KEY",
    "Lattice",
    "make_lattice",
    "extend_lattice",
    "lattice_key",
    "lattice_ij_from_key",
    "lattice_xy",
    "lattice_on_boundary",
    "depth_floor",
    "check_lattice_keys",
    "validate_build_args",
    "VertexCache",
    "empty_cache",
    "cache_size",
    "cache_lookup",
    "cache_missing",
    "cache_insert",
    "empty_active",
    "active_add_slots",
    "active_contains_slots",
    "active_contains",
    "LEAF_CONVERGED",
    "LEAF_CONVERGENCE_FAILED",
    "LEAF_APPROX_PARITY_UNRESOLVED",
    "LEAF_JACOBIAN_PARITY_UNRESOLVED",
    "LEAF_RAYTRACE_NONFINITE",
    "LEAF_JACOBIAN_NONFINITE",
    "LeafStore",
    "empty_store",
    "store_add",
    "store_remove",
    "store_compact",
    "initial_triangles",
    "ring_triangles",
    "midpoint_ij",
    "red_split",
    "edge_quarter_keys",
    "find_unbalanced",
    "make_raytrace",
    "trace_keys",
    "evaluate",
    "CriticalBand",
    "empty_band",
    "sample_jacobians",
    "band_from_samples",
    "band_leaf_index",
    "band_sample_keys",
    "merge_bands",
    "force_split",
    "refine",
    "balance",
    "canonical_order",
    "min_angle",
    "close",
    "invalidate_nonfinite_origins",
    "MeshIndex",
    "build_index",
    "AdaptiveMesh",
    "warn_depth_limited",
    "warn_cancellation_floor",
    "freeze",
    "build_adaptive_mesh",
    "seed_from_mesh",
    "extend_adaptive_mesh",
    "mesh_query",
    "mesh_seeds",
    "dedup_block_group",
    "dedup_representatives",
    "METHODS",
    "mesh_forward_raytrace",
)

# ---------------------------------------------------------------------------
# Host-side build kernels (backend, float64)
# ---------------------------------------------------------------------------

# Indices into the stacked six-point array [theta1, theta2, theta3, m1, m2, m3]
# giving the four children in the orientation-preserving order
#   C_1 = (t1, m3, m2)  C_2 = (t2, m1, m3)  C_3 = (t3, m2, m1)  C_4 = (m1, m2, m3)
# with m_i the midpoint opposite theta_i.
CHILD_VERTEX_INDICES = ((0, 5, 4), (1, 3, 5), (2, 4, 3), (3, 4, 5))

# The two triangles splitting the unit cell along the (0,0)-(1,1) diagonal, both
# positively oriented. Note ``func/base.py`` builds its pair with *opposite*
# handedness; this module fixes that so orientation is globally consistent.
ROOT_SHAPES = (
    ((0, 0), (1, 1), (0, 1)),
    ((0, 0), (1, 0), (1, 1)),
)

# `CHILD_VERTEX_INDICES` as a single backend int64 array, built once, so
# `red_split` can gather all four children with one fancy-indexing op rather
# than three separate per-axis index lists.
_CHILD_VERTEX_INDEX_TABLE = backend.as_array(CHILD_VERTEX_INDICES, dtype=backend.int64)


def shape_matrix(tri):
    """
    Edge matrix ``P = [v1 - v0 | v2 - v0]`` of a triangle.

    Parameters
    ----------
    tri: ndarray
        Triangle vertices, shape ``(..., 3, 2)``.

        *Unit: arcsec*

    Returns
    -------
    ndarray
        Shape ``(..., 2, 2)``, the two edge vectors as columns.

        *Unit: arcsec*
    """
    return backend.stack(
        (tri[..., 1, :] - tri[..., 0, :], tri[..., 2, :] - tri[..., 0, :]), dim=-1
    )


def affine_from_triangles(p, q):
    """
    Linear part of the affine map reproducing three vertex correspondences.

    Invariant under any relabelling applied simultaneously to both triangles,
    which is what makes the parity test immune to vertex ordering.

    Parameters
    ----------
    p: ndarray
        Lens-plane triangle, shape ``(..., 3, 2)``.

        *Unit: arcsec*

    q: ndarray
        Source-plane triangle, shape ``(..., 3, 2)``.

        *Unit: arcsec*

    Returns
    -------
    ndarray
        Shape ``(..., 2, 2)``.
    """
    return shape_matrix(q) @ backend.linalg.inv(shape_matrix(p))


def _derive_child_matrices():
    """Recover ``M_k`` from the child ordering, where ``P_k = (1/2) P M_k``."""
    parent = backend.as_array(
        [[0.0, 0.0], [3.0, 2.0], [-1.0, 5.0]], dtype=backend.float64
    )
    t1, t2, t3 = parent
    six = backend.stack([t1, t2, t3, (t2 + t3) / 2, (t3 + t1) / 2, (t1 + t2) / 2])
    P = shape_matrix(parent)
    rows = []
    for idx in CHILD_VERTEX_INDICES:
        Pk = shape_matrix(six[list(idx), :])
        exact = 2.0 * (backend.linalg.inv(P) @ Pk)
        rounded = backend.round(exact)
        if not bool(backend.allclose(exact, rounded, atol=1e-9)):
            raise AssertionError("child matrices are not integral")
        rows.append(backend.long(rounded))
    return backend.stack(rows)


def child_matrix_tables() -> (
    Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike, ArrayLike]
):
    """
    Fixed tables for red refinement.

    Red refinement gives ``P_k = (1/2) P M_k`` with ``M_k`` a fixed integer matrix
    per child index, so by induction ``P`` at level ``d`` is ``2**-d * h0 * R @ G[c]``
    for one of finitely many group elements ``G[c]``. The group generated by the four
    ``M_k`` is ``{+-I, +-M_2, +-M_3}``, of order six, and both root shapes lie in a
    single orbit -- so the inverse table has exactly six entries, not one per level.

    Returns
    -------
    M: ndarray
        ``(4, 2, 2)`` int64 child matrices; every ``det M_k == +1``.
    G: ndarray
        ``(6, 2, 2)`` int64 group elements in canonical order.
    COMPOSE: ndarray
        ``(6, 4)`` int64 with ``G[COMPOSE[c, k]] == G[c] @ M[k]``.
    PINV0: ndarray
        ``(6, 2, 2)`` float64 with ``PINV0[c] == inv(R @ G[c])``, ``R`` the class-0
        root shape. A triangle of class ``c`` at level ``d`` has
        ``inv(P_child_k) == (2**(d + 1) / h0) * PINV0[COMPOSE[c, k]]``.
    ROOT_CLASS: ndarray
        ``(2,)`` int64 class index of each entry of :data:`ROOT_SHAPES`.
    """
    M = _derive_child_matrices()
    eye = backend.eye(2, dtype=backend.int64)
    G = backend.stack([eye, M[1], M[2], -eye, -M[1], -M[2]])

    lookup = {tuple(backend.to_numpy(G[c]).reshape(-1).tolist()): c for c in range(6)}
    compose_rows = []
    for c in range(6):
        row = []
        for k in range(4):
            product = G[c] @ M[k]
            key = tuple(backend.to_numpy(product).reshape(-1).tolist())
            if key not in lookup:
                raise AssertionError("child matrix group is not closed")
            row.append(lookup[key])
        compose_rows.append(row)
    COMPOSE = backend.as_array(compose_rows, dtype=backend.int64)

    R = shape_matrix(backend.as_array(ROOT_SHAPES[0], dtype=backend.float64))
    PINV0 = backend.stack(
        [
            backend.linalg.inv(R @ backend.to(G[c], dtype=backend.float64))
            for c in range(6)
        ]
    )

    root_class_values = []
    for s, shape in enumerate(ROOT_SHAPES):
        target = shape_matrix(backend.as_array(shape, dtype=backend.float64))
        match = None
        for c in range(6):
            if bool(
                backend.allclose(R @ backend.to(G[c], dtype=backend.float64), target)
            ):
                match = c
                break
        if match is None:  # pragma: no cover - guarded by test_group_tables_close_...
            raise AssertionError(f"root shape {s} is not in the orbit of R")
        root_class_values.append(match)
    ROOT_CLASS = backend.as_array(root_class_values, dtype=backend.int64)
    return M, G, COMPOSE, PINV0, ROOT_CLASS


def sigma_min_2x2(A):
    """
    Smallest singular value of a batch of 2x2 matrices, in closed form.

    Pulling a source-plane displacement back to the lens plane gives
    ``||dtheta|| <= ||dbeta|| / sigma_min(A)``, so ``sigma_min`` is the correct
    anisotropic contraction scale. ``det A = sigma_min * sigma_max`` conflates the
    two directions and is wrong near folds, where ``sigma_min -> 0`` while
    ``sigma_max`` stays order one.

    With ``F = ||A||_F**2`` and ``D = |det A|``, ``(sigma_max +- sigma_min)**2 =
    F +- 2D``. ``F - 2D >= 0`` holds in exact arithmetic but **not** in floating
    point -- for a near-conformal ``A`` it goes negative by an ulp, and the
    unguarded ``sqrt`` then returns ``NaN`` for the near-circularly-symmetric core
    of essentially every lens model. Hence the clip.

    ``sigma_min`` is returned as ``D / sigma_max`` rather than
    ``(sqrt(F + 2D) - sqrt(F - 2D)) / 2`` because the former is stable in the
    near-degenerate limit, which is exactly the near-caustic regime that drives
    refinement.

    The ``sigma_max == 0`` branch is exact, not a tolerance: ``sigma_max == 0`` iff
    ``A == 0``, whose smallest singular value is exactly zero. Do not replace it
    with ``max(sigma_min, eps)`` -- both ``backend.maximum`` and ``backend.clamp``
    propagate ``NaN``, so a floor would fail open.

    Parameters
    ----------
    A: ndarray
        Shape ``(..., 2, 2)``, float64.

    Returns
    -------
    ndarray
        Shape ``(...)``. Exactly ``0.0`` where ``A == 0``; ``NaN`` where ``A`` is
        non-finite, so that step 7 fails closed.
    """
    F = backend.sum(A**2, dim=(-2, -1))
    D = backend.abs(A[..., 0, 0] * A[..., 1, 1] - A[..., 0, 1] * A[..., 1, 0])
    sigma_max = (
        backend.sqrt(F + 2 * D) + backend.sqrt(backend.clamp(F - 2 * D, 0.0, None))
    ) / 2
    zero = sigma_max == 0
    # Double `where` keeps the division away from 0/0 without masking a NaN input:
    # a NaN sigma_max fails the `== 0` test, so NaN reaches the output.
    return backend.where(zero, 0.0, D / backend.where(zero, 1.0, sigma_max))


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


def quadratic_vertex_parity_ok(beta_v, beta_m):
    """Check parity agreement at the quadratic interpolant's vertices.

    Parameters
    ----------
    beta_v : ndarray, shape (n, 3, 2)
        Mapped triangle vertices.
    beta_m : ndarray, shape (n, 3, 2)
        Mapped edge midpoints, each opposite its corresponding vertex.

    Returns
    -------
    ndarray, shape (n,)
        True when all three estimated vertex determinants are finite
        and have the same strictly nonzero sign.

    Notes
    -----
    Uses the existing six samples; no additional raytracing.
    This checks the interpolant at vertices, not the entire true mapping.
    """
    # At each vertex, estimate derivatives toward the next and
    # previous vertices in cyclic order. Using differences avoids
    # combining large absolute source-plane coordinates directly.
    d_next = 4.0 * (beta_m[:, [2, 0, 1]] - beta_v) - (beta_v[:, [1, 2, 0]] - beta_v)
    d_prev = 4.0 * (beta_m[:, [1, 2, 0]] - beta_v) - (beta_v[:, [2, 0, 1]] - beta_v)

    det = d_next[..., 0] * d_prev[..., 1] - d_next[..., 1] * d_prev[..., 0]

    # Cyclic lens-plane edge pairs have the same determinant.
    # Its common factor can be omitted when checking sign agreement.
    return backend.all(backend.isfinite(det), dim=1) & (
        backend.all(det > 0, dim=1) | backend.all(det < 0, dim=1)
    )


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
    :func:`parity_from_jacobians` on the result.

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
    if theta_v.shape != theta_m.shape or theta_v.shape[1:] != (3, 2):
        raise ValueError("theta_v and theta_m must both have shape (N, 3, 2)")

    n = theta_v.shape[0]
    if n == 0:
        empty = backend.zeros((0,), dtype=backend.bool, device=backend.device(theta_v))
        return (empty, empty) if return_details else empty

    theta = backend.concatenate((theta_v, theta_m), dim=1).reshape(-1, 2)

    J = jacobian_fn(theta[:, 0], theta[:, 1])
    if J.shape != (6 * n, 2, 2):
        raise ValueError("jacobian_fn must return shape (6 * N, 2, 2)")

    return parity_from_jacobians(J.reshape(n, 6, 2, 2), return_details=return_details)


def parity_from_jacobians(J, *, return_details=False):
    """
    True where ``sign(det A)`` is one strict sign at all six samples of a triangle.

    The arithmetic half of :func:`jacobian_parity_ok`, on Jacobians already
    evaluated, so a caller holding them makes no second call. The
    ``max_level`` pass of :func:`refine` is that caller: it evaluates each
    lattice point once, however many triangles share it, and keeps the
    values for the critical band.

    The sign is read off a row-scaled copy of ``A``. Dividing a row by a
    positive number scales ``det A`` by a positive factor, so the sign is
    unchanged, while the scaled entries are at most one in magnitude -- so a
    Jacobian with huge entries cannot overflow the determinant to ``inf``,
    and one with tiny entries cannot underflow it to an exact zero that would
    read as singular. A non-finite ``A`` is swapped for the identity before
    any arithmetic, so it never reaches the determinant; the finiteness mask
    carries its failure instead.

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
    if n == 0:
        empty = backend.zeros((0,), dtype=backend.bool, device=backend.device(J))
        return (empty, empty) if return_details else empty

    J = J.reshape(-1, 2, 2)
    finite_J = backend.all(backend.isfinite(J), dim=(-2, -1))

    # Keep nonfinite matrices out of subsequent arithmetic.
    # Their failure flags are retained through finite_J.
    identity = backend.to(backend.eye(2), dtype=J.dtype, device=backend.device(J))
    safe_J = backend.where(finite_J[:, None, None], J, identity)

    # Positive row scaling preserves determinant sign while avoiding
    # overflow/underflow caused by the overall matrix scale.
    row_scale = backend.max(backend.abs(safe_J), dim=-1)
    scaled = safe_J / backend.where(row_scale > 0, row_scale, 1.0)[..., None]

    det = scaled[:, 0, 0] * scaled[:, 1, 1] - scaled[:, 0, 1] * scaled[:, 1, 0]

    usable = finite_J & backend.isfinite(det) & (det != 0)
    jacobian_nonfinite = ~backend.all(usable.reshape(n, 6), dim=1)

    det = det.reshape(n, 6)
    same_sign = backend.all(det > 0, dim=1) | backend.all(det < 0, dim=1)
    parity_ok = ~jacobian_nonfinite & same_sign

    if return_details:
        return parity_ok, jacobian_nonfinite
    return parity_ok


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

    Q = child_shape_matrices(beta_v, beta_m)
    child_ok = parity_from_children(Q)

    A = Q @ pinv0[compose[classes]]
    s = backend.min(sigma_min_2x2(A), dim=1) * (2.0 ** (level + 1)) / h0

    r = midpoint_deviation(beta_v, beta_m)
    deviation_ok = converged_from_deviation(r, s, min_img_sep)

    finite_samples = backend.all(backend.isfinite(beta_v), dim=(1, 2)) & backend.all(
        backend.isfinite(beta_m), dim=(1, 2)
    )

    # Initial status from mapping samples and the approximate tests.
    status = (
        (backend.long(~child_ok) * LEAF_APPROX_PARITY_UNRESOLVED)
        | (backend.long(finite_samples & ~deviation_ok) * LEAF_CONVERGENCE_FAILED)
        | (backend.long(~finite_samples) * LEAF_RAYTRACE_NONFINITE)
    )

    # Normally check Jacobians only for candidates that would otherwise
    # converge. At max_level, check every finite triangle.
    candidate = status == LEAF_CONVERGED
    rows = backend.flatnonzero(finite_samples & (candidate | force_jacobian))

    if jacobian is None:
        jacobian_ok, jacobian_nonfinite = jacobian_parity_ok(
            jacobian_fn,
            theta_v[rows],
            theta_m[rows],
            return_details=True,
        )
    else:
        jacobian_ok, jacobian_nonfinite = parity_from_jacobians(
            jacobian[rows], return_details=True
        )

    jacobian_status = (
        backend.long(~jacobian_ok & ~jacobian_nonfinite)
        * LEAF_JACOBIAN_PARITY_UNRESOLVED
    ) | (backend.long(jacobian_nonfinite) * LEAF_JACOBIAN_NONFINITE)

    # Preserve existing failures; Jacobians can add failures but cannot
    # rescue a failed child-parity or deviation test.
    status = backend.fill_at_indices(status, rows, status[rows] | jacobian_status)

    # Used by refinement counters. Unchecked rows retain the child result;
    # checked rows must pass both child parity and Jacobian parity.
    parity_ok = backend.fill_at_indices(
        backend.copy(child_ok),
        rows,
        child_ok[rows] & jacobian_ok,
    )

    keep = status == LEAF_CONVERGED
    return keep, parity_ok, s, status


# ---------------------------------------------------------------------------
# Query kernels (backend-dispatched)
# ---------------------------------------------------------------------------


def triangle_weights(tri, beta) -> ArrayLike:
    """
    The three cross products that give containment and barycentric coordinates.

    With ``q1, q2, q3`` the triangle's mapped vertices, ``w_i`` is the cross
    product of the two edges of the sub-triangle opposite vertex ``i``. Then
    ``w1 + w2 + w3 == d`` identically, where ``d`` is twice the signed area, so
    ``w / d`` sums to one by construction and no matrix inverse is needed.

    Because a mesh vertex is stored once and referenced by index, two leaves
    sharing an edge form these products from bit-identical operands in opposite
    order. IEEE multiplication is commutative, so their weights on that edge are
    **exactly** negated and a query point can never fall through the seam between
    them.

    Parameters
    ----------
    tri: ArrayLike
        Source-plane triangle vertices, shape ``(T, 3, 2)``.

        *Unit: arcsec*

    beta: ArrayLike
        Query points, one per candidate, shape ``(T, 2)``.

        *Unit: arcsec*

    Returns
    -------
    ArrayLike
        Shape ``(T, 3)``.
    """
    q1 = tri[:, 0, :] - beta
    q2 = tri[:, 1, :] - beta
    q3 = tri[:, 2, :] - beta
    return backend.stack(
        (
            q2[:, 0] * q3[:, 1] - q2[:, 1] * q3[:, 0],
            q3[:, 0] * q1[:, 1] - q3[:, 1] * q1[:, 0],
            q1[:, 0] * q2[:, 1] - q1[:, 1] * q2[:, 0],
        ),
        dim=-1,
    )


def contains(w) -> ArrayLike:
    """
    Containment test: all three weights share a sign, zeros counting as inside.

    Source-plane images flip handedness across critical curves by design, so a
    fixed positive convention would be wrong on half the mesh. Since
    ``w1 + w2 + w3 == d``, "all three share a sign" implies that sign is
    ``sign(d)`` whenever ``d != 0`` -- so this is equivalent to testing against the
    triangle's own ``d``, while also being well defined when ``d`` is *exactly*
    zero. That case is reachable: three source-plane vertices can be exactly
    collinear in floating point, and a ``kappa = 1`` sheet maps every leaf to a
    point. Testing against ``sign(d) == 0`` would make every such leaf a hit for
    every query.

    Zeros counting as inside means a point on a shared edge returns both leaves.
    That is intended, and is the main reason candidate count is not image
    multiplicity.

    Parameters
    ----------
    w: ArrayLike
        Weights from :func:`triangle_weights`, shape ``(T, 3)``.

    Returns
    -------
    ArrayLike
        Shape ``(T,)`` bool.
    """
    nonneg = (w[..., 0] >= 0) & (w[..., 1] >= 0) & (w[..., 2] >= 0)
    nonpos = (w[..., 0] <= 0) & (w[..., 1] <= 0) & (w[..., 2] <= 0)
    return nonneg | nonpos


def sanitize_bary(w, d) -> ArrayLike:
    """
    Normalize weights to barycentric coordinates, clamping then falling back.

    ``d`` goes to zero on leaves straddling a critical curve, so ``w / d`` there is
    a ratio of two quantities at the roundoff floor -- unbounded, possibly outside
    the triangle, or ``NaN``. Two regimes, handled in order.

    The clip handles ordinary roundoff: a hit passed containment, so
    ``bary in [0, 1]**3`` holds mathematically and small excursions are numerical
    only. The centroid branch handles genuine breakdown, where ``w`` and ``d`` are
    both noise and the clipped components carry no information; it is also the only
    defined answer when they clip to all-zero. ``isfinite`` is checked *after* the
    clip, because clipping propagates ``NaN``.

    The centroid is a bounded fallback, not a guess. A leaf with ``d`` near zero
    reached the size floor, so its longest edge is at most ``min_img_sep`` and the
    centroid is within ``(2/3) * l_max`` of every point in it.

    Parameters
    ----------
    w: ArrayLike
        Weights of the hits, shape ``(K, 3)``.
    d: ArrayLike
        Twice the signed source-plane area of each hit leaf, shape ``(K,)``.

    Returns
    -------
    ArrayLike
        Shape ``(K, 3)``, guaranteed to lie in the simplex.
    """
    bary = backend.clamp(w / backend.unsqueeze(d, -1), 0.0, 1.0)
    total = bary[..., 0] + bary[..., 1] + bary[..., 2]
    ok = (
        (total > 0)
        & backend.isfinite(bary[..., 0])
        & backend.isfinite(bary[..., 1])
        & backend.isfinite(bary[..., 2])
    )
    safe = backend.where(ok, total, backend.ones_like(total))
    normed = bary / backend.unsqueeze(safe, -1)
    third = backend.ones_like(bary) / 3
    return backend.where(backend.unsqueeze(ok, -1), normed, third)


# ---------------------------------------------------------------------------
# Dyadic lattice and build-argument validation
# ---------------------------------------------------------------------------

MAX_KEY = 2**63 - 1


def depth_floor(fov, init_res, min_img_sep) -> int:
    """
    Level at which the longest leaf edge first falls to ``min_img_sep``.

    Red refinement makes every child similar to its parent with ratio 1/2, so
    ``l_max`` is a function of level alone and the size floor is a depth computable
    up front. ``sqrt(2) * fov / init_res`` is the level-0 hypotenuse.
    """
    l_max0 = math.sqrt(2.0) * fov / init_res
    if l_max0 <= min_img_sep:
        return 0
    return int(math.ceil(math.log2(l_max0 / min_img_sep)))


def check_lattice_keys(init_res, max_level, remedy) -> None:
    """
    Raise if the lattice for ``init_res`` cells at ``max_level`` overflows int64 keys.

    The lattice is built one level finer than ``max_level`` (see
    :class:`Lattice`), so that is the one sized here. ``remedy`` ends the
    message: what the caller can change.
    """
    n = init_res * (1 << (max_level + 1))
    if (n + 1) ** 2 >= MAX_KEY:
        raise ValueError(
            f"lattice too fine to key in int64: init_res={init_res} at level "
            f"{max_level} needs a lattice of {n + 1} points per axis, one level "
            f"finer than max_level so that max_level edge midpoints are lattice "
            f"points. {remedy}"
        )


def validate_build_args(
    fov, init_res, min_img_sep, max_depth, requested_min_img_sep=None
) -> None:
    """
    Reject impossible parameters, including a lattice that would overflow int64.

    ``min_img_sep`` is the value every check and the depth computation use.
    ``requested_min_img_sep`` names only the positivity message, so a caller who
    halved it (``build_adaptive_mesh``) is told the number they actually
    supplied rather than the halved one. Defaults to ``min_img_sep`` for a
    direct call, where the two coincide.
    """
    if requested_min_img_sep is None:
        requested_min_img_sep = min_img_sep
    if not fov > 0:
        raise ValueError(f"fov must be positive, got {fov}")
    if init_res < 1:
        raise ValueError(f"init_res must be at least 1, got {init_res}")
    if not min_img_sep > 0:
        raise ValueError(f"min_img_sep must be positive, got {requested_min_img_sep}")
    if max_depth < 0:
        raise ValueError(f"max_depth must be non-negative, got {max_depth}")
    max_level = min(max_depth, depth_floor(fov, init_res, min_img_sep))
    check_lattice_keys(
        init_res, max_level, "Raise min_img_sep, lower max_depth, or lower init_res."
    )


class Lattice(NamedTuple):
    """
    Dyadic integer lattice over the square domain, at the finest allowed level.

    Every mesh vertex is an integer pair, so midpoints are exact integer averages
    -- no float hashing, no rounding tolerance. The lens-plane position is a pure
    function of the integer pair, so two triangles sharing a vertex compute
    bit-identical coordinates. That is the root of the exact-negation property that
    stops a query falling through the seam between adjacent leaves.

    The lattice is built one level finer than ``max_level``, so a triangle's
    edge vectors are multiples of ``1 << (max_level + 1 - d)`` at level ``d``
    -- at least 2 for every ``d <= max_level``. That makes :func:`_midpoint_ij`
    exact at *every* level, ``max_level`` included, which is what lets the
    parity test run there.

    It also partitions the lattice. Every point the vertex cache holds -- a
    vertex at any level, or a midpoint below ``max_level`` -- has **even**
    coordinates; a ``max_level`` midpoint always has at least one **odd**
    coordinate. So the ``max_level`` midpoint pass can never re-trace a cached
    point, and a ``max_level`` midpoint can never be a triangle vertex.

    Widening does not move anything. ``fov / (2n)`` is exactly ``fl(fov / n) / 2``
    and ``(2 * ij) * (scale / 2)`` rounds the same exact real as ``ij * scale``,
    so every pre-existing point keeps a bit-identical position. Keys scale
    uniformly, so :func:`_canonical_order` is unchanged too.

    ``origin`` is the lattice index ``lo`` belongs to: :func:`lattice_xy`
    places ``ij`` at ``lo + (ij - origin) * scale``. A fresh lattice has
    ``origin == 0``, ``lo`` at its corner. :func:`extend_lattice` grows the
    lattice by shifting every index and ``origin`` together, never ``lo`` or
    ``scale``, so growth keeps every existing point bit-identical, as
    widening does.
    """

    level: int
    n: int
    stride: int
    scale: float
    lo: ArrayLike
    origin: int = 0


def make_lattice(fov, x0, y0, init_res, lattice_level) -> Lattice:
    """
    Build a :class:`Lattice` covering ``fov``, centred at ``(x0, y0)``.

    Parameters
    ----------
    fov: float
        Field of view.

        *Unit: arcsec*

    x0: float
        Domain centre, x.

        *Unit: arcsec*

    y0: float
        Domain centre, y.

        *Unit: arcsec*

    init_res: int
        Level-0 grid resolution.
    lattice_level: int
        Level at which the lattice itself is built -- one finer than
        ``max_level``, see :class:`Lattice`.

    Returns
    -------
    Lattice
        With origin 0: lo is the lattice's own corner.
    """
    level = int(lattice_level)
    n = int(init_res) * (1 << level)
    stride = n + 1
    scale = float(fov) / n
    lo = backend.as_array([x0 - fov / 2.0, y0 - fov / 2.0], dtype=backend.float64)
    return Lattice(level=level, n=n, stride=stride, scale=scale, lo=lo, origin=0)


def extend_lattice(lat, k) -> Lattice:
    """
    ``lat`` grown by ``k`` level-0 cells on every side, every old point in place.

    A point at ``ij`` on ``lat`` is at ``ij + (k << lat.level)`` on the
    result, and ``origin`` moves by the same amount, so ``ij - origin`` -- and
    with it :func:`lattice_xy` -- is unchanged bit for bit. ``lo`` and
    ``scale`` are never recomputed: a fresh lattice over the larger fov
    computes both from that fov and can round differently. The stride grows,
    so keys change, but a uniform shift keeps their ``(i, j)`` lexicographic
    order, and with it every key-sorted array.

    Parameters
    ----------
    lat: Lattice
    k: int
        Level-0 cells added on each side, at least zero.

    Returns
    -------
    Lattice

    Raises
    ------
    ValueError
        If ``k`` is negative.
    """
    k = int(k)
    if k < 0:
        raise ValueError(f"k must be non-negative, got {k}")
    pad = k << lat.level
    n = lat.n + 2 * pad
    return lat._replace(n=n, stride=n + 1, origin=lat.origin + pad)


def lattice_key(lat, ij):
    """Lattice key of integer coordinates, shape ``(..., 2) -> (...)``."""
    return ij[..., 0] * lat.stride + ij[..., 1]


def lattice_ij_from_key(lat, key):
    """Inverse of :func:`lattice_key`, shape ``(...) -> (..., 2)``."""
    return backend.stack((key // lat.stride, key % lat.stride), dim=-1)


def lattice_xy(lat, ij):
    """Lens-plane position, shape ``(..., 2) -> (..., 2)``.

    ``lo + (ij - origin) * scale``. For a fresh lattice ``origin == 0``, and
    subtracting zero from an integer is exact, so positions are what they were
    before the anchor existed.

    *Unit: arcsec*
    """
    return lat.lo + backend.to(ij - lat.origin, dtype=backend.float64) * lat.scale


def lattice_on_boundary(lat, ij):
    """True where the point lies on the edge of the domain."""
    return (
        (ij[..., 0] == 0)
        | (ij[..., 0] == lat.n)
        | (ij[..., 1] == 0)
        | (ij[..., 1] == lat.n)
    )


# ---------------------------------------------------------------------------
# Vertex cache and active-vertex set
# ---------------------------------------------------------------------------


class VertexCache(NamedTuple):
    """
    Lattice key to slot, with the source-plane image of every evaluated point.

    Lookup is ``backend.searchsorted`` against a sorted key array rather than
    a Python dict, so a whole level's worth of points resolves in one
    vectorized call. Slots are assigned monotonically in order of first
    evaluation and never move.

    ``ij`` and ``beta`` hold the lattice coordinates and source-plane image of
    every evaluated point, shape ``(N, 2)`` and indexed by slot. ``keys`` and
    ``slots`` are the sorted-key index: ``keys`` is ascending, and
    ``slots[i]`` is the slot of ``keys[i]``.
    """

    keys: ArrayLike
    slots: ArrayLike
    ij: ArrayLike
    beta: ArrayLike


def empty_cache(device=None) -> VertexCache:
    """An empty :class:`VertexCache`."""
    return VertexCache(
        keys=backend.empty((0,), dtype=backend.int64, device=device),
        slots=backend.empty((0,), dtype=backend.int64, device=device),
        ij=backend.empty((0, 2), dtype=backend.int64, device=device),
        beta=backend.empty((0, 2), dtype=backend.float64, device=device),
    )


def cache_size(cache) -> int:
    """Number of vertices held in the cache."""
    return int(cache.ij.shape[0])


def cache_lookup(cache, keys) -> ArrayLike:
    """Slot of each key, or ``-1`` where absent."""
    if cache.keys.shape[0] == 0:
        return backend.zeros_like(keys) - 1
    n = cache.keys.shape[0]
    pos = backend.clamp(backend.searchsorted(cache.keys, keys), 0, n - 1)
    return backend.where(cache.keys[pos] == keys, cache.slots[pos], -1)


def cache_missing(cache, keys) -> ArrayLike:
    """Unique keys not yet evaluated, ascending."""
    uniq = backend.unique(keys)
    return uniq[cache_lookup(cache, uniq) < 0]


def cache_insert(cache, keys, ij, beta) -> Tuple[VertexCache, ArrayLike]:
    """
    Assign slots to new keys. ``keys`` must be unique, absent, and sorted.

    The key index is merged rather than re-sorted. Both sides are already
    sorted -- ``keys`` comes from :func:`cache_missing`, which returns
    ``backend.unique`` output, and ``cache.keys`` is maintained sorted -- and
    no key appears on both sides, so ``searchsorted`` plus a running offset
    gives each new key its position in the merged array outright. The result
    is identical to sorting the concatenation, because with all keys distinct
    the sorted order is unique.

    Returns
    -------
    VertexCache
        The updated cache.
    ArrayLike
        The slots assigned to ``keys``, in the order given.
    """
    start = cache.ij.shape[0]
    k = int(keys.shape[0])
    slots = backend.arange(start, start + k, dtype=backend.int64)

    new_ij = backend.concatenate([cache.ij, ij], dim=0)
    new_beta = backend.concatenate([cache.beta, beta], dim=0)

    total = cache.keys.shape[0] + k
    dest = backend.searchsorted(cache.keys, keys) + backend.arange(
        k, dtype=backend.int64
    )
    new_keys = backend.fill_at_indices(
        backend.empty((total,), dtype=backend.int64), dest, keys
    )
    new_slots = backend.fill_at_indices(
        backend.empty((total,), dtype=backend.int64), dest, slots
    )
    # The complement of `dest`: where the pre-existing keys/slots land in the
    # merged array. Computed as a boolean mask, then converted to indices with
    # `flatnonzero` so the merge only ever calls `fill_at_indices` with
    # integer positions.
    stay = backend.fill_at_indices(
        backend.ones((total,), dtype=backend.bool), dest, False
    )
    old_dest = backend.flatnonzero(stay)
    new_keys = backend.fill_at_indices(new_keys, old_dest, cache.keys)
    new_slots = backend.fill_at_indices(new_slots, old_dest, cache.slots)

    return (
        VertexCache(keys=new_keys, slots=new_slots, ij=new_ij, beta=new_beta),
        slots,
    )


def empty_active(device=None) -> ArrayLike:
    """
    Lattice points that are currently vertices of some triangle in the mesh.

    Stored as one flag per **vertex-cache slot**, not as a sorted key set.
    Every active key is by construction a vertex of some triangle, so it has
    already been evaluated and is already in the cache -- a second sorted
    structure would duplicate the cache's own key index, and keeping it
    sorted cost an ``np.union1d`` over the whole active set on every
    insertion, which was the single largest term in the build.

    Separate from the vertex cache, which also holds midpoints of
    tested-but-never-split triangles. Only ever grows, since a parent's
    vertices are inherited by all its children.
    """
    return backend.zeros((0,), dtype=backend.bool, device=device)


def active_add_slots(active, n_slots, slots) -> ArrayLike:
    """
    Activate vertex-cache slots.

    Takes slots rather than keys because every caller already holds them:
    re-keying a triangle's vertices only to look them up again is exactly the
    work this function exists to avoid.
    """
    # Trip-wire for the `active subset of cache` invariant. A -1 slot -- what
    # `cache_lookup` returns for an absent key -- would negative-index into
    # the last cache entry and activate the wrong vertex, and the mesh would
    # come out unbalanced rather than raising.
    #
    # `raise AssertionError` rather than a bare `assert`: `python -O` strips
    # bare asserts, and this guards against silent geometric corruption.
    if not bool(backend.all(slots >= 0)):
        raise AssertionError("cannot activate an uncached vertex")
    # Concatenate unconditionally, even when `extra` is 0, so this always
    # scatters into a fresh array and never the caller's own. `torch.cat`/
    # `jnp.concatenate` both always allocate, unlike `fill_at_indices` on its
    # own: torch mutates its first argument in place and returns it, so
    # scattering directly into `active` on the no-growth path would silently
    # clobber the caller's array under torch while leaving it untouched under
    # jax -- the exact backend-divergence `cache_insert` avoids by always
    # writing into a fresh `backend.empty`/`backend.ones` buffer.
    extra = max(n_slots - active.shape[0], 0)
    active = backend.concatenate(
        [active, backend.zeros((extra,), dtype=backend.bool)], dim=0
    )
    return backend.fill_at_indices(active, slots, True)


def active_contains_slots(active, slots) -> ArrayLike:
    """True where the slot is an active vertex. ``-1`` reads as False."""
    if active.shape[0] == 0:
        return backend.zeros(slots.shape, dtype=backend.bool)
    present = slots >= 0
    return present & active[backend.where(present, slots, 0)]


def active_contains(active, cache, keys) -> ArrayLike:
    """
    True where the key is an active vertex.

    A key absent from the cache was never evaluated, so it cannot be a
    triangle vertex and cannot be active -- ``cache_lookup`` returns ``-1``
    and :func:`active_contains_slots` reads that as False.
    """
    return active_contains_slots(active, cache_lookup(cache, keys))


# ---------------------------------------------------------------------------
# Leaf store
# ---------------------------------------------------------------------------

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


class LeafStore(NamedTuple):
    """
    Terminal triangles, keyed by row index with a validity flag.

    Not append-only: a triangle marked converged at level ``d`` can be
    removed and replaced by descendants several levels later, when a distant
    refinement cascades back to it. Hence the flag and the single compaction
    at the end (:func:`store_compact`), rather than streaming into a flat
    array as we go.

    ``v`` holds the three vertex-cache slots of each leaf, shape ``(N, 3)``.
    ``level``, ``cls``, ``status`` and ``valid`` are per-row, shape ``(N,)``.
    ``status`` is a ``backend.int64`` bitmask of the ``LEAF_*`` flags above.
    """

    v: ArrayLike
    level: ArrayLike
    cls: ArrayLike
    status: ArrayLike
    valid: ArrayLike


def empty_store(device=None) -> LeafStore:
    """An empty :class:`LeafStore`."""
    return LeafStore(
        v=backend.empty((0, 3), dtype=backend.int64, device=device),
        level=backend.empty((0,), dtype=backend.int64, device=device),
        cls=backend.empty((0,), dtype=backend.int64, device=device),
        status=backend.empty((0,), dtype=backend.int64, device=device),
        valid=backend.empty((0,), dtype=backend.bool, device=device),
    )


def _broadcast_row_field(value, n_rows) -> ArrayLike:
    """A per-row array as-is, or a Python scalar broadcast to ``n_rows``."""
    if hasattr(value, "shape"):
        return value
    return backend.zeros((n_rows,), dtype=backend.int64) + value


def store_add(store, v, level, cls, status) -> Tuple[LeafStore, ArrayLike]:
    """
    Append triangles, returning their row indices.

    ``level``, ``cls`` and ``status`` may each be a Python scalar or a
    per-row array; a scalar is broadcast to the row count before
    concatenating.

    Returns
    -------
    LeafStore
        The updated store.
    ArrayLike
        The row indices assigned to ``v``, in the order given.
    """
    start = store.v.shape[0]
    k = v.shape[0]
    rows = backend.arange(start, start + k, dtype=backend.int64)

    new_store = LeafStore(
        v=backend.concatenate([store.v, v], dim=0),
        level=backend.concatenate([store.level, _broadcast_row_field(level, k)], dim=0),
        cls=backend.concatenate([store.cls, _broadcast_row_field(cls, k)], dim=0),
        status=backend.concatenate(
            [store.status, _broadcast_row_field(status, k)], dim=0
        ),
        valid=backend.concatenate(
            [store.valid, backend.ones((k,), dtype=backend.bool)], dim=0
        ),
    )
    return new_store, rows


def store_remove(store, rows) -> LeafStore:
    """
    Mark rows invalid, returning a NEW store; ``store`` itself is untouched.

    Scatters into a copy of ``store.valid`` rather than ``store.valid``
    itself: ``backend.fill_at_indices`` mutates its first argument in place
    and returns it under torch, so writing straight into ``store.valid``
    would silently clobber the input store's own array under torch while
    leaving it (and every alias of it a caller might still hold) untouched
    under jax -- the same backend divergence :func:`cache_insert` and
    :func:`active_add_slots` avoid by always writing into a fresh buffer.
    """
    valid = backend.fill_at_indices(backend.copy(store.valid), rows, False)
    return LeafStore(
        v=store.v, level=store.level, cls=store.cls, status=store.status, valid=valid
    )


def store_compact(store) -> Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
    """The surviving rows of ``v``, ``level``, ``cls`` and ``status``."""
    keep = backend.flatnonzero(store.valid)
    return store.v[keep], store.level[keep], store.cls[keep], store.status[keep]


# ---------------------------------------------------------------------------
# Triangle helpers and 2:1 balance detection
# ---------------------------------------------------------------------------


def initial_triangles(
    init_res, lattice_level, root_class
) -> Tuple[ArrayLike, ArrayLike]:
    """
    Level-0 triangles: two per cell, split on the ``(0,0)-(1,1)`` diagonal.

    Both are emitted positively oriented. ``func/base.py`` builds its pair with
    *opposite* handedness; this fixes that so orientation is globally consistent
    and downstream degree or winding-number arguments stay available.

    Built with a ``backend.meshgrid(..., indexing="ij")`` cell grid, flattened
    in the same row-major order ``numpy`` uses -- the oracle test compares the
    result element by element, so a transposed ordering fails loudly rather
    than quietly.

    Parameters
    ----------
    init_res: int
        Level-0 grid resolution.
    lattice_level: int
        Level the lattice is built at -- one finer than ``max_level``, see
        :class:`Lattice`.
    root_class: ArrayLike
        ``ROOT_CLASS`` from :func:`child_matrix_tables`, shape ``(2,)``.

    Returns
    -------
    ij: ArrayLike
        ``(2 * init_res**2, 3, 2)`` int64 lattice coordinates.
    cls: ArrayLike
        ``(2 * init_res**2,)`` int64 orientation classes.
    """
    step = 1 << int(lattice_level)
    axis = backend.arange(init_res, dtype=backend.int64)
    i, j = backend.meshgrid(axis, axis, indexing="ij")
    base = backend.stack((i.reshape(-1), j.reshape(-1)), dim=-1) * step  # (r**2, 2)
    blocks, classes = [], []
    for s, shape in enumerate(ROOT_SHAPES):
        offs = backend.as_array(shape, dtype=backend.int64) * step  # (3, 2)
        blocks.append(base[:, None, :] + offs[None, :, :])
        classes.append(
            backend.zeros((base.shape[0],), dtype=backend.int64) + root_class[s]
        )
    return backend.concatenate(blocks, dim=0), backend.concatenate(classes, dim=0)


def ring_triangles(
    init_res, k, lattice_level, root_class
) -> Tuple[ArrayLike, ArrayLike]:
    """
    Level-0 triangles of the cells within ``k`` cells of the domain's edge.

    :func:`initial_triangles` of the whole ``init_res x init_res`` grid, less
    the central ``(init_res - 2k) x (init_res - 2k)`` block of cells: the
    ring an extension by ``k`` adds around an existing mesh. A row subset of
    :func:`initial_triangles`, in its order.

    Parameters
    ----------
    init_res: int
        Level-0 cells per axis of the grown grid.
    k: int
        Width of the ring, in cells.
    lattice_level: int
    root_class: ArrayLike
        ``ROOT_CLASS`` from :func:`child_matrix_tables`.

    Returns
    -------
    ij: ArrayLike
        ``(T, 3, 2)`` int64 lattice coordinates.
    cls: ArrayLike
        ``(T,)`` int64 orientation classes.
    """
    ij, cls = initial_triangles(init_res, lattice_level, root_class)
    # Both root shapes put theta_1 at their cell's (0, 0) corner.
    cell = ij[:, 0, :] // (1 << int(lattice_level))
    inner = (cell >= k) & (cell < init_res - k)
    keep = backend.flatnonzero(~(inner[:, 0] & inner[:, 1]))
    return ij[keep], cls[keep]


def midpoint_ij(ij) -> ArrayLike:
    """
    Edge midpoints ``m1, m2, m3``, with ``m_i`` opposite ``theta_i``.

    Exact integer averaging via integer ``//``: the lattice is one level finer
    than ``max_level`` (see :class:`Lattice`), so the coordinate sums are even
    at every level up to and including ``max_level``.

    Parameters
    ----------
    ij: ArrayLike
        Triangle vertex coordinates, shape ``(n, 3, 2)`` int64.

    Returns
    -------
    ArrayLike
        Shape ``(n, 3, 2)`` int64.
    """
    return backend.stack(
        (
            (ij[:, 1] + ij[:, 2]) // 2,
            (ij[:, 2] + ij[:, 0]) // 2,
            (ij[:, 0] + ij[:, 1]) // 2,
        ),
        dim=1,
    )


def red_split(v, m, cls, compose) -> Tuple[ArrayLike, ArrayLike]:
    """
    Split into the four canonical children, triangle-major.

    Parameters
    ----------
    v: ArrayLike
        Triangle vertex slots, shape ``(n, 3)`` int64.
    m: ArrayLike
        Edge-midpoint slots ``m1, m2, m3``, shape ``(n, 3)`` int64.
    cls: ArrayLike
        Orientation class of each triangle, shape ``(n,)`` int64.
    compose: ArrayLike
        ``COMPOSE`` from :func:`child_matrix_tables`, shape ``(6, 4)``.

    Returns
    -------
    child_v: ArrayLike
        ``(4n, 3)`` vertex slots, ordered as all four children of triangle 0,
        then triangle 1, and so on.
    child_cls: ArrayLike
        ``(4n,)`` orientation classes, in the same order.
    """
    six = backend.concatenate([v, m], dim=1)  # (n, 6)
    child_v = six[:, _CHILD_VERTEX_INDEX_TABLE].reshape(-1, 3)
    child_cls = compose[cls].reshape(-1)
    return child_v, child_cls


def edge_quarter_keys(lat, ij) -> ArrayLike:
    """
    Keys of both quarter points on each of the three edges.

    A neighbour across an edge that is two or more levels finer has one of
    these as a vertex, so six hash lookups decide balance for a triangle --
    no edge-to-triangle adjacency table and no ancestry walk. Both quarter
    points are checked because the neighbour across an edge can itself be
    non-uniform.

    The caller must only pass triangles at level ``<= max_level - 2``, where
    the edge vectors are divisible by four and the quarter points are lattice
    points.

    Parameters
    ----------
    lat: Lattice
        Used to key the quarter points.
    ij: ArrayLike
        Lattice coordinates, shape ``(n, 3, 2)`` int64.

    Returns
    -------
    ArrayLike
        Shape ``(n, 6)`` int64.
    """
    a = ij[:, [0, 1, 2], :]
    b = ij[:, [1, 2, 0], :]
    delta = (b - a) // 4
    return backend.concatenate(
        (lattice_key(lat, a + delta), lattice_key(lat, b - delta)), dim=1
    )


def find_unbalanced(store, cache, lat, active, max_level, frontier_level) -> ArrayLike:
    """
    Rows of ``store`` carrying an active quarter point on some edge.

    Two independent level bounds apply. ``level <= frontier_level - 2`` is an
    optimization: only triangles at least two levels coarser than the
    frontier can have been invalidated by it, so the scan skips most of the
    store. ``level <= max_level - 2`` is the bound above which no neighbour
    can be two levels finer, since ``max_level`` is the finest level there is.

    Parameters
    ----------
    store: LeafStore
        Terminal triangles; only rows with ``valid`` set are scanned.
    cache: VertexCache
        Supplies the lattice coordinates of each row's vertices.
    lat: Lattice
        Used to key the quarter points.
    active: ArrayLike
        Pre-existing active-vertex set; membership decides a violation.
    max_level: int
        Finest allowed level.
    frontier_level: int
        Level of the finest triangles created in the current round.

    Returns
    -------
    ArrayLike
        Int64 indices into ``store``'s rows.
    """
    # `frontier_level <= max_level` holds for every call the level loop
    # makes, so the first term always binds and the second is unreachable
    # defensive code today. Keep the min(): `max_level - 2` is the level
    # above which no neighbour can be two levels finer, since `max_level` is
    # the finest level there is. (It used to double as an integrality
    # requirement for the quarter-point arithmetic below; on the widened
    # lattice quarter points stay lattice points down to `max_level - 1`, so
    # that role is gone and only the balance argument remains.)
    bound = min(frontier_level - 2, max_level - 2)
    cand = backend.flatnonzero(store.valid & (store.level <= bound))
    if cand.shape[0] == 0:
        return cand
    ij = cache.ij[store.v[cand]]  # (n, 3, 2)
    # One edge at a time, accumulating into a single `(n,)` mask, rather than
    # building the whole `(cand, 6)` key array the way `edge_quarter_keys`
    # does: the batched form held 153 MB at a million leaves -- the largest
    # transient in the level loop. The disjunction is over the same six keys
    # `edge_quarter_keys` returns; only the association changes.
    hit = backend.zeros((cand.shape[0],), dtype=backend.bool)
    for e in range(3):
        a = ij[:, e, :]
        b = ij[:, (e + 1) % 3, :]
        delta = (b - a) // 4
        for quarter in (a + delta, b - delta):
            hit = hit | active_contains(active, cache, lattice_key(lat, quarter))
    return cand[hit]


# ---------------------------------------------------------------------------
# Raytrace wrapper and point evaluation
# ---------------------------------------------------------------------------


def make_raytrace(raytrace, device) -> Callable[[ArrayLike], ArrayLike]:
    """
    Wrap a user ``raytrace(x, y) -> (bx, by)`` as a host-side ``(N, 2) -> (N, 2)``.

    Coordinates go out to the callback, and come back from it, as float64 --
    **unconditionally**. This is the single most important property of the
    build. The refinement criterion compares a midpoint deviation against an
    affine prediction, both ``O(fov)`` quantities, so its roundoff floor is
    ``eps * fov`` and the comparison is meaningless below
    ``h ~ sqrt(8 * eps * fov)``. At ``fov = 5`` that floor is about ``2e-3``
    arcsec in float32 -- comparable to a typical ``min_img_sep`` -- and below
    it the midpoint deviation cancels to *exactly* zero, which the criterion
    reads as "perfectly affine" and converges. That is the fail-**open**
    direction: the mesh silently stops refining exactly where it most needs
    to, and no downstream clamp can recover the resolution once lost. Hence
    the coercion on the way in, and the cast on the way out, regardless of
    what dtype the callback itself operates in or returns.

    The callback's own dtype is still recorded in ``info["dtype"]`` on every
    call -- not acted on here, but so that a caller several layers up (the
    mesh build) can tell a silently downgraded callback from a well-behaved
    one and warn accordingly.

    Parameters
    ----------
    raytrace: Callable[[ArrayLike, ArrayLike], Tuple[ArrayLike, ArrayLike]]
        ``raytrace(x, y) -> (bx, by)``, on 1-D arrays of shape ``(N,)``.
    device:
        Device for the coordinates handed to ``raytrace``.

    Returns
    -------
    Callable[[ArrayLike], ArrayLike]
        ``(N, 2) -> (N, 2)`` float64. Carries a mutable ``.info`` dict,
        ``{"done": bool, "dtype": Any}``. ``"done"`` becomes True once the
        2-tuple return shape has been validated, which happens only on the
        first call; every call, first or not, validates that the returned
        arrays are shape-preserving on the 1-D input. ``"dtype"`` is whatever
        dtype the callback itself returned on its most recent call.
    """
    info = {"done": False, "dtype": backend.float64}

    def call(xy):
        build_device = backend.device(xy)
        x = backend.as_array(xy[:, 0], dtype=backend.float64, device=device)
        y = backend.as_array(xy[:, 1], dtype=backend.float64, device=device)
        out = raytrace(x, y)
        if not info["done"]:
            if not isinstance(out, tuple) or len(out) != 2:
                raise ValueError(
                    "raytrace must return a 2-tuple (bx, by) of arrays with "
                    f"shape (N,); got {type(out).__name__}"
                )
            info["done"] = True
        bx, by = out
        info["dtype"] = bx.dtype
        if bx.shape != x.shape or by.shape != y.shape:
            raise ValueError(
                f"raytrace returned shape {tuple(bx.shape)}/{tuple(by.shape)} "
                f"for {tuple(x.shape)} inputs; it must be shape-preserving on "
                "1-D input"
            )
        return backend.to(
            backend.stack((bx, by), dim=-1),
            dtype=backend.float64,
            device=build_device,
        )

    call.info = info  # type: ignore[attr-defined]
    return call


def trace_keys(lat, ij, raytrace_fn, batch_size) -> ArrayLike:
    """
    Raytrace lattice points as one logical batch, chunked only for memory.

    The whole-array path is used whenever ``batch_size`` is ``None`` or the
    input already fits in a single batch; only otherwise does this chunk with
    :func:`backend.chunk` and concatenate. That ordering is what makes the
    result bit-identical for every ``batch_size``: ``raytrace_fn`` acts on
    each row independently, and concatenation preserves row order, so
    partitioning into chunks can change *how many* calls are made but never
    *what* they compute.

    Split out of :func:`evaluate` so a ``max_level`` midpoint pass -- points
    consumed by the parity test and never read again -- can reuse the
    chunking without touching the vertex cache.

    Parameters
    ----------
    lat: Lattice
        Used to convert ``ij`` to lens-plane positions.
    ij: ArrayLike
        Lattice coordinates, shape ``(n, 2)`` int64.
    raytrace_fn: Callable[[ArrayLike], ArrayLike]
        From :func:`make_raytrace`.
    batch_size: Optional[int]
        Maximum rows per ``raytrace_fn`` call, or ``None`` for a single call.

    Returns
    -------
    ArrayLike
        Shape ``(n, 2)`` float64.

        *Unit: arcsec*
    """
    xy = lattice_xy(lat, ij)
    if batch_size is None or xy.shape[0] <= batch_size:
        return raytrace_fn(xy)
    n_chunks = math.ceil(xy.shape[0] / batch_size)
    return backend.concatenate(
        [raytrace_fn(chunk) for chunk in backend.chunk(xy, n_chunks, dim=0)], dim=0
    )


def evaluate(cache, lat, keys, raytrace_fn, batch_size) -> VertexCache:
    """
    Evaluate every not-yet-cached key, in one logical batch per call.

    Deduplication happens before any point is traced: :func:`cache_missing`
    reduces ``keys`` to its unique, not-yet-cached elements first, so a caller
    that repeats a key -- two triangles sharing a vertex, say -- never causes
    it to be raytraced twice.

    Parameters
    ----------
    cache: VertexCache
    lat: Lattice
    keys: ArrayLike
        Lattice keys to ensure are cached, shape ``(n,)`` int64. Need not be
        unique or sorted.
    raytrace_fn: Callable[[ArrayLike], ArrayLike]
        From :func:`make_raytrace`.
    batch_size: Optional[int]
        Forwarded to :func:`trace_keys`.

    Returns
    -------
    VertexCache
        ``cache`` itself, unchanged, when every key is already cached;
        otherwise a new cache with the missing keys inserted.
    """
    todo = cache_missing(cache, keys)
    if todo.shape[0] == 0:
        return cache
    ij = lattice_ij_from_key(lat, todo)
    beta = trace_keys(lat, ij, raytrace_fn, batch_size)
    new_cache, _ = cache_insert(cache, todo, ij, beta)
    return new_cache


# ---------------------------------------------------------------------------
# The critical band
# ---------------------------------------------------------------------------


class CriticalBand(NamedTuple):
    """
    The ``max_level`` leaves ``det A`` changes sign across, with ``det A`` at their samples.

    A leaf is in the band when the determinants at its six samples -- the
    three vertices and the three edge midpoints, the corners of its four
    red-split children -- are all finite and not all of one class, where a
    sample's class is ``det >= 0``: an exact zero counts as positive. Every
    ``LEAF_JACOBIAN_PARITY_UNRESOLVED`` leaf is in the band. So is a leaf
    flagged ``LEAF_JACOBIAN_NONFINITE`` only because some sample's
    determinant is exactly zero, when zero counting as positive leaves its
    classes mixed -- which is what keeps a critical curve through lattice
    points from vanishing. The one exception is rounding, not overflow: the
    sign test behind the flags reads a row-scaled ``det A``, while this band
    stores the raw ``det A``, and the two can disagree whenever ``det A`` is
    within a few ulps of zero, even with O(1) entries. Measured on 1e6
    near-singular matrices, 154,070 had a raw determinant ``>= 0`` paired
    with a scaled determinant ``< 0``, and 48,349 had a nonzero raw
    determinant paired with a scaled determinant of exactly zero. So,
    rarely, a flagged leaf can be missing from the band, or a band leaf can
    carry neither flag. Tracing is unaffected, because the band is
    consistent with itself.

    Samples are deduplicated by lattice key: a sample shared by several band
    leaves is one row, with one ``det``, so every leaf sharing it agrees on
    its class. That consistency is what lets
    :func:`~caustics.lenses.func.adaptive_critical.mesh_critical_curves`
    chain the crossings of neighbouring leaves.

    Parameters
    ----------
    leaves: ArrayLike
        Shape ``(F,)`` int64 index into ``AdaptiveMesh.leaves``. As
        :func:`refine` returns it, an index into the :class:`LeafStore`'s
        rows instead; :func:`freeze` maps it onto the mesh and sorts the
        rows by it, so row order depends on the mesh alone.
    samples: ArrayLike
        Shape ``(F, 6)`` int64 index into ``lens``, ``source`` and ``det``:
        each leaf's ``theta_1, theta_2, theta_3, m_1, m_2, m_3``, the vertices
        in the leaf's own order and ``m_i`` opposite ``theta_i``.
        Samples are numbered in ascending lattice-key order.
    lens: ArrayLike
        Lens-plane position of each sample, shape ``(S, 2)``, at the mesh
        dtype.

        *Unit: arcsec*
    source: ArrayLike
        Source-plane image of each sample, shape ``(S, 2)``, at the mesh
        dtype.

        *Unit: arcsec*
    det: ArrayLike
        ``det A`` at each sample, shape ``(S,)``, always float64.
    """

    leaves: ArrayLike
    samples: ArrayLike
    lens: ArrayLike
    source: ArrayLike
    det: ArrayLike


def empty_band(device=None) -> CriticalBand:
    """A :class:`CriticalBand` with no leaves."""
    return CriticalBand(
        leaves=backend.zeros((0,), dtype=backend.int64, device=device),
        samples=backend.zeros((0, 6), dtype=backend.int64, device=device),
        lens=backend.zeros((0, 2), dtype=backend.float64, device=device),
        source=backend.zeros((0, 2), dtype=backend.float64, device=device),
        det=backend.zeros((0,), dtype=backend.float64, device=device),
    )


def sample_jacobians(
    lat, six_ij, jacobian_fn
) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
    """
    The lens Jacobian at every triangle's six samples, one call per lattice point.

    A vertex is shared by about six triangles and a midpoint by two, so
    evaluating per triangle, as :func:`jacobian_parity_ok` does, repeats
    most points. Deduplicating on the lattice key first cuts the batch by
    roughly ``2.5x`` on a refined region, and it gives every triangle that
    shares a sample the same ``A`` there -- which the critical band relies
    on.

    ``jacobian_fn`` is called exactly once, on the lattice's own float64
    coordinates, and not at all when there are no triangles.

    Parameters
    ----------
    lat: Lattice
    six_ij: ArrayLike
        Lattice coordinates of each triangle's ``theta_1, theta_2, theta_3,
        m_1, m_2, m_3``, shape ``(G, 6, 2)`` int64.
    jacobian_fn: Callable[[ArrayLike, ArrayLike], ArrayLike]
        ``jacobian_fn(x, y) -> (K, 2, 2)``.

    Returns
    -------
    keys: ArrayLike
        ``(U,)`` int64, the distinct sample keys, ascending.
    index: ArrayLike
        ``(G, 6)`` int64 index into ``keys`` of each triangle's samples.
    J: ArrayLike
        ``(U, 2, 2)``, the Jacobian at each distinct sample.

    Raises
    ------
    ValueError
        If ``jacobian_fn`` does not return ``(K, 2, 2)`` for ``K`` points.
    """
    n = six_ij.shape[0]
    if n == 0:
        return (
            backend.zeros((0,), dtype=backend.int64),
            backend.zeros((0, 6), dtype=backend.int64),
            backend.zeros((0, 2, 2), dtype=backend.float64),
        )
    keys, index = backend.unique(
        lattice_key(lat, six_ij).reshape(-1), return_inverse=True
    )
    xy = lattice_xy(lat, lattice_ij_from_key(lat, keys))
    J = jacobian_fn(xy[:, 0], xy[:, 1])
    if J.shape != (keys.shape[0], 2, 2):
        raise ValueError("jacobian_fn must return shape (K, 2, 2) for K points")
    return keys, index.reshape(n, 6), J


def band_from_samples(lat, cache, keys, index, J, mid_keys, mid_beta) -> CriticalBand:
    """
    The :class:`CriticalBand` among triangles whose six samples are evaluated.

    Parameters
    ----------
    lat: Lattice
    cache: VertexCache
        Supplies each vertex sample's image.
    keys, index, J: ArrayLike
        From :func:`sample_jacobians`.
    mid_keys: ArrayLike
        ``(M,)`` int64 keys of the traced ``max_level`` midpoints, ascending.
    mid_beta: ArrayLike
        ``(M, 2)`` float64, their images.

        *Unit: arcsec*

    Returns
    -------
    CriticalBand
        With ``leaves`` indexing the rows of ``index``, and positions and
        images at float64.

    Raises
    ------
    AssertionError
        If a band sample is neither a cached vertex nor a traced midpoint.
    """
    det = backend.to(
        J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0], dtype=backend.float64
    )
    det6 = det[index]
    positive = det6 >= 0
    in_band = (
        backend.all(backend.isfinite(det6), dim=1)
        & backend.any(positive, dim=1)
        & backend.any(~positive, dim=1)
    )
    rows = backend.flatnonzero(in_band)
    if rows.shape[0] == 0:
        return empty_band()

    used, samples = backend.unique(index[rows].reshape(-1), return_inverse=True)
    point_keys = keys[used]
    # A vertex key is always cached and a max_level midpoint key never is --
    # it has an odd coordinate -- so cache membership alone tells them apart.
    slot = cache_lookup(cache, point_keys)
    vertex = slot >= 0
    at = backend.clamp(
        backend.searchsorted(mid_keys, point_keys), 0, mid_keys.shape[0] - 1
    )
    # `raise AssertionError` rather than a bare `assert`, as in
    # `active_add_slots`: `python -O` strips bare asserts, and a sample
    # matched to the wrong image would silently misplace the caustic.
    if not bool(backend.all(vertex | (mid_keys[at] == point_keys))):
        raise AssertionError("band sample is neither a cached vertex nor a midpoint")
    source = backend.where(
        backend.unsqueeze(vertex, -1),
        cache.beta[backend.where(vertex, slot, 0)],
        mid_beta[at],
    )
    return CriticalBand(
        leaves=rows,
        samples=samples.reshape(-1, 6),
        lens=lattice_xy(lat, lattice_ij_from_key(lat, point_keys)),
        source=source,
        det=det[used],
    )


def band_leaf_index(valid, order, origin, rows) -> ArrayLike:
    """
    Frozen-mesh leaf of each ``max_level`` :class:`LeafStore` row.

    A ``max_level`` row is never invalidated -- :func:`refine` breaks right
    after adding it, before any cascade -- so its compact position is the
    count of valid rows before it. :func:`canonical_order` then permutes the
    compact rows, and :func:`close` emits exactly one leaf per ``max_level``
    origin, since none has a hanging node; ``origin`` is non-decreasing, so
    ``searchsorted`` finds it.

    Parameters
    ----------
    valid: ArrayLike
        ``LeafStore.valid``, shape ``(N,)`` bool.
    order: ArrayLike
        The :func:`canonical_order` permutation of the compact rows.
    origin: ArrayLike
        From :func:`close`, shape ``(L,)``, non-decreasing.
    rows: ArrayLike
        ``(F,)`` int64 store rows, each a ``max_level`` leaf.

    Returns
    -------
    ArrayLike
        ``(F,)`` int64 index into the closed leaves.
    """
    compact = backend.cumsum(backend.long(valid), dim=0)[rows] - 1
    canonical = backend.argsort(order)[compact]
    return backend.searchsorted(origin, canonical)


def band_sample_keys(lat, cache, store, rows) -> ArrayLike:
    """
    Lattice keys of the six samples of each store row, shape ``(F, 6)``.

    In :class:`CriticalBand` order, ``theta_1, theta_2, theta_3, m_1, m_2,
    m_3``: the vertices through the vertex cache, the midpoints exact on the
    lattice through :func:`midpoint_ij`.
    """
    ij = cache.ij[store.v[rows]]
    return lattice_key(lat, backend.concatenate((ij, midpoint_ij(ij)), dim=1))


def merge_bands(lat, cache, store, first, second) -> CriticalBand:
    """
    One :class:`CriticalBand` from two whose ``leaves`` are disjoint store rows.

    A band's samples are its distinct sample keys in ascending order -- true
    of every band :func:`band_from_samples` returns and of every band this
    returns -- so each band's own keys follow from its rows alone. The merged
    band has one sample per distinct key over both, ascending again. A key
    both hold -- a sample on the seam between an old mesh and the ring an
    extension adds -- takes ``first``'s ``det`` and ``source``, so every
    sample has one value, which is what
    :func:`~caustics.lenses.func.adaptive_critical.trace_band` chains on.
    ``lens`` is recomputed from the keys by :func:`lattice_xy`.

    Every scatter writes distinct indices: all of ``first``'s positions, and
    only those of ``second``'s that ``first`` lacks.

    Parameters
    ----------
    lat: Lattice
    cache: VertexCache
    store: LeafStore
    first, second: CriticalBand
        ``leaves`` index ``store``'s rows; ``source`` and ``det`` float64.

    Returns
    -------
    CriticalBand
        ``leaves`` is ``first.leaves`` then ``second.leaves``, still store
        rows; ``lens`` and ``source`` float64.

    Raises
    ------
    AssertionError
        If a band's rows do not hold exactly its samples.
    """
    if first.leaves.shape[0] + second.leaves.shape[0] == 0:
        return empty_band()
    k1 = band_sample_keys(lat, cache, store, first.leaves)
    k2 = band_sample_keys(lat, cache, store, second.leaves)
    own1 = backend.unique(k1.reshape(-1))
    own2 = backend.unique(k2.reshape(-1))
    # `raise AssertionError` rather than a bare `assert`: `python -O` strips
    # bare asserts, and a band read off the wrong keys would pair samples
    # with another point's values.
    if own1.shape[0] != first.det.shape[0] or own2.shape[0] != second.det.shape[0]:
        raise AssertionError("a band's rows do not hold exactly its samples")
    keys, samples = backend.unique(
        backend.concatenate((k1, k2), dim=0).reshape(-1), return_inverse=True
    )
    at1 = backend.searchsorted(keys, own1)
    at2 = backend.searchsorted(keys, own2)
    if own1.shape[0]:
        pos = backend.clamp(backend.searchsorted(own1, own2), 0, own1.shape[0] - 1)
        only2 = backend.flatnonzero(own1[pos] != own2)
    else:
        only2 = backend.arange(own2.shape[0], dtype=backend.int64)
    n = keys.shape[0]
    det = backend.zeros((n,), dtype=backend.float64)
    det = backend.fill_at_indices(det, at2[only2], second.det[only2])
    det = backend.fill_at_indices(det, at1, first.det)
    source = backend.zeros((n, 2), dtype=backend.float64)
    source = backend.fill_at_indices(source, at2[only2], second.source[only2])
    source = backend.fill_at_indices(source, at1, first.source)
    return CriticalBand(
        leaves=backend.concatenate((first.leaves, second.leaves), dim=0),
        samples=samples.reshape(-1, 6),
        lens=lattice_xy(lat, lattice_ij_from_key(lat, keys)),
        source=source,
        det=det,
    )


# ---------------------------------------------------------------------------
# The refinement loop
# ---------------------------------------------------------------------------


def force_split(
    store, cache, lat, active, violators, compose, raytrace_fn, batch_size
) -> Tuple[LeafStore, VertexCache, ArrayLike, ArrayLike, ArrayLike]:
    """
    Split balance violators into auto-converged children.

    The body of every balance cascade, shared by :func:`refine`'s per-level
    one and :func:`balance`'s fixed point. A forced child is auto-converged
    -- steps 3-7 are skipped so the cascade cannot re-enter the split
    machinery from inside itself -- and has no flag of its own: it inherits
    its parent's status. A violator is bounded to ``level <= max_level - 2``
    by :func:`find_unbalanced`, and only ``max_level`` rows carry a stored
    flag, so that status is always ``LEAF_CONVERGED``, transitively. A forced
    child can still reach freeze with a non-finite vertex;
    :func:`invalidate_nonfinite_origins` is left to catch that.

    The violators' midpoints are evaluated here when not yet cached. In a
    fresh build they always are -- every tested triangle cached its
    midpoints, and a forced child's are drained from ``deferred`` at the top
    of the next level, before any later cascade can re-force it -- so
    :func:`evaluate` finds nothing missing and makes no raytrace call. A leaf
    seeded from a frozen mesh has only its vertices cached, so an extension
    that forces one traces its midpoints here.

    Parameters
    ----------
    store: LeafStore
    cache: VertexCache
    lat: Lattice
    active: ArrayLike
    violators: ArrayLike
        Store rows to split, from :func:`find_unbalanced`.
    compose: ArrayLike
        ``COMPOSE`` from :func:`child_matrix_tables`.
    raytrace_fn: Callable[[ArrayLike], ArrayLike]
        From :func:`make_raytrace`, for midpoints not yet cached.
    batch_size: Optional[int]
        Forwarded to :func:`evaluate`.

    Returns
    -------
    store: LeafStore
        ``violators`` removed and their children added.
    cache: VertexCache
        With every violator midpoint evaluated.
    active: ArrayLike
        With the children's vertices activated.
    kid_mid: ArrayLike
        ``(12k,)`` int64 lattice keys of the children's edge midpoints, for
        the caller to defer or drop.
    kid_level: ArrayLike
        ``(4k,)`` int64 level of each child.

    Raises
    ------
    AssertionError
        "cascade hit an unevaluated midpoint" if a midpoint is uncached even
        after evaluation, where a ``-1`` slot would silently negative-index
        ``cache.ij`` into wrong geometry.
    """
    vv = store.v[violators]
    mid_keys = lattice_key(lat, midpoint_ij(cache.ij[vv]))
    cache = evaluate(cache, lat, mid_keys.reshape(-1), raytrace_fn, batch_size)
    vm = cache_lookup(cache, mid_keys)
    # `raise AssertionError` rather than a bare `assert`: `python -O` strips
    # bare asserts, and this one guards against silent geometric corruption.
    if not bool(backend.all(vm >= 0)):
        raise AssertionError("cascade hit an unevaluated midpoint")
    kid_v, kid_cls = red_split(vv, vm, store.cls[violators], compose)
    kid_level = backend.repeat(store.level[violators] + 1, 4, axis=0)
    kid_status = backend.repeat(store.status[violators], 4, axis=0)
    store = store_remove(store, violators)
    store, _ = store_add(store, kid_v, kid_level, kid_cls, kid_status)
    active = active_add_slots(active, cache_size(cache), kid_v.reshape(-1))
    kid_mid = lattice_key(lat, midpoint_ij(cache.ij[kid_v])).reshape(-1)
    return store, cache, active, kid_mid, kid_level


def refine(
    raytrace_fn,
    jacobian_fn,
    lat,
    init_res,
    h0,
    min_img_sep,
    max_level,
    tables,
    batch_size,
    *,
    roots=None,
    seed=None,
):
    """
    Level-synchronous refinement.

    Processes the whole active set one level at a time: gathers all unique new
    points for that level, calls ``raytrace`` once on the batch, applies the
    criterion vectorized, then partitions into converged and to-split. No
    Python-level recursion over individual triangles, no per-triangle ``raytrace``.

    **The criterion converges no triangle without the Jacobian's agreement, at
    any level.** :func:`evaluate_criterion` evaluates ``jacobian_fn`` on every
    triangle that passes all its other tests, and one whose six samples
    disagree on ``sign(det A)`` -- or where some ``A`` is non-finite or
    singular -- splits instead of converging. So no leaf the criterion
    converged has a critical curve running between its samples, while the
    Jacobian cost stays proportional to the triangles about to converge rather
    than to every triangle tested. A balance-cascade child is the exception:
    it inherits its parent's verdict without a Jacobian check of its own, and
    its three edge midpoints are points the parent's check never saw. Below
    ``max_level``, unlike raytraced points, Jacobian evaluations are not
    deduplicated: every checked triangle evaluates its own six points, shared
    vertices included. The per-level status bitmask is discarded below
    ``max_level``, since every failure there is a reason to split rather than
    a verdict.

    At ``max_level`` nothing splits, so there is no cascade, so no force-split.
    The midpoints are still evaluated: traced once, deduplicated on their
    lattice keys, and still never added to the vertex cache -- they are the
    only points with an odd coordinate, so they could never collide with a
    cache entry regardless. Their images are kept only where the critical
    band needs them. The full criterion still runs on them, with the
    Jacobian forced on every triangle whose six samples are finite, since
    here ``status`` is the leaf's final record: every test the leaf fails is
    OR-ed into it, and anything but ``LEAF_CONVERGED`` keeps the leaf out of
    the spatial index. A triangle that straddles a fold, in particular,
    contains it at a scale no further split can resolve. That forced pass
    evaluates each distinct lattice sample once (:func:`sample_jacobians`)
    rather than six times per triangle, and the same values build the
    :class:`CriticalBand` (:func:`band_from_samples`) -- the leaves ``det A``
    changes sign across, with ``det A`` and the image at their samples,
    midpoint images included, that would otherwise be dropped. A triangle
    with a non-finite sample skips the criterion and is stored with
    ``LEAF_RAYTRACE_NONFINITE`` alone.

    The ``max_level`` midpoint pass is the single largest batch in the build.
    For a mesh refining uniformly to ``max_level`` on an ``N x N`` cell grid it
    adds the ``3 * N**2 + 2 * N`` edge midpoints to the ``(N + 1)**2``
    vertices, which together are exactly the ``(2 * N + 1)**2`` points of the
    widened lattice -- so a full-depth build now evaluates every lattice point
    exactly once, where it used to evaluate only the even sublattice. Roughly
    ``4x`` the ``raytrace`` calls in that worst case, and less on a genuinely
    adaptive mesh. ``counters["max_level_midpoints"]`` is the measured cost.

    **Non-finite triangles split unconditionally.** A non-finite sample point is
    maximal ignorance about a triangle, so it triggers refinement like every other
    unresolved condition rather than terminating it -- the criterion simply cannot
    be evaluated there, which is why the split carries no verdict. A red split
    hands the bad vertex to exactly one of the four children, so the other three
    re-enter the criterion normally and the singularity ends up ringed by a band of
    ``LEAF_RAYTRACE_NONFINITE`` leaves at the smallest allowed size instead of a
    hexagon of ``fov / init_res``. Terminating on the spot instead would put the
    hole at whatever level the triangle was first sampled, and such leaves are
    excluded from the spatial index -- so on a singular model that hole is exactly
    the region where the mesh is most needed. The cost is
    ``counters["nonfinite_splits"]``: six triangles per level for a point
    singularity, but ``O(area * 4**max_level)`` should a ``raytrace`` return
    non-finite values over a whole region.

    Parameters
    ----------
    raytrace_fn: Callable[[ArrayLike], ArrayLike]
        From :func:`make_raytrace`.
    jacobian_fn: Callable[[ArrayLike, ArrayLike], ArrayLike]
        ``jacobian_fn(x, y) -> (K, 2, 2)``, the Jacobian of the map
        ``raytrace_fn`` traces, forwarded to :func:`evaluate_criterion`. Called
        directly on the lattice's float64 lens-plane coordinates: not through
        :func:`make_raytrace`, and not chunked by ``batch_size``.
    lat: Lattice
    init_res: int
        Level-0 grid resolution.
    h0: float
        Level-0 cell size ``fov / init_res``.

        *Unit: arcsec*
    min_img_sep: float
        Lens-plane tolerance.

        *Unit: arcsec*
    max_level: int
        Finest allowed level.
    tables:
        ``(M, G, COMPOSE, PINV0, ROOT_CLASS)`` from :func:`child_matrix_tables`.
    batch_size: Optional[int]
        Forwarded to :func:`evaluate` and :func:`trace_keys`.
    roots: Optional[Tuple[ArrayLike, ArrayLike]]
        Level-0 triangles to refine, ``(ij, cls)`` as
        :func:`initial_triangles` returns them. ``None`` refines every cell of
        the ``init_res x init_res`` grid; an extension passes only its ring,
        from :func:`ring_triangles`.
    seed: Optional[Tuple[VertexCache, ArrayLike, LeafStore]]
        ``(cache, active, store)`` to continue from, as
        :func:`seed_from_mesh` rebuilds them; ``None`` starts empty. Seeded
        rows take part in the balance cascade like any other, but only
        ``roots`` and their descendants are tested. The cascade's frontier
        bound assumes the leaves were balanced before each level, which a
        seed need not be, so a seeded run must be followed by
        :func:`balance`.

    Returns
    -------
    cache: VertexCache
    active: ArrayLike
        The active-vertex set.
    store: LeafStore
        Every row below ``max_level`` is ``LEAF_CONVERGED``; ``max_level`` rows
        carry the full status bitmask.
    counters: dict[str, int]
        ``converged_level0``: leaves converged at level 0.
        ``parity_splits``: level ``< max_level`` triangles split on parity --
        child parity, or Jacobian parity for a triangle that would otherwise
        have converged, a non-finite or singular Jacobian included.
        ``parity_invalid``: ``max_level`` triangles failing child parity or
        Jacobian parity, the Jacobian evaluated on all of them.
        ``deviation_splits``: triangles passing child parity but split by the
        deviation test. The Jacobian is never evaluated on these.
        ``sigma_zero``: triangles seen with ``s == 0`` (parity-ok or not).
        ``nonfinite_splits``: level ``< max_level`` triangles split because some
        sample was non-finite.
        ``max_level_midpoints``: unique ``max_level`` midpoints traced for the
        parity test.
        ``forced``: children produced by the balance cascade.
        ``cascade_rounds``: balance-cascade rounds run over the whole build.
    band: CriticalBand
        The critical band, with ``leaves`` indexing ``store``'s rows. Empty
        when the loop ends before ``max_level``.
    """
    M, G, COMPOSE, PINV0, ROOT_CLASS = tables
    if seed is None:
        cache, active, store = empty_cache(), empty_active(), empty_store()
    else:
        cache, active, store = seed
    counters = {
        "converged_level0": 0,
        "parity_splits": 0,
        "parity_invalid": 0,
        "deviation_splits": 0,
        "sigma_zero": 0,
        "nonfinite_splits": 0,
        "max_level_midpoints": 0,
        "forced": 0,
        "cascade_rounds": 0,
    }

    if roots is None:
        active_ij, active_cls = initial_triangles(init_res, lat.level, ROOT_CLASS)
    else:
        active_ij, active_cls = roots
    deferred = backend.empty((0,), dtype=backend.int64)
    band = empty_band()

    for level in range(max_level + 1):
        vert_keys = lattice_key(lat, active_ij)  # (n, 3)
        need = [vert_keys.reshape(-1)]
        if level < max_level:
            mid_ij = midpoint_ij(active_ij)
            need.append(lattice_key(lat, mid_ij).reshape(-1))
            need.append(deferred)
            deferred = backend.empty((0,), dtype=backend.int64)
        cache = evaluate(
            cache, lat, backend.concatenate(need, dim=0), raytrace_fn, batch_size
        )

        v = cache_lookup(cache, vert_keys)
        active = active_add_slots(active, cache_size(cache), v.reshape(-1))
        beta_v = cache.beta[v]
        finite_v = backend.all(backend.isfinite(beta_v), dim=(1, 2))

        if level == max_level:
            # has_converged = backend.zeros((active_ij.shape[0],), dtype=backend.bool)
            # finite_samples = backend.zeros((active_ij.shape[0],), dtype=backend.bool)
            status = (
                backend.zeros((active_ij.shape[0],), dtype=backend.int64)
                + LEAF_RAYTRACE_NONFINITE
            )
            rows = backend.flatnonzero(finite_v)

            if rows.shape[0]:
                # Trace unique midpoints without adding them to the vertex cache.
                # Flattened before `backend.unique` rather than leaning on a
                # shape-preserving `return_inverse`, so the reshapes below are
                # explicit and this does not depend on the backend's own
                # convention.
                keys = lattice_key(lat, midpoint_ij(active_ij[rows])).reshape(-1)
                uniq, inv = backend.unique(keys, return_inverse=True)
                inv = inv.reshape(-1)
                mid_beta = trace_keys(
                    lat, lattice_ij_from_key(lat, uniq), raytrace_fn, batch_size
                )
                beta_m = mid_beta[inv].reshape(-1, 3, 2)
                counters["max_level_midpoints"] = int(uniq.shape[0])

                # Only evaluate the criterion where all six samples are finite.
                finite_m = backend.all(backend.isfinite(beta_m), dim=(1, 2))
                good_rows = rows[finite_m]

                if good_rows.shape[0]:
                    good_ij = active_ij[good_rows]
                    six_ij = backend.concatenate((good_ij, midpoint_ij(good_ij)), dim=1)
                    theta = lattice_xy(lat, six_ij)
                    # One Jacobian call over the distinct samples, not six per
                    # triangle: shared samples then carry one `A`, which the
                    # critical band needs, and the batch shrinks ~2.5x.
                    sample_keys, sample_index, J = sample_jacobians(
                        lat, six_ij, jacobian_fn
                    )

                    keep, parity_ok, s, criterion_status = evaluate_criterion(
                        jacobian_fn,
                        theta[:, :3],
                        theta[:, 3:],
                        beta_v[good_rows],
                        beta_m[finite_m],
                        active_cls[good_rows],
                        level,
                        h0,
                        min_img_sep,
                        PINV0,
                        COMPOSE,
                        force_jacobian=True,
                        jacobian=J[sample_index],
                    )

                    status = backend.fill_at_indices(
                        status, good_rows, criterion_status
                    )
                    counters["parity_invalid"] = int(
                        backend.to_numpy(backend.sum(~parity_ok))
                    )
                    counters["sigma_zero"] += int(backend.to_numpy(backend.sum(s == 0)))

                    band = band_from_samples(
                        lat, cache, sample_keys, sample_index, J, uniq, mid_beta
                    )
                    band = band._replace(leaves=good_rows[band.leaves])

            has_converged = status == LEAF_CONVERGED

            store, max_rows = store_add(store, v, level, active_cls, status)
            band = band._replace(leaves=max_rows[band.leaves])

            if level == 0:
                counters["converged_level0"] = int(
                    backend.to_numpy(backend.sum(has_converged))
                )

            break

        m = cache_lookup(cache, lattice_key(lat, mid_ij))
        beta_m = cache.beta[m]
        good = finite_v & backend.all(backend.isfinite(beta_m), dim=(1, 2))
        counters["nonfinite_splits"] += int(backend.to_numpy(backend.sum(~good)))

        rows = backend.flatnonzero(good)

        theta_v = lattice_xy(lat, active_ij[rows])
        theta_m = lattice_xy(lat, mid_ij[rows])

        keep, parity_ok, s, _status = evaluate_criterion(
            jacobian_fn,
            theta_v,
            theta_m,
            beta_v[rows],
            beta_m[rows],
            active_cls[rows],
            level,
            h0,
            min_img_sep,
            PINV0,
            COMPOSE,
        )
        counters["parity_splits"] += int(backend.to_numpy(backend.sum(~parity_ok)))
        counters["deviation_splits"] += int(
            backend.to_numpy(backend.sum(parity_ok & ~keep))
        )
        counters["sigma_zero"] += int(backend.to_numpy(backend.sum(s == 0)))

        # A non-finite triangle joins the criterion's failures in `pending`
        # rather than terminating: the criterion cannot be evaluated on it, so
        # the split is unconditional. Sorted, so which reason condemned a
        # triangle never reaches the child ordering.
        done = rows[keep]
        pending = backend.sort(
            backend.concatenate((backend.flatnonzero(~good), rows[~keep]), dim=0)
        )
        store, _ = store_add(store, v[done], level, active_cls[done], LEAF_CONVERGED)
        if level == 0:
            counters["converged_level0"] = int(done.shape[0])

        child_v, child_cls = red_split(
            v[pending], m[pending], active_cls[pending], COMPOSE
        )
        child_ij = cache.ij[child_v]
        active = active_add_slots(active, cache_size(cache), child_v.reshape(-1))

        # Balance cascade. The children above are already registered as active
        # vertices, which is what makes the quarter-point test able to see them --
        # the split must precede the cascade, not follow it.
        frontier_level = level + 1
        while True:
            violators = find_unbalanced(
                store, cache, lat, active, max_level, frontier_level
            )
            if violators.shape[0] == 0:
                break
            counters["cascade_rounds"] += 1
            store, cache, active, kid_mid, kid_level = force_split(
                store, cache, lat, active, violators, COMPOSE, raytrace_fn, batch_size
            )
            counters["forced"] += int(kid_level.shape[0])
            # Forced children are produced after this level's raytrace call has
            # gone out, and land at levels the loop will never revisit. Queue
            # their midpoints and drain at the top of the next level, so the
            # one-batch-per-level structure survives. A forced child cannot be
            # re-forced within the same cascade, because the frontier only moves
            # coarser -- so the deferral is never more than one level deep.
            deferred = backend.concatenate((deferred, kid_mid), dim=0)
            # The MAXIMUM kid level, not the minimum: a kid at level L invalidates
            # neighbours at level <= L-2, so the minimum would skip violators.
            # The maximum strictly decreases each round, which terminates the loop.
            frontier_level = int(backend.to_numpy(backend.max(kid_level)))

        active_ij, active_cls = child_ij, child_cls
        if active_ij.shape[0] == 0:
            break

    return cache, active, store, counters, band


def balance(
    store, cache, lat, active, max_level, compose, raytrace_fn, batch_size
) -> Tuple[LeafStore, VertexCache, ArrayLike, int]:
    """
    Force-split until no leaf is unbalanced: the coarsest 2:1-balanced refinement.

    :func:`refine`'s cascade scans only rows at ``level <= frontier_level -
    2``, which is complete only when the leaves were balanced before each
    level. That holds in a fresh build, whose new vertices only ever appear
    at the frontier, and fails in an extension, whose ring starts at level 0
    beside old leaves as deep as ``max_level``. This scans every valid row at
    ``level <= max_level - 2`` instead, splits what it finds with
    :func:`force_split`, and repeats until a scan finds nothing.

    Every row split has an active quarter point, and active vertices are only
    ever added, so every balanced refinement containing the current leaves
    splits that row too. The fixed point is therefore the coarsest balanced
    refinement of the leaves it started from -- the unique mesh a fresh
    build's incremental cascade reaches -- whatever order rows are met in.
    Rounds are bounded by the largest level gap between neighbours, at most
    about ``max_level``, each one scan of the store. After a fresh
    :func:`refine` the first scan finds nothing. ``max_level`` rows are never
    split, so every critical-band row stays a leaf.

    Parameters
    ----------
    store: LeafStore
    cache: VertexCache
    lat: Lattice
    active: ArrayLike
    max_level: int
    compose: ArrayLike
        ``COMPOSE`` from :func:`child_matrix_tables`.
    raytrace_fn: Callable[[ArrayLike], ArrayLike]
        Forwarded to :func:`force_split`, for midpoints not yet cached.
    batch_size: Optional[int]
        Forwarded to :func:`force_split`.

    Returns
    -------
    store: LeafStore
    cache: VertexCache
    active: ArrayLike
    forced: int
        Children created; zero when the leaves were already balanced.
    """
    forced = 0
    while True:
        violators = find_unbalanced(store, cache, lat, active, max_level, max_level)
        if violators.shape[0] == 0:
            return store, cache, active, forced
        store, cache, active, _, kid_level = force_split(
            store, cache, lat, active, violators, compose, raytrace_fn, batch_size
        )
        forced += int(kid_level.shape[0])


# ---------------------------------------------------------------------------
# Canonical ordering and mesh closure
# ---------------------------------------------------------------------------


def canonical_order(lat, cache, v) -> ArrayLike:
    """
    Deterministic leaf ordering, independent of the order the cascade produced.

    Sorts by the row-wise-sorted triple of vertex lattice keys, which is unique per
    triangle since a triangle is determined by its vertex set. This makes
    byte-identical output a property of the data rather than of control flow.

    Parameters
    ----------
    lat: Lattice
    cache: VertexCache
    v: ArrayLike
        Leaf vertex slots, shape ``(n, 3)`` int64.

    Returns
    -------
    ArrayLike
        Int64 permutation of ``0 .. n - 1`` that sorts ``v`` into canonical order.
    """
    keys = backend.sort(lattice_key(lat, cache.ij[v]), dim=1)
    # `backend.lexsort` follows NumPy's convention: the LAST key given is the
    # PRIMARY sort key, so the row's smallest key (`keys[:, 0]`) goes last.
    return backend.lexsort([keys[:, 2], keys[:, 1], keys[:, 0]])


def min_angle(tri) -> ArrayLike:
    """
    Smallest interior angle of each triangle, in radians.

    Parameters
    ----------
    tri: ArrayLike
        Shape ``(n, 3, 2)``.

    Returns
    -------
    ArrayLike
        Shape ``(n,)``. Degenerate triangles give ``0.0`` rather than ``NaN``.
    """
    a = backend.norm(tri[:, 2] - tri[:, 1], dim=-1)
    b = backend.norm(tri[:, 0] - tri[:, 2], dim=-1)
    c = backend.norm(tri[:, 1] - tri[:, 0], dim=-1)
    # A degenerate triangle makes one of these a 0/0 division. Neither backend
    # warns the way NumPy's default error mode does, so there is no
    # `np.errstate` context to drop into an equivalent.
    cosines = backend.stack(
        (
            (b * b + c * c - a * a) / (2 * b * c),
            (c * c + a * a - b * b) / (2 * c * a),
            (a * a + b * b - c * c) / (2 * a * b),
        ),
        dim=1,
    )
    angles = backend.arccos(backend.clamp(cosines, -1.0, 1.0))
    # Both backends' `nan_to_num` already default a bare NaN to 0.0 when `nan`
    # is left unset -- matching the oracle's explicit `nan=0.0` -- and the
    # wrapper here does not itself expose a `nan` keyword.
    return backend.nan_to_num(backend.min(angles, dim=1))


def close(
    lat, cache, active, v, level, status
) -> Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
    """
    Make the balanced mesh conforming, using the pre-closure active-vertex set.

    Fixed pattern table, all vertices already cached:

    - 1 hanging node at ``m_i``: bisect from the opposite vertex into two triangles.
    - 2 hanging nodes: emit the corner triangle at the vertex opposite the whole
      edge, then split the remaining quadrilateral along whichever of its two
      diagonals maximizes the minimum angle.
    - 3 hanging nodes: the canonical red split, free.

    Parameters
    ----------
    lat: Lattice
    cache: VertexCache
    active: ArrayLike
        Pre-closure active-vertex set.
    v: ArrayLike
        Pre-closure leaf vertex slots in canonical order, shape ``(L0, 3)``.
    level, status: ArrayLike
        Shape ``(L0,)``, inherited by the emitted triangles.

    Returns
    -------
    leaves: ArrayLike
        ``(L, 3)`` vertex slots, positively oriented.
    origin: ArrayLike
        ``(L,)`` index into ``v``. Non-decreasing, so each origin's terminal
        triangles are contiguous and grouping is a slice.
    out_level, out_status: ArrayLike
        ``(L,)``, inherited from the origin.
    """
    v_ij = cache.ij[v]
    mid_ij = midpoint_ij(v_ij)
    mid_keys = lattice_key(lat, mid_ij)
    m = cache_lookup(cache, mid_keys)
    # No exactness gate. The lattice is one level finer than max_level, so
    # `midpoint_ij` is exact at every level and `mid_keys` always names the
    # true midpoint. At max_level that midpoint has an odd coordinate, is
    # traced transiently by `refine` and never cached, so `cache_lookup`
    # returns -1 and `active_contains_slots` reads it as False. Even were it
    # cached it could not be *active*: an active slot is by definition a
    # triangle vertex, and every triangle vertex has even coordinates. So no
    # max_level leaf has a hanging node -- the invariant `refine` relies on --
    # and the count == 0 pass-through below is still every max_level leaf's
    # only route through.
    #
    # `m` is already `cache_lookup(cache, mid_keys)`; asking by slot skips a
    # second searchsorted over the same keys.
    hanging = active_contains_slots(active, m)  # (L0, 3)
    # `v_ij` and the midpoint coordinates are dead from here: the pattern
    # tables below work in slots, and `geom` re-gathers from the cache.
    del v_ij, mid_ij, mid_keys
    count = backend.to(backend.sum(hanging, dim=1), dtype=backend.int64)

    # `count` is always one of {0, 1, 2, 3} -- three possibly-hanging edge
    # midpoints -- so `count + 1` is the same map as
    # `np.choose(count, [1, 2, 3, 4])`: 0 -> 1, 1 -> 2, 2 -> 3, 3 -> 4.
    n_children = count + 1
    offsets = backend.concatenate(
        [backend.zeros((1,), dtype=backend.int64), backend.cumsum(n_children, dim=0)],
        dim=0,
    )
    total = int(backend.to_numpy(offsets[-1]))
    leaves = backend.empty((total, 3), dtype=backend.int64)
    origin = backend.repeat(
        backend.arange(v.shape[0], dtype=backend.int64), n_children, axis=0
    )

    def geom(slots):
        return lattice_xy(lat, cache.ij[slots])

    sel = backend.flatnonzero(count == 0)
    leaves = backend.fill_at_indices(leaves, offsets[sel], v[sel])

    for i in range(3):
        j, k = (i + 1) % 3, (i + 2) % 3
        sel = backend.flatnonzero((count == 1) & hanging[:, i])
        if sel.shape[0] == 0:
            continue
        o = offsets[sel]
        leaves = backend.fill_at_indices(
            leaves, o, backend.stack((v[sel, i], v[sel, j], m[sel, i]), dim=1)
        )
        leaves = backend.fill_at_indices(
            leaves, o + 1, backend.stack((v[sel, i], m[sel, i], v[sel, k]), dim=1)
        )

    for c in range(3):
        a, b = (c + 1) % 3, (c + 2) % 3
        sel = backend.flatnonzero((count == 2) & ~hanging[:, c])
        if sel.shape[0] == 0:
            continue
        o = offsets[sel]
        # Corner triangle at theta_c: exactly red-split child C_c, so positively
        # oriented by construction.
        leaves = backend.fill_at_indices(
            leaves, o, backend.stack((v[sel, c], m[sel, b], m[sel, a]), dim=1)
        )
        a1 = backend.stack((v[sel, a], v[sel, b], m[sel, a]), dim=1)
        a2 = backend.stack((v[sel, a], m[sel, a], m[sel, b]), dim=1)
        b1 = backend.stack((v[sel, a], v[sel, b], m[sel, b]), dim=1)
        b2 = backend.stack((v[sel, b], m[sel, a], m[sel, b]), dim=1)
        score_a = backend.minimum(min_angle(geom(a1)), min_angle(geom(a2)))
        score_b = backend.minimum(min_angle(geom(b1)), min_angle(geom(b2)))
        use_b = score_b > score_a  # ties take candidate A, deterministically
        leaves = backend.fill_at_indices(
            leaves, o + 1, backend.where(use_b[:, None], b1, a1)
        )
        leaves = backend.fill_at_indices(
            leaves, o + 2, backend.where(use_b[:, None], b2, a2)
        )

    sel = backend.flatnonzero(count == 3)
    if sel.shape[0]:
        o = offsets[sel]
        six = backend.concatenate([v[sel], m[sel]], dim=1)  # (k, 6)
        # Raw concatenate + fancy-index gather, deliberately not a call to
        # `red_split`: this is the only path that reaches the count == 3
        # branch, and going through `red_split` would give it zero coverage
        # of its own from this branch.
        kids = six[:, _CHILD_VERTEX_INDEX_TABLE]  # (k, 4, 3)
        for t in range(4):
            leaves = backend.fill_at_indices(leaves, o + t, kids[:, t, :])

    return leaves, origin, level[origin], status[origin]


# ---------------------------------------------------------------------------
# Freeze-time invalidation and the spatial index
# ---------------------------------------------------------------------------


def invalidate_nonfinite_origins(vs, leaves, origin, pre_status) -> ArrayLike:
    """
    Re-check finiteness at freeze and flag non-finite origins.

    A balance-cascade child inherits its vertices from a parent whose
    midpoints were never finiteness-tested, and a closure triangle can pick
    up a midpoint no criterion ever saw, so a non-finite vertex can reach
    freeze on a leaf not already flagged. Without this it would enter the
    spatial index and swallow every query in its cell.

    The flag is propagated UP to the origin and then back DOWN to every one
    of its leaves, rather than applied to the bad leaf alone: the
    termination-reason counts are pre-closure, so marking only the leaf would
    leave them disagreeing with the pre-closure leaf count. It is also the
    conservative direction -- if one triangle of a region has a bad vertex,
    the region is not trustworthy.

    ``LEAF_RAYTRACE_NONFINITE`` is OR-ed in rather than assigned, so an origin
    keeps every flag it already carried and gains this one alongside. The
    oracle instead overwrote the status with its single ``NONFINITE`` code.

    The oracle (``old_adaptive._invalidate_nonfinite_origins``) computes the
    per-origin flag with ``np.logical_or.at(origin_bad, origin, ~leaf_finite)``,
    an OR-scatter over possibly-repeated ``origin`` indices. That scatter has
    no backend-agnostic form: torch's index assignment keeps the LAST write on
    a repeated index, while jax's analogous ``.at[]`` update ACCUMULATES --
    neither reproduces an OR-reduction on both backends. Counting bad leaves
    per origin with ``bincount`` and testing ``> 0`` sidesteps the scatter
    entirely: it *is* an OR-reduction over duplicates, and it is a primitive
    both backends already agree on. ``minlength=n_origins`` is an exact upper
    bound here -- every ``origin`` value is a valid index into ``pre_status``
    by construction. ``minlength`` is a lower bound on both backends, while
    the construction here also guarantees the result has exactly that length.

    Parameters
    ----------
    vs: ArrayLike
        Vertex-cache positions, shape ``(V, 2)``.

        *Unit: arcsec*
    leaves: ArrayLike
        Pre-closure or closure leaf vertex slots, shape ``(L, 3)``, int64 into
        ``vs``.
    origin: ArrayLike
        Shape ``(L,)``, int64 index into ``pre_status``. Need not be sorted.
    pre_status: ArrayLike
        Shape ``(N,)``, one status bitmask per origin.

    Returns
    -------
    ArrayLike
        ``pre_status`` with ``LEAF_RAYTRACE_NONFINITE`` OR-ed into every origin
        owning a non-finite leaf, and every other origin unchanged.
    """
    n_origins = pre_status.shape[0]
    leaf_finite = backend.all(backend.isfinite(vs[leaves]), dim=(1, 2))
    bad_rows = backend.flatnonzero(~leaf_finite)
    origin_bad = backend.bincount(origin[bad_rows], minlength=n_origins) > 0
    return pre_status | (backend.long(origin_bad) * LEAF_RAYTRACE_NONFINITE)


class MeshIndex(NamedTuple):
    """
    Uniform-grid CSR index over source-plane leaf bounding boxes.

    One cell lookup per query is complete: ``beta`` lies in the triangle,
    which lies in its AABB, which is covered by the cells the leaf registered
    in, so ``beta``'s own cell always contains any leaf containing ``beta``.
    No neighbour search is needed. Per-cell lists are stored ascending, which
    gives ``query`` its sorted CSR blocks with no sort at query time.

    ``lo``/``hi`` are the source-plane bounding box of every indexed leaf's
    vertices and ``cell`` is the per-axis cell size, all shape ``(2,)``.
    ``cell_offsets`` (shape ``(nx * ny + 1,)``) and ``cell_leaves`` are the
    CSR arrays: cell ``c``'s leaves are
    ``cell_leaves[cell_offsets[c]:cell_offsets[c + 1]]``, ascending.
    """

    lo: ArrayLike
    hi: ArrayLike
    cell: ArrayLike
    nx: int
    ny: int
    cell_offsets: ArrayLike
    cell_leaves: ArrayLike


def build_index(vs, leaves, valid_rows, index_cells) -> MeshIndex:
    """
    Build the uniform-grid CSR spatial index over ``valid_rows``' AABBs.

    Deliberately float64 regardless of the mesh's own dtype: build-side and
    query-side cell arithmetic (``mesh_query``'s ``(chunk - lo) / cell``) must
    agree by construction, and forcing this to the mesh's own dtype would let
    the two sides round independently right at cell boundaries -- worse, not
    cleaner.

    Parameters
    ----------
    vs: ArrayLike
        Vertex-cache positions, shape ``(V, 2)``.

        *Unit: arcsec*
    leaves: ArrayLike
        All leaf vertex slots, shape ``(L, 3)``, int64 into ``vs``.
    valid_rows: ArrayLike
        Ascending row indices of ``leaves`` to index, shape ``(K,)``.
    index_cells: int or None
        Target cell count along the larger span axis. ``None`` sizes cells
        from the mean leaf density instead.

    Returns
    -------
    MeshIndex
    """
    # `vs` is already float64 unless the caller asked for a float32 mesh, and
    # this gather is the largest single array in the function -- `backend.to`
    # skips the copy when the dtype already matches (as does `Tensor.to`; a
    # jax array is immutable regardless), so casting unconditionally does not
    # double it for nothing.
    tri = backend.to(vs[leaves[valid_rows]], dtype=backend.float64)
    if tri.shape[0] == 0:
        return MeshIndex(
            lo=backend.zeros((2,), dtype=backend.float64),
            hi=backend.ones((2,), dtype=backend.float64),
            cell=backend.ones((2,), dtype=backend.float64),
            nx=1,
            ny=1,
            cell_offsets=backend.zeros((2,), dtype=backend.int64),
            cell_leaves=backend.empty((0,), dtype=backend.int64),
        )
    flat = tri.reshape(-1, 2)
    lo, hi = backend.min(flat, dim=0), backend.max(flat, dim=0)
    span = backend.where(hi > lo, hi - lo, 1.0)  # a degenerate axis becomes one cell

    # `nx`, `ny` and the cell size are grid *shape*, not mesh data, and every
    # other host-side kernel in this module already drops to a Python scalar
    # for exactly this (see `depth_floor`, `make_lattice`). One `to_numpy`
    # call here, not a separate one for each of the handful of scalars it
    # takes to get there.
    span_np = backend.to_numpy(span)
    if index_cells is None:
        c = float(math.sqrt(span_np[0] * span_np[1] / tri.shape[0]))
    else:
        c = float(span_np.max()) / int(index_cells)
    c = max(c, float(backend.finfo(backend.float64).tiny))
    nx = max(1, int(math.ceil(span_np[0] / c)))
    ny = max(1, int(math.ceil(span_np[1] / c)))
    cell = span / backend.as_array([nx, ny], dtype=backend.float64)

    upper = backend.as_array([nx - 1, ny - 1], dtype=backend.int64)
    # `backend.clamp` requires min and max to both be Tensors when either one
    # is, on torch (a bare Python `0` alongside the array `upper` raises); a
    # zeros array makes both bounds a Tensor on both backends.
    lower = backend.zeros((2,), dtype=backend.int64)
    i0 = backend.clamp(
        backend.long((backend.min(tri, dim=1) - lo) / cell), lower, upper
    )
    i1 = backend.clamp(
        backend.long((backend.max(tri, dim=1) - lo) / cell), lower, upper
    )
    tall = i1[:, 1] - i0[:, 1] + 1
    counts = (i1[:, 0] - i0[:, 0] + 1) * tall
    owner = backend.repeat(
        backend.arange(counts.shape[0], dtype=backend.int64), counts, axis=0
    )
    total_pairs = int(backend.to_numpy(backend.sum(counts)))
    within = backend.arange(total_pairs, dtype=backend.int64) - backend.repeat(
        backend.cumsum(counts, dim=0) - counts, counts, axis=0
    )
    cell_id = (i0[owner, 0] + within // tall[owner]) * ny + (
        i0[owner, 1] + within % tall[owner]
    )
    # A stable sort on `cell_id` alone, not a lexsort on `(leaf_id, cell_id)`.
    # `owner` is non-decreasing by construction and `valid_rows` is ascending,
    # so `leaf_id = valid_rows[owner]` is already sorted in generation order
    # and a stable sort reproduces the lexsort's tie-breaking exactly. Ties in
    # the pair cannot occur at all: within one leaf the cell rectangle is
    # enumerated bijectively, so a leaf never registers in a cell twice.
    order = backend.argsort(cell_id)
    # `leaf_id` is deferred past the sort. It is only needed to fill
    # `cell_leaves`, and materialising it beforehand costs another array as
    # long as the (cell, leaf) pair list -- in the function that already
    # dominates the build's peak memory.
    cell_leaves = valid_rows[owner[order]]
    # Counting the pairs per cell is the same thing as `searchsorted` over a
    # sorted key array, by the definition of a CSR offset -- and `bincount`
    # runs on the *unsorted* ids, so the sorted copy `cell_id[order]` and a
    # `nx * ny + 1`-long probe array are both never built. `minlength=nx*ny`
    # is an exact upper bound: `i0`/`i1` are clamped to
    # `[0, nx - 1] x [0, ny - 1]` by construction, so `cell_id < nx * ny`
    # always, so the minimum-length result is exactly `nx * ny` entries.
    counts_per_cell = backend.long(backend.bincount(cell_id, minlength=nx * ny))
    cell_offsets = backend.concatenate(
        [
            backend.zeros((1,), dtype=backend.int64),
            backend.cumsum(counts_per_cell, dim=0),
        ],
        dim=0,
    )
    return MeshIndex(
        lo=lo,
        hi=hi,
        cell=cell,
        nx=nx,
        ny=ny,
        cell_offsets=cell_offsets,
        cell_leaves=cell_leaves,
    )


# ---------------------------------------------------------------------------
# Assembly: the frozen mesh and its public build entry point
# ---------------------------------------------------------------------------


class AdaptiveMesh(NamedTuple):
    """
    A frozen adaptive mesh of the lens plane, queryable from the source plane.

    One topology, two embeddings. Vertex index ``v`` is shared across both
    planes -- ``vertices_lens[v]`` and ``vertices_source[v]`` are the same
    point's two positions -- so there is no separate source-plane triangle
    table; the correspondence is structural rather than an invariant kept in
    sync.

    Leaves with any failure flag set -- every status but ``LEAF_CONVERGED`` --
    remain in ``leaves`` and ``leaf_status`` but are never registered in
    ``index``, so a source-plane query cannot return them. That is a genuine
    coverage hole in the lens plane, sized at ``min_img_sep`` scale rather
    than ``init_res`` scale, since a flag is only ever stored at ``max_level``
    (or OR-ed in at freeze, for a non-finite vertex): a leaf failing either
    parity test straddles a fold or critical curve the mesh cannot resolve, one
    failing the deviation test has no affine model accurate to
    ``min_img_sep``, and one with a non-finite sample could never be tested at
    all. Reporting no coverage there is the conservative, deliberate answer --
    correctness over completeness exactly where images merge or diverge.
    ``leaf_status`` keeps every reason, so the hole can be taken apart
    afterwards: ``(leaf_status & LEAF_JACOBIAN_PARITY_UNRESOLVED) != 0``, for
    instance, selects the leaves with a critical curve running between their
    samples. That flag misses a leaf whose samples straddle the curve only
    through an exact ``det A == 0``, so ``critical_band`` is the complete
    record of every ``max_level`` leaf ``det A`` changes sign across;
    :func:`~caustics.lenses.func.adaptive_critical.mesh_critical_curves`
    turns it into the ordered curves themselves.

    ``min_img_sep`` is stored because it is the mesh's own defining tolerance
    -- the halved value :func:`build_adaptive_mesh` actually refined to, not
    the value the caller passed. The lens deliberately is **not** stored, nor
    its ``raytrace``: a callable carries no identity the mesh could check, so
    holding one would imply a guarantee that it matches the build when nothing
    can enforce it.

    Parameters
    ----------
    vertices_lens: ArrayLike
        Lens-plane position of every vertex, shape ``(V, 2)``.

        *Unit: arcsec*
    vertices_source: ArrayLike
        Source-plane image of every vertex, shape ``(V, 2)``, at ``dtype``.

        *Unit: arcsec*
    vertices_ij: ArrayLike
        Lattice coordinates of every vertex, shape ``(V, 2)`` int64 -- the
        integer pair ``vertices_lens`` is computed from, exact at any
        ``dtype``. Vertices are in ascending lattice-key order.
    leaves: ArrayLike
        Terminal-triangle vertex indices, shape ``(L, 3)`` int64 into both
        vertex arrays, positively oriented.
    leaf_area2: ArrayLike
        Twice the signed source-plane area of each leaf, shape ``(L,)``.
        Computed from ``vertices_source`` at ``dtype``, so it is exact for
        whatever precision the mesh was frozen at -- not a higher-precision
        value cast down afterwards. Never consumed on a leaf outside the index,
        and meaningless on one with ``LEAF_RAYTRACE_NONFINITE`` set.

        *Unit: arcsec^2*
    leaf_origin: ArrayLike
        Shape ``(L,)`` int64 index into ``origin_leaves``' row axis (and into
        the pre-closure leaf list conceptually): the triangle each leaf was
        closed from. Non-decreasing, so one origin's terminal triangles are
        contiguous.
    leaf_status: ArrayLike
        Shape ``(L,)`` int64 bitmask of ``LEAF_*`` failure flags,
        ``LEAF_CONVERGED`` (zero) where none is set, inherited from the leaf's
        origin.
    leaf_level: ArrayLike
        Shape ``(L,)`` int64 refinement level, inherited from the leaf's
        origin.
    origin_leaves: ArrayLike
        Shape ``(N, 3)`` int64, the pre-closure leaves' own vertex slots --
        what ``leaf_origin`` indexes into.
    origin_cls: ArrayLike
        Shape ``(N,)`` int64 orientation class of each pre-closure leaf, as
        :func:`child_matrix_tables` defines it. With ``origin_leaves``,
        ``leaf_level`` and ``leaf_status`` it is the whole pre-closure leaf
        store, which :func:`seed_from_mesh` rebuilds from.
    index: MeshIndex
        Spatial index over the source-plane bounding boxes of the
        ``LEAF_CONVERGED`` leaves, and of no other.
    critical_band: CriticalBand
        The ``max_level`` leaves ``det A`` changes sign across, with ``det A``
        and the image at their six samples, kept from the build's own
        ``max_level`` pass so that
        :func:`~caustics.lenses.func.adaptive_critical.mesh_critical_curves`
        needs no lens. A superset of the ``LEAF_JACOBIAN_PARITY_UNRESOLVED``
        leaves; see :class:`CriticalBand`.
    lattice: Lattice
        The lattice the mesh lives on, ``lo`` on ``device``. An extended
        mesh keeps its original build's ``lo`` and ``scale`` with
        ``origin > 0``; see :func:`extend_lattice`.
    fov: float
        Side length of the square domain actually meshed.

        *Unit: arcsec*
    init_res: int
        Level-0 cells per axis actually meshed.
    d_floor: int
        Level at which the level-0 hypotenuse first falls to ``min_img_sep``,
        uncapped by ``max_depth``.
    max_level: int
        Finest level actually reached, ``min(max_depth, d_floor)``.
    min_img_sep: float
        The halved tolerance the build refined to.

        *Unit: arcsec*
    dtype: Any
        Backend float dtype ``vertices_lens`` and ``vertices_source`` are
        stored at.
    device: Any
        Device the mesh's arrays live on.
    """

    vertices_lens: ArrayLike
    vertices_source: ArrayLike
    vertices_ij: ArrayLike
    leaves: ArrayLike
    leaf_area2: ArrayLike
    leaf_origin: ArrayLike
    leaf_status: ArrayLike
    leaf_level: ArrayLike
    origin_leaves: ArrayLike
    origin_cls: ArrayLike
    # `index` shadows `tuple.index` (the element-lookup method) by name --
    # deliberately, per the interface this task specifies -- which mypy
    # flags as incompatible with the inherited method's type. Runtime is
    # unaffected: `NamedTuple` fields become properties on the subclass, so
    # `mesh.index` always resolves to the field; nothing in this module ever
    # calls the shadowed `.index(value)` lookup method.
    index: MeshIndex  # type: ignore[assignment]
    critical_band: CriticalBand
    lattice: Lattice
    fov: float
    init_res: int
    d_floor: int
    max_level: int
    min_img_sep: float
    dtype: Any
    device: Any


def warn_depth_limited(fov, init_res, min_img_sep, d_floor, max_level) -> None:
    """
    Warn when ``max_depth`` stopped refinement short of the size floor.

    Shared by :func:`build_adaptive_mesh` and :func:`extend_adaptive_mesh`, so
    an extension warns exactly as a fresh build of its fov would. Refinement
    is depth-limited exactly when ``d_floor > max_level``, and ``max_level``
    is then ``max_depth``, so the message names ``max_depth`` without being
    given it. ``min_img_sep`` is the halved tolerance; the caller's request
    is ``2 * min_img_sep`` exactly, since halving is exact.
    """
    if d_floor <= max_level:
        return
    l_max_final = float(math.sqrt(2.0) * fov / (init_res * 2**max_level))
    warn(
        f"Adaptive mesh is depth-limited: max_depth={max_level} is below "
        f"d_floor={d_floor}, the depth required to reach "
        f"min_img_sep={2 * min_img_sep:g} arcsec (refined internally "
        f"to {min_img_sep:g}). Refinement stops at level {max_level}, "
        f"where the maximum leaf edge is {l_max_final:.3g} arcsec. Set "
        f"max_depth >= {d_floor} to restore the size-floor guarantee, or "
        f"raise init_res / min_img_sep."
    )


def warn_cancellation_floor(raytrace_fn, fov, min_img_sep) -> None:
    """
    Warn when the dtype ``raytrace`` returned cannot resolve ``min_img_sep``.

    The midpoint deviation compares ``O(fov)`` quantities, so its roundoff
    floor is ``sqrt(8 * eps * fov)`` (see :func:`make_raytrace`). ``fov`` is
    the fov being meshed, so an extension checks the larger one.
    ``min_img_sep`` is the halved tolerance, as for
    :func:`warn_depth_limited`.
    """
    # `.info` is attached dynamically by `make_raytrace` (documented on its
    # own `# type: ignore[attr-defined]` there); mypy has no way to see it on
    # the `Callable[[ArrayLike], ArrayLike]` return annotation.
    raytrace_dtype = raytrace_fn.info["dtype"]  # type: ignore[attr-defined]
    eps = float(backend.finfo(raytrace_dtype).eps)
    cancellation_floor = float(math.sqrt(8.0 * eps * fov))
    if cancellation_floor > min_img_sep:
        warn(
            f"raytrace returned {raytrace_dtype}, whose "
            f"cancellation floor sqrt(8*eps*fov) = {cancellation_floor:.3g} "
            f"arcsec exceeds min_img_sep={2 * min_img_sep:g} (refined "
            f"internally to {min_img_sep:g}). Below that scale the midpoint "
            "deviation cancels to zero, which the criterion reads as "
            "'affine' and converges. Supply a raytrace that preserves "
            "float64."
        )


def freeze(
    lat,
    cache,
    active,
    store,
    band,
    seed_band,
    *,
    fov,
    init_res,
    min_img_sep,
    d_floor,
    max_level,
    dtype,
    device,
    index_cells,
) -> AdaptiveMesh:
    """
    Close, order, index and freeze a balanced refinement.

    The last stage of every build: canonical ordering and closure, the
    critical band merged and mapped onto the closed leaves, vertex
    compaction, freeze-time invalidation, and the spatial index. It makes no
    lens call. :func:`build_adaptive_mesh` and :func:`extend_adaptive_mesh`
    both end here, so the frozen mesh is a function of the refinement alone,
    never of the order it was produced in.

    Parameters
    ----------
    lat: Lattice
    cache: VertexCache
    active: ArrayLike
    store: LeafStore
        Balanced, so :func:`balance` has run.
    band: CriticalBand
        :func:`refine`'s own, ``leaves`` indexing ``store``'s rows.
    seed_band: CriticalBand
        The band carried over from a seeding mesh, ``leaves`` indexing
        ``store``'s rows too; :func:`empty_band` for a fresh build. It wins
        :func:`merge_bands`' tie on a shared sample.
    fov: float
        *Unit: arcsec*
    init_res: int
    min_img_sep: float
        The halved tolerance the build refined to.

        *Unit: arcsec*
    d_floor, max_level: int
    dtype: Optional
        Frozen-mesh dtype, ``backend.float64`` when ``None``.
    device: Optional
    index_cells: Optional[int]
        Forwarded to :func:`build_index`.

    Returns
    -------
    AdaptiveMesh
    """
    pre_v, pre_level, pre_cls, pre_status = store_compact(store)
    order = canonical_order(lat, cache, pre_v)
    pre_v, pre_level = pre_v[order], pre_level[order]
    pre_cls, pre_status = pre_cls[order], pre_status[order]
    leaf_v, origin, leaf_level, leaf_status = close(
        lat, cache, active, pre_v, pre_level, pre_status
    )
    band = merge_bands(lat, cache, store, seed_band, band)
    band_leaves = band_leaf_index(store.valid, order, origin, band.leaves)
    # Trip-wire for the index chase above: a band row that landed on any leaf
    # but its own would pair one leaf's samples with another's vertices.
    if not bool(backend.all(leaf_v[band_leaves] == store.v[band.leaves])):
        raise AssertionError("a critical-band row did not map onto its own leaf")
    # Rows in leaf order, so band row order depends on the mesh alone, not on
    # the order refinement met the leaves in. One band row per leaf: no ties.
    rows = backend.argsort(band_leaves)
    band_leaves, band_samples = band_leaves[rows], band.samples[rows]

    # Compaction: sorting by lattice key makes vertex order a function of the
    # geometry alone and gives row-major locality for query-time gathers.
    # `leaf_v` alone: every closure pattern re-emits all three of its origin's
    # vertices, so `pre_v`'s slots are a subset of `leaf_v`'s and unioning them
    # would sort in millions of redundant entries on a large build. Guarded by
    # `test_closure_re_emits_every_origin_vertex`, which is what makes this a
    # checked property rather than an argument.
    used = backend.unique(leaf_v.reshape(-1))
    used = used[backend.argsort(lattice_key(lat, cache.ij[used]))]
    remap = backend.fill_at_indices(
        backend.zeros((cache_size(cache),), dtype=backend.int64),
        used,
        backend.arange(used.shape[0], dtype=backend.int64),
    )
    leaves = remap[leaf_v]
    origin_leaves = remap[pre_v]

    if dtype is None:
        dtype = backend.float64
    # Kept on the ambient (pre-`device`) array world here, deliberately: the
    # gathers and the freeze-time re-check just below index `vs` with
    # `leaves`/`origin`, which are still on that same ambient world, so moving
    # `vs` to `device` before them would risk indexing a `device` array with
    # an off-`device` index array. `device` is applied uniformly to every
    # returned field in one pass at the very end instead, once nothing further
    # indexes across them.
    vl = backend.to(lattice_xy(lat, cache.ij[used]), dtype=dtype)
    vs = backend.to(cache.beta[used], dtype=dtype)

    # Re-check finiteness at freeze. A LEAF_CONVERGED leaf produced by the
    # balance cascade inherits vertices from a parent whose midpoints were
    # never finiteness-tested, and a closure triangle can pick up a midpoint
    # no criterion ever saw, so a non-finite vertex can reach here on a leaf
    # not already flagged. Without this it would enter the index and swallow
    # every query in its cell.
    #
    # `LEAF_RAYTRACE_NONFINITE` is propagated UP to the origin and then back
    # DOWN to every leaf, rather than being applied to the leaf alone: the
    # termination-reason counts are pre-closure, so marking only the leaf
    # would leave them disagreeing with the pre-closure leaf count. It is
    # also the conservative direction -- if one triangle of a region has a
    # bad vertex, the region is not trustworthy.
    pre_status = invalidate_nonfinite_origins(vs, leaves, origin, pre_status)
    leaf_status = pre_status[origin]

    P = shape_matrix(vs[leaves])
    leaf_area2 = P[:, 0, 0] * P[:, 1, 1] - P[:, 0, 1] * P[:, 1, 0]
    valid_rows = backend.flatnonzero(leaf_status == LEAF_CONVERGED)
    index = build_index(vs, leaves, valid_rows, index_cells)

    def to_device(array):
        return backend.to(array, device=device)

    critical_band = CriticalBand(
        leaves=to_device(band_leaves),
        samples=to_device(band_samples),
        lens=to_device(backend.to(band.lens, dtype=dtype)),
        source=to_device(backend.to(band.source, dtype=dtype)),
        det=to_device(band.det),
    )

    return AdaptiveMesh(
        vertices_lens=to_device(vl),
        vertices_source=to_device(vs),
        vertices_ij=to_device(cache.ij[used]),
        leaves=to_device(leaves),
        leaf_area2=to_device(leaf_area2),
        leaf_origin=to_device(origin),
        leaf_status=to_device(leaf_status),
        leaf_level=to_device(leaf_level),
        origin_leaves=to_device(origin_leaves),
        origin_cls=to_device(pre_cls),
        index=MeshIndex(
            lo=to_device(index.lo),
            hi=to_device(index.hi),
            cell=to_device(index.cell),
            nx=index.nx,
            ny=index.ny,
            cell_offsets=to_device(index.cell_offsets),
            cell_leaves=to_device(index.cell_leaves),
        ),
        critical_band=critical_band,
        lattice=lat._replace(lo=to_device(lat.lo)),
        fov=float(fov),
        init_res=int(init_res),
        d_floor=d_floor,
        max_level=max_level,
        min_img_sep=float(min_img_sep),
        dtype=dtype,
        device=device,
    )


def _grow(
    raytrace_fn,
    jacobian_fn,
    lat,
    tables,
    *,
    roots,
    seed,
    seed_band,
    fov,
    init_res,
    h0,
    min_img_sep,
    d_floor,
    max_level,
    device,
    dtype,
    raytrace_batch_size,
    index_cells,
) -> AdaptiveMesh:
    """
    Refine ``roots`` over ``seed``, balance, and freeze: the stages both builds share.

    :func:`build_adaptive_mesh` passes every cell and no seed;
    :func:`extend_adaptive_mesh` passes the ring and the old mesh. Running
    both through here is what makes an extension a fresh build of the larger
    fov, rather than a second algorithm that happens to agree with one.
    """
    cache, active, store, _counters, band = refine(
        raytrace_fn,
        jacobian_fn,
        lat,
        init_res,
        h0,
        min_img_sep,
        max_level,
        tables,
        raytrace_batch_size,
        roots=roots,
        seed=seed,
    )
    warn_cancellation_floor(raytrace_fn, fov, min_img_sep)
    store, cache, active, _forced = balance(
        store,
        cache,
        lat,
        active,
        max_level,
        tables[2],
        raytrace_fn,
        raytrace_batch_size,
    )
    return freeze(
        lat,
        cache,
        active,
        store,
        band,
        seed_band,
        fov=fov,
        init_res=init_res,
        min_img_sep=min_img_sep,
        d_floor=d_floor,
        max_level=max_level,
        dtype=dtype,
        device=device,
        index_cells=index_cells,
    )


def build_adaptive_mesh(
    lens,
    fov,
    init_res,
    min_img_sep,
    max_depth=25,
    *,
    x0=0.0,
    y0=0.0,
    device=None,
    dtype=None,
    raytrace_batch_size=None,
    index_cells=None,
) -> AdaptiveMesh:
    """
    Build an adaptively refined triangular mesh of the lens plane.

    The mesh is built once and reused across many queries; it does not depend
    on any query point. The build runs four stages, all on ``backend``
    arrays: :func:`refine` over every level-0 cell, :func:`balance`, and
    :func:`freeze` -- canonical ordering and closure, the critical band,
    freeze-time invalidation and the spatial index. :func:`extend_adaptive_mesh`
    runs the same stages from a built mesh, seeded by :func:`seed_from_mesh`,
    which is what makes an extension a fresh build of the larger fov.

    Parameters
    ----------
    lens:
        The lens to mesh -- any caustics lens, or anything else exposing the
        same two methods on 1-D arrays of shape ``(N,)``:
        ``lens.raytrace(x, y) -> (bx, by)``, mapping lens-plane to
        source-plane coordinates, and ``lens.jacobian_lens_equation(x, y)``,
        returning that map's ``(N, 2, 2)`` Jacobian ``d(beta) / d(theta)``. The
        Jacobian decides :func:`jacobian_parity_ok`, which the criterion
        requires before it converges any leaf. ``raytrace`` goes through
        :func:`make_raytrace` -- float64 in and out, on ``device``, chunked by
        ``raytrace_batch_size`` -- while ``jacobian_lens_equation`` is called
        directly on the build's own float64 coordinates.
    fov: float
        Side length of the square lens-plane domain.

        *Unit: arcsec*
    init_res: int
        Number of **cells** per axis, giving ``2 * init_res**2`` level-0
        triangles. Differs from ``forward_raytrace``'s ``divisions``, which
        counts ``linspace`` *points* and yields ``(n - 1)**2`` cells.

        This also carries the completeness obligation: the criterion samples
        six points per triangle, so structure below the level-0 scale is
        invisible to it. ``init_res`` must already resolve the smallest
        curvature scale in the lens.
    min_img_sep: float
        Requested lens-plane tolerance. The parity-condemned band at
        ``max_level`` is model-dependent, so the build refines to
        ``min_img_sep / 2`` internally to narrow it; this is not a universal
        numerical bound on the band width. The
        halved value is the one stored on the returned :class:`AdaptiveMesh`,
        not the value passed in.

        *Unit: arcsec*
    max_depth: int
        Hard cap on refinement level. Refinement runs to
        ``min(max_depth, d_floor)``; a warning is raised if ``max_depth``
        binds.
    x0, y0: float
        Centre of the domain.

        *Unit: arcsec*
    device: Optional
        Device for the coordinates handed to ``raytrace`` and for the frozen
        mesh.
    dtype: Optional
        Frozen-mesh dtype, a **backend** dtype such as ``backend.float32`` --
        not a NumPy one. Defaults to ``backend.float64``, the build dtype,
        and is stored on the mesh exactly as given.
    raytrace_batch_size: Optional[int]
        Splits each per-level ``raytrace`` call for memory. Forwarded to
        :func:`refine`.
    index_cells: Optional[int]
        Spatial-index cells along the longer axis of the source-plane
        bounding box. Forwarded to :func:`build_index`.

    Returns
    -------
    AdaptiveMesh
    """

    raytrace = lens.raytrace
    jacobian = lens.jacobian_lens_equation

    # The parity-condemned band at max_level is model-dependent. Refining to
    # half the requested separation narrows it, without implying a universal
    # bound on its width. Halved once, here, before any use, so every stage
    # below -- validation, the depth floor, max_level, both warnings, refine,
    # and the value stored on the returned mesh -- sees this one halved value
    # and never the caller's original. `requested_min_img_sep` is kept so
    # `validate_build_args` can name what the caller actually passed; the
    # warning helpers recover it as `2 * min_img_sep`, exact since halving is.
    requested_min_img_sep = min_img_sep
    min_img_sep = min_img_sep / 2
    validate_build_args(fov, init_res, min_img_sep, max_depth, requested_min_img_sep)
    d_floor = depth_floor(fov, init_res, min_img_sep)
    max_level = min(int(max_depth), d_floor)
    warn_depth_limited(fov, init_res, min_img_sep, d_floor, max_level)

    tables = child_matrix_tables()
    lat = make_lattice(fov, x0, y0, init_res, max_level + 1)
    raytrace_fn = make_raytrace(raytrace, device)
    return _grow(
        raytrace_fn,
        jacobian,
        lat,
        tables,
        roots=None,
        seed=None,
        seed_band=empty_band(),
        fov=fov,
        init_res=init_res,
        h0=fov / init_res,
        min_img_sep=min_img_sep,
        d_floor=d_floor,
        max_level=max_level,
        device=device,
        dtype=dtype,
        raytrace_batch_size=raytrace_batch_size,
        index_cells=index_cells,
    )


def _ambient(array) -> ArrayLike:
    """``array`` on the build's ambient device, where :func:`refine` makes its arrays."""
    return backend.to(array, device=backend.device(backend.zeros((0,))))


def seed_from_mesh(
    mesh, lat, k, raytrace_fn, batch_size
) -> Tuple[VertexCache, ArrayLike, LeafStore, CriticalBand]:
    """
    Rebuild :func:`refine`'s state from a frozen mesh, on its lattice grown by ``k``.

    A frozen mesh keeps every leaf vertex with its lattice coordinates, and
    every pre-closure leaf with its level, status and orientation class --
    everything :func:`refine`, :func:`balance` and :func:`freeze` read, but
    the midpoints of leaves that never split. :func:`force_split` evaluates
    those on demand, for the few leaves an extension forces.

    - The cache holds the mesh's vertices at their shifted coordinates, slot
      ``i`` being mesh vertex ``i``. The vertices are in lattice-key order and
      the shift keeps it, so the key index is already sorted. ``beta`` is
      ``vertices_source`` at float64. For a mesh not stored at float64 the
      vertices on its outer boundary are raytraced again: they are the only
      old points the ring's criterion reads, and it must read float64 values
      straight from ``raytrace``, as :func:`make_raytrace` explains.
    - Every vertex is active: a mesh's vertices are exactly its leaves'.
    - The store has one row per origin, in ``origin_leaves`` order, with the
      level and status of its first leaf: ``leaf_origin`` is non-decreasing
      and every origin has a leaf, so ``searchsorted`` finds it with no
      scatter. Below ``max_level`` the only flag a status can carry is the
      freeze-time ``LEAF_RAYTRACE_NONFINITE``, which :func:`freeze` derives
      again, so those rows are reset to ``LEAF_CONVERGED`` -- which is also
      what a forced child must inherit, as in a fresh build.
    - The band is the mesh's own, its ``leaves`` mapped to store rows, with
      ``lens`` and ``source`` at float64.

    Parameters
    ----------
    mesh: AdaptiveMesh
    lat: Lattice
        ``extend_lattice(mesh.lattice, k)``, ``lo`` on the ambient device.
    k: int
        Level-0 cells added on each side.
    raytrace_fn: Callable[[ArrayLike], ArrayLike]
        From :func:`make_raytrace`, for the boundary of a non-float64 mesh.
    batch_size: Optional[int]
        Forwarded to :func:`trace_keys`.

    Returns
    -------
    cache: VertexCache
    active: ArrayLike
    store: LeafStore
    band: CriticalBand
        ``leaves`` indexing ``store``'s rows.
    """
    pad = int(k) << lat.level
    ij = _ambient(mesh.vertices_ij) + pad
    n_vertices = ij.shape[0]
    beta = backend.to(_ambient(mesh.vertices_source), dtype=backend.float64)
    if mesh.vertices_source.dtype != backend.float64:
        old = ij - pad
        n_old = lat.n - 2 * pad
        edge = backend.flatnonzero(backend.any((old == 0) | (old == n_old), dim=1))
        if edge.shape[0]:
            beta = backend.fill_at_indices(
                backend.copy(beta),
                edge,
                trace_keys(lat, ij[edge], raytrace_fn, batch_size),
            )
    cache = VertexCache(
        keys=lattice_key(lat, ij),
        slots=backend.arange(n_vertices, dtype=backend.int64),
        ij=ij,
        beta=beta,
    )
    active = backend.ones((n_vertices,), dtype=backend.bool)

    n_origins = mesh.origin_leaves.shape[0]
    leaf_origin = _ambient(mesh.leaf_origin)
    first = backend.searchsorted(
        leaf_origin, backend.arange(n_origins, dtype=backend.int64)
    )
    level = _ambient(mesh.leaf_level)[first]
    status = backend.where(
        level == mesh.max_level, _ambient(mesh.leaf_status)[first], LEAF_CONVERGED
    )
    store = LeafStore(
        v=_ambient(mesh.origin_leaves),
        level=level,
        cls=_ambient(mesh.origin_cls),
        status=status,
        valid=backend.ones((n_origins,), dtype=backend.bool),
    )

    old_band = mesh.critical_band
    band = CriticalBand(
        leaves=leaf_origin[_ambient(old_band.leaves)],
        samples=_ambient(old_band.samples),
        lens=backend.to(_ambient(old_band.lens), dtype=backend.float64),
        source=backend.to(_ambient(old_band.source), dtype=backend.float64),
        det=_ambient(old_band.det),
    )
    return cache, active, store, band


def _extension_cells(mesh, fov) -> int:
    """
    Level-0 cells per side that grow ``mesh`` to at least ``fov``.

    The smallest ``k`` with ``mesh.fov + 2 * k * h0 >= fov``, less a
    tolerance of ``1e-9`` cells, so an fov already on a cell boundary is not
    pushed out a further ring by one ulp of rounding.

    Raises
    ------
    ValueError
        If ``fov`` is not finite, or is smaller than ``mesh.fov``.
    """
    fov = float(fov)
    if not math.isfinite(fov):
        raise ValueError(f"fov must be finite, got {fov}")
    h0 = mesh.lattice.scale * (1 << mesh.lattice.level)
    cells = (fov - mesh.fov) / (2.0 * h0)
    if cells < -1e-9:
        raise ValueError(
            f"fov={fov:g} is smaller than the mesh's fov={mesh.fov:g}; an "
            "adaptive mesh can only grow"
        )
    return max(0, math.ceil(cells - 1e-9))


def extend_adaptive_mesh(
    mesh, lens, fov, *, raytrace_batch_size=None, index_cells=None
) -> AdaptiveMesh:
    """
    Grow an adaptive mesh to a larger fov about the same centre, reusing its refinement.

    For a mesh whose critical curves the fov cuts open -- see
    :func:`~caustics.lenses.func.adaptive_critical.mesh_critical_curves` --
    this adds whole level-0 cells around it instead of rebuilding. The result
    is bit for bit the mesh :func:`build_adaptive_mesh` produces on the same
    lattice: both run the same seed, refine, balance and freeze stages, a
    fresh build seeding from nothing and refining every cell, this seeding
    from ``mesh`` and refining only the new ring. The refinement criterion
    reads only a triangle's own samples, so every verdict inside the old
    domain still holds; only the 2:1 balance reaches across the seam, and
    :func:`balance` settles it to the same coarsest balanced refinement a
    fresh build's cascade reaches.

    A fresh :func:`build_adaptive_mesh` at the new fov computes its
    lattice's corner and spacing from that fov. Those match this mesh's --
    anchored at the original build, see :func:`extend_lattice` -- exactly
    when the arithmetic is exact, as with a dyadic fov, ``init_res`` and
    centre. Otherwise positions differ by rounding, which can in rare cases
    flip a verdict sitting at a threshold.

    Lens calls: the ring's own refinement, exactly what a fresh build spends
    there; midpoints inside old leaves the balance force-splits; and, for a
    mesh not stored at float64, the vertices on its outer boundary, raytraced
    again so the ring's criterion reads float64 values. No Jacobian call
    lands strictly inside the old domain. Closure, ordering and indexing
    still run over the whole mesh, without a lens call.

    Parameters
    ----------
    mesh: AdaptiveMesh
        The mesh to grow. It is not modified.
    lens:
        The lens ``mesh`` was built from, with the contract of
        :func:`build_adaptive_mesh`. Nothing can check that it is the same
        lens: the mesh deliberately stores none.
    fov: float
        Requested side length, about the mesh's own centre. Rounded up to the
        smallest ``mesh.fov + 2 * k * h0`` reaching it, ``h0`` being the
        level-0 cell size, within ``1e-9`` cells; the fov meshed is the
        result's ``fov``.

        *Unit: arcsec*
    raytrace_batch_size: Optional[int]
        As for :func:`build_adaptive_mesh`.
    index_cells: Optional[int]
        As for :func:`build_adaptive_mesh`. Not stored on the mesh, so pass
        the build's value again to match a fresh build.

    Returns
    -------
    AdaptiveMesh
        ``mesh`` itself when ``fov`` needs no new cell. Otherwise a new mesh
        with ``mesh``'s ``min_img_sep``, ``max_level``, ``d_floor``,
        ``dtype``, ``device`` and centre, and ``init_res`` grown by ``2 * k``.

    Raises
    ------
    ValueError
        If ``fov`` is not finite or is smaller than ``mesh.fov``, or if the
        grown lattice would overflow int64 keys.

    Warns
    -----
    UserWarning
        Exactly as :func:`build_adaptive_mesh` of the grown fov would: when
        ``max_depth`` limited the build, and when ``raytrace`` returns a
        dtype whose cancellation floor at the grown fov exceeds
        ``min_img_sep``.
    """
    k = _extension_cells(mesh, fov)
    if k == 0:
        return mesh
    h0 = mesh.lattice.scale * (1 << mesh.lattice.level)
    init_res = mesh.init_res + 2 * k
    fov = mesh.fov + 2 * k * h0
    check_lattice_keys(
        init_res,
        mesh.max_level,
        "Extend by less, or rebuild with a larger min_img_sep or a lower max_depth.",
    )
    warn_depth_limited(fov, init_res, mesh.min_img_sep, mesh.d_floor, mesh.max_level)

    tables = child_matrix_tables()
    lat = extend_lattice(mesh.lattice, k)._replace(lo=_ambient(mesh.lattice.lo))
    raytrace_fn = make_raytrace(lens.raytrace, mesh.device)
    cache, active, store, seed_band = seed_from_mesh(
        mesh, lat, k, raytrace_fn, raytrace_batch_size
    )
    return _grow(
        raytrace_fn,
        lens.jacobian_lens_equation,
        lat,
        tables,
        roots=ring_triangles(init_res, k, lat.level, tables[4]),
        seed=(cache, active, store),
        seed_band=seed_band,
        fov=fov,
        init_res=init_res,
        h0=h0,
        min_img_sep=mesh.min_img_sep,
        d_floor=mesh.d_floor,
        max_level=mesh.max_level,
        device=mesh.device,
        dtype=mesh.dtype,
        raytrace_batch_size=raytrace_batch_size,
        index_cells=index_cells,
    )


# ---------------------------------------------------------------------------
# Source-plane query
# ---------------------------------------------------------------------------


def _as_beta(mesh, beta) -> ArrayLike:
    """
    Coerce query points to the mesh's dtype and device, shape ``(B, 2)``.

    A bare ``(2,)`` raises rather than being promoted, so :func:`mesh_query`'s
    output shapes are never ambiguous.
    """
    beta = backend.as_array(beta, dtype=mesh.vertices_source.dtype, device=mesh.device)
    if len(beta.shape) != 2 or beta.shape[1] != 2:
        raise ValueError(
            f"beta must have shape (B, 2), got {tuple(beta.shape)}. A single "
            "point must be passed as shape (1, 2)."
        )
    return beta


def _empty_query_result(mesh, n_queries) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
    """The ``mesh_query`` result for ``n_queries`` points that all miss."""
    int64 = backend.int64
    return (
        backend.zeros((0,), dtype=int64, device=mesh.device),
        backend.zeros((n_queries + 1,), dtype=int64, device=mesh.device),
        backend.zeros((0, 3), dtype=mesh.vertices_source.dtype, device=mesh.device),
    )


def mesh_query(mesh, beta, batch_size=None) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
    """
    Terminal leaves whose source-plane image contains each query point.

    Returns candidate regions, not images. A point on a shared edge returns
    both leaves -- zeros count as inside, which is what guarantees no query
    falls through a seam -- and near-critical leaves overlap, so **candidate
    count is not image multiplicity**.

    Parameters
    ----------
    mesh: AdaptiveMesh
        The frozen mesh to query.
    beta: ArrayLike
        Source-plane query points, shape ``(B, 2)`` strictly. A bare ``(2,)``
        raises rather than being promoted, so output shapes are never
        ambiguous.

        *Unit: arcsec*
    batch_size: Optional[int]
        Chunk size over query points. ``None`` processes all at once. Results
        are byte-identical for every value; this only bounds peak memory,
        which spikes for chunks landing near a caustic.

    Returns
    -------
    leaf_indices: ArrayLike
        ``(K,)`` indices into ``mesh.leaves``, strictly ascending within each
        block.
    offsets: ArrayLike
        ``(B + 1,)`` CSR offsets, ``offsets[0] == 0`` and ``offsets[B] == K``.
    bary: ArrayLike
        ``(K, 3)`` barycentric coordinates of ``beta`` in the source-plane
        image of the hit triangle, guaranteed to lie in the simplex.
    """
    beta = _as_beta(mesh, beta)
    n = beta.shape[0]
    if n == 0:
        return _empty_query_result(mesh, 0)

    int64 = backend.int64
    index = mesh.index
    step = n if batch_size is None else max(1, int(batch_size))
    idx_parts, bary_parts, count_parts = [], [], []

    for lo in range(0, n, step):
        chunk = beta[lo : lo + step]
        b = chunk.shape[0]
        u = backend.long(backend.floor((chunk - index.lo) / index.cell))
        # Containment is a coordinate test against the stored exact `hi`, never a
        # cell-index test on `u`. Because `cell = span / [nx, ny]`, a point sitting
        # exactly on the upper bbox edge (x == hi_x) gives `u_x == nx`, which
        # fails `u_x < nx` -- even though `build_index` clips leaf registration
        # to column `nx - 1`, i.e. the leaf IS indexed, in the very column that
        # test would reject. Do NOT recompute `hi` as `lo + cell * [nx, ny]`:
        # `(span / n) * n` need not equal `span` to the ulp, so the exact
        # `mesh.index.hi` from the build is used instead. A leaf containing a
        # `beta` with x == hi has its `i1_x` clipped to `nx - 1`, the column the
        # clamp below selects, so this coordinate test is provably complete.
        inside = (
            (chunk[:, 0] >= index.lo[0])
            & (chunk[:, 0] <= index.hi[0])
            & (chunk[:, 1] >= index.lo[1])
            & (chunk[:, 1] <= index.hi[1])
        )
        # Clamped only to keep the gather in range; `inside` forces an empty
        # block for out-of-bbox points.
        cell = backend.clamp(u[:, 0], 0, index.nx - 1) * index.ny + backend.clamp(
            u[:, 1], 0, index.ny - 1
        )
        start = index.cell_offsets[cell]
        count = backend.where(
            inside, index.cell_offsets[cell + 1] - start, backend.zeros_like(start)
        )
        total = int(backend.to_numpy(backend.sum(count)))
        if total == 0:
            count_parts.append(backend.zeros((b,), dtype=int64, device=mesh.device))
            continue

        qidx = backend.repeat(
            backend.arange(b, dtype=int64, device=mesh.device), count, axis=0
        )
        base = backend.cumsum(count, dim=0) - count
        within = backend.arange(
            total, dtype=int64, device=mesh.device
        ) - backend.repeat(base, count, axis=0)
        cand = index.cell_leaves[start[qidx] + within]

        w = triangle_weights(mesh.vertices_source[mesh.leaves[cand]], chunk[qidx])
        hit = contains(w)

        # Per-query hit counts by cumsum differences. Never add_at_indices:
        # torch keeps the last write on duplicate indices, jax accumulates --
        # so a scatter-add would mean two different things on the two backends.
        csum = backend.concatenate(
            (
                backend.zeros((1,), dtype=int64, device=mesh.device),
                backend.cumsum(backend.long(hit), dim=0),
            ),
            dim=0,
        )
        count_parts.append(csum[base + count] - csum[base])
        idx_parts.append(cand[hit])
        # Gather `leaf_area2` after masking by `hit`, not before: `cand` is the
        # pre-containment candidate list, so `leaf_area2[cand]` would be a
        # full-length temporary thrown away by the mask on the very next
        # operation.
        bary_parts.append(sanitize_bary(w[hit], mesh.leaf_area2[cand[hit]]))

    counts = backend.concatenate(count_parts, dim=0)
    offsets = backend.concatenate(
        (
            backend.zeros((1,), dtype=int64, device=mesh.device),
            backend.cumsum(counts, dim=0),
        ),
        dim=0,
    )
    if not idx_parts:
        empty = _empty_query_result(mesh, 0)
        return empty[0], offsets, empty[2]
    return (
        backend.concatenate(idx_parts, dim=0),
        offsets,
        backend.concatenate(bary_parts, dim=0),
    )


def mesh_seeds(mesh, leaf_indices, bary) -> ArrayLike:
    """
    Lens-plane preimage of hit leaves under each leaf's own affine map, shape ``(K, 2)``.

    A ``bary``-weighted gather of ``vertices_lens``: the pure contraction
    ``sum(tri * bary[..., None], axis=1)`` where ``tri = vertices_lens[leaves[
    leaf_indices]]``. That map is exactly the one step 7's refinement criterion
    bounds, so the result is a Newton seed accurate to ``min_img_sep`` by
    construction. Because ``bary`` is guaranteed to lie in the simplex (see
    :func:`sanitize_bary`), the seed always lies inside its leaf.

    There is no ``beta`` mode here, unlike the oracle's ``Mesh.seeds``, and no
    guard rejecting an ambiguous call: a caller wanting query-then-seed
    composes :func:`mesh_query` and this function directly --
    ``leaf_indices, _, bary = mesh_query(mesh, beta)`` then
    ``mesh_seeds(mesh, leaf_indices, bary)`` -- rather than this function
    accepting both forms behind a runtime check.

    Parameters
    ----------
    mesh: AdaptiveMesh
        The frozen mesh ``leaf_indices`` indexes into.
    leaf_indices: ArrayLike
        ``(K,)`` indices into ``mesh.leaves``, e.g. from :func:`mesh_query`.
    bary: ArrayLike
        ``(K, 3)`` barycentric coordinates within each indexed leaf, e.g. from
        :func:`mesh_query`. The caller is responsible for ensuring it lies in
        the simplex; :func:`mesh_query` already guarantees this.

    Returns
    -------
    ArrayLike
        ``(K, 2)`` lens-plane seed positions.

        *Unit: arcsec*
    """
    tri = mesh.vertices_lens[mesh.leaves[leaf_indices]]
    return backend.sum(tri * backend.unsqueeze(bary, -1), dim=1)


def dedup_block_group(points, rows, n_blocks, m, tol) -> ArrayLike:
    """
    Connected-component representatives for blocks of exactly ``m`` points.

    Every slot is real, so this carries none of the padding machinery a
    ragged formulation needs: no validity mask, no clipped gather, and the
    "no label" sentinel is ``m`` rather than a global maximum. Grouping the
    caller's blocks by count and calling this once per distinct count is what
    keeps the ``(n_blocks, m, m)`` intermediate proportional to
    ``sum_c B_c * c**2`` instead of ``B * max(c)**2``.

    Parameters
    ----------
    points: ArrayLike
        The caller's full point array, shape ``(K, 2)``.

        *Unit: arcsec*
    rows: ArrayLike
        ``(n_blocks * m,)`` int64 indices into ``points``, block-major. A
        host-side sequence is accepted too; see :func:`dedup_representatives`.
    n_blocks, m: int
        Block count and the common per-block point count.
    tol: float
        Separation below which two points are the same image.

        *Unit: arcsec*

    Returns
    -------
    ArrayLike
        ``(n_blocks * m,)`` bool, in the order of ``rows``.
    """
    device = backend.device(points)
    int64 = backend.int64
    p = points[backend.as_array(rows, dtype=int64, device=device)]
    p = p.reshape(n_blocks, m, 2)

    delta = backend.unsqueeze(p, 2) - backend.unsqueeze(p, 1)
    # Squared distances against a squared tolerance: no sqrt, and the
    # comparison is exact on the diagonal, so every point is its own
    # neighbour and the label update below is a true minimum over the closed
    # neighbourhood.
    adjacent = backend.long(backend.sum(delta * delta, dim=-1) < tol * tol)

    # Min-label propagation. `m` is the sentinel for "no label": it exceeds
    # every real slot index, so it never wins a minimum against a neighbour.
    index = backend.unsqueeze(backend.arange(m, dtype=int64, device=device), 0)
    labels = index + backend.zeros((n_blocks, m), dtype=int64, device=device)
    for _ in range(m):
        neighbour = adjacent * backend.unsqueeze(labels, 1) + (1 - adjacent) * m
        updated = backend.min(neighbour, dim=2)
        if bool(backend.to_numpy(backend.all(updated == labels))):
            break
        labels = updated

    # Each component now carries the lowest slot index it contains, and that
    # slot is its own label -- so the fixed points are exactly one per
    # component.
    return (labels == index).reshape(-1)


def dedup_representatives(points, counts, tol) -> ArrayLike:
    """
    One representative per cluster of near-coincident points, within each block.

    Clusters are the **connected components** of the ``distance < tol`` graph,
    not the greedy clusters :func:`~caustics.lenses.func.base.remove_duplicate_points`
    produces. The difference is order dependence: for three collinear points
    spaced ``0.9 * tol`` apart, greedy returns two representatives in one input
    order and one in another, so the image count would depend on the order
    :func:`mesh_query` happened to emit candidates in. Components are a
    function of the point set alone, which is what makes a multiplicity map
    reproducible.

    Adjacency is strict ``<``, so a pair separated by exactly ``tol`` stays
    distinct. That matches the build contract, where ``min_img_sep`` is a size
    floor the mesh resolves *to* rather than a scale it merges away.

    Vectorized by grouping blocks that share a count and running each group at
    its own width, because a greedy loop is one Python iteration per point --
    fine for the handful of images of a single source, hopeless for the
    ``nx * ny`` blocks of a multiplicity map. Blocks of zero or one point never
    reach the kernel; their answer is already known. The cost is a
    ``(B_c, c, c)`` intermediate per distinct count ``c``, which is why callers
    may still want to chunk over query points when a single block is enormous.

    Block-major order is restored by an inverse permutation rather than a
    scatter: every row belongs to exactly one group, so concatenating the
    groups' row indices gives a permutation of ``range(K)``, and ``argsort`` of
    a permutation *is* its inverse -- computed by sorting, never by an indexed
    assignment. That matters because torch keeps the last write on duplicate
    indices and jax accumulates, so a scatter would mean two different things
    on the two backends; a gather (indexing by the inverse permutation) means
    the same thing on both.

    Parameters
    ----------
    points: ArrayLike
        Shape ``(K, 2)``, laid out block-major: block ``b`` occupies the
        ``counts[b]`` rows following those of blocks ``0 .. b - 1``.

        *Unit: arcsec*
    counts: ArrayLike
        Shape ``(B,)`` int, with ``counts.sum() == K``. A host-side sequence
        is accepted directly and coerced on entry -- that is the array-like
        input the no-``numpy``-import rule permits, not an exception to it.
        Zero-length blocks are allowed.
    tol: float
        Separation below which two points are the same image.

        *Unit: arcsec*

    Returns
    -------
    ArrayLike
        ``(K,)`` bool, True on exactly one point per cluster.
    """
    device = backend.device(points)
    counts = backend.as_array(counts, dtype=backend.int64, device=device)
    total = int(backend.to_numpy(backend.sum(counts)))
    if total == 0:
        return backend.zeros((0,), dtype=backend.bool, device=device)

    starts = backend.cumsum(counts, dim=0) - counts
    row_groups, keep_groups = [], []

    # A block of one point is its own representative and a block of none
    # contributes nothing, so neither reaches the clustering kernel at all.
    # On a multiplicity map those are the large majority, and skipping them is
    # the single biggest reduction in what the kernel has to hold.
    singles = backend.flatnonzero(counts == 1)
    if singles.shape[0] > 0:
        row_groups.append(starts[singles])
        keep_groups.append(
            backend.ones((singles.shape[0],), dtype=backend.bool, device=device)
        )

    # The rest are grouped by *equal* count so each group runs at its own M.
    # Padding every block to the global maximum is what made the intermediate
    # `B * max(c)**2` and put a fine multiplicity map out of memory. The
    # distinct counts themselves are pulled to the host: there are at most a
    # handful of them (multiplicities are small integers), and each drives a
    # Python-level `dedup_block_group` call with its own static shape anyway.
    distinct = backend.to_numpy(backend.unique(counts[counts > 1])).tolist()
    for m in distinct:
        blocks = backend.flatnonzero(counts == m)
        rows = (
            backend.unsqueeze(starts[blocks], 1)
            + backend.unsqueeze(
                backend.arange(m, dtype=backend.int64, device=device), 0
            )
        ).reshape(-1)
        row_groups.append(rows)
        keep_groups.append(dedup_block_group(points, rows, blocks.shape[0], m, tol))

    # Every row belongs to exactly one group, so `perm` is a permutation of
    # `range(total)` and its `argsort` is exactly its inverse.
    perm = backend.concatenate(row_groups, dim=0)
    inverse = backend.argsort(perm)
    stacked = backend.concatenate(keep_groups, dim=0)
    return stacked[inverse]


# ---------------------------------------------------------------------------
# Forward raytrace
# ---------------------------------------------------------------------------

METHODS = ("rootfind", "dedup")


def _check_method(method) -> None:
    """Reject an unrecognised image-finding method, naming the alternatives."""
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")


def _to_source(raytrace):
    """Wrap ``raytrace(x, y)`` as a ``(..., 2) -> (..., 2)`` map."""

    def to_source(xy):
        return backend.stack(raytrace(xy[..., 0], xy[..., 1]), dim=-1)

    return to_source


def _forward_chunk(mesh, chunk, raytrace, method, tol, lm_kwargs):
    """
    Images and per-source counts for one chunk of query points.

    Returns
    -------
    images: ArrayLike or None
        ``(K, 2)`` lens-plane positions, or ``None`` when the chunk found
        none.
    counts: ArrayLike
        ``(b,)`` ``backend`` int64 image multiplicity.
    """
    b = chunk.shape[0]
    int64 = backend.int64
    device = mesh.device
    none = (None, backend.zeros((b,), dtype=int64, device=device))

    idx, offsets, bary = mesh_query(mesh, chunk)
    seed = mesh_seeds(mesh, idx, bary)
    if seed.shape[0] == 0:
        return none

    if method == "dedup":
        # No residual filter and no displacement filter. Both exist to
        # reject a root that *wandered* away from its seed -- see the Notes
        # on `mesh_forward_raytrace`. A seed cannot wander: `bary` lies in
        # the simplex, so the seed lies inside its leaf, and that leaf's
        # source-plane image contains `beta`. Every seed is therefore
        # already an approximate image, and filtering would be testing a
        # property the construction guarantees.
        survivors, kept = seed, offsets[1:] - offsets[:-1]
    else:
        to_source = _to_source(raytrace)
        # One target per seed, so a source with several candidate leaves
        # root-finds each of them against its own beta.
        spans = offsets[1:] - offsets[:-1]
        target = backend.repeat(chunk, spans, axis=0)
        root, _, _ = batch_lm(seed, target, to_source, **lm_kwargs)

        converged = backend.sum((to_source(root) - target) ** 2, dim=-1) < tol * tol
        # See the Notes on `mesh_forward_raytrace` for why containment and
        # the ball are OR-ed.
        tri = mesh.vertices_lens[mesh.leaves[idx]]
        near = contains(triangle_weights(tri, root)) | (
            backend.sum((root - seed) ** 2, dim=-1) <= mesh.min_img_sep**2
        )
        keep = converged & near

        # Chunk bookkeeping stays on `backend` int64 arrays: `kept` is read
        # off `keep` by the same cumsum-difference trick `mesh_query` uses
        # for its own per-query hit counts, never a host round trip.
        keep_i = backend.long(keep)
        csum = backend.concatenate(
            (
                backend.zeros((1,), dtype=int64, device=device),
                backend.cumsum(keep_i, dim=0),
            ),
            dim=0,
        )
        kept = csum[offsets[1:]] - csum[offsets[:-1]]
        if int(backend.to_numpy(backend.sum(kept))) == 0:
            return none
        survivors = root[keep]

    unique = dedup_representatives(survivors, kept, mesh.min_img_sep)
    unique_i = backend.long(unique)
    kept_off = backend.concatenate(
        (backend.zeros((1,), dtype=int64, device=device), backend.cumsum(kept, dim=0)),
        dim=0,
    )
    csum = backend.concatenate(
        (
            backend.zeros((1,), dtype=int64, device=device),
            backend.cumsum(unique_i, dim=0),
        ),
        dim=0,
    )
    counts = csum[kept_off[1:]] - csum[kept_off[:-1]]
    return survivors[unique], counts


def _image_chunks(mesh, beta, raytrace, batch_size, method, residual_tol, lm_kwargs):
    """Yield :func:`_forward_chunk`'s ``(images, counts)`` per chunk."""
    tol = mesh.min_img_sep if residual_tol is None else float(residual_tol)
    lm_kwargs = {} if lm_kwargs is None else dict(lm_kwargs)
    n = beta.shape[0]
    step = max(1, n) if batch_size is None else max(1, int(batch_size))
    for lo in range(0, max(n, 1), step):
        yield _forward_chunk(
            mesh, beta[lo : lo + step], raytrace, method, tol, lm_kwargs
        )


def mesh_forward_raytrace(
    mesh,
    beta,
    raytrace: Callable[[ArrayLike, ArrayLike], Tuple[ArrayLike, ArrayLike]],
    batch_size: Optional[int] = None,
    *,
    method: str = "rootfind",
    residual_tol: Optional[float] = None,
    lm_kwargs: Optional[dict] = None,
) -> Tuple[ArrayLike, ArrayLike]:
    """
    Image-plane positions of every image of each source-plane point.

    :func:`mesh_seeds` supplies a Newton seed per candidate leaf, accurate to
    ``min_img_sep`` by construction. Under ``method="rootfind"``,
    Levenberg-Marquardt refines each seed to a root of the lens equation,
    unconverged roots are discarded, and the survivors are deduplicated at
    ``min_img_sep``; under ``method="dedup"`` the seeds themselves are
    deduplicated directly -- see the ``method`` parameter below.

    Parameters
    ----------
    mesh: AdaptiveMesh
        The frozen mesh to raytrace against.
    beta: ArrayLike
        Source-plane points, shape ``(B, 2)`` strictly. A single point must be
        passed as ``(1, 2)``.

        *Unit: arcsec*

    raytrace: Callable
        **Must be the raytrace of the lens this mesh was built from**, i.e.
        ``lens.raytrace``, called as ``raytrace(x, y) -> (bx, by)``. The seeds
        handed to the root finder are preimages under *this* mesh's leaves, so
        a different lens would be root-found from meaningless starting points
        -- silently, since the residual filter would simply reject most of them
        and return too few images rather than raising. This cannot be checked:
        a callable carries no identity the mesh could have recorded at build
        time.
    batch_size: Optional[int]
        Chunk size over source points. Bounds peak memory for the whole
        pipeline, not just :func:`mesh_query` -- the root finder holds
        ``(K, 2)`` states and the dedup a ``(B_c, c, c)`` intermediate per
        distinct candidate count ``c`` (see :func:`dedup_representatives`).
        Image counts are invariant to this chunking. Levenberg-Marquardt's
        shared damping schedule can shift root positions within solver
        accuracy when the batch partition changes.
    method: str
        ``"rootfind"`` (default) refines every seed with
        Levenberg-Marquardt and returns machine-precision image positions.
        ``"dedup"`` skips the root finder entirely and deduplicates the
        seeds, which are already accurate to ``min_img_sep`` by
        construction. It **never calls** ``raytrace``, which is why it is
        roughly two orders of magnitude faster; ``raytrace``,
        ``residual_tol`` and ``lm_kwargs`` are accepted and ignored.

        Positions from ``"dedup"`` are accurate to ``min_img_sep``, not to
        machine precision. Counts agree with ``"rootfind"`` except within
        about ``min_img_sep`` of a caustic -- measured at 12 pixels in
        24656 on an EPL-plus-shear lens. Neither method's counts are
        guaranteed to satisfy the odd-image theorem any longer:
        ``"dedup"`` can merge a near-tangential pair closer than
        ``min_img_sep`` into one, and ``"rootfind"`` can miss an image
        outright whose seed would have fallen in a non-converged
        ``max_level`` leaf, which was never in the index to seed the root
        finder in the first place. Measured on the cored SIE fixture
        (``fov=5``, ``init_res=32``) over a 25x25 source grid of 0.08 arcsec
        pixels centred on the lens, ``"rootfind"``: even image counts --
        impossible for this non-singular lens -- turn up at 371 of 625
        pixels at ``min_img_sep=0.04``, 263 at ``0.02``, and 4 at ``0.01``.
        Use ``"rootfind"`` when the position itself matters, ``"dedup"``
        when the count does.
    residual_tol: Optional[float]
        Source-plane tolerance on ``|raytrace(x) - beta|`` for accepting a
        root. Defaults to ``min_img_sep``.

        *Unit: arcsec*

    lm_kwargs: Optional[dict]
        Extra keyword arguments for :func:`~caustics.utils.batch_lm`, e.g.
        ``max_iter``.

    Returns
    -------
    images: ArrayLike
        ``(K, 2)`` lens-plane image positions, laid out block-major: the
        ``counts[b]`` images of source ``b`` follow those of sources
        ``0 .. b - 1``.

        *Unit: arcsec*

    counts: ArrayLike
        ``(B,)`` int64 image multiplicity of each source point.

    Notes
    -----
    Two filters decide that a root is an image, and both are needed.

    The **residual** test alone is weak near a fold caustic, where the lens
    map is quadratic: a point sitting well over ``min_img_sep`` from the true
    image in the lens plane can still have a small source-plane residual, so
    it survives the residual test, escapes the dedup, and inflates the count
    exactly where multiplicity structure matters most.

    The **displacement** test closes that hole using a guarantee the mesh
    already makes -- the seed lies inside its leaf and is accurate to
    ``min_img_sep`` -- so a root that left its own neighbourhood is not the
    root its seed was pointing at. It is a disjunction rather than plain
    containment because a leaf at the size floor is itself only about
    ``min_img_sep`` across, so a genuine root near a leaf edge can
    legitimately land just outside it; requiring containment alone would
    drop real images.

    Neither filter applies under ``method="dedup"``. Both reject a root
    that *wandered* -- the residual test catches a solve that converged to
    nothing, the displacement test one that converged to a different image.
    A seed cannot wander: it lies inside its own leaf, whose source-plane
    image contains ``beta``. Filtering it would test a property the
    construction already guarantees.

    Root finding runs in the dtype of the frozen mesh, so a mesh built with
    ``dtype=backend.float32`` caps the achievable accuracy near the
    cancellation floor :func:`build_adaptive_mesh` already warns about.
    """
    _check_method(method)
    beta = _as_beta(mesh, beta)
    n = beta.shape[0]
    int64 = backend.int64

    def no_images():
        return backend.zeros((0, 2), dtype=mesh.vertices_lens.dtype, device=mesh.device)

    if n == 0:
        return no_images(), backend.zeros((0,), dtype=int64, device=mesh.device)

    image_parts, count_parts = [], []
    for images, counts in _image_chunks(
        mesh, beta, raytrace, batch_size, method, residual_tol, lm_kwargs
    ):
        count_parts.append(counts)
        if images is not None:
            image_parts.append(images)

    counts = backend.concatenate(count_parts, dim=0)
    if not image_parts:
        return no_images(), counts
    return backend.concatenate(image_parts, dim=0), counts
