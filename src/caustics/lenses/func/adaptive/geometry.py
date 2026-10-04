"""
Triangle maths shared by both meshes, and small array helpers.

The red-refinement child ordering and its integer group tables, the smallest
singular value that scales the lens criterion, the minimum angle closure
compares, and the containment test and barycentric coordinates every
source-plane lookup reads. Nothing here calls the lens or knows about a mesh.
"""

from ....backend_obj import backend
from .mesh_backend import map_arrays

# Indices into the stacked six-point array [theta1, theta2, theta3, m1, m2, m3]
# giving the four children in the orientation-preserving order
#   C_1 = (t1, m3, m2)  C_2 = (t2, m1, m3)  C_3 = (t3, m2, m1)  C_4 = (m1, m2, m3)
# with m_i the midpoint opposite theta_i.
CHILD_VERTEX_INDICES = ((0, 5, 4), (1, 3, 5), (2, 4, 3), (3, 4, 5))

# The two triangles splitting the unit cell along its (0,0)-(1,1) diagonal, both
# positively oriented.
ROOT_SHAPES = (
    ((0, 0), (1, 1), (0, 1)),
    ((0, 0), (1, 0), (1, 1)),
)

# `CHILD_VERTEX_INDICES` as one int64 array, to gather all four children at once.
_CHILD_VERTEX_INDEX_TABLE = backend.as_array(CHILD_VERTEX_INDICES, dtype=backend.int64)


def shape_matrix(tri):
    """
    Edge matrix ``P = [v1 - v0 | v2 - v0]`` of a triangle.

    Parameters
    ----------
    tri: ArrayLike
        Triangle vertices, shape ``(..., 3, 2)``.

        *Unit: arcsec*

    Returns
    -------
    ArrayLike
        Shape ``(..., 2, 2)``, the two edge vectors as columns.

        *Unit: arcsec*
    """
    return backend.stack(
        (tri[..., 1, :] - tri[..., 0, :], tri[..., 2, :] - tri[..., 0, :]), dim=-1
    )


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
        rows.append(backend.long(backend.round(2.0 * (backend.linalg.inv(P) @ Pk))))
    return backend.stack(rows)


def child_matrix_tables():
    """
    Fixed tables for red refinement.

    Red refinement gives ``P_k = (1/2) P M_k`` with ``M_k`` a fixed integer matrix
    per child index, so ``P`` at level ``d`` is ``2**-d * h0 * R @ G[c]`` for one
    element ``G[c]`` of the group the four ``M_k`` generate,
    ``{+-I, +-M_2, +-M_3}``. Both root shapes lie in one orbit, so six classes
    cover every triangle at every level.

    Returns
    -------
    M: ArrayLike
        ``(4, 2, 2)`` int64 child matrices; every ``det M_k == +1``.
    G: ArrayLike
        ``(6, 2, 2)`` int64 group elements in canonical order.
    COMPOSE: ArrayLike
        ``(6, 4)`` int64 with ``G[COMPOSE[c, k]] == G[c] @ M[k]``.
    PINV0: ArrayLike
        ``(6, 2, 2)`` float64 with ``PINV0[c] == inv(R @ G[c])``, ``R`` the class-0
        root shape. A triangle of class ``c`` at level ``d`` has
        ``inv(P_child_k) == (2**(d + 1) / h0) * PINV0[COMPOSE[c, k]]``.
    ROOT_CLASS: ArrayLike
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
            row.append(lookup[tuple(backend.to_numpy(product).reshape(-1).tolist())])
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
    for shape in ROOT_SHAPES:
        target = shape_matrix(backend.as_array(shape, dtype=backend.float64))
        root_class_values.append(
            next(
                c
                for c in range(6)
                if bool(
                    backend.allclose(
                        R @ backend.to(G[c], dtype=backend.float64), target
                    )
                )
            )
        )
    ROOT_CLASS = backend.as_array(root_class_values, dtype=backend.int64)
    return M, G, COMPOSE, PINV0, ROOT_CLASS


_, _, COMPOSE, PINV0, ROOT_CLASS = child_matrix_tables()


def area2(tri):
    """
    Twice the signed area of each triangle.

    Parameters
    ----------
    tri: ArrayLike
        ``(K, 3, 2)`` vertices.

        *Unit: arcsec*

    Returns
    -------
    ArrayLike
        ``(K,)``, positive for a counter-clockwise triangle.

        *Unit: arcsec^2*
    """
    P = shape_matrix(tri)
    return P[:, 0, 0] * P[:, 1, 1] - P[:, 0, 1] * P[:, 1, 0]


def build_device():
    """The backend's default device, where every build runs."""
    return backend.device(backend.zeros((0,)))


def to_device(value, device):
    """
    ``value`` on ``device``: an array, or a NamedTuple of arrays and NamedTuples, field by field.

    Anything else is returned as it is.
    """
    return map_arrays(value, lambda a: backend.to(a, device=device))


def is_member(sorted_values, values):
    """True where each of ``values`` is in the ascending ``sorted_values``."""
    if sorted_values.shape[0] == 0:
        return backend.zeros(
            values.shape, dtype=backend.bool, device=backend.device(values)
        )
    pos = backend.clamp(
        backend.searchsorted(sorted_values, values), 0, sorted_values.shape[0] - 1
    )
    return sorted_values[pos] == values


def csr_offsets(counts):
    """CSR offsets ``(n + 1,)`` of blocks of ``counts`` ``(n,)`` rows each."""
    zero = backend.zeros((1,), dtype=backend.int64, device=backend.device(counts))
    return backend.concatenate(
        (zero, backend.cumsum(backend.long(counts), dim=0)), dim=0
    )


def sigma_min_2x2(A):
    """
    Smallest singular value of a batch of 2x2 matrices, in closed form.

    A source-plane displacement pulls back to at most ``||dbeta|| / sigma_min``
    in the lens plane. With ``F = ||A||_F**2`` and ``D = |det A|``,
    ``(sigma_max +- sigma_min)**2 = F +- 2D``; ``F - 2D`` is clipped at zero,
    since rounding takes it an ulp negative for a near-conformal ``A``, and
    ``sigma_min = D / sigma_max`` stays accurate as ``A`` nears singular.

    Parameters
    ----------
    A: ArrayLike
        Shape ``(..., 2, 2)``, float64.

    Returns
    -------
    ArrayLike
        Shape ``(...)``. Exactly ``0.0`` where ``A == 0``; ``NaN`` where ``A`` is
        non-finite, so the criterion fails closed.
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


def triangle_weights(tri, beta):
    """
    The three cross products that give containment and barycentric coordinates.

    ``w_i`` is the cross product of the two edges of the sub-triangle
    opposite vertex ``i``, so ``w1 + w2 + w3`` is twice the signed area ``d``
    and ``w / d`` are the barycentric coordinates. Two triangles sharing an
    edge form its weight from the same operands in opposite order, so the
    two weights are exactly negated and no point falls through the seam.

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


def contains(w):
    """
    Containment test: all three weights share a sign, zeros counting as inside.

    Source-plane triangles flip orientation across critical curves, so the
    test asks for one shared sign rather than a positive one. A point on a
    shared edge is inside both triangles.

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


def sanitize_bary(w, d):
    """
    Normalize weights to barycentric coordinates, clamping then falling back.

    A hit passed containment, so its coordinates lie in ``[0, 1]`` up to
    rounding, which the clip removes. Where ``d`` is so near zero that the
    clipped coordinates are non-finite or all zero, the centroid stands in.

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


def min_angle(tri):
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
    cosines = backend.stack(
        (
            (b * b + c * c - a * a) / (2 * b * c),
            (c * c + a * a - b * b) / (2 * c * a),
            (a * a + b * b - c * c) / (2 * a * b),
        ),
        dim=1,
    )
    angles = backend.arccos(backend.clamp(cosines, -1.0, 1.0))
    return backend.nan_to_num(backend.min(angles, dim=1))
