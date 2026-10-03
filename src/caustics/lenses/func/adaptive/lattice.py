"""
The dyadic integer lattice every mesh vertex lives on, and triangles on it.

Keys and lens-plane positions, growth by whole level-0 cells, the size floor
and the int64 key limit, the level-0 triangles of a grid or of the ring an
extension adds, exact edge midpoints, and the quarter points the 2:1 balance
test reads.
"""

import math
from typing import NamedTuple, Tuple
from warnings import warn

from ....backend_obj import ArrayLike, backend
from .geometry import ROOT_SHAPES

__all__ = (
    "MAX_KEY",
    "depth_floor",
    "warn_depth_limited",
    "lattice_h0",
    "lattice_fov",
    "lattice_init_res",
    "check_lattice_keys",
    "Lattice",
    "make_lattice",
    "extend_lattice",
    "lattice_key",
    "lattice_ij_from_key",
    "lattice_xy",
    "lattice_on_boundary",
    "initial_triangles",
    "ring_triangles",
    "midpoint_ij",
)


MAX_KEY = 2**63 - 1


def depth_floor(h0, tol):
    """Level at which the longest leaf edge, ``sqrt(2) * h0 / 2**level``, first falls to ``tol``."""
    l_max0 = math.sqrt(2.0) * h0
    if l_max0 <= tol:
        return 0
    return int(math.ceil(math.log2(l_max0 / tol)))


def warn_depth_limited(what, request, h0, tol, max_level):
    """
    Warn when ``max_depth`` stopped refinement before the longest leaf edge fell to ``tol``.

    ``what`` names the mesh and ``request`` the tolerance as the caller gave it.
    """
    d_floor = depth_floor(h0, tol)
    if d_floor <= max_level:
        return
    l_max = math.sqrt(2.0) * h0 / 2**max_level
    warn(
        f"{what} is depth-limited: max_depth={max_level} is below d_floor={d_floor}, "
        f"the depth required to reach {request}. Refinement stops at level "
        f"{max_level}, where the longest leaf edge is {l_max:.3g} arcsec. Set "
        f"max_depth >= {d_floor}, or raise init_res or the tolerance."
    )


def lattice_h0(lat):
    """Level-0 cell size, ``fov / init_res``.

    *Unit: arcsec*
    """
    return lat.scale * (1 << lat.level)


def lattice_fov(lat):
    """Side of the square the lattice covers.

    *Unit: arcsec*
    """
    return lat.n * lat.scale


def lattice_init_res(lat):
    """Level-0 cells per axis."""
    return lat.n >> lat.level


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
    scale: float
    lo: ArrayLike
    origin: int = 0


def make_lattice(fov, x0, y0, init_res, lattice_level) -> Lattice:
    """
    Build a :class:`Lattice` covering ``fov``, centered at ``(x0, y0)``.

    Parameters
    ----------
    fov: float
        Field of view.

        *Unit: arcsec*

    x0: float
        Domain center, x.

        *Unit: arcsec*

    y0: float
        Domain center, y.

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
    scale = float(fov) / n
    lo = backend.as_array([x0 - fov / 2.0, y0 - fov / 2.0], dtype=backend.float64)
    return Lattice(level=level, n=n, scale=scale, lo=lo, origin=0)


def extend_lattice(lat, k) -> Lattice:
    """
    ``lat`` grown by ``k`` level-0 cells on every side, every old point in place.

    A point at ``ij`` on ``lat`` is at ``ij + (k << lat.level)`` on the
    result, and ``origin`` moves by the same amount, so ``ij - origin`` -- and
    with it :func:`lattice_xy` -- is unchanged bit for bit. ``lo`` and
    ``scale`` are never recomputed: a fresh lattice over the larger fov
    computes both from that fov and can round differently. ``n`` grows,
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
    return lat._replace(n=n, origin=lat.origin + pad)


def lattice_key(lat, ij):
    """Lattice key of integer coordinates, shape ``(..., 2) -> (...)``."""
    return ij[..., 0] * (lat.n + 1) + ij[..., 1]


def lattice_ij_from_key(lat, key):
    """Inverse of :func:`lattice_key`, shape ``(...) -> (..., 2)``."""
    stride = lat.n + 1
    return backend.stack((key // stride, key % stride), dim=-1)


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
