"""
A uniform-grid spatial index over source-plane triangles, and lookups in it.

A triangle is registered in every cell its bounding box covers, so one cell
lookup per point finds every triangle that can contain it.
"""

import math
from typing import NamedTuple

from ....backend_obj import ArrayLike, backend
from .geometry import contains, csr_offsets, triangle_weights


class MeshIndex(NamedTuple):
    """
    Uniform-grid CSR index over triangle bounding boxes.

    Cell ``c``'s triangles are ``cell_leaves[cell_offsets[c]:cell_offsets[c + 1]]``,
    ascending.

    Parameters
    ----------
    lo, hi: ArrayLike
        ``(2,)`` float64 corners of the bounding box of every indexed triangle.

        *Unit: arcsec*
    cell: ArrayLike
        ``(2,)`` float64 cell size per axis.

        *Unit: arcsec*
    nx, ny: int
        Cells per axis.
    cell_offsets: ArrayLike
        ``(nx * ny + 1,)`` int64 CSR offsets.
    cell_leaves: ArrayLike
        int64 triangle indices, cell by cell.
    """

    lo: ArrayLike
    hi: ArrayLike
    cell: ArrayLike
    nx: int
    ny: int
    cell_offsets: ArrayLike
    cell_leaves: ArrayLike


def build_index(vertices, triangles, rows):
    """
    The :class:`MeshIndex` over the bounding boxes of ``triangles[rows]``.

    Cells are sized from the mean triangle density, and the cell arithmetic
    is float64 so that it agrees exactly with :func:`index_hits`'.

    Parameters
    ----------
    vertices: ArrayLike
        ``(V, 2)`` positions.

        *Unit: arcsec*
    triangles: ArrayLike
        ``(T, 3)`` int64 vertex indices.
    rows: ArrayLike
        ``(K,)`` int64 ascending rows of ``triangles`` to index.

    Returns
    -------
    MeshIndex
    """
    tri = backend.to(vertices[triangles[rows]], dtype=backend.float64)
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
    span = backend.where(hi > lo, hi - lo, 1.0)  # a degenerate axis is one cell

    span_np = backend.to_numpy(span)
    c = float(math.sqrt(span_np[0] * span_np[1] / tri.shape[0]))
    c = max(c, float(backend.finfo(backend.float64).tiny))
    nx = max(1, int(math.ceil(span_np[0] / c)))
    ny = max(1, int(math.ceil(span_np[1] / c)))
    cell = span / backend.as_array([nx, ny], dtype=backend.float64)

    upper = backend.as_array([nx - 1, ny - 1], dtype=backend.int64)
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
    # A stable sort keeps each cell's triangles in ascending row order.
    order = backend.argsort(cell_id)
    counts_per_cell = backend.long(backend.bincount(cell_id, minlength=nx * ny))
    cell_offsets = csr_offsets(counts_per_cell)
    return MeshIndex(
        lo=lo,
        hi=hi,
        cell=cell,
        nx=nx,
        ny=ny,
        cell_offsets=cell_offsets,
        cell_leaves=rows[owner[order]],
    )


def index_hits(index, vertices, triangles, beta):
    """
    Triangles of ``index`` whose image contains each point, zeros counting as inside.

    One cell lookup per point, then :func:`~.geometry.contains` on that
    cell's triangles. A point is tested against the stored ``hi``, never a
    recomputed ``lo + cell * n``, so a point on the upper edge of the
    bounding box still finds the triangles registered in the last column.

    Parameters
    ----------
    index: MeshIndex
        Built over ``triangles`` by :func:`build_index`.
    vertices: ArrayLike
        ``(V, 2)`` positions ``triangles`` indexes.

        *Unit: arcsec*
    triangles: ArrayLike
        ``(T, 3)`` int64 vertex indices.
    beta: ArrayLike
        ``(B, 2)`` points.

        *Unit: arcsec*

    Returns
    -------
    qidx: ArrayLike
        ``(K,)`` int64 point of each hit, non-decreasing.
    tri: ArrayLike
        ``(K,)`` int64 triangle of each hit, ascending within each point.
    w: ArrayLike
        ``(K, 3)`` :func:`~.geometry.triangle_weights` of each hit.
    """
    int64 = backend.int64
    device = backend.device(beta)
    b = beta.shape[0]
    u = backend.long(backend.floor((beta - index.lo) / index.cell))
    inside = (
        (beta[:, 0] >= index.lo[0])
        & (beta[:, 0] <= index.hi[0])
        & (beta[:, 1] >= index.lo[1])
        & (beta[:, 1] <= index.hi[1])
    )
    cell = backend.clamp(u[:, 0], 0, index.nx - 1) * index.ny + backend.clamp(
        u[:, 1], 0, index.ny - 1
    )
    start = index.cell_offsets[cell]
    count = backend.where(
        inside, index.cell_offsets[cell + 1] - start, backend.zeros_like(start)
    )
    total = int(backend.to_numpy(backend.sum(count)))
    if total == 0:
        return (
            backend.zeros((0,), dtype=int64, device=device),
            backend.zeros((0,), dtype=int64, device=device),
            backend.zeros((0, 3), dtype=vertices.dtype, device=device),
        )
    qidx = backend.repeat(backend.arange(b, dtype=int64, device=device), count, axis=0)
    base = backend.cumsum(count, dim=0) - count
    within = backend.arange(total, dtype=int64, device=device) - backend.repeat(
        base, count, axis=0
    )
    cand = index.cell_leaves[start[qidx] + within]
    w = triangle_weights(vertices[triangles[cand]], beta[qidx])
    hit = backend.flatnonzero(contains(w))
    return qidx[hit], cand[hit], w[hit]


def as_points(bx, by, device):
    """``bx`` and ``by``, flattened, as ``(B, 2)`` float64 points on ``device``."""
    f64 = backend.float64
    x = backend.as_array(bx, dtype=f64, device=device).reshape(-1)
    y = backend.as_array(by, dtype=f64, device=device).reshape(-1)
    return backend.stack((x, y), dim=-1)
