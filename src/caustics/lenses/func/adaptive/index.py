"""
A uniform-grid spatial index over source-plane triangles, and lookups in it.

A triangle is registered in every cell its bounding box covers, so one cell
lookup per point finds every triangle that can contain it.
"""

import math
from typing import NamedTuple

from ....backend_obj import ArrayLike
from .mesh_backend import mesh_backend
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
    device = mesh_backend.device(vertices)
    f64, int64 = mesh_backend.float64, mesh_backend.int64
    tri = mesh_backend.to(vertices[triangles[rows]], dtype=f64)
    if tri.shape[0] == 0:
        return MeshIndex(
            lo=mesh_backend.zeros((2,), dtype=f64, device=device),
            hi=mesh_backend.ones((2,), dtype=f64, device=device),
            cell=mesh_backend.ones((2,), dtype=f64, device=device),
            nx=1,
            ny=1,
            cell_offsets=mesh_backend.zeros((2,), dtype=int64, device=device),
            cell_leaves=mesh_backend.zeros((0,), dtype=int64, device=device),
        )
    flat = tri.reshape(-1, 2)
    lo, hi = mesh_backend.min(flat, dim=0), mesh_backend.max(flat, dim=0)
    span = mesh_backend.where(hi > lo, hi - lo, 1.0)  # a degenerate axis is one cell

    span_np = mesh_backend.to_numpy(span)
    c = float(math.sqrt(span_np[0] * span_np[1] / tri.shape[0]))
    c = max(c, float(mesh_backend.finfo(f64).tiny))
    nx = max(1, int(math.ceil(span_np[0] / c)))
    ny = max(1, int(math.ceil(span_np[1] / c)))
    cell = span / mesh_backend.as_array([nx, ny], dtype=f64, device=device)

    upper = mesh_backend.as_array([nx - 1, ny - 1], dtype=int64, device=device)
    lower = mesh_backend.zeros((2,), dtype=int64, device=device)
    i0 = mesh_backend.clamp(
        mesh_backend.long((mesh_backend.min(tri, dim=1) - lo) / cell), lower, upper
    )
    i1 = mesh_backend.clamp(
        mesh_backend.long((mesh_backend.max(tri, dim=1) - lo) / cell), lower, upper
    )
    tall = i1[:, 1] - i0[:, 1] + 1
    counts = (i1[:, 0] - i0[:, 0] + 1) * tall
    owner = mesh_backend.repeat(
        mesh_backend.arange(counts.shape[0], dtype=int64, device=device), counts, axis=0
    )
    total_pairs = int(mesh_backend.to_numpy(mesh_backend.sum(counts)))
    within = mesh_backend.arange(
        total_pairs, dtype=int64, device=device
    ) - mesh_backend.repeat(mesh_backend.cumsum(counts, dim=0) - counts, counts, axis=0)
    cell_id = (i0[owner, 0] + within // tall[owner]) * ny + (
        i0[owner, 1] + within % tall[owner]
    )
    # A stable sort keeps each cell's triangles in ascending row order.
    order = mesh_backend.argsort(cell_id)
    counts_per_cell = mesh_backend.long(
        mesh_backend.bincount(cell_id, minlength=nx * ny)
    )
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
    int64 = mesh_backend.int64
    device = mesh_backend.device(beta)
    b = beta.shape[0]
    u = mesh_backend.long(mesh_backend.floor((beta - index.lo) / index.cell))
    inside = (
        (beta[:, 0] >= index.lo[0])
        & (beta[:, 0] <= index.hi[0])
        & (beta[:, 1] >= index.lo[1])
        & (beta[:, 1] <= index.hi[1])
    )
    cell = mesh_backend.clamp(u[:, 0], 0, index.nx - 1) * index.ny + mesh_backend.clamp(
        u[:, 1], 0, index.ny - 1
    )
    start = index.cell_offsets[cell]
    count = mesh_backend.where(
        inside, index.cell_offsets[cell + 1] - start, mesh_backend.zeros_like(start)
    )
    total = int(mesh_backend.to_numpy(mesh_backend.sum(count)))
    if total == 0:
        return (
            mesh_backend.zeros((0,), dtype=int64, device=device),
            mesh_backend.zeros((0,), dtype=int64, device=device),
            mesh_backend.zeros((0, 3), dtype=vertices.dtype, device=device),
        )
    qidx = mesh_backend.repeat(
        mesh_backend.arange(b, dtype=int64, device=device), count, axis=0
    )
    base = mesh_backend.cumsum(count, dim=0) - count
    within = mesh_backend.arange(
        total, dtype=int64, device=device
    ) - mesh_backend.repeat(base, count, axis=0)
    cand = index.cell_leaves[start[qidx] + within]
    w = triangle_weights(vertices[triangles[cand]], beta[qidx])
    hit = mesh_backend.flatnonzero(contains(w))
    return qidx[hit], cand[hit], w[hit]


def as_points(bx, by, device):
    """``bx`` and ``by``, flattened, as ``(B, 2)`` float64 points on ``device``."""
    f64 = mesh_backend.float64
    x = mesh_backend.as_array(bx, dtype=f64, device=device).reshape(-1)
    y = mesh_backend.as_array(by, dtype=f64, device=device).reshape(-1)
    return mesh_backend.stack((x, y), dim=-1)
