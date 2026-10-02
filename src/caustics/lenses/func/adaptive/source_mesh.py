"""
The source-plane magnification mesh, and the rasterization its refinement reads.

:func:`segment_cells` and :func:`level_cells` find the square cells of a
lattice level that segments touch, and :func:`triangle_cells` the cell each
lattice triangle of that level is half of, so that a triangle a sheet edge
touches can be found without a segment-triangle test.
"""

from ....backend_obj import ArrayLike, backend

__all__ = (
    "segment_cells",
    "level_cells",
    "triangle_cells",
    "cell_member",
)


def _expand(start, count):
    """Each ``start[e] + 0 .. count[e] - 1``, with its owner ``e``."""
    int64 = backend.int64
    device = backend.device(count)
    total = int(backend.to_numpy(backend.sum(count)))
    owner = backend.repeat(
        backend.arange(count.shape[0], dtype=int64, device=device), count, axis=0
    )
    within = backend.arange(total, dtype=int64, device=device) - backend.repeat(
        backend.cumsum(count, dim=0) - count, count, axis=0
    )
    return owner, start[owner] + within


def segment_cells(a, b, lo, size, n_cells, pad) -> ArrayLike:
    """
    Cells of a square grid that segments touch, each segment padded by ``pad``.

    Column by column: for each column a segment's padded x-range spans, the
    segment is clipped to the column's slab, and every row its clipped
    y-range, padded, spans is taken. That is the supercover of the padded
    segment, so every cell the segment touches is flagged -- conservative
    under rounding for ``pad`` above a few ulps of the coordinates -- and no
    cell farther than ``pad`` from it is. A segment wholly outside the grid
    flags nothing.

    Parameters
    ----------
    a, b: ArrayLike
        ``(E, 2)`` float64 segment ends.

        *Unit: arcsec*
    lo: ArrayLike
        ``(2,)`` float64 corner of the grid.

        *Unit: arcsec*
    size: float
        Cell side.

        *Unit: arcsec*
    n_cells: int
        Cells per axis.
    pad: float
        Padding.

        *Unit: arcsec*

    Returns
    -------
    ArrayLike
        ``(K,)`` int64 cell ids ``i * n_cells + j``, ``i`` along x, unique and
        ascending.
    """
    f64 = backend.float64
    hi = lo + size * n_cells
    xa, xb = backend.minimum(a[:, 0], b[:, 0]), backend.maximum(a[:, 0], b[:, 0])
    ya, yb = backend.minimum(a[:, 1], b[:, 1]), backend.maximum(a[:, 1], b[:, 1])
    meets = (
        (xb + pad >= lo[0])
        & (xa - pad <= hi[0])
        & (yb + pad >= lo[1])
        & (ya - pad <= hi[1])
    )
    keep = backend.flatnonzero(meets)
    if keep.shape[0] == 0:
        return backend.zeros((0,), dtype=backend.int64, device=backend.device(a))
    a, b, xa, xb = a[keep], b[keep], xa[keep], xb[keep]
    top = n_cells - 1
    i0 = backend.clamp(backend.long(backend.floor((xa - pad - lo[0]) / size)), 0, top)
    i1 = backend.clamp(backend.long(backend.floor((xb + pad - lo[0]) / size)), 0, top)
    seg, col = _expand(i0, i1 - i0 + 1)

    # Clip each segment to its column's padded slab.
    x_lo = lo[0] + backend.to(col, dtype=f64) * size - pad
    x_hi = x_lo + size + 2.0 * pad
    ax, ay = a[seg, 0], a[seg, 1]
    dx, dy = b[seg, 0] - ax, b[seg, 1] - ay
    vertical = dx == 0
    safe = backend.where(vertical, backend.ones_like(dx), dx)
    ta, tb = (x_lo - ax) / safe, (x_hi - ax) / safe
    t0 = backend.where(
        vertical,
        backend.zeros_like(dx),
        backend.clamp(backend.minimum(ta, tb), 0.0, 1.0),
    )
    t1 = backend.where(
        vertical,
        backend.ones_like(dx),
        backend.clamp(backend.maximum(ta, tb), 0.0, 1.0),
    )
    y0, y1 = ay + t0 * dy, ay + t1 * dy
    y_lo = backend.minimum(y0, y1) - pad
    y_hi = backend.maximum(y0, y1) + pad
    rows = backend.flatnonzero((y_hi >= lo[1]) & (y_lo <= hi[1]))
    j0 = backend.clamp(backend.long(backend.floor((y_lo[rows] - lo[1]) / size)), 0, top)
    j1 = backend.clamp(backend.long(backend.floor((y_hi[rows] - lo[1]) / size)), 0, top)
    owner, row = _expand(j0, j1 - j0 + 1)
    return backend.unique(col[rows][owner] * n_cells + row)


def level_cells(lat, segments, pad, level) -> ArrayLike:
    """
    :func:`segment_cells` on the square cells of lattice level ``level``.

    The grid is the lattice's own: corner ``lat.lo``, side
    ``lat.scale * 2**(lat.level - level)``, ``lat.n >> (lat.level - level)``
    cells per axis. A relative padding of ``1e-9`` of a cell side is added to
    ``pad``.

    Parameters
    ----------
    lat: Lattice
    segments: ArrayLike
        ``(E, 2, 2)`` float64.

        *Unit: arcsec*
    pad: float
        *Unit: arcsec*
    level: int

    Returns
    -------
    ArrayLike
        ``(K,)`` int64 cell ids, unique and ascending.
    """
    shift = lat.level - level
    size = lat.scale * (1 << shift)
    return segment_cells(
        segments[:, 0], segments[:, 1], lat.lo, size, lat.n >> shift, pad + 1e-9 * size
    )


def triangle_cells(lat, ij, level) -> ArrayLike:
    """
    The level-``level`` square cell each lattice triangle of that level is half of.

    Red refinement keeps the two root shapes up to translation and point
    reflection, so a level-``level`` triangle is half of one level-``level``
    square cell, and its smallest coordinates are that cell's corner. A
    closure triangle lies inside its origin's cell, and its smallest
    coordinates lie below that cell's far corner, so it maps to its
    origin's cell at the origin's level too.

    Parameters
    ----------
    lat: Lattice
        A fresh lattice, ``origin == 0``.
    ij: ArrayLike
        ``(n, 3, 2)`` int64 lattice coordinates.
    level: int

    Returns
    -------
    ArrayLike
        ``(n,)`` int64 cell ids, as :func:`level_cells` numbers them.
    """
    shift = lat.level - level
    corner = backend.min(ij, dim=1) // (1 << shift)
    return corner[:, 0] * (lat.n >> shift) + corner[:, 1]


def cell_member(sorted_ids, ids) -> ArrayLike:
    """True where each of ``ids`` is in the ascending ``sorted_ids``."""
    if sorted_ids.shape[0] == 0:
        return backend.zeros(ids.shape, dtype=backend.bool, device=backend.device(ids))
    pos = backend.clamp(
        backend.searchsorted(sorted_ids, ids), 0, sorted_ids.shape[0] - 1
    )
    return sorted_ids[pos] == ids
