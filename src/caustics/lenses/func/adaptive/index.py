"""
A multi-level grid index over source-plane triangles, and lookups in it.

Cells come in power-of-two levels. Each triangle is registered at the one
level whose cells match its bounding box, grown by how far the triangle
reaches beyond itself, in the cells of that box it reaches, grown: in each
column of the box's cells, from the first to the last cell it reaches
there. A point reads one cell per occupied level, so the cells it reads
are as fine as the triangles registered in them wherever those crowd:
along caustics, and where outliers stretch the extent.
"""

from typing import NamedTuple

import numpy as np

from ....backend_obj import ArrayLike
from .mesh_backend import mesh_backend
from .geometry import contains, edge_nearest, expand, sorted_find, triangle_weights

# Level-0 cells divide the larger side of the index's extent into 2**KEY_BITS:
# the finest grid whose cell keys, with their level, fit in an int64.
KEY_BITS = 28
# A triangle's box spans at most about this many cells per axis at its level.
CELLS_PER_BOX = 6
# Most (point, candidate) pairs one step of a query holds.
PAIR_CAP = 1 << 20
# Most entries one step of a build holds.
ENTRY_CAP = 1 << 18
# The low bits of a packed column, holding its box: the codes of a level and
# x take the KEY_BITS + 1 bits above them, below an int64's sign bit.
ROW_BITS = 62 - KEY_BITS
# A step sorts int32 local keys where every one, and every entry's position
# in the step, is below this many; int64 otherwise.
INT32_KEYS = 1 << 31
# The margin by which the cells a triangle reaches are widened, against
# rounding: this much of a cell, far above the rounding of coordinates of the
# order of the index's extent...
PAD = 2.0**-10
# ...plus this many ulps of the index's largest coordinate, for an index far
# from the origin against its extent.
PAD_ULPS = 8
# Columns the eager step finding the cells each triangle reaches takes at once.
REACH_BLOCK = 1 << 18


class MeshIndex(NamedTuple):
    """
    Multi-level CSR index over triangles, each registered at the level its grown bounding box sets, in the cells of that box it reaches.

    Level ``l`` has square cells of side ``fine * 2**l``, counted from
    ``lo``; cell ``(ix, iy)`` of it has key ``l * 2**56 + ix * 2**28 + iy``.
    Cell ``keys[u]``'s triangles are ``leaves[offsets[u]:offsets[u + 1]]``,
    ascending.

    Parameters
    ----------
    lo, hi: ArrayLike
        ``(2,)`` float64 corners of the bounding box of every indexed
        triangle, grown.

        *Unit: arcsec*
    fine: ArrayLike
        ``()`` float64 side of the level-0 cells: the larger side of the
        bounding box over ``2**KEY_BITS``.

        *Unit: arcsec*
    levels: ArrayLike
        ``(L,)`` int64 levels holding a triangle, ascending.
    keys: ArrayLike
        ``(U,)`` int64 keys of the cells holding a triangle, ascending.
    offsets: ArrayLike
        ``(U + 1,)`` int64 CSR offsets.
    leaves: ArrayLike
        int64 triangle indices, cell by cell.
    """

    lo: ArrayLike
    hi: ArrayLike
    fine: ArrayLike
    levels: ArrayLike
    keys: ArrayLike
    offsets: ArrayLike
    leaves: ArrayLike


def _level_tables(device):
    """``2.0**l`` ``(KEY_BITS + 1,)`` float64, and the cells per axis ``2**(KEY_BITS - l)`` int64, of every level ``l``."""
    scale = mesh_backend.as_array(
        [2.0**l for l in range(KEY_BITS + 1)], dtype=mesh_backend.float64, device=device
    )
    span = mesh_backend.as_array(
        [1 << (KEY_BITS - l) for l in range(KEY_BITS + 1)],
        dtype=mesh_backend.int64,
        device=device,
    )
    return scale, span


def _cell_of(x, lo, side, top):
    """
    Cell coordinates of points ``x`` ``(..., 2)`` in cells of side ``side``
    counted from ``lo``, clamped to ``[0, top]``.

    ``side`` and ``top`` broadcast against ``x``. :func:`build_index` and
    :func:`index_cells` both locate points here, so they agree exactly.
    """
    u = mesh_backend.long(mesh_backend.floor((x - lo) / side))
    return mesh_backend.minimum(mesh_backend.clamp(u, 0, None), top)


def _cell_key(level, ix, iy):
    """The int64 key of cell ``(ix, iy)`` of ``level``."""
    return level * (1 << (2 * KEY_BITS)) + ix * (1 << KEY_BITS) + iy


def _code_offsets(device):
    """``(KEY_BITS + 1,)`` int64 offset of each level ``l``'s codes: the sum of ``2**(KEY_BITS - k)`` over ``k < l``."""
    return mesh_backend.as_array(
        [(2 << KEY_BITS) - ((2 << KEY_BITS) >> l) for l in range(KEY_BITS + 1)],
        dtype=mesh_backend.int64,
        device=device,
    )


def _level_code(level, x):
    """
    The int64 code of column ``x`` of ``level``: ``offset[level] + x``.

    ``offset[l]`` is the sum of the cells per axis ``2**(KEY_BITS - k)`` of
    the levels ``k < l``, so codes ascend in level, then in x, and stay
    below ``2**(KEY_BITS + 1)``.
    """
    return _code_offsets(mesh_backend.device(level))[level] + x


def _code_base(code):
    """The int64 key of the cell at ``y = 0`` of the level and x of each :func:`_level_code` ``code``."""
    offset = _code_offsets(mesh_backend.device(code))
    # The last level whose offset is at most the code.
    level = mesh_backend.searchsorted(offset, code + 1) - 1
    return _cell_key(level, code - offset[level], 0)


def _blocks(counts, total):
    """
    The block of each entry of consecutive blocks of ``counts`` ``(n,)`` int64 entries, each at least 1.

    A cumulative sum of flags set where each block but the first starts.
    ``total`` is ``counts.sum()``, read on the host by the caller.

    Returns
    -------
    ArrayLike
        ``(total,)`` int64.
    """
    flag = mesh_backend.zeros(
        (total,), dtype=mesh_backend.module.uint8, device=mesh_backend.device(counts)
    )
    flag = mesh_backend.fill_at_indices(
        flag, mesh_backend.cumsum(counts, dim=0)[:-1], 1
    )
    return mesh_backend.cumsum(flag, dim=0)


def _boxes(vertices, triangles, rows, grow):
    """
    The level of each grown box of ``triangles[rows]``, and the cells it covers there.

    Returns
    -------
    lo, hi, fine: ArrayLike
        As :class:`MeshIndex`'s.
    level: ArrayLike
        ``(K,)`` int64.
    lower: ArrayLike
        ``(K, 2)`` int64 cell coordinates of each box's lower-left cell.
    cells: ArrayLike
        ``(K, 2)`` int64 cells each box covers per axis.
    """
    device = mesh_backend.device(vertices)
    f64 = mesh_backend.float64
    tri = mesh_backend.to(vertices[triangles[rows]], dtype=f64)
    box_lo, box_hi = mesh_backend.min(tri, dim=1), mesh_backend.max(tri, dim=1)
    if grow is not None:
        reach = mesh_backend.unsqueeze(mesh_backend.to(grow, dtype=f64), -1)
        box_lo, box_hi = box_lo - reach, box_hi + reach
    lo, hi = mesh_backend.min(box_lo, dim=0), mesh_backend.max(box_hi, dim=0)
    extent = mesh_backend.max(hi - lo)
    extent = mesh_backend.where(extent > 0, extent, 1.0)  # every box one point
    fine = mesh_backend.clamp(
        extent / (1 << KEY_BITS), float(mesh_backend.finfo(f64).tiny), None
    )

    scale, span = _level_tables(device)
    size = mesh_backend.max(box_hi - box_lo, dim=1) / (CELLS_PER_BOX * fine)
    # The number of powers of two below `size`: ceil(log2(size)), from 0 to KEY_BITS.
    level = mesh_backend.searchsorted(scale[:KEY_BITS], size)
    side = mesh_backend.unsqueeze(fine * scale[level], -1)
    top = mesh_backend.unsqueeze(span[level] - 1, -1)
    i0 = _cell_of(box_lo, lo, side, top)
    i1 = _cell_of(box_hi, lo, side, top)
    return lo, hi, fine, level, i0, i1 - i0 + 1


def _run_flags(values):
    """``(n,)`` bool, True at the first entry of each run of equal values of ``values`` ``(n,)``, ``n >= 1``."""
    return mesh_backend.concatenate(
        (
            mesh_backend.ones(
                (1,), dtype=mesh_backend.bool, device=mesh_backend.device(values)
            ),
            values[1:] != values[:-1],
        ),
        dim=0,
    )


def _by_x(px, py, qx, qy):
    """Points ``p`` and ``q``, by their coordinates, in order of x, ``p`` first where they tie: a compare-and-swap of arrays."""
    swap = px > qx
    where = mesh_backend.where
    return (
        where(swap, qx, px),
        where(swap, qy, py),
        where(swap, px, qx),
        where(swap, py, qy),
    )


def _slope(px, py, qx, qy):
    """The slope of the line from ``p`` to ``q`` where ``qx > px``; 0 elsewhere."""
    wide = qx > px
    return mesh_backend.where(
        wide, (qy - py) / mesh_backend.where(wide, qx - px, 1.0), 0.0
    )


def _reached(tx, ty, side, reach, i0x, i0y, i1y, box, dx, lo):
    """
    The cells of each column that its box's triangle, grown, reaches: the first in y, and how many.

    Column ``dx`` of box ``box`` is the box's column ``i0x + dx`` in x, a
    slab of x. With A, B, C the triangle's vertices in order of x, the
    triangle lies between its long edge AC and the chain A-B-C. Over the
    slab, widened by ``reach`` on both sides and clipped to ``[Ax, Cx]``,
    both are extreme at the slab's ends, and the chain also at B where B
    lies in it: the triangle there spans the interval of y from the least
    to the greatest of those values. That interval, widened by ``reach``,
    gives the column's first and last cell, located as :func:`_cell_of`
    locates points and clamped to the box's cells ``[i0y, i1y]``. As the
    triangle is convex, with no grow every cell between them holds a point
    of it.

    No cell a lookup can find the triangle from is dropped. A point within
    its grow of the triangle, in this column, is within that grow in x and
    in y of a point of the triangle; that point lies in the widened slab,
    so its y lies in the interval, and the first point's y in the widened
    one. The slab is where :func:`_cell_of` puts a point, its clamp
    included: a point on the index's upper edge is put in the last cell,
    and its x is that column's upper end, which the slab holds. Beyond the
    grow, ``reach`` holds a margin of ``PAD`` of a cell and ``PAD_ULPS``
    ulps of the index's largest coordinate, above the rounding of the slab,
    of the interval and of where a lookup puts a point; and the interval's
    ends are located by the arithmetic of :func:`_cell_of`, whose rounding
    never reverses the order of two values. A triangle of no width in x,
    and a column whose interval comes out empty or not a number, keep the
    box's cells.

    Gathers and arithmetic only: no search, no sort, no branch on data,
    nothing written in place.

    Parameters
    ----------
    tx, ty: ArrayLike
        ``(K, 3)`` float64 coordinates of each box's triangle's vertices.

        *Unit: arcsec*
    side: ArrayLike
        ``(K,)`` float64 side of the cells of each box's level.

        *Unit: arcsec*
    reach: ArrayLike
        ``(K,)`` float64 each triangle's grow, plus the margin.

        *Unit: arcsec*
    i0x, i0y, i1y: ArrayLike
        ``(K,)`` int64 each box's first cell in x, and its first and last
        cell in y.
    box, dx: ArrayLike
        ``(n,)`` int64 each column's box, and its offset in x from the
        box's first column.
    lo: ArrayLike
        ``(2,)`` float64 the index's lower corner.

        *Unit: arcsec*

    Returns
    -------
    first, count: ArrayLike
        ``(n,)`` int64 each column's first cell in y and its number of
        cells, within ``[i0y, i1y]``.
    """
    where, minimum, maximum = (
        mesh_backend.where,
        mesh_backend.minimum,
        mesh_backend.maximum,
    )
    # Per box: A, B, C by three compare-and-swaps, and the slopes of AC, AB
    # and BC.
    ax, ay, bx, by = _by_x(tx[:, 0], ty[:, 0], tx[:, 1], ty[:, 1])
    bx, by, cx, cy = _by_x(bx, by, tx[:, 2], ty[:, 2])
    ax, ay, bx, by = _by_x(ax, ay, bx, by)
    s_ac, s_ab, s_bc = (
        _slope(ax, ay, cx, cy),
        _slope(ax, ay, bx, by),
        _slope(bx, by, cx, cy),
    )
    # Per column.
    side, reach, i0y, i1y = side[box], reach[box], i0y[box], i1y[box]
    ax, ay, bx, by, cx = ax[box], ay[box], bx[box], by[box], cx[box]
    s_ac, s_ab, s_bc = s_ac[box], s_ab[box], s_bc[box]
    xa = (
        lo[0]
        + mesh_backend.to(i0x[box] + dx, dtype=mesh_backend.float64) * side
        - reach
    )
    xl = maximum(xa, ax)
    xr = minimum(xa + side + 2.0 * reach, cx)
    long_l, long_r = ay + s_ac * (xl - ax), ay + s_ac * (xr - ax)
    chain_l = where(xl <= bx, ay + s_ab * (xl - ax), by + s_bc * (xl - bx))
    chain_r = where(xr <= bx, ay + s_ab * (xr - ax), by + s_bc * (xr - bx))
    ymin = minimum(minimum(long_l, long_r), minimum(chain_l, chain_r))
    ymax = maximum(maximum(long_l, long_r), maximum(chain_l, chain_r))
    at_b = (xl <= bx) & (bx <= xr)
    ymin = where(at_b, minimum(ymin, by), ymin)
    ymax = where(at_b, maximum(ymax, by), ymax)
    c0 = maximum(
        mesh_backend.long(mesh_backend.floor((ymin - reach - lo[1]) / side)), i0y
    )
    c1 = minimum(
        mesh_backend.long(mesh_backend.floor((ymax + reach - lo[1]) / side)), i1y
    )
    # NaN, from a slope too steep for a float, fails the last comparison.
    keep = (c1 >= c0) & (cx > ax) & (ymin <= ymax)
    first = where(keep, c0, i0y)
    return first, where(keep, c1, i1y) - first + 1


def _reached_in_blocks(tx, ty, side, reach, i0x, i0y, i1y, box, dx, lo):
    """
    :func:`_reached`, ``REACH_BLOCK`` columns at a time, so that its temporaries stay bounded.

    Each block is handed only its boxes, from the lowest to the highest it
    holds: few where ``box`` ascends, as :func:`_columns`' does.

    Returns
    -------
    first, count: ArrayLike
        As :func:`_reached`'s.
    """
    n = box.shape[0]
    device = mesh_backend.device(box)
    first = mesh_backend.empty((n,), dtype=mesh_backend.int64, device=device)
    count = mesh_backend.empty((n,), dtype=mesh_backend.int64, device=device)
    for start in range(0, n, REACH_BLOCK):
        block = slice(start, start + REACH_BLOCK)
        b = box[block]
        low, high = mesh_backend.to_numpy(
            mesh_backend.stack((mesh_backend.min(b), mesh_backend.max(b)))
        ).tolist()
        boxes = slice(low, high + 1)
        f, c = _reached(
            tx[boxes],
            ty[boxes],
            side[boxes],
            reach[boxes],
            i0x[boxes],
            i0y[boxes],
            i1y[boxes],
            b - low,
            dx[block],
            lo,
        )
        first = mesh_backend.fill_at_indices(first, block, f)
        count = mesh_backend.fill_at_indices(count, block, c)
    return first, count


def _columns(vertices, triangles, rows, grow):
    """
    The columns of cells of the grown boxes of ``triangles[rows]``, by level and x, each holding the cells its triangle reaches.

    A box's column holds the cells from the first to the last in y that
    the box's triangle, grown, reaches there (:func:`_reached`). Each
    column is packed in one int64, ``code << ROW_BITS | box``, with
    ``code`` the :func:`_level_code` of its level and x and ``box`` its
    box's position in ``rows``, and these are sorted by value: by level,
    x, then row, since ``rows`` ascend. ``code`` and ``box`` are read off
    the sorted values, and the row is gathered per box; a column's first
    cell and count, packed in one int64, are gathered through the sort's
    order.

    Returns
    -------
    lo, hi, fine, levels: ArrayLike
        As :class:`MeshIndex`'s.
    code: ArrayLike
        ``(C,)`` int64 code of each column's level and x, non-decreasing.
    first: ArrayLike
        ``(C,)`` int64 lowest cell in y of each column.
    height: ArrayLike
        ``(C,)`` int64 cells in each column.
    column_rows: ArrayLike
        ``(C,)`` int64 row of each column's triangle.
    """
    lo, hi, fine, level, lower, cells = _boxes(vertices, triangles, rows, grow)
    device = mesh_backend.device(lower)
    f64, int64 = mesh_backend.float64, mesh_backend.int64
    wide = cells[:, 0]
    n_columns = int(mesh_backend.to_numpy(mesh_backend.sum(wide)))
    box_of = _blocks(wide, n_columns)
    dx = (
        mesh_backend.arange(n_columns, dtype=int64, device=device)
        - (mesh_backend.cumsum(wide, dim=0) - wide)[box_of]
    )
    # The triangles' vertices in float64, as in _boxes; freed with the rest
    # of the step's inputs once it has run.
    corner = triangles[rows]
    tx = mesh_backend.to(vertices[:, 0][corner], dtype=f64)
    ty = mesh_backend.to(vertices[:, 1][corner], dtype=f64)
    del corner
    scale, _ = _level_tables(device)
    side = fine * scale[level]
    largest = mesh_backend.max(mesh_backend.abs(mesh_backend.concatenate((lo, hi))))
    margin = PAD * side + PAD_ULPS * mesh_backend.finfo(f64).eps * largest
    reach = margin if grow is None else mesh_backend.to(grow, dtype=f64) + margin
    del margin
    first, height = _reached_in_blocks(
        tx,
        ty,
        side,
        reach,
        lower[:, 0],
        lower[:, 1],
        lower[:, 1] + cells[:, 1] - 1,
        box_of,
        dx,
        lo,
    )
    del tx, ty, side, reach
    # A count is at most CELLS_PER_BOX + 2.
    bits = (CELLS_PER_BOX + 2).bit_length()
    reached = (first << bits) | height
    del first, height
    packed = (_level_code(level, lower[:, 0]) << ROW_BITS) + mesh_backend.arange(
        rows.shape[0], dtype=int64, device=device
    )
    packed = packed[box_of] + (dx << ROW_BITS)
    # Freed before the columns' own arrays are made.
    del box_of, dx
    packed, order = mesh_backend.module.sort(packed, stable=True)
    reached = reached[order]
    del order
    box = packed & ((1 << ROW_BITS) - 1)
    code = packed >> ROW_BITS
    del packed
    column_rows = rows[box]
    del box
    levels = mesh_backend.flatnonzero(
        mesh_backend.bincount(level, minlength=KEY_BITS + 1)
    )
    first, height = reached >> bits, reached & ((1 << bits) - 1)
    return lo, hi, fine, levels, code, first, height, column_rows


def _steps(code, height):
    """
    Where each run of columns of one level and x starts, and the entries before it, on the host.

    Returns
    -------
    starts, before: np.ndarray
        ``(G + 1,)`` int64 for ``G`` runs, closed by the column and entry
        totals.
    """
    run_start = mesh_backend.flatnonzero(_run_flags(code))
    ends = mesh_backend.cumsum(height, dim=0)
    starts = np.append(mesh_backend.to_numpy(run_start), code.shape[0])
    before = np.append(
        mesh_backend.to_numpy((ends - height)[run_start]),
        mesh_backend.to_numpy(ends[-1]),
    )
    return starts, before


def _step_cells(code, first, height, column_rows, total):
    """
    One step of :func:`build_index`: the cells of columns ``code``, from cell ``first`` in y, ``height`` cells tall.

    The columns come as :func:`_columns` gives them; ``total`` is
    ``height.sum()``, read on the host by the caller. Each entry is put in
    order by a stable sort on a local key, ``run * span + (y - ymin)``:
    ``run`` the index of its run of one code in the step, ``y`` its cell,
    ``ymin`` and ``span`` the step's lowest cell in y and its range. That
    orders the entries as their cells' keys do, and keeps each cell's
    triangles in ascending row order. The local keys are int32 where every
    one, and every entry's position in the step, is below ``INT32_KEYS``,
    int64 otherwise, and the cells' keys are rebuilt from them.

    Returns
    -------
    keys: ArrayLike
        ``(U,)`` int64 keys of the cells, ascending.
    starts: ArrayLike
        ``(U,)`` int64 first entry of each cell.
    leaves: ArrayLike
        ``(total,)`` int64 triangle rows, cell by cell, ascending.
    """
    int64 = mesh_backend.int64
    new_run = _run_flags(code)
    run_code = code[mesh_backend.flatnonzero(new_run)]
    ymin, top = mesh_backend.to_numpy(
        mesh_backend.stack((mesh_backend.min(first), mesh_backend.max(first + height)))
    ).tolist()
    span = top - ymin
    fits = max(run_code.shape[0] * span, total) <= INT32_KEYS
    dtype = mesh_backend.int32 if fits else int64
    column = _blocks(height, total)
    # The local key of each column's lowest cell, less the position of the
    # column's first entry: an entry's local key is this plus its position.
    shift = (
        (mesh_backend.cumsum(new_run, dim=0) - 1) * span
        + (first - ymin)
        - (mesh_backend.cumsum(height, dim=0) - height)
    )
    local = mesh_backend.to(shift, dtype=dtype)[column] + mesh_backend.arange(
        total, dtype=dtype, device=mesh_backend.device(code)
    )
    # A stable sort keeps each cell's triangles in ascending row order.
    local, order = mesh_backend.module.sort(local, stable=True)
    starts = mesh_backend.flatnonzero(_run_flags(local))
    cell = mesh_backend.to(local[starts], dtype=int64)
    keys = _code_base(run_code)[cell // span] + (ymin + cell % span)
    return keys, starts, column_rows[column[order]]


def build_index(vertices, triangles, rows, grow=None):
    """
    The :class:`MeshIndex` over ``triangles[rows]``, each grown by its ``grow``.

    A triangle's level is set by its grown bounding box: the fewest
    doublings of ``CELLS_PER_BOX * fine`` that reach the box's longer side,
    so the box spans at most about ``CELLS_PER_BOX`` cells per axis there.
    The triangle is registered in the cells of that box it reaches, grown:
    in each column of the box's cells, from the first to the last cell it
    reaches there (:func:`_reached`). No cell a lookup can find it from is
    left out.

    The boxes' columns of cells are put in order of level, x and row by
    one sort of int64s, each packing a column's level and x above its
    box (:func:`_columns`). Their cells are then sorted in steps of whole
    runs of one level and x, holding at most ``ENTRY_CAP`` entries or one
    run holding more, each on a local key of the run in the step and the
    cell's y (:func:`_step_cells`): no two steps share a cell, so memory
    stays bounded and the index does not depend on the steps.

    Parameters
    ----------
    vertices: ArrayLike
        ``(V, 2)`` positions.

        *Unit: arcsec*
    triangles: ArrayLike
        ``(T, 3)`` int64 vertex indices.
    rows: ArrayLike
        ``(K,)`` int64 ascending rows of ``triangles`` to index.
    grow: Optional[ArrayLike]
        ``(K,)`` float64, how far each of ``triangles[rows]`` reaches beyond
        itself: its bounding box is grown by it on every side. None grows
        nothing.

        *Unit: arcsec*

    Returns
    -------
    MeshIndex

    Raises
    ------
    ValueError
        With ``2**ROW_BITS`` rows or more: a packed column holds fewer.
    """
    device = mesh_backend.device(vertices)
    f64, int64 = mesh_backend.float64, mesh_backend.int64
    n = rows.shape[0]
    if n >= 1 << ROW_BITS:
        raise ValueError(f"build_index takes fewer than {1 << ROW_BITS} rows; got {n}")
    if n == 0:
        return MeshIndex(
            lo=mesh_backend.zeros((2,), dtype=f64, device=device),
            hi=mesh_backend.ones((2,), dtype=f64, device=device),
            fine=mesh_backend.as_array(1.0 / (1 << KEY_BITS), dtype=f64, device=device),
            levels=mesh_backend.zeros((0,), dtype=int64, device=device),
            keys=mesh_backend.zeros((0,), dtype=int64, device=device),
            offsets=mesh_backend.zeros((1,), dtype=int64, device=device),
            leaves=mesh_backend.zeros((0,), dtype=int64, device=device),
        )
    lo, hi, fine, levels, code, first, height, column_rows = _columns(
        vertices, triangles, rows, grow
    )
    starts, before = _steps(code, height)
    total = int(before[-1])
    keys, offsets = [], []
    leaves = mesh_backend.zeros((total,), dtype=int64, device=device)
    run, n_runs = 0, starts.shape[0] - 1
    while run < n_runs:
        end = max(
            run + 1,
            int(np.searchsorted(before, before[run] + ENTRY_CAP, side="right")) - 1,
        )
        a, b = int(starts[run]), int(starts[end])
        e0, e1 = int(before[run]), int(before[end])
        cell_keys, cell_starts, cell_leaves = _step_cells(
            code[a:b], first[a:b], height[a:b], column_rows[a:b], e1 - e0
        )
        keys.append(cell_keys)
        offsets.append(cell_starts + e0)
        leaves = mesh_backend.fill_at_indices(leaves, slice(e0, e1), cell_leaves)
        run = end
    offsets.append(mesh_backend.as_array([total], dtype=int64, device=device))
    return MeshIndex(
        lo=lo,
        hi=hi,
        fine=fine,
        levels=levels,
        keys=mesh_backend.concatenate(keys, dim=0),
        offsets=mesh_backend.concatenate(offsets, dim=0),
        leaves=leaves,
    )


def index_cells(index, beta):
    """
    The cell of each point at every level of ``index``, as CSR ranges into ``index.leaves``.

    A point outside the stored ``[lo, hi]``, NaN included, gets empty
    ranges, as does a cell that holds nothing.

    Parameters
    ----------
    index: MeshIndex
    beta: ArrayLike
        ``(B, 2)`` points.

        *Unit: arcsec*

    Returns
    -------
    start, count: ArrayLike
        ``(B, L)`` int64, one column per entry of ``index.levels``.
    """
    int64 = mesh_backend.int64
    device = mesh_backend.device(beta)
    n_levels = index.levels.shape[0]
    if n_levels == 0:
        empty = mesh_backend.zeros((beta.shape[0], 0), dtype=int64, device=device)
        return empty, empty
    scale, span = _level_tables(device)
    side = (index.fine * scale[index.levels]).reshape(1, n_levels, 1)
    top = (span[index.levels] - 1).reshape(1, n_levels, 1)
    u = _cell_of(mesh_backend.unsqueeze(beta, 1), index.lo, side, top)
    key = _cell_key(mesh_backend.unsqueeze(index.levels, 0), u[..., 0], u[..., 1])
    pos, found = sorted_find(index.keys, key)
    inside = (
        (beta[:, 0] >= index.lo[0])
        & (beta[:, 0] <= index.hi[0])
        & (beta[:, 1] >= index.lo[1])
        & (beta[:, 1] <= index.hi[1])
    )
    found = found & mesh_backend.unsqueeze(inside, -1)
    start = index.offsets[pos]
    count = mesh_backend.where(
        found, index.offsets[pos + 1] - start, mesh_backend.zeros_like(start)
    )
    return start, count


def _chunk_hits(index, vertices, triangles, beta, start, count, total, grow):
    """
    :func:`index_hits` for points ``beta`` whose cells are ``start``, ``count`` ``(B, L)``.

    ``total`` is ``count.sum()``, read on the host by the caller.
    """
    block, within = expand(count.reshape(-1), total)
    cand = index.leaves[start.reshape(-1)[block] + within]
    qidx = block // start.shape[1]
    tri = vertices[triangles[cand]]
    w = triangle_weights(tri, beta[qidx])
    hit = contains(w)
    if grow is not None:
        reach = mesh_backend.flatnonzero(~hit & (grow[cand] > 0))
        dist, _ = edge_nearest(tri[reach], beta[qidx[reach]])
        hit = mesh_backend.fill_at_indices(hit, reach, dist <= grow[cand[reach]])
    hit = mesh_backend.flatnonzero(hit)
    qidx, cand, w = qidx[hit], cand[hit], w[hit]
    # Each level's hits come in ascending order; merge them per point.
    order = mesh_backend.lexsort((cand, qidx))
    return qidx[order], cand[order], w[order]


def index_hits(index, vertices, triangles, beta, grow=None):
    """
    Triangles of ``index`` whose image contains each point, zeros counting as inside, or reaches it.

    One cell per level per point, then :func:`~.geometry.contains` on its
    triangles. With ``grow``, a triangle that does not contain the point is
    a hit when the point lies within its ``grow`` of it
    (:func:`~.geometry.edge_nearest`); the index must have been built with
    a ``grow`` at least as large. Points run in chunks of whole points
    holding at most ``PAIR_CAP`` (point, candidate) pairs, or one point
    holding more, so memory stays bounded and the result does not depend
    on the chunking.

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
    grow: Optional[ArrayLike]
        ``(T,)`` float64, how far each triangle reaches beyond itself. None
        reaches nothing.

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
    qidx = [mesh_backend.zeros((0,), dtype=int64, device=device)]
    tri = [mesh_backend.zeros((0,), dtype=int64, device=device)]
    w = [mesh_backend.zeros((0, 3), dtype=vertices.dtype, device=device)]
    step = max(1, PAIR_CAP // max(index.levels.shape[0], 1))
    for first in range(0, beta.shape[0], step):
        chunk = beta[first : first + step]
        start, count = index_cells(index, chunk)
        # Running candidate totals, read on the host once per chunk.
        ends = np.cumsum(mesh_backend.to_numpy(mesh_backend.sum(count, dim=1)))
        lo = 0
        while lo < chunk.shape[0]:
            before = int(ends[lo - 1]) if lo else 0
            hi = max(
                lo + 1, int(np.searchsorted(ends, before + PAIR_CAP, side="right"))
            )
            total = int(ends[hi - 1]) - before
            if total:
                found = _chunk_hits(
                    index,
                    vertices,
                    triangles,
                    chunk[lo:hi],
                    start[lo:hi],
                    count[lo:hi],
                    total,
                    grow,
                )
                qidx.append(found[0] + (first + lo))
                tri.append(found[1])
                w.append(found[2])
            lo = hi
    return (
        mesh_backend.concatenate(qidx, dim=0),
        mesh_backend.concatenate(tri, dim=0),
        mesh_backend.concatenate(w, dim=0),
    )


def as_points(bx, by, device):
    """``bx`` and ``by``, flattened, as ``(B, 2)`` float64 points on ``device``."""
    f64 = mesh_backend.float64
    x = mesh_backend.as_array(bx, dtype=f64, device=device).reshape(-1)
    y = mesh_backend.as_array(by, dtype=f64, device=device).reshape(-1)
    return mesh_backend.stack((x, y), dim=-1)
