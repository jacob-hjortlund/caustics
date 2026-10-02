"""
The source-plane magnification mesh: an adaptive lattice sampled with the total magnification.

:func:`build_magnification_mesh` refines a dyadic triangulation of the
source plane, with
:func:`~caustics.lenses.func.adaptive.magnification.make_sampler` in place
of ``raytrace``: every lattice point it evaluates holds the total
magnification there and the image count. It reuses the lens build's lattice,
vertex cache, leaf store, red split, balance and closure, and makes no lens
call. :func:`segment_cells`, :func:`level_cells` and :func:`triangle_cells`
find the triangles a sheet edge touches without a segment-triangle test.
"""

import math
from typing import Any, NamedTuple, Optional, Tuple
from warnings import warn

from ....backend_obj import ArrayLike, backend
from .geometry import child_matrix_tables
from .state import (
    _ambient,
    active_add_slots,
    cache_lookup,
    cache_size,
    empty_active,
    empty_cache,
    empty_store,
    store_add,
    store_compact,
    store_remove,
)
from .lattice import (
    Lattice,
    check_lattice_keys,
    depth_floor,
    initial_triangles,
    lattice_key,
    lattice_xy,
    make_lattice,
    midpoint_ij,
)
from .sampling import evaluate
from .refinement import balance, red_split
from .closure import canonical_order, close
from .mesh import MeshIndex, build_index
from .magnification import band_magnification_floor, make_sampler, sheet_edges

__all__ = (
    "segment_cells",
    "level_cells",
    "triangle_cells",
    "cell_member",
    "split_mask",
    "validate_source_args",
    "warn_source_depth_limited",
    "MagnificationMesh",
    "build_magnification_mesh",
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


def split_mask(mu6, touched, targets, rtol) -> ArrayLike:
    """
    The source build's refinement criterion, vectorized over triangles.

    A triangle splits when a non-fov sheet edge touches its cell
    (``touched``); when, for some targeted threshold, its six samples are not
    all on one side, ``mu >= t`` counting as inside -- the tracing rule of
    :mod:`~caustics.lenses.func.adaptive.regions`; or, given ``rtol``, when a
    midpoint's ``log(1 + mu)`` deviates from the mean of its edge's ends by
    ``rtol`` or more. That last test is written ``all(dev < rtol)``, as in
    :func:`~caustics.lenses.func.adaptive.criterion.converged_from_deviation`,
    so a NaN deviation -- ``inf - inf`` next to the critical band -- splits.

    Parameters
    ----------
    mu6: ArrayLike
        ``(n, 6)`` float64 ``mu`` at ``theta_1, theta_2, theta_3, m_1, m_2,
        m_3``, ``m_i`` opposite ``theta_i``.
    touched: ArrayLike
        ``(n,)`` bool.
    targets: Tuple[float, ...]
    rtol: Optional[float]

    Returns
    -------
    ArrayLike
        ``(n,)`` bool, True where the triangle splits.
    """
    split = touched
    for t in targets:
        inside = mu6 >= t
        split = split | (backend.any(inside, dim=1) & backend.any(~inside, dim=1))
    if rtol is not None:
        log_mu = backend.log(1.0 + mu6)
        lv, lm = log_mu[:, :3], log_mu[:, 3:]
        dev = backend.abs(lm - 0.5 * (lv[:, [1, 2, 0]] + lv[:, [2, 0, 1]]))
        split = split | ~backend.all(dev < rtol, dim=1)
    return split


def validate_source_args(fov, init_res, src_tol, max_depth) -> None:
    """Reject impossible parameters, including a lattice that would overflow int64."""
    if not fov > 0:
        raise ValueError(f"fov must be positive, got {fov}")
    if init_res < 1:
        raise ValueError(f"init_res must be at least 1, got {init_res}")
    if not src_tol > 0:
        raise ValueError(f"src_tol must be positive, got {src_tol}")
    if max_depth < 0:
        raise ValueError(f"max_depth must be non-negative, got {max_depth}")
    max_level = min(max_depth, depth_floor(fov, init_res, src_tol))
    check_lattice_keys(
        init_res, max_level, "Raise src_tol, lower max_depth, or lower init_res."
    )


def warn_source_depth_limited(fov, init_res, src_tol, d_floor, max_level) -> None:
    """Warn when ``max_depth`` stopped refinement short of ``src_tol``."""
    if d_floor <= max_level:
        return
    l_max = float(math.sqrt(2.0) * fov / (init_res * 2**max_level))
    warn(
        f"Magnification mesh is depth-limited: max_depth={max_level} is below "
        f"d_floor={d_floor}, the depth required to reach src_tol={src_tol:g} "
        f"arcsec. Refinement stops at level {max_level}, where the maximum "
        f"leaf edge is {l_max:.3g} arcsec. Set max_depth >= {d_floor} to "
        "restore the src_tol guarantee, or raise init_res / src_tol."
    )


def _targets(mu_min) -> Tuple[float, ...]:
    """``mu_min`` as a tuple of floats: ``()`` for ``None``."""
    if mu_min is None:
        return ()
    values = backend.to_numpy(backend.as_array(mu_min, dtype=backend.float64))
    return tuple(float(v) for v in values.reshape(-1).tolist())


class MagnificationMesh(NamedTuple):
    """
    An adaptive source-plane mesh sampled with the total magnification.

    Built by :func:`build_magnification_mesh` and read by
    :mod:`~caustics.lenses.func.adaptive.regions`. Positions and values are
    always float64, so there is no ``dtype`` field.

    Parameters
    ----------
    vertices: ArrayLike
        ``(V, 2)`` source-plane position of every vertex, in lattice-key
        order.

        *Unit: arcsec*
    vertices_ij: ArrayLike
        ``(V, 2)`` int64 lattice coordinates of every vertex.
    mu: ArrayLike
        ``(V,)`` float64
        :func:`~caustics.lenses.func.adaptive.magnification.mesh_total_magnification`
        at every vertex, ``+inf`` in the critical band's images.
    n: ArrayLike
        ``(V,)`` int64 image count at every vertex.
    leaves: ArrayLike
        ``(L, 3)`` int64 closed, conforming, positively oriented leaves.
    leaf_level: ArrayLike
        ``(L,)`` int64 refinement level, inherited from the pre-closure leaf.
    incomplete: ArrayLike
        ``(L,)`` bool, True where the image of the lens fov's boundary
        touches the leaf's pre-closure cell: on one side of it an image lies
        outside the lens fov, so ``mu`` there is too low.
    index: MeshIndex
        Over every leaf, for :func:`~caustics.lenses.func.adaptive.regions.in_magnified_region`.
    lattice: Lattice
    fov: float
        Side of the square window meshed.

        *Unit: arcsec*
    init_res: int
    src_tol: float
        The source-plane size floor; see :func:`build_magnification_mesh`.

        *Unit: arcsec*
    d_floor: int
        Level at which the level-0 hypotenuse first falls to ``src_tol``.
    max_level: int
        ``min(max_depth, d_floor)``.
    mu_min: Tuple[float, ...]
        The thresholds the refinement targeted, possibly none.
    rtol: Optional[float]
        The sweep tolerance, or ``None``.
    mu_band: float
        :func:`~caustics.lenses.func.adaptive.magnification.band_magnification_floor`
        of the lens mesh's critical band.
    device: Any
    """

    vertices: ArrayLike
    vertices_ij: ArrayLike
    mu: ArrayLike
    n: ArrayLike
    leaves: ArrayLike
    leaf_level: ArrayLike
    incomplete: ArrayLike
    # Shadows `tuple.index`, as `AdaptiveMesh.index` does.
    index: MeshIndex  # type: ignore[assignment]
    lattice: Lattice
    fov: float
    init_res: int
    src_tol: float
    d_floor: int
    max_level: int
    mu_min: Tuple[float, ...]
    rtol: Optional[float]
    mu_band: float
    device: Any


def _test_round(
    store,
    cache,
    active,
    rows,
    lat,
    sample,
    inner_cells,
    targets,
    rtol,
    compose,
    batch_size,
):
    """
    Test ``rows`` of ``store``, all below ``max_level``, and split the failures.

    One :func:`evaluate` call over every row's midpoints; the criterion is
    :func:`split_mask`. A failing row is replaced by its four red-split
    children, which are returned as the next untested rows.
    """
    v = store.v[rows]
    ij = cache.ij[v]
    mid_keys = lattice_key(lat, midpoint_ij(ij))
    cache = evaluate(cache, lat, mid_keys.reshape(-1), sample, batch_size)
    m = cache_lookup(cache, mid_keys)
    level = store.level[rows]
    touched = backend.zeros((rows.shape[0],), dtype=backend.bool)
    for d in backend.to_numpy(backend.unique(level)).tolist():
        at = backend.flatnonzero(level == d)
        hit = cell_member(inner_cells[int(d)], triangle_cells(lat, ij[at], int(d)))
        touched = backend.fill_at_indices(touched, at, hit)
    mu6 = cache.beta[backend.concatenate((v, m), dim=1)][..., 0]
    split = backend.flatnonzero(split_mask(mu6, touched, targets, rtol))
    failing = rows[split]
    kid_v, kid_cls = red_split(v[split], m[split], store.cls[failing], compose)
    kid_level = backend.repeat(store.level[failing] + 1, 4, axis=0)
    store = store_remove(store, failing)
    store, untested = store_add(store, kid_v, kid_level, kid_cls, 0)
    active = active_add_slots(active, cache_size(cache), kid_v.reshape(-1))
    return store, cache, active, untested


def _refine(
    lat,
    sample,
    inner_cells,
    targets,
    rtol,
    init_res,
    max_level,
    compose,
    root_class,
    batch_size,
):
    """
    Test every leaf below ``max_level`` until none fails, balance, and repeat.

    The worklist holds the untested leaves: at first the level-0 triangles,
    then each round's children, then whatever :func:`balance` forces. Testing
    forced leaves too is what makes the floor guarantee hold: under 2:1
    balance a closure triangle's vertices are among its origin's six
    samples, so once every leaf below ``max_level`` has passed
    :func:`split_mask`, only ``max_level`` leaves straddle a targeted
    threshold. ``max_level`` leaves are never tested.
    """
    ij0, cls0 = initial_triangles(init_res, lat.level, root_class)
    keys0 = lattice_key(lat, ij0)
    cache = evaluate(empty_cache(), lat, keys0.reshape(-1), sample, batch_size)
    v0 = cache_lookup(cache, keys0)
    active = active_add_slots(empty_active(), cache_size(cache), v0.reshape(-1))
    store, untested = store_add(empty_store(), v0, 0, cls0, 0)
    while True:
        while True:
            rows = untested[store.level[untested] < max_level]
            if rows.shape[0] == 0:
                break
            store, cache, active, untested = _test_round(
                store,
                cache,
                active,
                rows,
                lat,
                sample,
                inner_cells,
                targets,
                rtol,
                compose,
                batch_size,
            )
        n_rows = store.v.shape[0]
        store, cache, active, forced = balance(
            store, cache, lat, active, max_level, compose, sample, batch_size
        )
        if forced == 0:
            return cache, active, store
        new = backend.arange(n_rows, store.v.shape[0], dtype=backend.int64)
        untested = new[store.valid[new]]


def _freeze(lat, cache, active, store, outer_cells, index_cells):
    """Close, mark ``incomplete``, compact and index the refined store."""
    pre_v, pre_level, _, pre_status = store_compact(store)
    order = canonical_order(lat, cache, pre_v)
    pre_v, pre_level, pre_status = pre_v[order], pre_level[order], pre_status[order]
    pre_incomplete = backend.zeros((pre_v.shape[0],), dtype=backend.bool)
    for d in backend.to_numpy(backend.unique(pre_level)).tolist():
        at = backend.flatnonzero(pre_level == d)
        hit = cell_member(
            outer_cells[int(d)], triangle_cells(lat, cache.ij[pre_v[at]], int(d))
        )
        pre_incomplete = backend.fill_at_indices(pre_incomplete, at, hit)
    leaf_v, origin, leaf_level, _ = close(
        lat, cache, active, pre_v, pre_level, pre_status
    )
    # Lattice-key order, as `freeze` orders a lens mesh's vertices.
    used = backend.unique(leaf_v.reshape(-1))
    used = used[backend.argsort(lattice_key(lat, cache.ij[used]))]
    remap = backend.fill_at_indices(
        backend.zeros((cache_size(cache),), dtype=backend.int64),
        used,
        backend.arange(used.shape[0], dtype=backend.int64),
    )
    leaves = remap[leaf_v]
    vertices = lattice_xy(lat, cache.ij[used])
    index = build_index(
        vertices,
        leaves,
        backend.arange(leaves.shape[0], dtype=backend.int64),
        index_cells,
    )
    return (
        vertices,
        cache.ij[used],
        cache.beta[used, 0],
        backend.long(cache.beta[used, 1]),
        leaves,
        leaf_level,
        pre_incomplete[origin],
        index,
    )


def build_magnification_mesh(
    mesh,
    init_res,
    src_tol,
    max_depth=25,
    *,
    mu_min=None,
    rtol=None,
    fov=None,
    x0=None,
    y0=None,
    batch_size=None,
    index_cells=None,
) -> MagnificationMesh:
    """
    Build an adaptive source-plane mesh sampled with the total magnification.

    A dyadic lattice triangulation of a square source-plane window, refined
    as the lens build refines the lens plane, with
    :func:`~caustics.lenses.func.adaptive.magnification.make_sampler` in
    place of ``raytrace``: every lattice point evaluated holds
    :func:`~caustics.lenses.func.adaptive.magnification.mesh_total_magnification`
    there, and the image count. No lens call.

    Refinement tests every leaf below ``max_level`` -- the three vertices
    and three edge midpoints of each -- and splits it when any of these
    holds (:func:`split_mask`):

    1. A sheet edge of the lens mesh that is not on its fov boundary touches
       the leaf's cell
       (:func:`~caustics.lenses.func.adaptive.magnification.sheet_edges`).
       Those are where the image count changes -- the critical band's edges,
       with the ``+inf`` strip along each caustic between them, folds, and
       failed leaves -- so they reach the floor in every mode, and a region
       boundary always runs along every caustic.
    2. For a threshold in ``mu_min``, the six samples are not all on one
       side.
    3. Given ``rtol``, a midpoint's ``log(1 + mu)`` deviates from the mean of
       its edge's ends by ``rtol`` or more: a relative tolerance on ``mu``.

    The leaves :func:`balance` forces are tested too, until none fails, so
    every leaf below ``max_level`` has passed the criterion. The closed mesh
    is conforming. The image of the lens fov's boundary is not refined along
    -- a boundary there is wrong anyway, since on one side an image lies
    outside the lens fov -- but the leaves it touches are marked
    ``incomplete``.

    ``src_tol`` is the source-plane size floor, the counterpart of the lens
    build's ``min_img_sep``. Refinement runs until the longest leaf edge is
    at most ``src_tol``: :func:`depth_floor` turns it into ``d_floor``, and
    ``max_level = min(max_depth, d_floor)``. Each traced boundary point lies
    on a floor-leaf edge whose ends lie on opposite sides of the threshold,
    so it is within one edge, at most ``src_tol``, of where the sampled
    field crosses it. In threshold mode that holds at every boundary point
    of a targeted threshold, and the area error is bounded by the perimeter
    times ``src_tol``, usually far less; in sweep mode only cells next to
    sheet edges are forced to the floor, and elsewhere ``rtol`` governs.
    ``src_tol`` is measured against the sampled field -- the lens mesh's
    piecewise-affine ``mu_tot``, whose accuracy
    :func:`~caustics.lenses.func.adaptive.magnification.mesh_total_magnification`
    states -- not against the true lens: below the scale at which the lens
    mesh's own leaf images change ``mu``, a smaller ``src_tol`` costs more
    and adds nothing. It is a geometric tolerance, separate from ``rtol``
    (how accurately ``mu`` is resolved) and ``mu_min`` (where to look).

    ``init_res`` carries the same completeness obligation as the lens
    build's: a high-magnification island with no sheet edge in it, smaller
    than the level-0 sample spacing, is invisible to the criterion.

    Parameters
    ----------
    mesh: AdaptiveMesh
    init_res: int
        Level-0 cells per axis of the window.
    src_tol: float
        The source-plane size floor, as above.

        *Unit: arcsec*
    max_depth: int
        Hard cap on refinement level; a warning is raised when it binds.
    mu_min: Optional[float or Sequence[float]]
        Thresholds the refinement targets (threshold mode). Tracing another
        threshold on the result is allowed, but only as accurate as the
        refinement happens to be there.
    rtol: Optional[float]
        Relative tolerance on ``mu`` everywhere (sweep mode). Either, both,
        or neither of ``mu_min`` and ``rtol`` may be given; with neither,
        only sheet edges refine.
    fov, x0, y0: Optional[float]
        The square window: side and centre. Default: centred on the
        bounding box of the converged leaves' source-plane images,
        ``mesh.index``, with its longer side. A region the window cuts comes
        back open. The default window's edges usually lie on the image of
        the lens fov's boundary, so the leaves along them are
        ``incomplete`` -- up to a level-0 cell deep where no criterion
        refines them -- and regions and points there are not known. A
        window strictly inside that image avoids it.

        *Unit: arcsec*
    batch_size: Optional[int]
        Chunks sampler calls. The result is bit-identical for every value.
    index_cells: Optional[int]
        Forwarded to :func:`build_index`, for the source leaves and the
        band cover.

    Returns
    -------
    MagnificationMesh

    Raises
    ------
    ValueError
        For a non-positive ``fov`` or ``src_tol``, an ``init_res`` below 1, a
        negative ``max_depth``, or a lattice too fine for int64 keys.

    Warns
    -----
    UserWarning
        When ``max_depth`` binds before ``src_tol`` is reached, and for each
        targeted threshold above ``mu_band``.
    """
    f64 = backend.float64
    targets = _targets(mu_min)
    (lx, ly), (hx, hy) = backend.to_numpy(
        backend.stack((mesh.index.lo, mesh.index.hi))
    ).tolist()
    fov = max(hx - lx, hy - ly) if fov is None else float(fov)
    x0 = 0.5 * (lx + hx) if x0 is None else float(x0)
    y0 = 0.5 * (ly + hy) if y0 is None else float(y0)
    validate_source_args(fov, init_res, src_tol, max_depth)
    d_floor = depth_floor(fov, init_res, src_tol)
    max_level = min(int(max_depth), d_floor)
    warn_source_depth_limited(fov, init_res, src_tol, d_floor, max_level)
    mu_band = band_magnification_floor(mesh.critical_band)
    for t in targets:
        if t > mu_band:
            warn(
                f"mu_min={t:g} exceeds mu_band={mu_band:.3g}, the smallest "
                "magnification the critical band resolves: the region of "
                f"mu_tot >= {t:g} is narrower than the band there and comes "
                "out as the band strip."
            )

    _, _, compose, _, root_class = child_matrix_tables()
    lat = make_lattice(fov, x0, y0, init_res, max_level + 1)
    sample = make_sampler(mesh, index_cells)
    edges = sheet_edges(mesh)
    segments = _ambient(backend.to(edges.source, dtype=f64))
    on_fov = _ambient(edges.fov)
    # A few ulps of the window's coordinates, so that rasterization stays
    # conservative under rounding at every level.
    pad = 64.0 * float(backend.finfo(f64).eps) * (fov + max(abs(x0), abs(y0)))
    inner = segments[backend.flatnonzero(~on_fov)]
    outer = segments[backend.flatnonzero(on_fov)]
    inner_cells = [level_cells(lat, inner, pad, d) for d in range(max_level + 1)]
    outer_cells = [level_cells(lat, outer, pad, d) for d in range(max_level + 1)]

    cache, active, store = _refine(
        lat,
        sample,
        inner_cells,
        targets,
        rtol,
        init_res,
        max_level,
        compose,
        root_class,
        batch_size,
    )
    vertices, vertices_ij, mu, n, leaves, leaf_level, incomplete, index = _freeze(
        lat, cache, active, store, outer_cells, index_cells
    )

    def to_device(array):
        return backend.to(array, device=mesh.device)

    return MagnificationMesh(
        vertices=to_device(vertices),
        vertices_ij=to_device(vertices_ij),
        mu=to_device(mu),
        n=to_device(n),
        leaves=to_device(leaves),
        leaf_level=to_device(leaf_level),
        incomplete=to_device(incomplete),
        index=MeshIndex(
            lo=to_device(index.lo),
            hi=to_device(index.hi),
            cell=to_device(index.cell),
            nx=index.nx,
            ny=index.ny,
            cell_offsets=to_device(index.cell_offsets),
            cell_leaves=to_device(index.cell_leaves),
        ),
        lattice=lat._replace(lo=to_device(lat.lo)),
        fov=float(fov),
        init_res=int(init_res),
        src_tol=float(src_tol),
        d_floor=d_floor,
        max_level=max_level,
        mu_min=targets,
        rtol=None if rtol is None else float(rtol),
        mu_band=mu_band,
        device=mesh.device,
    )
