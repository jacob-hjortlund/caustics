"""
The magnification map: an adaptive mesh of the source plane, sampled with the total magnification.

:func:`build_magnification_map` refines a square source-plane window on the
shared loop (:func:`~.refine.refine`), sampling
:func:`~.magnification.total_magnification` and the image count at every
lattice point. A leaf splits where an image-count edge of the lens mesh
touches its cell, where a targeted threshold runs between its samples, or
where ``mu`` deviates from linear by ``rtol`` (:func:`split_mask`). No lens
call is made.
"""

from typing import NamedTuple
from warnings import warn

from ....backend_obj import ArrayLike, backend
from .mesh_backend import mesh_backend, to_mesh, to_user
from .geometry import ROOT_CLASS, build_device, is_member, to_device
from .lattice import (
    Lattice,
    check_lattice_keys,
    depth_floor,
    initial_triangles,
    lattice_xy,
    make_lattice,
    warn_depth_limited,
)
from .refine import add_roots, close, empty_cache, empty_store, refine
from .index import MeshIndex, build_index
from .magnification import band_magnification_floor, magnification_sampler, sheet_edges


def _expand(start, count):
    """Each ``start[e] + 0 .. count[e] - 1``, with its owner ``e``."""
    int64 = mesh_backend.int64
    device = mesh_backend.device(count)
    total = int(mesh_backend.to_numpy(mesh_backend.sum(count)))
    owner = mesh_backend.repeat(
        mesh_backend.arange(count.shape[0], dtype=int64, device=device), count, axis=0
    )
    within = mesh_backend.arange(
        total, dtype=int64, device=device
    ) - mesh_backend.repeat(mesh_backend.cumsum(count, dim=0) - count, count, axis=0)
    return owner, start[owner] + within


def segment_cells(a, b, lo, size, n_cells, pad):
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
    f64 = mesh_backend.float64
    hi = lo + size * n_cells
    xa, xb = mesh_backend.minimum(a[:, 0], b[:, 0]), mesh_backend.maximum(
        a[:, 0], b[:, 0]
    )
    ya, yb = mesh_backend.minimum(a[:, 1], b[:, 1]), mesh_backend.maximum(
        a[:, 1], b[:, 1]
    )
    meets = (
        (xb + pad >= lo[0])
        & (xa - pad <= hi[0])
        & (yb + pad >= lo[1])
        & (ya - pad <= hi[1])
    )
    keep = mesh_backend.flatnonzero(meets)
    if keep.shape[0] == 0:
        return mesh_backend.zeros(
            (0,), dtype=mesh_backend.int64, device=mesh_backend.device(a)
        )
    a, b, xa, xb = a[keep], b[keep], xa[keep], xb[keep]
    top = n_cells - 1
    i0 = mesh_backend.clamp(
        mesh_backend.long(mesh_backend.floor((xa - pad - lo[0]) / size)), 0, top
    )
    i1 = mesh_backend.clamp(
        mesh_backend.long(mesh_backend.floor((xb + pad - lo[0]) / size)), 0, top
    )
    seg, col = _expand(i0, i1 - i0 + 1)

    # Clip each segment to its column's padded slab.
    x_lo = lo[0] + mesh_backend.to(col, dtype=f64) * size - pad
    x_hi = x_lo + size + 2.0 * pad
    ax, ay = a[seg, 0], a[seg, 1]
    dx, dy = b[seg, 0] - ax, b[seg, 1] - ay
    vertical = dx == 0
    safe = mesh_backend.where(vertical, mesh_backend.ones_like(dx), dx)
    ta, tb = (x_lo - ax) / safe, (x_hi - ax) / safe
    t0 = mesh_backend.where(
        vertical,
        mesh_backend.zeros_like(dx),
        mesh_backend.clamp(mesh_backend.minimum(ta, tb), 0.0, 1.0),
    )
    t1 = mesh_backend.where(
        vertical,
        mesh_backend.ones_like(dx),
        mesh_backend.clamp(mesh_backend.maximum(ta, tb), 0.0, 1.0),
    )
    y0, y1 = ay + t0 * dy, ay + t1 * dy
    y_lo = mesh_backend.minimum(y0, y1) - pad
    y_hi = mesh_backend.maximum(y0, y1) + pad
    rows = mesh_backend.flatnonzero((y_hi >= lo[1]) & (y_lo <= hi[1]))
    j0 = mesh_backend.clamp(
        mesh_backend.long(mesh_backend.floor((y_lo[rows] - lo[1]) / size)), 0, top
    )
    j1 = mesh_backend.clamp(
        mesh_backend.long(mesh_backend.floor((y_hi[rows] - lo[1]) / size)), 0, top
    )
    owner, row = _expand(j0, j1 - j0 + 1)
    return mesh_backend.unique(col[rows][owner] * n_cells + row)


def level_cells(lat, segments, pad, level):
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


def triangle_cells(lat, ij, level):
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
    corner = mesh_backend.min(ij, dim=1) // (1 << shift)
    return corner[:, 0] * (lat.n >> shift) + corner[:, 1]


def split_mask(mu6, touched, targets, rtol):
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
        split = split | (
            mesh_backend.any(inside, dim=1) & mesh_backend.any(~inside, dim=1)
        )
    if rtol is not None:
        log_mu = mesh_backend.log(1.0 + mu6)
        lv, lm = log_mu[:, :3], log_mu[:, 3:]
        dev = mesh_backend.abs(lm - 0.5 * (lv[:, [1, 2, 0]] + lv[:, [2, 0, 1]]))
        split = split | ~mesh_backend.all(dev < rtol, dim=1)
    return split


def _targets(mu_min):
    """``mu_min`` as a tuple of floats: ``()`` for ``None``."""
    if mu_min is None:
        return ()
    values = mesh_backend.to_numpy(
        mesh_backend.as_array(mu_min, dtype=mesh_backend.float64)
    )
    return tuple(float(v) for v in values.reshape(-1).tolist())


class MagnificationMap(NamedTuple):
    """
    An adaptive mesh of a source-plane window, with the total magnification at its vertices.

    Parameters
    ----------
    lattice: Lattice
    vertices_ij: ArrayLike
        ``(V, 2)`` int64 lattice coordinates, in lattice-key order.
    vertices: ArrayLike
        ``(V, 2)`` float64 source-plane positions.

        *Unit: arcsec*
    mu: ArrayLike
        ``(V,)`` float64 total magnification, ``inf`` in the critical band's images.
    n: ArrayLike
        ``(V,)`` int64 image count.
    leaves: ArrayLike
        ``(L, 3)`` int64 closed, conforming, positively oriented leaves.
    incomplete: ArrayLike
        ``(L,)`` bool, True where the image of the lens fov's boundary
        touches the leaf's pre-closure cell: an image there lies outside
        the lens fov, so ``mu`` is too low.
    index: MeshIndex
        Over every leaf.
    """

    lattice: Lattice
    vertices_ij: ArrayLike
    vertices: ArrayLike
    mu: ArrayLike
    n: ArrayLike
    leaves: ArrayLike
    incomplete: ArrayLike
    index: MeshIndex  # type: ignore[assignment]  # shadows tuple.index


def _touches(lat, cells, ij, level):
    """True where a triangle ``ij`` ``(n, 3, 2)`` lies in a cell of ``cells[level]``, its level's ascending cell ids."""
    hit = mesh_backend.zeros((ij.shape[0],), dtype=mesh_backend.bool)
    for d in mesh_backend.to_numpy(mesh_backend.unique(level)).tolist():
        at = mesh_backend.flatnonzero(level == d)
        cell_ids = triangle_cells(lat, ij[at], int(d))
        hit = mesh_backend.fill_at_indices(hit, at, is_member(cells[int(d)], cell_ids))
    return hit


def build_magnification_map(
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
):
    """
    Build an adaptive source-plane mesh sampled with the total magnification.

    A leaf below the finest level splits (:func:`split_mask`) when:

    1. an image-count edge of the lens mesh not on its fov boundary
       (:func:`~.magnification.sheet_edges`) touches the leaf's cell, so a
       region boundary always runs along every caustic;
    2. for a threshold in ``mu_min``, its six samples are not all on one side;
    3. given ``rtol``, a midpoint's ``log(1 + mu)`` deviates from the mean of
       its edge's ends by ``rtol`` or more.

    The finest level is where the longest leaf edge first falls to
    ``src_tol``, capped at ``max_depth``, so each boundary point of a
    targeted threshold lies within ``src_tol`` of where the sampled field
    crosses it. Leaves the image of the lens fov's boundary touches are
    marked ``incomplete``.

    Parameters
    ----------
    mesh: LensMesh
    init_res: int
        Level-0 cells per axis of the window.
    src_tol: float
        Source-plane size floor.

        *Unit: arcsec*
    max_depth: int
        Hard cap on the refinement level; a warning is raised when it binds.
    mu_min: Optional[float or Sequence[float]]
        Thresholds the refinement targets.
    rtol: Optional[float]
        Relative tolerance on ``mu`` everywhere.
    fov, x0, y0: Optional[float]
        The square window, side and center. By default, the bounding box of
        the converged leaves' source-plane images, at its longer side.

        *Unit: arcsec*
    batch_size: Optional[int]
        Most points per sampler call; the result does not depend on it.

    Returns
    -------
    MagnificationMap
    """
    mag = _build_magnification_map(
        to_mesh(mesh),
        to_mesh(init_res),
        to_mesh(src_tol),
        to_mesh(max_depth),
        mu_min=to_mesh(mu_min),
        rtol=to_mesh(rtol),
        fov=to_mesh(fov),
        x0=to_mesh(x0),
        y0=to_mesh(y0),
        batch_size=batch_size,
    )
    return to_user(mag, backend.device(mesh.vertices_lens))


def _build_magnification_map(
    mesh, init_res, src_tol, max_depth, *, mu_min, rtol, fov, x0, y0, batch_size
):
    """:func:`build_magnification_map` on ``mesh_backend`` arrays, on the build device."""
    f64 = mesh_backend.float64
    targets = _targets(mu_min)
    (lx, ly), (hx, hy) = mesh_backend.to_numpy(
        mesh_backend.stack((mesh.index.lo, mesh.index.hi))
    ).tolist()
    fov = max(hx - lx, hy - ly) if fov is None else float(fov)
    x0 = 0.5 * (lx + hx) if x0 is None else float(x0)
    y0 = 0.5 * (ly + hy) if y0 is None else float(y0)
    h0 = fov / init_res
    max_level = min(int(max_depth), depth_floor(h0, src_tol))
    check_lattice_keys(
        init_res, max_level, "Raise src_tol, lower max_depth, or lower init_res."
    )
    warn_depth_limited(
        "Magnification map", f"src_tol={src_tol:g} arcsec", h0, src_tol, max_level
    )
    mu_band = band_magnification_floor(mesh.critical_band)
    for t in targets:
        if t > mu_band:
            warn(
                f"mu_min={t:g} exceeds mu_band={mu_band:.3g}, the smallest "
                "magnification the critical band resolves: the region of "
                f"mu_tot >= {t:g} is narrower than the band there and comes "
                "out as the band strip."
            )

    lat = make_lattice(fov, x0, y0, init_res, max_level + 1)
    segments, on_fov = sheet_edges(mesh)
    segments = to_device(mesh_backend.to(segments, dtype=f64), build_device())
    on_fov = to_device(on_fov, build_device())
    # A few ulps of the window's coordinates keep the rasterization
    # conservative under rounding at every level.
    pad = 64.0 * float(mesh_backend.finfo(f64).eps) * (fov + max(abs(x0), abs(y0)))
    inner = segments[mesh_backend.flatnonzero(~on_fov)]
    outer = segments[mesh_backend.flatnonzero(on_fov)]
    inner_cells = [level_cells(lat, inner, pad, d) for d in range(max_level + 1)]
    outer_cells = [level_cells(lat, outer, pad, d) for d in range(max_level + 1)]

    def split(ij, values6, cls, level):
        touched = _touches(lat, inner_cells, ij, level)
        return split_mask(values6[..., 0], touched, targets, rtol)

    sample = magnification_sampler(mesh)
    ij, cls = initial_triangles(init_res, lat.level, ROOT_CLASS)
    cache, store, rows = add_roots(
        empty_cache(2), empty_store(), lat, ij, cls, sample, batch_size
    )
    cache, store = refine(cache, store, rows, lat, sample, split, max_level, batch_size)
    used, leaves, leaf_origin, origin, _ = close(lat, cache, store)
    origin_ij = cache.ij[store.v[origin]]
    incomplete = _touches(lat, outer_cells, origin_ij, store.level[origin])
    vertices = lattice_xy(lat, cache.ij[used])
    mag = MagnificationMap(
        lattice=lat,
        vertices_ij=cache.ij[used],
        vertices=vertices,
        mu=cache.values[used][:, 0],
        n=mesh_backend.long(cache.values[used][:, 1]),
        leaves=leaves,
        incomplete=incomplete[leaf_origin],
        index=build_index(
            vertices,
            leaves,
            mesh_backend.arange(leaves.shape[0], dtype=mesh_backend.int64),
        ),
    )
    return mag
