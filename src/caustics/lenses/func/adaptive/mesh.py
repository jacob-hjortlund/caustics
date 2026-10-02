"""
The frozen :class:`AdaptiveMesh`, its spatial index, and :func:`freeze`.

:func:`freeze` is the last stage of every build: canonical ordering and
closure, the critical band mapped onto the closed leaves, vertex compaction,
freeze-time invalidation, and the source-plane index (:func:`build_index`).
Its one lens call, made only when needed, evaluates the Jacobian at the
vertices no criterion reached.
"""

import math
from typing import Any, NamedTuple

from ....backend_obj import ArrayLike, backend
from .geometry import jacobian_det, shape_matrix
from .state import cache_size, store_compact
from .sampling import call_jacobian
from .lattice import Lattice, lattice_key, lattice_xy
from .criterion import LEAF_CONVERGED, LEAF_RAYTRACE_NONFINITE
from .band import CriticalBand, band_leaf_index, merge_bands
from .holes import CentreHoles, empty_holes
from .closure import canonical_order, close

__all__ = (
    "invalidate_nonfinite_origins",
    "MeshIndex",
    "build_index",
    "AdaptiveMesh",
    "freeze",
)


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
    :func:`~caustics.lenses.func.adaptive.curves.mesh_critical_curves_and_caustics`
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
    vertices_det: ArrayLike
        ``det A`` of the lens Jacobian at every vertex, shape ``(V,)``,
        float64 whatever ``dtype`` is. It is the value the build evaluated at
        the vertex (:func:`~caustics.lenses.func.adaptive.sampling.sample_jacobians`)
        or, at a vertex no criterion reached, the one :func:`freeze`
        evaluated, formed as
        :func:`~caustics.lenses.func.adaptive.geometry.jacobian_det` forms
        it, so a critical-band sample at a vertex carries this same value.
        :func:`~caustics.lenses.func.adaptive.magnification.mesh_total_magnification`
        interpolates it. Non-finite wherever the lens Jacobian is.
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
        :func:`~caustics.lenses.func.adaptive.curves.mesh_critical_curves_and_caustics`
        needs no lens. A superset of the ``LEAF_JACOBIAN_PARITY_UNRESOLVED``
        leaves; see :class:`CriticalBand`.
    holes: CentreHoles
        Holes of radius ``min_img_sep`` around the centres passed to the
        build, with the images of their boundary circles, so that
        :func:`~caustics.lenses.func.adaptive.curves.mesh_critical_curves_and_caustics`
        can repair curves through lens centres without a lens call. Empty
        when no centres were given. See :class:`CentreHoles`.
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
    vertices_det: ArrayLike
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
    holes: CentreHoles
    lattice: Lattice
    fov: float
    init_res: int
    d_floor: int
    max_level: int
    min_img_sep: float
    dtype: Any
    device: Any


def freeze(
    lat,
    cache,
    active,
    store,
    band,
    seed_band,
    *,
    jacobian_fn,
    fov,
    init_res,
    min_img_sep,
    d_floor,
    max_level,
    dtype,
    device,
    index_cells,
    holes=None,
) -> AdaptiveMesh:
    """
    Close, order, index and freeze a balanced refinement.

    The last stage of every build: canonical ordering and closure, the
    critical band merged and mapped onto the closed leaves, vertex
    compaction, freeze-time invalidation, and the spatial index. Its only
    lens call is the Jacobian fill below. :func:`build_adaptive_mesh` and :func:`extend_adaptive_mesh`
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
    jacobian_fn: Callable[[ArrayLike, ArrayLike], ArrayLike]
        The Jacobian ``refine`` evaluated. Called once, on the vertices whose
        ``det A`` the cache lacks -- the midpoints of balance-forced leaves no
        criterion sampled, and in an extension those of old leaves the balance
        split -- and not at all when there are none.
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
    holes: Optional[CentreHoles]
        Stored on the mesh with ``centres``, ``lens`` and ``source`` at
        ``dtype``; :func:`empty_holes` when ``None``.

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

    # One Jacobian call, at most, for the vertices without a `det A`. Every
    # other vertex already holds one: evaluated once in this build, or
    # carried over from a seeding mesh.
    missing = used[backend.flatnonzero(~cache.has_det[used])]
    det = cache.det
    if missing.shape[0]:
        det = backend.fill_at_indices(
            backend.copy(det),
            missing,
            jacobian_det(call_jacobian(lat, cache.ij[missing], jacobian_fn)),
        )

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

    if holes is None:
        holes = empty_holes()
    centre_holes = CentreHoles(
        centres=to_device(backend.to(holes.centres, dtype=dtype)),
        radius=to_device(holes.radius),
        offsets=to_device(holes.offsets),
        angle=to_device(holes.angle),
        lens=to_device(backend.to(holes.lens, dtype=dtype)),
        source=to_device(backend.to(holes.source, dtype=dtype)),
        growth=to_device(holes.growth),
        growth_err=to_device(holes.growth_err),
        pseudo_caustic=to_device(holes.pseudo_caustic),
    )

    return AdaptiveMesh(
        vertices_lens=to_device(vl),
        vertices_source=to_device(vs),
        vertices_ij=to_device(cache.ij[used]),
        vertices_det=to_device(det[used]),
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
        holes=centre_holes,
        lattice=lat._replace(lo=to_device(lat.lo)),
        fov=float(fov),
        init_res=int(init_res),
        d_floor=d_floor,
        max_level=max_level,
        min_img_sep=float(min_img_sep),
        dtype=dtype,
        device=device,
    )
