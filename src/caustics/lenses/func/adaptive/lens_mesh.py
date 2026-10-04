"""
The lens mesh: an adaptive triangulation of the lens plane, carrying the lens map at its vertices.

:func:`build_lens_mesh` refines the lens plane until each leaf's affine
model of the lens map is accurate to ``min_img_sep`` and no critical curve
runs between its samples (:func:`~.criterion.lens_status`), down to a size
floor. :func:`extend_lens_mesh` grows a mesh to a larger fov, reusing every
lens evaluation, and :func:`build_closed_lens_mesh` grows it until the fov
cuts no critical curve.
"""

import math
from typing import NamedTuple
from warnings import warn

from ....backend_obj import ArrayLike, backend
from .mesh_backend import mesh_backend, to_mesh, to_user
from .geometry import ROOT_CLASS, build_device, is_member, to_device
from .lattice import (
    Lattice,
    check_lattice_keys,
    depth_floor,
    extend_lattice,
    initial_triangles,
    lattice_fov,
    lattice_h0,
    lattice_ij_from_key,
    lattice_init_res,
    lattice_key,
    lattice_xy,
    make_lattice,
    midpoint_ij,
    ring_triangles,
    warn_depth_limited,
)
from .refine import (
    LeafStore,
    VertexCache,
    add_roots,
    cache_lookup,
    close,
    empty_cache,
    empty_store,
    refine,
    sample_points,
)
from .index import MeshIndex, build_index
from .criterion import LEAF_CONVERGED, lens_status
from .band import CriticalBand, build_band, in_band
from .holes import CenterHoles, merge_centers, sample_holes
from .curves import _critical_curves_and_caustics


class LensMesh(NamedTuple):
    """
    An adaptive triangulation of the lens plane, queryable from the source plane.

    Vertex ``v`` has a lens-plane position ``vertices_lens[v]`` and its image
    ``vertices_source[v]``, so the two embeddings share one topology. Only
    leaves whose origin is ``LEAF_CONVERGED`` are in ``index``: a leaf
    failing the criterion at the finest level straddles a fold, a critical
    curve or a non-finite point the mesh cannot resolve, and no query
    returns it. The pre-closure leaves -- the origins -- are kept with their
    level, class and status, which is what an extension refines from.

    Parameters
    ----------
    lattice: Lattice
        The lattice every vertex lives on. ``lattice.level - 1`` is the
        finest level, :func:`~.lattice.lattice_fov` the fov and
        :func:`~.lattice.lattice_init_res` the level-0 cells per axis.
    vertices_ij: ArrayLike
        ``(V, 2)`` int64 lattice coordinates, in ascending lattice-key order.
    vertices_lens: ArrayLike
        ``(V, 2)`` lens-plane positions.

        *Unit: arcsec*
    vertices_source: ArrayLike
        ``(V, 2)`` their images.

        *Unit: arcsec*
    vertices_det: ArrayLike
        ``(V,)`` ``det A`` of the lens Jacobian at each vertex.
    leaves: ArrayLike
        ``(L, 3)`` int64 vertex indices of the closed, conforming leaves,
        positively oriented in the lens plane.
    leaf_origin: ArrayLike
        ``(L,)`` int64 origin of each leaf, non-decreasing.
    origin_leaves: ArrayLike
        ``(N, 3)`` int64 vertex indices of the pre-closure leaves.
    origin_level: ArrayLike
        ``(N,)`` int64 refinement level of each origin.
    origin_cls: ArrayLike
        ``(N,)`` int64 orientation class of each origin, see
        :func:`~.geometry.child_matrix_tables`.
    origin_status: ArrayLike
        ``(N,)`` int64 ``LEAF_*`` bitmask: ``LEAF_CONVERGED`` below the
        finest level, where every leaf passed the criterion, and the
        criterion's verdict at it.
    index: MeshIndex
        Over the source-plane images of the leaves whose origin converged.
    critical_band: CriticalBand
        The finest-level origins ``det A`` changes sign across.
    holes: CenterHoles
        Holes around the lens centers, with their hole curves.
    min_img_sep: float
        The tolerance the mesh was refined to, half the one requested.

        *Unit: arcsec*
    """

    lattice: Lattice
    vertices_ij: ArrayLike
    vertices_lens: ArrayLike
    vertices_source: ArrayLike
    vertices_det: ArrayLike
    leaves: ArrayLike
    leaf_origin: ArrayLike
    origin_leaves: ArrayLike
    origin_level: ArrayLike
    origin_cls: ArrayLike
    origin_status: ArrayLike
    index: MeshIndex  # type: ignore[assignment]  # shadows tuple.index
    critical_band: CriticalBand
    holes: CenterHoles
    min_img_sep: float


def make_sampler(raytrace, jacobian, device, batch_size=None):
    """
    The lens map as ``(N, 2) -> (N, 3)`` float64 ``(bx, by, det A)``, or ``(N, 2)`` images when ``jacobian`` is None.

    ``raytrace`` and ``jacobian`` receive the same float64 coordinates on
    ``device``, in the user's backend, padded to
    ``backend.padded_size(N, batch_size)`` rows by repeating the last. Float64
    matters: the criterion compares midpoint deviations of ``O(fov)``
    quantities, which cancel to exactly zero below about
    ``sqrt(8 * eps * fov)`` and would read as converged. ``det A`` is formed
    in the Jacobian's dtype, then cast; the values come back on the
    positions' device, one row per position.
    """
    f64 = backend.float64

    def sample(xy):
        n = xy.shape[0]
        size = backend.padded_size(n, batch_size)
        padded = mesh_backend.pad_rows(xy, size) if size > n else xy
        x = backend.as_array(to_user(padded[:, 0]), dtype=f64, device=device)
        y = backend.as_array(to_user(padded[:, 1]), dtype=f64, device=device)
        columns = list(raytrace(x, y))
        if jacobian is not None:
            J = jacobian(x, y)
            columns.append(J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0])
        out = backend.stack([backend.to(c, dtype=f64) for c in columns], dim=-1)
        out = mesh_backend.to(to_mesh(out), device=mesh_backend.device(xy))
        return out[:n] if size > n else out

    return sample


def _lens_split(h0, min_img_sep):
    """The lens criterion as :func:`~.refine.refine`'s ``split``."""

    def split(ij, values6, cls, level):
        status = lens_status(
            values6[..., :2], values6[..., 2], cls, level, h0, min_img_sep
        )
        return status != LEAF_CONVERGED

    return split


def _warn_depth_limited(h0, min_img_sep, max_level):
    request = f"min_img_sep={2 * min_img_sep:g} arcsec (refined internally to {min_img_sep:g})"
    warn_depth_limited("Lens mesh", request, h0, min_img_sep, max_level)


def build_lens_mesh(
    raytrace,
    jacobian,
    fov,
    init_res,
    min_img_sep,
    max_depth=25,
    *,
    x0=0.0,
    y0=0.0,
    centers=None,
    device=None,
    batch_size=None,
):
    """
    Build an adaptive mesh of the lens plane.

    Both triangles of every cell of an ``init_res x init_res`` grid over the
    square fov are refined until each leaf passes
    :func:`~.criterion.lens_status` or reaches the finest level, where the
    longest leaf edge first falls to ``min_img_sep / 2`` -- capped at
    ``max_depth``. The leaves are then 2:1 balanced, closed and indexed. The
    criterion samples six points per triangle, so ``init_res`` must resolve
    the smallest structure of the lens.

    Parameters
    ----------
    raytrace: Callable[[ArrayLike, ArrayLike], Tuple[ArrayLike, ArrayLike]]
        ``raytrace(x, y) -> (bx, by)`` on 1-D arrays, e.g. ``lens.raytrace``.
    jacobian: Callable[[ArrayLike, ArrayLike], ArrayLike]
        ``jacobian(x, y) -> (N, 2, 2)``, the Jacobian of ``raytrace``, e.g.
        ``lens.jacobian_lens_equation``. Under jax that one re-traces its
        autodiff at every call. A jitted Jacobian taking the lens's
        parameters as an input is much cheaper over many builds and follows
        the parameters as they change: after ``lens.to_dynamic()``,
        ``jac = jax.jit(lambda x, y, p: lens.jacobian_lens_equation(x, y, params=p))``
        passed as ``lambda x, y: jac(x, y, lens.get_values())``. It must not
        close over parameter values.
    fov: float
        Side of the square lens-plane domain.

        *Unit: arcsec*
    init_res: int
        Level-0 cells per axis.
    min_img_sep: float
        Requested lens-plane tolerance. The mesh refines to half of it, and
        stores that.

        *Unit: arcsec*
    max_depth: int
        Hard cap on the refinement level; a warning is raised when it binds.
    x0, y0: float
        Center of the domain.

        *Unit: arcsec*
    centers: Optional[ArrayLike]
        ``(S, 2)`` positions where the lens map may jump, such as the center
        of a singular isothermal profile or a point mass. Each gets a hole
        of radius ``min_img_sep / 2``, merged where holes overlap, whose
        boundary is traced so that critical curves can be re-joined around
        it.

        *Unit: arcsec*
    device: Optional
        Device the lens lives on: ``raytrace`` and ``jacobian`` receive
        coordinates there, and the mesh is stored there.
    batch_size: Optional[int]
        Most points per ``raytrace`` and ``jacobian`` call.

    Returns
    -------
    LensMesh
    """
    mesh = _build_lens_mesh(
        raytrace,
        jacobian,
        fov,
        init_res,
        min_img_sep,
        max_depth,
        x0=x0,
        y0=y0,
        centers=to_mesh(centers),
        device=device,
        batch_size=batch_size,
    )
    return to_user(mesh, device)


def _build_lens_mesh(
    raytrace,
    jacobian,
    fov,
    init_res,
    min_img_sep,
    max_depth,
    *,
    x0,
    y0,
    centers,
    device,
    batch_size,
):
    """:func:`build_lens_mesh` on ``mesh_backend`` arrays, on the build device."""
    min_img_sep = min_img_sep / 2
    h0 = fov / init_res
    max_level = min(int(max_depth), depth_floor(h0, min_img_sep))
    check_lattice_keys(
        init_res, max_level, "Raise min_img_sep, lower max_depth, or lower init_res."
    )
    _warn_depth_limited(h0, min_img_sep, max_level)
    lat = make_lattice(fov, x0, y0, init_res, max_level + 1)
    hole_centers, hole_radius = merge_centers(centers, min_img_sep)
    holes = sample_holes(
        make_sampler(raytrace, None, device, batch_size),
        hole_centers,
        hole_radius,
        min_img_sep,
        batch_size,
    )
    sample = make_sampler(raytrace, jacobian, device, batch_size)
    ij, cls = initial_triangles(init_res, lat.level, ROOT_CLASS)
    cache, store, rows = add_roots(
        empty_cache(3), empty_store(), lat, ij, cls, sample, batch_size
    )
    cache, store = refine(
        cache,
        store,
        rows,
        lat,
        sample,
        _lens_split(h0, min_img_sep),
        max_level,
        batch_size,
    )
    return _freeze(lat, cache, store, sample, min_img_sep, holes, batch_size)


def _freeze(lat, cache, store, sample, min_img_sep, holes, batch_size, seed=None):
    """
    Close and index a refinement, finishing its finest level.

    Each finest-level origin not carried over in ``seed`` gets its status
    from :func:`~.criterion.lens_status` on its six samples, its three
    midpoints sampled here and never cached, and the band is built from the
    same values. ``seed`` is ``(status, band_rows, mid_keys, mid_values)``
    from :func:`_seed`: the statuses of the store's first rows, the store
    rows of the old band, and the old band's midpoint samples, read here
    rather than sampled again.
    """
    int64, f64 = mesh_backend.int64, mesh_backend.float64
    used, leaves, leaf_origin, origin, origin_leaves = close(lat, cache, store)
    n_rows = store.v.shape[0]
    if seed is None:
        seed = (
            mesh_backend.zeros((0,), dtype=int64),
            mesh_backend.zeros((0,), dtype=int64),
            mesh_backend.zeros((0,), dtype=int64),
            mesh_backend.zeros((0, 3), dtype=f64),
        )
    seed_status, band_rows, mid_keys, mid_values = seed
    n_seed = seed_status.shape[0]
    status = mesh_backend.concatenate(
        (seed_status, mesh_backend.zeros((n_rows - n_seed,), dtype=int64)), dim=0
    )
    row = mesh_backend.arange(n_rows, dtype=int64)
    fresh = mesh_backend.flatnonzero(
        store.valid & (store.level == lat.level - 1) & (row >= n_seed)
    )

    ij = cache.ij[store.v[fresh]]
    mid = lattice_key(lat, midpoint_ij(ij))
    todo = mesh_backend.unique(mid.reshape(-1))
    todo = todo[~is_member(mid_keys, todo)]
    if todo.shape[0]:
        xy = lattice_xy(lat, lattice_ij_from_key(lat, todo))
        new_values = sample_points(xy, sample, batch_size)
    else:
        new_values = mesh_backend.zeros((0, 3), dtype=f64)
    table_keys = mesh_backend.concatenate((cache.keys, mid_keys, todo), dim=0)
    table_values = mesh_backend.concatenate(
        (cache.values[cache.slots], mid_values, new_values), dim=0
    )
    order = mesh_backend.argsort(table_keys)
    table_keys, table_values = table_keys[order], table_values[order]

    keys6 = mesh_backend.concatenate((lattice_key(lat, ij), mid), dim=1)
    values6 = table_values[mesh_backend.searchsorted(table_keys, keys6)]
    status = mesh_backend.fill_at_indices(
        status,
        fresh,
        lens_status(
            values6[..., :2],
            values6[..., 2],
            store.cls[fresh],
            store.level[fresh],
            lattice_h0(lat),
            min_img_sep,
        ),
    )
    band_rows = mesh_backend.concatenate(
        (band_rows, fresh[mesh_backend.flatnonzero(in_band(values6[..., 2]))]), dim=0
    )

    origin_of_row = mesh_backend.fill_at_indices(
        mesh_backend.zeros((n_rows,), dtype=int64) - 1,
        origin,
        mesh_backend.arange(origin.shape[0], dtype=int64),
    )
    band_leaves = origin_of_row[band_rows]
    order = mesh_backend.argsort(band_leaves)
    band_leaves, band_rows = band_leaves[order], band_rows[order]
    bij = cache.ij[store.v[band_rows]]
    band_keys6 = mesh_backend.concatenate(
        (lattice_key(lat, bij), lattice_key(lat, midpoint_ij(bij))), dim=1
    )
    band = build_band(lat, band_leaves, band_keys6, table_keys, table_values)

    origin_status = status[origin]
    vertices_source = cache.values[used][:, :2]
    converged = mesh_backend.flatnonzero(origin_status[leaf_origin] == LEAF_CONVERGED)
    mesh = LensMesh(
        lattice=lat,
        vertices_ij=cache.ij[used],
        vertices_lens=lattice_xy(lat, cache.ij[used]),
        vertices_source=vertices_source,
        vertices_det=cache.values[used][:, 2],
        leaves=leaves,
        leaf_origin=leaf_origin,
        origin_leaves=origin_leaves,
        origin_level=store.level[origin],
        origin_cls=store.cls[origin],
        origin_status=origin_status,
        index=build_index(vertices_source, leaves, converged),
        critical_band=band,
        holes=holes,
        min_img_sep=float(min_img_sep),
    )
    return mesh


def _seed(mesh, lat, k):
    """
    :func:`~.refine.refine`'s state, and :func:`_freeze`'s ``seed``, from a mesh on its lattice grown by ``k``.

    The mesh's vertices are in lattice-key order, which a uniform shift
    keeps, so they are the cache as they stand, every one a leaf vertex; its
    origins are the store, row for row.
    """
    mesh = to_device(mesh, build_device())
    int64 = mesh_backend.int64
    ij = mesh.vertices_ij + (int(k) << lat.level)
    n_vertices, n_origins = ij.shape[0], mesh.origin_leaves.shape[0]
    cache = VertexCache(
        keys=lattice_key(lat, ij),
        slots=mesh_backend.arange(n_vertices, dtype=int64),
        ij=ij,
        values=mesh_backend.concatenate(
            (mesh.vertices_source, mesh_backend.unsqueeze(mesh.vertices_det, -1)), dim=1
        ),
        active=mesh_backend.ones((n_vertices,), dtype=mesh_backend.bool),
    )
    store = LeafStore(
        v=mesh.origin_leaves,
        level=mesh.origin_level,
        cls=mesh.origin_cls,
        valid=mesh_backend.ones((n_origins,), dtype=mesh_backend.bool),
    )
    band = mesh.critical_band
    bij = ij[mesh.origin_leaves[band.leaves]]
    keys6 = mesh_backend.concatenate(
        (lattice_key(lat, bij), lattice_key(lat, midpoint_ij(bij))), dim=1
    )
    # Band samples are numbered in ascending key order, so these keys ascend.
    sample_keys = mesh_backend.fill_at_indices(
        mesh_backend.zeros((band.det.shape[0],), dtype=int64),
        band.samples.reshape(-1),
        keys6.reshape(-1),
    )
    mid = mesh_backend.flatnonzero(cache_lookup(cache, sample_keys) < 0)
    values = mesh_backend.concatenate(
        (band.source, mesh_backend.unsqueeze(band.det, -1)), dim=1
    )
    seed = (mesh.origin_status, band.leaves, sample_keys[mid], values[mid])
    return cache, store, seed


def extend_lens_mesh(mesh, raytrace, jacobian, fov, *, batch_size=None):
    """
    Grow a lens mesh to a larger fov about the same center, reusing its refinement.

    Whole level-0 cells are added on every side: the fewest that reach
    ``fov``, to within ``1e-9`` of a cell. The ring is refined as a fresh
    build would refine it, and the balance settles the seam, so the result is
    the mesh :func:`build_lens_mesh` gives on the grown lattice -- bit for bit
    when the lens returns the same value at a point whatever batch it arrives
    in, and the grown fov, center and cell size are dyadic. No old vertex or
    band sample is sampled again.

    Parameters
    ----------
    mesh: LensMesh
    raytrace, jacobian:
        The pair ``mesh`` was built from.
    fov: float
        Requested side length.

        *Unit: arcsec*
    batch_size: Optional[int]
        As for :func:`build_lens_mesh`.

    Returns
    -------
    LensMesh
        ``mesh`` itself when ``fov`` needs no new cell.
    """
    device = backend.device(mesh.vertices_lens)
    on_mesh = to_mesh(mesh)
    out = _extend_lens_mesh(
        on_mesh, raytrace, jacobian, fov, device=device, batch_size=batch_size
    )
    return mesh if out is on_mesh else to_user(out, device)


def _extend_lens_mesh(mesh, raytrace, jacobian, fov, *, device, batch_size):
    """:func:`extend_lens_mesh` on ``mesh_backend`` arrays; ``device`` is the lens's."""
    h0 = lattice_h0(mesh.lattice)
    k = max(0, math.ceil((float(fov) - lattice_fov(mesh.lattice)) / (2.0 * h0) - 1e-9))
    if k == 0:
        return mesh
    lat = extend_lattice(to_device(mesh.lattice, build_device()), k)
    init_res, max_level = lattice_init_res(lat), lat.level - 1
    check_lattice_keys(
        init_res,
        max_level,
        "Extend by less, or rebuild with a larger min_img_sep or a lower max_depth.",
    )
    _warn_depth_limited(h0, mesh.min_img_sep, max_level)
    sample = make_sampler(raytrace, jacobian, device, batch_size)
    cache, store, seed = _seed(mesh, lat, k)
    ij, cls = ring_triangles(init_res, k, lat.level, ROOT_CLASS)
    cache, store, rows = add_roots(cache, store, lat, ij, cls, sample, batch_size)
    split = _lens_split(h0, mesh.min_img_sep)
    cache, store = refine(cache, store, rows, lat, sample, split, max_level, batch_size)
    return _freeze(
        lat,
        cache,
        store,
        sample,
        mesh.min_img_sep,
        mesh.holes,
        batch_size,
        seed,
    )


def _hole_rings(fov, init_res, x0, y0, centers, min_img_sep):
    """
    Level-0 cells per side that put every center's hole strictly inside the fov.

    The holes are :func:`merge_centers`'s, at the halved ``min_img_sep`` the
    build gives it, so a merged hole reaches as far as its own radius. The
    smallest ``k`` with ``fov / 2 + k * h0`` beyond every hole's reach --
    its center's larger offset from ``(x0, y0)``, plus its radius.

    Returns
    -------
    k: int
        Zero when every hole already lies strictly inside.
    outside: int
        How many holes did not.
    """
    hole_centers, radius = merge_centers(centers, min_img_sep)
    if hole_centers.shape[0] == 0:
        return 0, 0
    center = mesh_backend.as_array([x0, y0], dtype=mesh_backend.float64)
    reach = mesh_backend.max(mesh_backend.abs(hole_centers - center), dim=1) + radius
    outside = int(mesh_backend.to_numpy(mesh_backend.sum(reach >= fov / 2)))
    if outside == 0:
        return 0, 0
    excess = float(mesh_backend.to_numpy(mesh_backend.max(reach))) - fov / 2
    return math.floor(excess / (fov / init_res)) + 1, outside


def _curves_cut_by_fov(mesh, curves):
    """
    How many open curves have an end on ``mesh``'s fov boundary.

    Exact, with no tolerance: every vertex and band sample is placed by
    :func:`~.lattice.lattice_xy`, so the boundary is the vertices' own
    extreme coordinates, and a crossing on a boundary child edge, both of
    whose samples share that coordinate, reproduces it bit for bit
    (:func:`~.curves.edge_zeros`). An end at a band gap inside the fov does
    not count.
    """
    open_curves = mesh_backend.flatnonzero(~curves.closed)
    n_open = open_curves.shape[0]
    if n_open == 0:
        return 0
    ends = mesh_backend.concatenate(
        (curves.offsets[open_curves], curves.offsets[open_curves + 1] - 1)
    )
    xy = curves.lens[ends]
    lo = mesh_backend.min(mesh.vertices_lens, dim=0)
    hi = mesh_backend.max(mesh.vertices_lens, dim=0)
    on_boundary = mesh_backend.any((xy == lo) | (xy == hi), dim=1)
    cut = on_boundary[:n_open] | on_boundary[n_open:]
    return int(mesh_backend.to_numpy(mesh_backend.sum(cut)))


def build_closed_lens_mesh(
    raytrace,
    jacobian,
    fov,
    init_res,
    min_img_sep,
    max_depth=25,
    *,
    growth=1.5,
    max_iters=10,
    x0=0.0,
    y0=0.0,
    centers=None,
    device=None,
    batch_size=None,
):
    """
    Build a lens mesh, then grow its fov until it cuts no critical curve.

    When a hole around one of ``centers`` does not lie strictly inside
    ``fov``, the fov is first widened by the fewest whole level-0 cells
    that take every hole inside, with a warning. After the build, while an
    open curve ends on the fov boundary and fewer than ``max_iters``
    extensions have run, :func:`extend_lens_mesh` grows the fov by
    ``growth``, rounded up to whole cells, and the curves are traced again.
    A curve ending inside the fov -- next to a non-finite leaf, at an
    unjoined hole, or where ``max_depth`` bound -- grows nothing.

    Parameters
    ----------
    raytrace, jacobian, fov, init_res, min_img_sep, max_depth, x0, y0, centers, device, batch_size:
        As for :func:`build_lens_mesh`.
    growth: float
        Factor each extension asks to multiply the fov by.
    max_iters: int
        Most extensions to run.

    Returns
    -------
    mesh: LensMesh
    curves: CriticalCurvesAndCaustics
        The last mesh's critical curves and caustics.
    """
    mesh, curves = _build_closed_lens_mesh(
        raytrace,
        jacobian,
        fov,
        init_res,
        min_img_sep,
        max_depth,
        growth=growth,
        max_iters=max_iters,
        x0=x0,
        y0=y0,
        centers=to_mesh(centers),
        device=device,
        batch_size=batch_size,
    )
    return to_user(mesh, device), to_user(curves, device)


def _build_closed_lens_mesh(
    raytrace,
    jacobian,
    fov,
    init_res,
    min_img_sep,
    max_depth,
    *,
    growth,
    max_iters,
    x0,
    y0,
    centers,
    device,
    batch_size,
):
    """:func:`build_closed_lens_mesh` on ``mesh_backend`` arrays, on the build device."""
    k, outside = _hole_rings(fov, init_res, x0, y0, centers, min_img_sep / 2)
    if k:
        widened = fov + 2 * k * (fov / init_res)
        warn(
            f"{outside} hole(s) around centers reach outside fov={fov:g}; building "
            f"at fov={widened:g}, init_res={init_res + 2 * k} so that every hole "
            "lies inside it.",
            stacklevel=3,
        )
        fov, init_res = widened, init_res + 2 * k
    mesh = _build_lens_mesh(
        raytrace,
        jacobian,
        fov,
        init_res,
        min_img_sep,
        max_depth,
        x0=x0,
        y0=y0,
        centers=centers,
        device=device,
        batch_size=batch_size,
    )
    curves = _critical_curves_and_caustics(mesh)
    cut = _curves_cut_by_fov(mesh, curves)
    for _ in range(max_iters):
        if cut == 0:
            break
        mesh = _extend_lens_mesh(
            mesh,
            raytrace,
            jacobian,
            growth * lattice_fov(mesh.lattice),
            device=device,
            batch_size=batch_size,
        )
        curves = _critical_curves_and_caustics(mesh)
        cut = _curves_cut_by_fov(mesh, curves)
    if cut:
        warn(
            f"The fov still cuts {cut} critical curve(s) after {max_iters} "
            f"extension(s), at fov={lattice_fov(mesh.lattice):g}; raise max_iters "
            "or growth to grow it further.",
            stacklevel=3,
        )
    return mesh, curves
