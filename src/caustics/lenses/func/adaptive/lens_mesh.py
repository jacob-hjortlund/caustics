"""
The lens mesh: an adaptive triangulation of the lens plane, carrying the lens map at its vertices.

:func:`build_lens_mesh` refines the lens plane until each leaf's affine
model of the lens map is accurate to ``min_img_sep`` and no critical curve
runs between its samples (:func:`~.criterion.lens_status`), down to a size
floor. :func:`extend_lens_mesh` grows a mesh to a larger fov, reusing every
lens evaluation, and :func:`build_closed_lens_mesh` grows it until the fov
cuts no critical curve.
"""

from typing import NamedTuple

from ....backend_obj import ArrayLike, backend
from .geometry import ROOT_CLASS, is_member, to_device
from .lattice import (
    Lattice,
    check_lattice_keys,
    depth_floor,
    initial_triangles,
    lattice_h0,
    lattice_ij_from_key,
    lattice_key,
    lattice_xy,
    make_lattice,
    midpoint_ij,
    warn_depth_limited,
)
from .refine import add_roots, close, empty_cache, empty_store, refine, sample_points
from .index import MeshIndex, build_index
from .criterion import LEAF_CONVERGED, lens_status
from .band import CriticalBand, build_band, in_band
from .holes import CentreHoles, merge_centres, sample_holes


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
    holes: CentreHoles
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
    holes: CentreHoles
    min_img_sep: float


def make_sampler(raytrace, jacobian, device):
    """
    The lens map as ``(N, 2) -> (N, 3)`` float64 ``(bx, by, det A)``, or ``(N, 2)`` images when ``jacobian`` is None.

    ``raytrace`` and ``jacobian`` receive the same float64 coordinates on
    ``device``. Float64 matters: the criterion compares midpoint deviations
    of ``O(fov)`` quantities, which cancel to exactly zero below about
    ``sqrt(8 * eps * fov)`` and would read as converged. ``det A`` is formed
    in the Jacobian's dtype, then cast; the values come back on the
    positions' device.
    """
    f64 = backend.float64

    def sample(xy):
        x = backend.as_array(xy[:, 0], dtype=f64, device=device)
        y = backend.as_array(xy[:, 1], dtype=f64, device=device)
        columns = list(raytrace(x, y))
        if jacobian is not None:
            J = jacobian(x, y)
            columns.append(J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0])
        out = backend.stack([backend.to(c, dtype=f64) for c in columns], dim=-1)
        return backend.to(out, device=backend.device(xy))

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
        ``lens.jacobian_lens_equation``.
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
    min_img_sep = min_img_sep / 2
    h0 = fov / init_res
    max_level = min(int(max_depth), depth_floor(h0, min_img_sep))
    check_lattice_keys(
        init_res, max_level, "Raise min_img_sep, lower max_depth, or lower init_res."
    )
    _warn_depth_limited(h0, min_img_sep, max_level)
    lat = make_lattice(fov, x0, y0, init_res, max_level + 1)
    hole_centers, hole_radius = merge_centres(centers, min_img_sep)
    holes = sample_holes(
        make_sampler(raytrace, None, device),
        hole_centers,
        hole_radius,
        min_img_sep,
        batch_size,
    )
    sample = make_sampler(raytrace, jacobian, device)
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
    return _freeze(lat, cache, store, sample, min_img_sep, holes, device, batch_size)


def _freeze(
    lat, cache, store, sample, min_img_sep, holes, device, batch_size, seed=None
):
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
    int64, f64 = backend.int64, backend.float64
    used, leaves, leaf_origin, origin, origin_leaves = close(lat, cache, store)
    n_rows = store.v.shape[0]
    if seed is None:
        seed = (
            backend.zeros((0,), dtype=int64),
            backend.zeros((0,), dtype=int64),
            backend.zeros((0,), dtype=int64),
            backend.zeros((0, 3), dtype=f64),
        )
    seed_status, band_rows, mid_keys, mid_values = seed
    n_seed = seed_status.shape[0]
    status = backend.concatenate(
        (seed_status, backend.zeros((n_rows - n_seed,), dtype=int64)), dim=0
    )
    row = backend.arange(n_rows, dtype=int64)
    fresh = backend.flatnonzero(
        store.valid & (store.level == lat.level - 1) & (row >= n_seed)
    )

    ij = cache.ij[store.v[fresh]]
    mid = lattice_key(lat, midpoint_ij(ij))
    todo = backend.unique(mid.reshape(-1))
    todo = todo[~is_member(mid_keys, todo)]
    if todo.shape[0]:
        xy = lattice_xy(lat, lattice_ij_from_key(lat, todo))
        new_values = sample_points(xy, sample, batch_size)
    else:
        new_values = backend.zeros((0, 3), dtype=f64)
    table_keys = backend.concatenate((cache.keys, mid_keys, todo), dim=0)
    table_values = backend.concatenate(
        (cache.values[cache.slots], mid_values, new_values), dim=0
    )
    order = backend.argsort(table_keys)
    table_keys, table_values = table_keys[order], table_values[order]

    keys6 = backend.concatenate((lattice_key(lat, ij), mid), dim=1)
    values6 = table_values[backend.searchsorted(table_keys, keys6)]
    status = backend.fill_at_indices(
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
    band_rows = backend.concatenate(
        (band_rows, fresh[backend.flatnonzero(in_band(values6[..., 2]))]), dim=0
    )

    origin_of_row = backend.fill_at_indices(
        backend.zeros((n_rows,), dtype=int64) - 1,
        origin,
        backend.arange(origin.shape[0], dtype=int64),
    )
    band_leaves = origin_of_row[band_rows]
    order = backend.argsort(band_leaves)
    band_leaves, band_rows = band_leaves[order], band_rows[order]
    bij = cache.ij[store.v[band_rows]]
    band_keys6 = backend.concatenate(
        (lattice_key(lat, bij), lattice_key(lat, midpoint_ij(bij))), dim=1
    )
    band = build_band(lat, band_leaves, band_keys6, table_keys, table_values)

    origin_status = status[origin]
    vertices_source = cache.values[used][:, :2]
    converged = backend.flatnonzero(origin_status[leaf_origin] == LEAF_CONVERGED)
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
    return to_device(mesh, device)
