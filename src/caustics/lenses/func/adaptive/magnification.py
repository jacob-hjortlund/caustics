"""
Total magnification at source-plane points, read off an adaptive mesh.

:func:`mesh_total_magnification` sums, over the converged leaves whose
source-plane image contains a point, each leaf's magnification -- the ratio
of its lens- and source-plane areas -- counting a point on a shared edge
once (:func:`counts_once`), and reads the images of the critical band as
infinitely magnified. Nothing here calls the lens.
"""

import math
from typing import Callable, NamedTuple, Tuple

from ....backend_obj import ArrayLike, backend
from .geometry import _CHILD_VERTEX_INDEX_TABLE, shape_matrix
from .mesh import MeshIndex, build_index
from .query import _as_beta, index_hits

__all__ = (
    "leaf_magnification",
    "counts_once",
    "BandCover",
    "band_cover",
    "band_magnification_floor",
    "total_magnification",
    "make_sampler",
    "mesh_total_magnification",
)


def leaf_magnification(mesh, leaves) -> ArrayLike:
    """
    Magnification of each leaf: its lens-plane area over its source-plane area.

    The density of lens-plane area per unit source-plane area under the
    leaf's affine map, ``1 / |det|`` of that map. Leaves are positively
    oriented in the lens plane, so the lens area is positive; the source
    area's sign is the leaf's parity and is dropped. Computed in float64.

    Parameters
    ----------
    mesh: AdaptiveMesh
    leaves: ArrayLike
        ``(K,)`` int64 indices into ``mesh.leaves``, converged ones.

    Returns
    -------
    ArrayLike
        ``(K,)`` float64.
    """
    f64 = backend.float64
    P = shape_matrix(backend.to(mesh.vertices_lens[mesh.leaves[leaves]], dtype=f64))
    area2_lens = P[:, 0, 0] * P[:, 1, 1] - P[:, 0, 1] * P[:, 1, 0]
    return area2_lens / backend.abs(backend.to(mesh.leaf_area2[leaves], dtype=f64))


def counts_once(tri, w, area2) -> ArrayLike:
    """
    True where a hit counts, so that a point on a shared edge counts once.

    :func:`~caustics.lenses.func.adaptive.geometry.contains` counts zeros as
    inside, so a point on an edge two leaves share hits both. A weight
    ``w[:, k]`` that is exactly zero puts the point on the edge opposite
    vertex ``k``; the hit counts only if the point shifted by
    ``delta * (1, eps)``, ``delta`` and ``eps`` infinitesimal, lies inside the
    triangle -- if the edge's inward normal has a positive x component, or a
    zero one and a positive y component. At a vertex both zero weights are
    tested.

    Exact: two leaves sharing an edge form its vector from the same stored
    vertices in opposite orders, and IEEE subtraction negates exactly, so
    their inward normals are exact negatives and exactly one passes -- the
    property :func:`~caustics.lenses.func.adaptive.geometry.triangle_weights`
    already relies on. Around a vertex inside one sheet exactly one leaf
    contains the direction ``(1, eps)``. On a fold edge, both leaves on one
    side, both count or neither does, as they should. A zero-length edge
    never passes; converged leaves have none.

    Parameters
    ----------
    tri: ArrayLike
        ``(K, 3, 2)`` source-plane vertices of the hit leaves.

        *Unit: arcsec*
    w: ArrayLike
        ``(K, 3)`` their raw weights, from :func:`index_hits`.
    area2: ArrayLike
        ``(K,)`` twice their signed source-plane areas, whose sign orients
        the normals.

    Returns
    -------
    ArrayLike
        ``(K,)`` bool.
    """
    positive = area2 > 0
    ok = backend.ones((tri.shape[0],), dtype=backend.bool, device=backend.device(tri))
    for k in range(3):
        j, l = (k + 1) % 3, (k + 2) % 3
        e = tri[:, l] - tri[:, j]
        # The left normal (-e_y, e_x) points inward on a counter-clockwise
        # triangle, the right one on a clockwise triangle.
        nx = backend.where(positive, -e[:, 1], e[:, 1])
        ny = backend.where(positive, e[:, 0], -e[:, 0])
        inward = (nx > 0) | ((nx == 0) & (ny > 0))
        ok = ok & ((w[:, k] != 0) | inward)
    return ok


class BandCover(NamedTuple):
    """
    The critical band's red-split children in the source plane, indexed.

    ``vertices`` is ``CriticalBand.source``; ``triangles`` holds every band
    leaf's four children, ``(4F, 3)`` int64 into it; ``index`` covers the
    children whose three images are finite, and no other, so a non-finite
    sample cannot reach the index's bounding box.
    """

    vertices: ArrayLike
    triangles: ArrayLike
    # Shadows `tuple.index`, as `AdaptiveMesh.index` does.
    index: MeshIndex  # type: ignore[assignment]


def band_cover(mesh, index_cells=None) -> BandCover:
    """
    Index the source-plane images of the critical band's children.

    Parameters
    ----------
    mesh: AdaptiveMesh
    index_cells: Optional[int]
        Forwarded to :func:`build_index`.

    Returns
    -------
    BandCover
    """
    band = mesh.critical_band
    triangles = band.samples[:, _CHILD_VERTEX_INDEX_TABLE].reshape(-1, 3)
    finite = backend.all(backend.isfinite(band.source[triangles]), dim=(1, 2))
    return BandCover(
        vertices=band.source,
        triangles=triangles,
        index=build_index(
            band.source, triangles, backend.flatnonzero(finite), index_cells
        ),
    )


def band_magnification_floor(band) -> float:
    """
    The smallest magnification the critical band resolves, ``mu_band``.

    For each band leaf, ``1 / max |det A|`` over its six samples bounds the
    magnification of each image inside it from below; this is the minimum of
    that bound over the band. A threshold above it describes a region
    narrower than the band, which the band's ``+inf`` reading cannot
    resolve. ``math.inf`` for a band with no leaf.

    Parameters
    ----------
    band: CriticalBand

    Returns
    -------
    float
    """
    if band.leaves.shape[0] == 0:
        return math.inf
    largest = backend.max(backend.abs(band.det[band.samples]), dim=1)
    return float(backend.to_numpy(backend.min(1.0 / largest)))


def _per_point_sum(qidx, values, n_points) -> Tuple[ArrayLike, ArrayLike]:
    """
    Sum and count of ``values`` per point, ``qidx`` non-decreasing.

    No scatter-add: torch keeps the last write on a repeated index and jax
    accumulates. Each point's values are laid out in a dense row by their
    rank within the point -- distinct indices, so a plain scatter -- and the
    columns are added in rank order, so a point's sum does not depend on how
    many other points share its chunk.
    """
    device = backend.device(qidx)
    counts = backend.long(backend.bincount(qidx, minlength=n_points))
    total = backend.zeros((n_points,), dtype=backend.float64, device=device)
    if qidx.shape[0] == 0:
        return total, counts
    width = int(backend.to_numpy(backend.max(counts)))
    starts = backend.cumsum(counts, dim=0) - counts
    rank = (
        backend.arange(qidx.shape[0], dtype=backend.int64, device=device) - starts[qidx]
    )
    dense = backend.fill_at_indices(
        backend.zeros((n_points * width,), dtype=backend.float64, device=device),
        qidx * width + rank,
        values,
    ).reshape(n_points, width)
    for c in range(width):
        total = total + dense[:, c]
    return total, counts


def total_magnification(mesh, cover, beta) -> Tuple[ArrayLike, ArrayLike]:
    """
    :func:`mesh_total_magnification` on one chunk.

    Parameters
    ----------
    mesh: AdaptiveMesh
    cover: BandCover
        From :func:`band_cover` of ``mesh``.
    beta: ArrayLike
        ``(B, 2)`` at the mesh's dtype and device.

        *Unit: arcsec*

    Returns
    -------
    mu: ArrayLike
        ``(B,)`` float64.
    n: ArrayLike
        ``(B,)`` int64.
    """
    b = beta.shape[0]
    qidx, cand, w = index_hits(mesh.index, mesh.vertices_source, mesh.leaves, beta)
    once = backend.flatnonzero(
        counts_once(mesh.vertices_source[mesh.leaves[cand]], w, mesh.leaf_area2[cand])
    )
    qidx, cand = qidx[once], cand[once]
    mu, n = _per_point_sum(qidx, leaf_magnification(mesh, cand), b)
    band, _, _ = index_hits(cover.index, cover.vertices, cover.triangles, beta)
    in_band = backend.bincount(band, minlength=b) > 0
    return backend.where(in_band, backend.inf, mu), n


def make_sampler(mesh, index_cells=None) -> Callable[[ArrayLike], ArrayLike]:
    """
    The ``(N, 2) -> (N, 2)`` map from source points to ``(mu, n)``.

    The stand-in for ``raytrace`` that
    :func:`~caustics.lenses.func.adaptive.source_mesh.build_magnification_mesh`
    hands to :func:`~caustics.lenses.func.adaptive.sampling.evaluate`. The
    band cover is built once. Points are cast to the mesh's dtype and
    device, and the result comes back as float64 on the caller's device, as
    :func:`~caustics.lenses.func.adaptive.sampling.make_raytrace`'s does.

    Parameters
    ----------
    mesh: AdaptiveMesh
    index_cells: Optional[int]
        Forwarded to :func:`band_cover`.

    Returns
    -------
    Callable[[ArrayLike], ArrayLike]
    """
    cover = band_cover(mesh, index_cells)

    def sample(xy):
        mu, n = total_magnification(mesh, cover, _as_beta(mesh, xy))
        out = backend.stack((mu, backend.to(n, dtype=backend.float64)), dim=-1)
        return backend.to(out, dtype=backend.float64, device=backend.device(xy))

    return sample


def mesh_total_magnification(
    mesh, beta, batch_size=None
) -> Tuple[ArrayLike, ArrayLike]:
    """
    Total point-source magnification at source-plane points, and the image count.

    ``mu`` sums ``mu_i``, the ratio of a leaf's lens- and source-plane areas
    (:func:`leaf_magnification`), over the converged leaves whose
    source-plane image contains the point, a point on a shared edge or
    vertex counted once (:func:`counts_once`); ``n`` counts them. It is the
    point-source total magnification of the mesh's piecewise-affine lens
    map. A point outside every leaf image has ``mu = 0`` and ``n = 0``.

    A point inside the source-plane image of a critical-band leaf's red-split
    children reads ``mu = +inf``, above every threshold: near a fold
    ``det A ~ g d``, so every image there is magnified by at least about
    ``1 / (g h)``, ``h`` the ``max_level`` leaf edge, and the band's images
    form a strip along the inside of each caustic about ``g h**2 / 2`` wide.
    :func:`band_magnification_floor` gives the bound. ``n`` does not count
    band hits.

    Missing contributions: leaves that failed the criterion outside the band
    -- typically next to a singular centre -- contribute nothing; their lens
    area is at the ``min_img_sep`` scale. Holes remove no leaf.

    Accuracy: the lens build bounds each leaf's position error, of order
    ``h**2``, not the error of its slope, of order ``h``, so ``mu_i`` errs
    like ``sqrt(min_img_sep)``. Measured on an analytic fold, the median
    relative error is 8%, 4% and 2% at ``min_img_sep`` 0.02, 0.005 and
    0.001 (95th percentile 18%, 10% and 4%); on an SIS and a cored
    isothermal lens, 1-5% median at 0.01-0.002. Averaging over source cells
    does not remove it: it is a bias at the leaf scale, not noise.

    Parameters
    ----------
    mesh: AdaptiveMesh
    beta: ArrayLike
        ``(B, 2)`` strictly, as for :func:`mesh_query`.

        *Unit: arcsec*
    batch_size: Optional[int]
        Chunk size over points. Results are bit-identical for every value.

    Returns
    -------
    mu: ArrayLike
        ``(B,)`` float64.
    n: ArrayLike
        ``(B,)`` int64.
    """
    beta = _as_beta(mesh, beta)
    cover = band_cover(mesh)
    n_points = beta.shape[0]
    step = max(n_points, 1) if batch_size is None else max(1, int(batch_size))
    mus, ns = [], []
    for lo in range(0, n_points, step):
        mu, n = total_magnification(mesh, cover, beta[lo : lo + step])
        mus.append(mu)
        ns.append(n)
    if not mus:
        return (
            backend.zeros((0,), dtype=backend.float64, device=mesh.device),
            backend.zeros((0,), dtype=backend.int64, device=mesh.device),
        )
    return backend.concatenate(mus, dim=0), backend.concatenate(ns, dim=0)
