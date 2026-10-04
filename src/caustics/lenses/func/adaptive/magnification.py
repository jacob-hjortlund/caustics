"""
Total magnification at source-plane points, read off a lens mesh.

Each image is one converged leaf whose source-plane image contains the
point, counted once on a shared edge (:func:`counts_once`), magnified by
``1 / |det A|`` with ``det A`` interpolated from the leaf's vertices
(:func:`hit_magnification`). A point in the image of the critical band reads
as infinitely magnified. Nothing here calls the lens.
"""

import math

from .mesh_backend import mesh_backend
from .geometry import _CHILD_VERTEX_INDEX_TABLE, area2, csr_offsets, sanitize_bary
from .criterion import LEAF_CONVERGED
from .index import as_points, build_index, index_hits


def hit_magnification(mesh, leaves, bary):
    """
    ``1 / |det A|`` of the image in each hit leaf, ``det A`` interpolated at ``bary``.

    An affine map preserves barycentric coordinates, so this is ``det A``
    where the leaf's affine map places the image. A converged leaf has
    ``det A`` of one strict sign at its vertices, so the interpolant is
    nonzero. It is summed term by term in a fixed order, so a hit reads one
    value whichever chunk of a query it arrives in.

    Parameters
    ----------
    mesh: LensMesh
    leaves: ArrayLike
        ``(K,)`` int64 converged leaves.
    bary: ArrayLike
        ``(K, 3)`` float64 barycentric coordinates in the simplex.

    Returns
    -------
    ArrayLike
        ``(K,)`` float64.
    """
    d3 = mesh.vertices_det[mesh.leaves[leaves]]
    det = bary[:, 0] * d3[:, 0] + bary[:, 1] * d3[:, 1] + bary[:, 2] * d3[:, 2]
    return 1.0 / mesh_backend.abs(det)


def counts_once(tri, w, area2):
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

    Two leaves sharing an edge form it from the same stored vertices in
    opposite orders, so their inward normals are exact negatives and exactly
    one passes. On a fold edge, with both leaves on one side, both count or
    neither does.

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
    ok = mesh_backend.ones(
        (tri.shape[0],), dtype=mesh_backend.bool, device=mesh_backend.device(tri)
    )
    for k in range(3):
        j, l = (k + 1) % 3, (k + 2) % 3
        e = tri[:, l] - tri[:, j]
        # The left normal (-e_y, e_x) points inward on a counter-clockwise
        # triangle, the right one on a clockwise triangle.
        nx = mesh_backend.where(positive, -e[:, 1], e[:, 1])
        ny = mesh_backend.where(positive, e[:, 0], -e[:, 0])
        inward = (nx > 0) | ((nx == 0) & (ny > 0))
        ok = ok & ((w[:, k] != 0) | inward)
    return ok


def band_cover(mesh):
    """
    The critical band's red-split children, and an index over their source-plane images.

    Returns
    -------
    triangles: ArrayLike
        ``(4F, 3)`` int64 indices into the band's samples.
    index: MeshIndex
        Over the children whose three images are finite.
    """
    band = mesh.critical_band
    triangles = band.samples[:, _CHILD_VERTEX_INDEX_TABLE].reshape(-1, 3)
    finite = mesh_backend.all(mesh_backend.isfinite(band.source[triangles]), dim=(1, 2))
    return triangles, build_index(
        band.source, triangles, mesh_backend.flatnonzero(finite)
    )


def band_magnification_floor(band):
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
    largest = mesh_backend.max(mesh_backend.abs(band.det[band.samples]), dim=1)
    return float(mesh_backend.to_numpy(mesh_backend.min(1.0 / largest)))


def _per_point_sum(qidx, values, n_points):
    """
    Sum and count of ``values`` per point, ``qidx`` non-decreasing.

    Values are laid out by rank within their point -- distinct indices, as
    torch and jax disagree on a scatter-add -- and summed in rank order, so a
    point's sum does not depend on its chunk.
    """
    device = mesh_backend.device(qidx)
    counts = mesh_backend.long(mesh_backend.bincount(qidx, minlength=n_points))
    total = mesh_backend.zeros((n_points,), dtype=mesh_backend.float64, device=device)
    if qidx.shape[0] == 0:
        return total, counts
    width = int(mesh_backend.to_numpy(mesh_backend.max(counts)))
    starts = mesh_backend.cumsum(counts, dim=0) - counts
    rank = (
        mesh_backend.arange(qidx.shape[0], dtype=mesh_backend.int64, device=device)
        - starts[qidx]
    )
    dense = mesh_backend.fill_at_indices(
        mesh_backend.zeros(
            (n_points * width,), dtype=mesh_backend.float64, device=device
        ),
        qidx * width + rank,
        values,
    ).reshape(n_points, width)
    for c in range(width):
        total = total + dense[:, c]
    return total, counts


def _total(mesh, cover, beta):
    """``(mu, n)`` at ``(B, 2)`` points; ``cover`` from :func:`band_cover`."""
    b = beta.shape[0]
    qidx, cand, w = index_hits(mesh.index, mesh.vertices_source, mesh.leaves, beta)
    tri = mesh.vertices_source[mesh.leaves[cand]]
    a2 = area2(tri)
    once = mesh_backend.flatnonzero(counts_once(tri, w, a2))
    qidx, cand, w, a2 = qidx[once], cand[once], w[once], a2[once]
    mu, n = _per_point_sum(qidx, hit_magnification(mesh, cand, sanitize_bary(w, a2)), b)
    triangles, index = cover
    band, _, _ = index_hits(index, mesh.critical_band.source, triangles, beta)
    in_band = mesh_backend.bincount(band, minlength=b) > 0
    return mesh_backend.where(in_band, mesh_backend.inf, mu), n


def total_magnification(bx, by, mesh, *, batch_size=None):
    """
    Total point-source magnification at source-plane points, and the image count.

    ``mu`` sums ``1 / |det A|`` over the images -- one per converged leaf
    whose source-plane image contains the point, a point on a shared edge
    or vertex counted once -- with ``det A`` interpolated at the image the
    leaf's affine map locates. ``n`` counts them. Within an image sheet
    ``mu`` is continuous; it jumps across sheet edges (:func:`sheet_edges`)
    and the critical band. A point in the image of the band reads
    ``mu = inf`` and adds nothing to ``n``. A point outside every leaf image
    has ``mu = 0`` and ``n = 0``. The interpolation and the image position
    both err at order ``h**2``, ``h`` the leaf size.

    Parameters
    ----------
    bx, by: ArrayLike
        Source-plane points, any shape, flattened to ``(B,)``.

        *Unit: arcsec*
    mesh: LensMesh
    batch_size: Optional[int]
        Most points per chunk; the result does not depend on it.

    Returns
    -------
    mu: ArrayLike
        ``(B,)`` float64.
    n: ArrayLike
        ``(B,)`` int64.
    """
    device = mesh_backend.device(mesh.vertices_lens)
    beta = as_points(bx, by, device)
    cover = band_cover(mesh)
    n_points = beta.shape[0]
    step = max(n_points, 1) if batch_size is None else max(1, int(batch_size))
    mus = [mesh_backend.zeros((0,), dtype=mesh_backend.float64, device=device)]
    ns = [mesh_backend.zeros((0,), dtype=mesh_backend.int64, device=device)]
    for lo in range(0, n_points, step):
        mu, n = _total(mesh, cover, beta[lo : lo + step])
        mus.append(mu)
        ns.append(n)
    return mesh_backend.concatenate(mus, dim=0), mesh_backend.concatenate(ns, dim=0)


def magnification_sampler(mesh):
    """
    ``(N, 2) -> (N, 2)`` float64 ``(mu, n)`` at source-plane positions, for :func:`~.refine.refine`.

    The band cover is built once.
    """
    cover = band_cover(mesh)
    device = mesh_backend.device(mesh.vertices_lens)

    def sample(xy):
        mu, n = _total(mesh, cover, mesh_backend.to(xy, device=device))
        out = mesh_backend.stack(
            (mu, mesh_backend.to(n, dtype=mesh_backend.float64)), dim=-1
        )
        return mesh_backend.to(out, device=mesh_backend.device(xy))

    return sample


def sheet_edges(mesh):
    """
    Source-plane images of the lens-mesh edges across which the image count changes.

    Each converged leaf contributes its three edges, with the side of each
    its image lies on, by the sign of its source-plane area. Summed per
    undirected edge, two converged leaves of one parity cancel. The edges
    left are the fov boundary, the edges around the critical band and other
    failed leaves, and folds. An edge only one leaf of the whole mesh has
    lies on the fov boundary.

    Returns
    -------
    segments: ArrayLike
        ``(E, 2, 2)`` the edges' two ends, lower vertex index first.

        *Unit: arcsec*
    on_fov: ArrayLike
        ``(E,)`` bool, True on the fov boundary.
    """
    leaves, vs = mesh.leaves, mesh.vertices_source
    n_vertices = vs.shape[0]
    a = leaves.reshape(-1)
    b = leaves[:, [1, 2, 0]].reshape(-1)
    key = mesh_backend.minimum(a, b) * n_vertices + mesh_backend.maximum(a, b)
    converged = mesh_backend.long(
        mesh.origin_status[mesh.leaf_origin] == LEAF_CONVERGED
    )
    side = (
        mesh_backend.long(mesh_backend.where(area2(vs[leaves]) > 0, 1, -1)) * converged
    )
    eps = mesh_backend.repeat(side, 3, axis=0) * mesh_backend.long(
        mesh_backend.where(a < b, 1, -1)
    )
    keys, inverse = mesh_backend.unique(key, return_inverse=True)
    counts = mesh_backend.long(mesh_backend.bincount(inverse, minlength=keys.shape[0]))
    csum = csr_offsets(eps[mesh_backend.argsort(inverse)])
    ends = mesh_backend.cumsum(counts, dim=0)
    dn = csum[ends] - csum[ends - counts]
    sheet = mesh_backend.flatnonzero(dn != 0)
    lo, hi = keys[sheet] // n_vertices, keys[sheet] % n_vertices
    return mesh_backend.stack((vs[lo], vs[hi]), dim=1), counts[sheet] == 1
