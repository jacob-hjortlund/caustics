"""
Source-plane queries of a frozen mesh: containing leaves, and seeds in them.
"""

from typing import Tuple

from ....backend_obj import ArrayLike, backend
from .geometry import contains, sanitize_bary, triangle_weights

__all__ = (
    "mesh_query",
    "mesh_seeds",
)


def _as_beta(mesh, beta) -> ArrayLike:
    """
    Coerce query points to the mesh's dtype and device, shape ``(B, 2)``.

    A bare ``(2,)`` raises rather than being promoted, so :func:`mesh_query`'s
    output shapes are never ambiguous.
    """
    beta = backend.as_array(beta, dtype=mesh.vertices_source.dtype, device=mesh.device)
    if len(beta.shape) != 2 or beta.shape[1] != 2:
        raise ValueError(
            f"beta must have shape (B, 2), got {tuple(beta.shape)}. A single "
            "point must be passed as shape (1, 2)."
        )
    return beta


def _empty_query_result(mesh, n_queries) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
    """The ``mesh_query`` result for ``n_queries`` points that all miss."""
    int64 = backend.int64
    return (
        backend.zeros((0,), dtype=int64, device=mesh.device),
        backend.zeros((n_queries + 1,), dtype=int64, device=mesh.device),
        backend.zeros((0, 3), dtype=mesh.vertices_source.dtype, device=mesh.device),
    )


def mesh_query(mesh, beta, batch_size=None) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
    """
    Terminal leaves whose source-plane image contains each query point.

    Returns candidate regions, not images. A point on a shared edge returns
    both leaves -- zeros count as inside, which is what guarantees no query
    falls through a seam -- and near-critical leaves overlap, so **candidate
    count is not image multiplicity**.

    Parameters
    ----------
    mesh: AdaptiveMesh
        The frozen mesh to query.
    beta: ArrayLike
        Source-plane query points, shape ``(B, 2)`` strictly. A bare ``(2,)``
        raises rather than being promoted, so output shapes are never
        ambiguous.

        *Unit: arcsec*
    batch_size: Optional[int]
        Chunk size over query points. ``None`` processes all at once. Results
        are byte-identical for every value; this only bounds peak memory,
        which spikes for chunks landing near a caustic.

    Returns
    -------
    leaf_indices: ArrayLike
        ``(K,)`` indices into ``mesh.leaves``, strictly ascending within each
        block.
    offsets: ArrayLike
        ``(B + 1,)`` CSR offsets, ``offsets[0] == 0`` and ``offsets[B] == K``.
    bary: ArrayLike
        ``(K, 3)`` barycentric coordinates of ``beta`` in the source-plane
        image of the hit triangle, guaranteed to lie in the simplex.
    """
    beta = _as_beta(mesh, beta)
    n = beta.shape[0]
    if n == 0:
        return _empty_query_result(mesh, 0)

    int64 = backend.int64
    index = mesh.index
    step = n if batch_size is None else max(1, int(batch_size))
    idx_parts, bary_parts, count_parts = [], [], []

    for lo in range(0, n, step):
        chunk = beta[lo : lo + step]
        b = chunk.shape[0]
        u = backend.long(backend.floor((chunk - index.lo) / index.cell))
        # Containment is a coordinate test against the stored exact `hi`, never a
        # cell-index test on `u`. Because `cell = span / [nx, ny]`, a point sitting
        # exactly on the upper bbox edge (x == hi_x) gives `u_x == nx`, which
        # fails `u_x < nx` -- even though `build_index` clips leaf registration
        # to column `nx - 1`, i.e. the leaf IS indexed, in the very column that
        # test would reject. Do NOT recompute `hi` as `lo + cell * [nx, ny]`:
        # `(span / n) * n` need not equal `span` to the ulp, so the exact
        # `mesh.index.hi` from the build is used instead. A leaf containing a
        # `beta` with x == hi has its `i1_x` clipped to `nx - 1`, the column the
        # clamp below selects, so this coordinate test is provably complete.
        inside = (
            (chunk[:, 0] >= index.lo[0])
            & (chunk[:, 0] <= index.hi[0])
            & (chunk[:, 1] >= index.lo[1])
            & (chunk[:, 1] <= index.hi[1])
        )
        # Clamped only to keep the gather in range; `inside` forces an empty
        # block for out-of-bbox points.
        cell = backend.clamp(u[:, 0], 0, index.nx - 1) * index.ny + backend.clamp(
            u[:, 1], 0, index.ny - 1
        )
        start = index.cell_offsets[cell]
        count = backend.where(
            inside, index.cell_offsets[cell + 1] - start, backend.zeros_like(start)
        )
        total = int(backend.to_numpy(backend.sum(count)))
        if total == 0:
            count_parts.append(backend.zeros((b,), dtype=int64, device=mesh.device))
            continue

        qidx = backend.repeat(
            backend.arange(b, dtype=int64, device=mesh.device), count, axis=0
        )
        base = backend.cumsum(count, dim=0) - count
        within = backend.arange(
            total, dtype=int64, device=mesh.device
        ) - backend.repeat(base, count, axis=0)
        cand = index.cell_leaves[start[qidx] + within]

        w = triangle_weights(mesh.vertices_source[mesh.leaves[cand]], chunk[qidx])
        hit = contains(w)

        # Per-query hit counts by cumsum differences. Never add_at_indices:
        # torch keeps the last write on duplicate indices, jax accumulates --
        # so a scatter-add would mean two different things on the two backends.
        csum = backend.concatenate(
            (
                backend.zeros((1,), dtype=int64, device=mesh.device),
                backend.cumsum(backend.long(hit), dim=0),
            ),
            dim=0,
        )
        count_parts.append(csum[base + count] - csum[base])
        idx_parts.append(cand[hit])
        # Gather `leaf_area2` after masking by `hit`, not before: `cand` is the
        # pre-containment candidate list, so `leaf_area2[cand]` would be a
        # full-length temporary thrown away by the mask on the very next
        # operation.
        bary_parts.append(sanitize_bary(w[hit], mesh.leaf_area2[cand[hit]]))

    counts = backend.concatenate(count_parts, dim=0)
    offsets = backend.concatenate(
        (
            backend.zeros((1,), dtype=int64, device=mesh.device),
            backend.cumsum(counts, dim=0),
        ),
        dim=0,
    )
    if not idx_parts:
        empty = _empty_query_result(mesh, 0)
        return empty[0], offsets, empty[2]
    return (
        backend.concatenate(idx_parts, dim=0),
        offsets,
        backend.concatenate(bary_parts, dim=0),
    )


def mesh_seeds(mesh, leaf_indices, bary) -> ArrayLike:
    """
    Lens-plane preimage of hit leaves under each leaf's own affine map, shape ``(K, 2)``.

    A ``bary``-weighted gather of ``vertices_lens``: the pure contraction
    ``sum(tri * bary[..., None], axis=1)`` where ``tri = vertices_lens[leaves[
    leaf_indices]]``. That map is exactly the one step 7's refinement criterion
    bounds, so the result is a Newton seed accurate to ``min_img_sep`` by
    construction. Because ``bary`` is guaranteed to lie in the simplex (see
    :func:`sanitize_bary`), the seed always lies inside its leaf.

    There is no ``beta`` mode here, unlike the oracle's ``Mesh.seeds``, and no
    guard rejecting an ambiguous call: a caller wanting query-then-seed
    composes :func:`mesh_query` and this function directly --
    ``leaf_indices, _, bary = mesh_query(mesh, beta)`` then
    ``mesh_seeds(mesh, leaf_indices, bary)`` -- rather than this function
    accepting both forms behind a runtime check.

    Parameters
    ----------
    mesh: AdaptiveMesh
        The frozen mesh ``leaf_indices`` indexes into.
    leaf_indices: ArrayLike
        ``(K,)`` indices into ``mesh.leaves``, e.g. from :func:`mesh_query`.
    bary: ArrayLike
        ``(K, 3)`` barycentric coordinates within each indexed leaf, e.g. from
        :func:`mesh_query`. The caller is responsible for ensuring it lies in
        the simplex; :func:`mesh_query` already guarantees this.

    Returns
    -------
    ArrayLike
        ``(K, 2)`` lens-plane seed positions.

        *Unit: arcsec*
    """
    tri = mesh.vertices_lens[mesh.leaves[leaf_indices]]
    return backend.sum(tri * backend.unsqueeze(bary, -1), dim=1)
