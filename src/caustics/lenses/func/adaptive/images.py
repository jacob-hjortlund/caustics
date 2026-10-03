"""
Every image of each source-plane point, from a lens mesh.

:func:`forward_raytrace` seeds Levenberg-Marquardt from each leaf whose
source-plane image contains the point (:func:`mesh_query`,
:func:`mesh_seeds`), keeps the roots that converge near their seed, and
merges near-coincident roots (:func:`dedup_representatives`).
"""

from ....backend_obj import backend
from ....utils import batch_lm
from .geometry import area2, contains, csr_offsets, sanitize_bary, triangle_weights
from .index import as_points, index_hits


def mesh_query(mesh, beta, batch_size=None):
    """
    Leaves whose source-plane image contains each point.

    A point on an edge two leaves share returns both, and leaves near a
    critical curve overlap, so the number of hits is not the image count.

    Parameters
    ----------
    mesh: LensMesh
    beta: ArrayLike
        ``(B, 2)`` source-plane points.

        *Unit: arcsec*
    batch_size: Optional[int]
        Most points per chunk; the result does not depend on it.

    Returns
    -------
    leaf_indices: ArrayLike
        ``(K,)`` int64 leaves, ascending within each point.
    offsets: ArrayLike
        ``(B + 1,)`` int64 CSR offsets into ``leaf_indices``.
    bary: ArrayLike
        ``(K, 3)`` float64 barycentric coordinates of each point in its
        leaf's image, in the simplex.
    """
    device = backend.device(mesh.vertices_lens)
    beta = backend.as_array(beta, dtype=backend.float64, device=device)
    n = beta.shape[0]
    step = max(n, 1) if batch_size is None else max(1, int(batch_size))
    leaves = [backend.zeros((0,), dtype=backend.int64, device=device)]
    bary = [backend.zeros((0, 3), dtype=backend.float64, device=device)]
    counts = [backend.zeros((0,), dtype=backend.int64, device=device)]
    for lo in range(0, n, step):
        chunk = beta[lo : lo + step]
        qidx, cand, w = index_hits(mesh.index, mesh.vertices_source, mesh.leaves, chunk)
        counts.append(backend.long(backend.bincount(qidx, minlength=chunk.shape[0])))
        leaves.append(cand)
        tri = mesh.vertices_source[mesh.leaves[cand]]
        bary.append(sanitize_bary(w, area2(tri)))
    return (
        backend.concatenate(leaves, dim=0),
        csr_offsets(backend.concatenate(counts, dim=0)),
        backend.concatenate(bary, dim=0),
    )


def mesh_seeds(mesh, leaf_indices, bary):
    """
    Lens-plane preimage of each hit under its leaf's affine map, ``(K, 2)``.

    Accurate to ``min_img_sep`` by the refinement criterion, and inside its
    leaf, since ``bary`` lies in the simplex.

    *Unit: arcsec*
    """
    tri = mesh.vertices_lens[mesh.leaves[leaf_indices]]
    return backend.sum(tri * backend.unsqueeze(bary, -1), dim=1)


def dedup_block_group(points, rows, n_blocks, m, tol):
    """
    Connected-component representatives for blocks of exactly ``m`` points.

    Every slot is real, so this carries none of the padding machinery a
    ragged formulation needs: no validity mask, no clipped gather, and the
    "no label" sentinel is ``m`` rather than a global maximum. Grouping the
    caller's blocks by count and calling this once per distinct count is what
    keeps the ``(n_blocks, m, m)`` intermediate proportional to
    ``sum_c B_c * c**2`` instead of ``B * max(c)**2``.

    Parameters
    ----------
    points: ArrayLike
        The caller's full point array, shape ``(K, 2)``.

        *Unit: arcsec*
    rows: ArrayLike
        ``(n_blocks * m,)`` int64 indices into ``points``, block-major. A
        host-side sequence is accepted too; see :func:`dedup_representatives`.
    n_blocks, m: int
        Block count and the common per-block point count.
    tol: float
        Separation below which two points are the same image.

        *Unit: arcsec*

    Returns
    -------
    ArrayLike
        ``(n_blocks * m,)`` bool, in the order of ``rows``.
    """
    device = backend.device(points)
    int64 = backend.int64
    p = points[backend.as_array(rows, dtype=int64, device=device)]
    p = p.reshape(n_blocks, m, 2)

    delta = backend.unsqueeze(p, 2) - backend.unsqueeze(p, 1)
    # Squared distances against a squared tolerance: no sqrt, and the
    # comparison is exact on the diagonal, so every point is its own
    # neighbour and the label update below is a true minimum over the closed
    # neighbourhood.
    adjacent = backend.long(backend.sum(delta * delta, dim=-1) < tol * tol)

    # Min-label propagation. `m` is the sentinel for "no label": it exceeds
    # every real slot index, so it never wins a minimum against a neighbour.
    index = backend.unsqueeze(backend.arange(m, dtype=int64, device=device), 0)
    labels = index + backend.zeros((n_blocks, m), dtype=int64, device=device)
    for _ in range(m):
        neighbour = adjacent * backend.unsqueeze(labels, 1) + (1 - adjacent) * m
        updated = backend.min(neighbour, dim=2)
        if bool(backend.to_numpy(backend.all(updated == labels))):
            break
        labels = updated

    # Each component now carries the lowest slot index it contains, and that
    # slot is its own label -- so the fixed points are exactly one per
    # component.
    return (labels == index).reshape(-1)


def dedup_representatives(points, counts, tol):
    """
    One representative per cluster of near-coincident points, within each block.

    Clusters are the **connected components** of the ``distance < tol`` graph,
    not the greedy clusters :func:`~caustics.lenses.func.base.remove_duplicate_points`
    produces. The difference is order dependence: for three collinear points
    spaced ``0.9 * tol`` apart, greedy returns two representatives in one input
    order and one in another, so the image count would depend on the order
    :func:`mesh_query` happened to emit candidates in. Components are a
    function of the point set alone, which is what makes a multiplicity map
    reproducible.

    Adjacency is strict ``<``, so a pair separated by exactly ``tol`` stays
    distinct. That matches the build contract, where ``min_img_sep`` is a size
    floor the mesh resolves *to* rather than a scale it merges away.

    Vectorized by grouping blocks that share a count and running each group at
    its own width, because a greedy loop is one Python iteration per point --
    fine for the handful of images of a single source, hopeless for the
    ``nx * ny`` blocks of a multiplicity map. Blocks of zero or one point never
    reach the kernel; their answer is already known. The cost is a
    ``(B_c, c, c)`` intermediate per distinct count ``c``, which is why callers
    may still want to chunk over query points when a single block is enormous.

    Block-major order is restored by an inverse permutation rather than a
    scatter: every row belongs to exactly one group, so concatenating the
    groups' row indices gives a permutation of ``range(K)``, and ``argsort`` of
    a permutation *is* its inverse -- computed by sorting, never by an indexed
    assignment. That matters because torch keeps the last write on duplicate
    indices and jax accumulates, so a scatter would mean two different things
    on the two backends; a gather (indexing by the inverse permutation) means
    the same thing on both.

    Parameters
    ----------
    points: ArrayLike
        Shape ``(K, 2)``, laid out block-major: block ``b`` occupies the
        ``counts[b]`` rows following those of blocks ``0 .. b - 1``.

        *Unit: arcsec*
    counts: ArrayLike
        Shape ``(B,)`` int, with ``counts.sum() == K``. A host-side sequence
        is accepted directly and coerced on entry -- that is the array-like
        input the no-``numpy``-import rule permits, not an exception to it.
        Zero-length blocks are allowed.
    tol: float
        Separation below which two points are the same image.

        *Unit: arcsec*

    Returns
    -------
    ArrayLike
        ``(K,)`` bool, True on exactly one point per cluster.
    """
    device = backend.device(points)
    counts = backend.as_array(counts, dtype=backend.int64, device=device)
    total = int(backend.to_numpy(backend.sum(counts)))
    if total == 0:
        return backend.zeros((0,), dtype=backend.bool, device=device)

    starts = backend.cumsum(counts, dim=0) - counts
    row_groups, keep_groups = [], []

    # A block of one point is its own representative and a block of none
    # contributes nothing, so neither reaches the clustering kernel at all.
    # On a multiplicity map those are the large majority, and skipping them is
    # the single biggest reduction in what the kernel has to hold.
    singles = backend.flatnonzero(counts == 1)
    if singles.shape[0] > 0:
        row_groups.append(starts[singles])
        keep_groups.append(
            backend.ones((singles.shape[0],), dtype=backend.bool, device=device)
        )

    # The rest are grouped by *equal* count so each group runs at its own M.
    # Padding every block to the global maximum is what made the intermediate
    # `B * max(c)**2` and put a fine multiplicity map out of memory. The
    # distinct counts themselves are pulled to the host: there are at most a
    # handful of them (multiplicities are small integers), and each drives a
    # Python-level `dedup_block_group` call with its own static shape anyway.
    distinct = backend.to_numpy(backend.unique(counts[counts > 1])).tolist()
    for m in distinct:
        blocks = backend.flatnonzero(counts == m)
        rows = (
            backend.unsqueeze(starts[blocks], 1)
            + backend.unsqueeze(
                backend.arange(m, dtype=backend.int64, device=device), 0
            )
        ).reshape(-1)
        row_groups.append(rows)
        keep_groups.append(dedup_block_group(points, rows, blocks.shape[0], m, tol))

    # Every row belongs to exactly one group, so `perm` is a permutation of
    # `range(total)` and its `argsort` is exactly its inverse.
    perm = backend.concatenate(row_groups, dim=0)
    inverse = backend.argsort(perm)
    stacked = backend.concatenate(keep_groups, dim=0)
    return stacked[inverse]


def _block_sums(values, offsets):
    """Sum of ``values`` over each CSR block of ``offsets``."""
    csum = csr_offsets(values)
    return csum[offsets[1:]] - csum[offsets[:-1]]


def _images(mesh, beta, raytrace, tol, lm_kwargs):
    """Images and their count per point, for one chunk of ``(B, 2)`` points."""
    device = backend.device(mesh.vertices_lens)
    idx, offsets, bary = mesh_query(mesh, beta)
    seed = mesh_seeds(mesh, idx, bary)
    if seed.shape[0] == 0:
        return seed, backend.zeros((beta.shape[0],), dtype=backend.int64, device=device)

    def to_source(xy):
        return backend.stack(raytrace(xy[..., 0], xy[..., 1]), dim=-1)

    target = backend.repeat(beta, offsets[1:] - offsets[:-1], axis=0)
    root, _, _ = batch_lm(seed, target, to_source, **lm_kwargs)
    converged = backend.sum((to_source(root) - target) ** 2, dim=-1) < tol * tol
    tri = mesh.vertices_lens[mesh.leaves[idx]]
    near = contains(triangle_weights(tri, root)) | (
        backend.sum((root - seed) ** 2, dim=-1) <= mesh.min_img_sep**2
    )
    keep = converged & near
    kept = _block_sums(backend.long(keep), offsets)
    root = root[backend.flatnonzero(keep)]
    unique = dedup_representatives(root, kept, mesh.min_img_sep)
    return root[unique], _block_sums(backend.long(unique), csr_offsets(kept))


def forward_raytrace(
    bx, by, raytrace, mesh, *, batch_size=None, residual_tol=None, lm_kwargs=None
):
    """
    Lens-plane positions of every image of each source-plane point.

    Each leaf whose source-plane image contains the point seeds
    Levenberg-Marquardt (:func:`~caustics.utils.batch_lm`) with the point's
    preimage under the leaf's affine map, accurate to ``min_img_sep``. A
    root is kept when it maps to within ``residual_tol`` of the point and
    lies in its seed's leaf or within ``min_img_sep`` of the seed: near a
    fold, a small residual alone admits points far from any image. Kept
    roots closer than ``min_img_sep`` are one image.

    Parameters
    ----------
    bx, by: ArrayLike
        Source-plane points, any shape, flattened to ``(B,)``.

        *Unit: arcsec*
    raytrace: Callable[[ArrayLike, ArrayLike], Tuple[ArrayLike, ArrayLike]]
        The raytrace ``mesh`` was built from.
    mesh: LensMesh
    batch_size: Optional[int]
        Most source points per chunk, bounding memory. Image counts do not
        depend on it; positions can move within solver accuracy.
    residual_tol: Optional[float]
        Largest ``|raytrace(x) - beta|`` of a kept root, ``mesh.min_img_sep``
        by default.

        *Unit: arcsec*
    lm_kwargs: Optional[dict]
        Extra keyword arguments for :func:`~caustics.utils.batch_lm`.

    Returns
    -------
    x, y: ArrayLike
        ``(K,)`` image positions, source by source: the ``counts[b]`` images
        of source ``b`` follow those of sources ``0 .. b - 1``.

        *Unit: arcsec*
    counts: ArrayLike
        ``(B,)`` int64 images per source.
    """
    device = backend.device(mesh.vertices_lens)
    beta = as_points(bx, by, device)
    tol = mesh.min_img_sep if residual_tol is None else float(residual_tol)
    lm_kwargs = {} if lm_kwargs is None else dict(lm_kwargs)
    n = beta.shape[0]
    step = max(n, 1) if batch_size is None else max(1, int(batch_size))
    images = [backend.zeros((0, 2), dtype=backend.float64, device=device)]
    counts = [backend.zeros((0,), dtype=backend.int64, device=device)]
    for lo in range(0, n, step):
        found, count = _images(mesh, beta[lo : lo + step], raytrace, tol, lm_kwargs)
        images.append(found)
        counts.append(count)
    images = backend.concatenate(images, dim=0)
    return images[:, 0], images[:, 1], backend.concatenate(counts, dim=0)
