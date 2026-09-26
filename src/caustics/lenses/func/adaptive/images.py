"""
Every image of each source-plane point, from the mesh's seeds.

:func:`mesh_forward_raytrace` seeds from
:func:`~caustics.lenses.func.adaptive.query.mesh_seeds`, refines each seed
with Levenberg-Marquardt under ``method="rootfind"``, and merges
near-coincident images of each source with :func:`dedup_representatives`.
"""

from typing import Callable, Optional, Tuple

from ....backend_obj import ArrayLike, backend
from ....utils import batch_lm
from .geometry import contains, triangle_weights
from .query import _as_beta, mesh_query, mesh_seeds

__all__ = (
    "dedup_block_group",
    "dedup_representatives",
    "METHODS",
    "mesh_forward_raytrace",
)


def dedup_block_group(points, rows, n_blocks, m, tol) -> ArrayLike:
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


def dedup_representatives(points, counts, tol) -> ArrayLike:
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


METHODS = ("rootfind", "dedup")


def _check_method(method) -> None:
    """Reject an unrecognised image-finding method, naming the alternatives."""
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")


def _to_source(raytrace):
    """Wrap ``raytrace(x, y)`` as a ``(..., 2) -> (..., 2)`` map."""

    def to_source(xy):
        return backend.stack(raytrace(xy[..., 0], xy[..., 1]), dim=-1)

    return to_source


def _forward_chunk(mesh, chunk, raytrace, method, tol, lm_kwargs):
    """
    Images and per-source counts for one chunk of query points.

    Returns
    -------
    images: ArrayLike or None
        ``(K, 2)`` lens-plane positions, or ``None`` when the chunk found
        none.
    counts: ArrayLike
        ``(b,)`` ``backend`` int64 image multiplicity.
    """
    b = chunk.shape[0]
    int64 = backend.int64
    device = mesh.device
    none = (None, backend.zeros((b,), dtype=int64, device=device))

    idx, offsets, bary = mesh_query(mesh, chunk)
    seed = mesh_seeds(mesh, idx, bary)
    if seed.shape[0] == 0:
        return none

    if method == "dedup":
        # No residual filter and no displacement filter. Both exist to
        # reject a root that *wandered* away from its seed -- see the Notes
        # on `mesh_forward_raytrace`. A seed cannot wander: `bary` lies in
        # the simplex, so the seed lies inside its leaf, and that leaf's
        # source-plane image contains `beta`. Every seed is therefore
        # already an approximate image, and filtering would be testing a
        # property the construction guarantees.
        survivors, kept = seed, offsets[1:] - offsets[:-1]
    else:
        to_source = _to_source(raytrace)
        # One target per seed, so a source with several candidate leaves
        # root-finds each of them against its own beta.
        spans = offsets[1:] - offsets[:-1]
        target = backend.repeat(chunk, spans, axis=0)
        root, _, _ = batch_lm(seed, target, to_source, **lm_kwargs)

        converged = backend.sum((to_source(root) - target) ** 2, dim=-1) < tol * tol
        # See the Notes on `mesh_forward_raytrace` for why containment and
        # the ball are OR-ed.
        tri = mesh.vertices_lens[mesh.leaves[idx]]
        near = contains(triangle_weights(tri, root)) | (
            backend.sum((root - seed) ** 2, dim=-1) <= mesh.min_img_sep**2
        )
        keep = converged & near

        # Chunk bookkeeping stays on `backend` int64 arrays: `kept` is read
        # off `keep` by the same cumsum-difference trick `mesh_query` uses
        # for its own per-query hit counts, never a host round trip.
        keep_i = backend.long(keep)
        csum = backend.concatenate(
            (
                backend.zeros((1,), dtype=int64, device=device),
                backend.cumsum(keep_i, dim=0),
            ),
            dim=0,
        )
        kept = csum[offsets[1:]] - csum[offsets[:-1]]
        if int(backend.to_numpy(backend.sum(kept))) == 0:
            return none
        survivors = root[keep]

    unique = dedup_representatives(survivors, kept, mesh.min_img_sep)
    unique_i = backend.long(unique)
    kept_off = backend.concatenate(
        (backend.zeros((1,), dtype=int64, device=device), backend.cumsum(kept, dim=0)),
        dim=0,
    )
    csum = backend.concatenate(
        (
            backend.zeros((1,), dtype=int64, device=device),
            backend.cumsum(unique_i, dim=0),
        ),
        dim=0,
    )
    counts = csum[kept_off[1:]] - csum[kept_off[:-1]]
    return survivors[unique], counts


def _image_chunks(mesh, beta, raytrace, batch_size, method, residual_tol, lm_kwargs):
    """Yield :func:`_forward_chunk`'s ``(images, counts)`` per chunk."""
    tol = mesh.min_img_sep if residual_tol is None else float(residual_tol)
    lm_kwargs = {} if lm_kwargs is None else dict(lm_kwargs)
    n = beta.shape[0]
    step = max(1, n) if batch_size is None else max(1, int(batch_size))
    for lo in range(0, max(n, 1), step):
        yield _forward_chunk(
            mesh, beta[lo : lo + step], raytrace, method, tol, lm_kwargs
        )


def mesh_forward_raytrace(
    mesh,
    beta,
    raytrace: Callable[[ArrayLike, ArrayLike], Tuple[ArrayLike, ArrayLike]],
    batch_size: Optional[int] = None,
    *,
    method: str = "rootfind",
    residual_tol: Optional[float] = None,
    lm_kwargs: Optional[dict] = None,
) -> Tuple[ArrayLike, ArrayLike]:
    """
    Image-plane positions of every image of each source-plane point.

    :func:`mesh_seeds` supplies a Newton seed per candidate leaf, accurate to
    ``min_img_sep`` by construction. Under ``method="rootfind"``,
    Levenberg-Marquardt refines each seed to a root of the lens equation,
    unconverged roots are discarded, and the survivors are deduplicated at
    ``min_img_sep``; under ``method="dedup"`` the seeds themselves are
    deduplicated directly -- see the ``method`` parameter below.

    Parameters
    ----------
    mesh: AdaptiveMesh
        The frozen mesh to raytrace against.
    beta: ArrayLike
        Source-plane points, shape ``(B, 2)`` strictly. A single point must be
        passed as ``(1, 2)``.

        *Unit: arcsec*

    raytrace: Callable
        **Must be the raytrace of the lens this mesh was built from**, i.e.
        ``lens.raytrace``, called as ``raytrace(x, y) -> (bx, by)``. The seeds
        handed to the root finder are preimages under *this* mesh's leaves, so
        a different lens would be root-found from meaningless starting points
        -- silently, since the residual filter would simply reject most of them
        and return too few images rather than raising. This cannot be checked:
        a callable carries no identity the mesh could have recorded at build
        time.
    batch_size: Optional[int]
        Chunk size over source points. Bounds peak memory for the whole
        pipeline, not just :func:`mesh_query` -- the root finder holds
        ``(K, 2)`` states and the dedup a ``(B_c, c, c)`` intermediate per
        distinct candidate count ``c`` (see :func:`dedup_representatives`).
        Image counts are invariant to this chunking. Levenberg-Marquardt's
        shared damping schedule can shift root positions within solver
        accuracy when the batch partition changes.
    method: str
        ``"rootfind"`` (default) refines every seed with
        Levenberg-Marquardt and returns machine-precision image positions.
        ``"dedup"`` skips the root finder entirely and deduplicates the
        seeds, which are already accurate to ``min_img_sep`` by
        construction. It **never calls** ``raytrace``, which is why it is
        roughly two orders of magnitude faster; ``raytrace``,
        ``residual_tol`` and ``lm_kwargs`` are accepted and ignored.

        Positions from ``"dedup"`` are accurate to ``min_img_sep``, not to
        machine precision. Counts agree with ``"rootfind"`` except within
        about ``min_img_sep`` of a caustic -- measured at 12 pixels in
        24656 on an EPL-plus-shear lens. Neither method's counts are
        guaranteed to satisfy the odd-image theorem any longer:
        ``"dedup"`` can merge a near-tangential pair closer than
        ``min_img_sep`` into one, and ``"rootfind"`` can miss an image
        outright whose seed would have fallen in a non-converged
        ``max_level`` leaf, which was never in the index to seed the root
        finder in the first place. Measured on the cored SIE fixture
        (``fov=5``, ``init_res=32``) over a 25x25 source grid of 0.08 arcsec
        pixels centred on the lens, ``"rootfind"``: even image counts --
        impossible for this non-singular lens -- turn up at 371 of 625
        pixels at ``min_img_sep=0.04``, 263 at ``0.02``, and 4 at ``0.01``.
        Use ``"rootfind"`` when the position itself matters, ``"dedup"``
        when the count does.
    residual_tol: Optional[float]
        Source-plane tolerance on ``|raytrace(x) - beta|`` for accepting a
        root. Defaults to ``min_img_sep``.

        *Unit: arcsec*

    lm_kwargs: Optional[dict]
        Extra keyword arguments for :func:`~caustics.utils.batch_lm`, e.g.
        ``max_iter``.

    Returns
    -------
    images: ArrayLike
        ``(K, 2)`` lens-plane image positions, laid out block-major: the
        ``counts[b]`` images of source ``b`` follow those of sources
        ``0 .. b - 1``.

        *Unit: arcsec*

    counts: ArrayLike
        ``(B,)`` int64 image multiplicity of each source point.

    Notes
    -----
    Two filters decide that a root is an image, and both are needed.

    The **residual** test alone is weak near a fold caustic, where the lens
    map is quadratic: a point sitting well over ``min_img_sep`` from the true
    image in the lens plane can still have a small source-plane residual, so
    it survives the residual test, escapes the dedup, and inflates the count
    exactly where multiplicity structure matters most.

    The **displacement** test closes that hole using a guarantee the mesh
    already makes -- the seed lies inside its leaf and is accurate to
    ``min_img_sep`` -- so a root that left its own neighbourhood is not the
    root its seed was pointing at. It is a disjunction rather than plain
    containment because a leaf at the size floor is itself only about
    ``min_img_sep`` across, so a genuine root near a leaf edge can
    legitimately land just outside it; requiring containment alone would
    drop real images.

    Neither filter applies under ``method="dedup"``. Both reject a root
    that *wandered* -- the residual test catches a solve that converged to
    nothing, the displacement test one that converged to a different image.
    A seed cannot wander: it lies inside its own leaf, whose source-plane
    image contains ``beta``. Filtering it would test a property the
    construction already guarantees.

    Root finding runs in the dtype of the frozen mesh, so a mesh built with
    ``dtype=backend.float32`` caps the achievable accuracy near the
    cancellation floor :func:`build_adaptive_mesh` already warns about.
    """
    _check_method(method)
    beta = _as_beta(mesh, beta)
    n = beta.shape[0]
    int64 = backend.int64

    def no_images():
        return backend.zeros((0, 2), dtype=mesh.vertices_lens.dtype, device=mesh.device)

    if n == 0:
        return no_images(), backend.zeros((0,), dtype=int64, device=mesh.device)

    image_parts, count_parts = [], []
    for images, counts in _image_chunks(
        mesh, beta, raytrace, batch_size, method, residual_tol, lm_kwargs
    ):
        count_parts.append(counts)
        if images is not None:
            image_parts.append(images)

    counts = backend.concatenate(count_parts, dim=0)
    if not image_parts:
        return no_images(), counts
    return backend.concatenate(image_parts, dim=0), counts
