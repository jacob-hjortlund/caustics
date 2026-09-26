"""
Evaluating the lens on lattice points.

:func:`make_raytrace` wraps a user ``raytrace`` so that coordinates go out and
come back as float64 -- the single most important property of the build.
:func:`evaluate` fills the vertex cache without ever tracing a point twice,
and :func:`sample_jacobians` evaluates the Jacobian once per distinct lattice
point.
"""

import math
from typing import Callable, Tuple

from ....backend_obj import ArrayLike, backend
from .state import VertexCache, cache_insert, cache_missing
from .lattice import lattice_ij_from_key, lattice_key, lattice_xy

__all__ = (
    "make_raytrace",
    "trace_points",
    "trace_keys",
    "evaluate",
    "sample_jacobians",
)


def make_raytrace(raytrace, device) -> Callable[[ArrayLike], ArrayLike]:
    """
    Wrap a user ``raytrace(x, y) -> (bx, by)`` as a host-side ``(N, 2) -> (N, 2)``.

    Coordinates go out to the callback, and come back from it, as float64 --
    **unconditionally**. This is the single most important property of the
    build. The refinement criterion compares a midpoint deviation against an
    affine prediction, both ``O(fov)`` quantities, so its roundoff floor is
    ``eps * fov`` and the comparison is meaningless below
    ``h ~ sqrt(8 * eps * fov)``. At ``fov = 5`` that floor is about ``2e-3``
    arcsec in float32 -- comparable to a typical ``min_img_sep`` -- and below
    it the midpoint deviation cancels to *exactly* zero, which the criterion
    reads as "perfectly affine" and converges. That is the fail-**open**
    direction: the mesh silently stops refining exactly where it most needs
    to, and no downstream clamp can recover the resolution once lost. Hence
    the coercion on the way in, and the cast on the way out, regardless of
    what dtype the callback itself operates in or returns.

    The callback's own dtype is still recorded in ``info["dtype"]`` on every
    call -- not acted on here, but so that a caller several layers up (the
    mesh build) can tell a silently downgraded callback from a well-behaved
    one and warn accordingly.

    Parameters
    ----------
    raytrace: Callable[[ArrayLike, ArrayLike], Tuple[ArrayLike, ArrayLike]]
        ``raytrace(x, y) -> (bx, by)``, on 1-D arrays of shape ``(N,)``.
    device:
        Device for the coordinates handed to ``raytrace``.

    Returns
    -------
    Callable[[ArrayLike], ArrayLike]
        ``(N, 2) -> (N, 2)`` float64. Carries a mutable ``.info`` dict,
        ``{"done": bool, "dtype": Any}``. ``"done"`` becomes True once the
        2-tuple return shape has been validated, which happens only on the
        first call; every call, first or not, validates that the returned
        arrays are shape-preserving on the 1-D input. ``"dtype"`` is whatever
        dtype the callback itself returned on its most recent call.
    """
    info = {"done": False, "dtype": backend.float64}

    def call(xy):
        build_device = backend.device(xy)
        x = backend.as_array(xy[:, 0], dtype=backend.float64, device=device)
        y = backend.as_array(xy[:, 1], dtype=backend.float64, device=device)
        out = raytrace(x, y)
        if not info["done"]:
            if not isinstance(out, tuple) or len(out) != 2:
                raise ValueError(
                    "raytrace must return a 2-tuple (bx, by) of arrays with "
                    f"shape (N,); got {type(out).__name__}"
                )
            info["done"] = True
        bx, by = out
        info["dtype"] = bx.dtype
        if bx.shape != x.shape or by.shape != y.shape:
            raise ValueError(
                f"raytrace returned shape {tuple(bx.shape)}/{tuple(by.shape)} "
                f"for {tuple(x.shape)} inputs; it must be shape-preserving on "
                "1-D input"
            )
        return backend.to(
            backend.stack((bx, by), dim=-1),
            dtype=backend.float64,
            device=build_device,
        )

    call.info = info  # type: ignore[attr-defined]
    return call


def trace_points(xy, raytrace_fn, batch_size) -> ArrayLike:
    """
    ``raytrace_fn`` on float64 lens-plane points ``(N, 2)``, chunked only for memory.

    The whole-array path is used whenever ``batch_size`` is ``None`` or the
    input already fits in one batch. ``raytrace_fn`` acts on each row
    independently and concatenation keeps row order, so the result is
    bit-identical for every ``batch_size``.

    Parameters
    ----------
    xy: ArrayLike
        ``(N, 2)`` float64 points.

        *Unit: arcsec*
    raytrace_fn: Callable[[ArrayLike], ArrayLike]
        From :func:`make_raytrace`.
    batch_size: Optional[int]
        Maximum rows per ``raytrace_fn`` call, or ``None`` for a single call.

    Returns
    -------
    ArrayLike
        ``(N, 2)`` float64 images.

        *Unit: arcsec*
    """
    if batch_size is None or xy.shape[0] <= batch_size:
        return raytrace_fn(xy)
    n_chunks = math.ceil(xy.shape[0] / batch_size)
    return backend.concatenate(
        [raytrace_fn(chunk) for chunk in backend.chunk(xy, n_chunks, dim=0)], dim=0
    )


def trace_keys(lat, ij, raytrace_fn, batch_size) -> ArrayLike:
    """
    Raytrace lattice points as one logical batch, chunked only for memory.

    The whole-array path is used whenever ``batch_size`` is ``None`` or the
    input already fits in a single batch; only otherwise does this chunk with
    :func:`backend.chunk` and concatenate. That ordering is what makes the
    result bit-identical for every ``batch_size``: ``raytrace_fn`` acts on
    each row independently, and concatenation preserves row order, so
    partitioning into chunks can change *how many* calls are made but never
    *what* they compute.

    Split out of :func:`evaluate` so a ``max_level`` midpoint pass -- points
    consumed by the parity test and never read again -- can reuse the
    chunking without touching the vertex cache.

    Parameters
    ----------
    lat: Lattice
        Used to convert ``ij`` to lens-plane positions.
    ij: ArrayLike
        Lattice coordinates, shape ``(n, 2)`` int64.
    raytrace_fn: Callable[[ArrayLike], ArrayLike]
        From :func:`make_raytrace`.
    batch_size: Optional[int]
        Maximum rows per ``raytrace_fn`` call, or ``None`` for a single call.

    Returns
    -------
    ArrayLike
        Shape ``(n, 2)`` float64.

        *Unit: arcsec*
    """
    return trace_points(lattice_xy(lat, ij), raytrace_fn, batch_size)


def evaluate(cache, lat, keys, raytrace_fn, batch_size) -> VertexCache:
    """
    Evaluate every not-yet-cached key, in one logical batch per call.

    Deduplication happens before any point is traced: :func:`cache_missing`
    reduces ``keys`` to its unique, not-yet-cached elements first, so a caller
    that repeats a key -- two triangles sharing a vertex, say -- never causes
    it to be raytraced twice.

    Parameters
    ----------
    cache: VertexCache
    lat: Lattice
    keys: ArrayLike
        Lattice keys to ensure are cached, shape ``(n,)`` int64. Need not be
        unique or sorted.
    raytrace_fn: Callable[[ArrayLike], ArrayLike]
        From :func:`make_raytrace`.
    batch_size: Optional[int]
        Forwarded to :func:`trace_keys`.

    Returns
    -------
    VertexCache
        ``cache`` itself, unchanged, when every key is already cached;
        otherwise a new cache with the missing keys inserted.
    """
    todo = cache_missing(cache, keys)
    if todo.shape[0] == 0:
        return cache
    ij = lattice_ij_from_key(lat, todo)
    beta = trace_keys(lat, ij, raytrace_fn, batch_size)
    new_cache, _ = cache_insert(cache, todo, ij, beta)
    return new_cache


def sample_jacobians(
    lat, six_ij, jacobian_fn
) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
    """
    The lens Jacobian at every triangle's six samples, one call per lattice point.

    A vertex is shared by about six triangles and a midpoint by two, so
    evaluating per triangle, as :func:`jacobian_parity_ok` does, repeats
    most points. Deduplicating on the lattice key first cuts the batch by
    roughly ``2.5x`` on a refined region, and it gives every triangle that
    shares a sample the same ``A`` there -- which the critical band relies
    on.

    ``jacobian_fn`` is called exactly once, on the lattice's own float64
    coordinates, and not at all when there are no triangles.

    Parameters
    ----------
    lat: Lattice
    six_ij: ArrayLike
        Lattice coordinates of each triangle's ``theta_1, theta_2, theta_3,
        m_1, m_2, m_3``, shape ``(G, 6, 2)`` int64.
    jacobian_fn: Callable[[ArrayLike, ArrayLike], ArrayLike]
        ``jacobian_fn(x, y) -> (K, 2, 2)``.

    Returns
    -------
    keys: ArrayLike
        ``(U,)`` int64, the distinct sample keys, ascending.
    index: ArrayLike
        ``(G, 6)`` int64 index into ``keys`` of each triangle's samples.
    J: ArrayLike
        ``(U, 2, 2)``, the Jacobian at each distinct sample.

    Raises
    ------
    ValueError
        If ``jacobian_fn`` does not return ``(K, 2, 2)`` for ``K`` points.
    """
    n = six_ij.shape[0]
    if n == 0:
        return (
            backend.zeros((0,), dtype=backend.int64),
            backend.zeros((0, 6), dtype=backend.int64),
            backend.zeros((0, 2, 2), dtype=backend.float64),
        )
    keys, index = backend.unique(
        lattice_key(lat, six_ij).reshape(-1), return_inverse=True
    )
    xy = lattice_xy(lat, lattice_ij_from_key(lat, keys))
    J = jacobian_fn(xy[:, 0], xy[:, 1])
    if J.shape != (keys.shape[0], 2, 2):
        raise ValueError("jacobian_fn must return shape (K, 2, 2) for K points")
    return keys, index.reshape(n, 6), J
