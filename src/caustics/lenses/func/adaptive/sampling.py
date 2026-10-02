"""
Evaluating the lens on lattice points.

:func:`make_raytrace` wraps a user ``raytrace`` so that coordinates go out and
come back as float64 -- the single most important property of the build.
:func:`evaluate` fills the vertex cache without ever tracing a point twice,
and :func:`sample_jacobians` evaluates the Jacobian at most once per lattice
point per build, keeping ``det A`` and the criterion's sign of it in the
vertex cache.
"""

import math
from typing import Callable, Tuple

from ....backend_obj import ArrayLike, backend
from .geometry import jacobian_det, jacobian_signs
from .state import (
    VertexCache,
    cache_insert,
    cache_lookup,
    cache_missing,
    cache_set_jacobian,
    cache_size,
)
from .lattice import lattice_ij_from_key, lattice_key, lattice_xy

__all__ = (
    "make_raytrace",
    "trace_points",
    "trace_keys",
    "evaluate",
    "call_jacobian",
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


def call_jacobian(lat, ij, jacobian_fn) -> ArrayLike:
    """
    ``jacobian_fn`` once, on the float64 lens-plane positions of lattice points.

    Parameters
    ----------
    lat: Lattice
    ij: ArrayLike
        ``(K, 2)`` int64 lattice coordinates.
    jacobian_fn: Callable[[ArrayLike, ArrayLike], ArrayLike]
        ``jacobian_fn(x, y) -> (K, 2, 2)``.

    Returns
    -------
    ArrayLike
        ``(K, 2, 2)``, as ``jacobian_fn`` returned it.

    Raises
    ------
    ValueError
        If ``jacobian_fn`` does not return ``(K, 2, 2)`` for ``K`` points.
    """
    xy = lattice_xy(lat, ij)
    J = jacobian_fn(xy[:, 0], xy[:, 1])
    # `getattr`, not `J.shape`: a raytrace passed as the Jacobian returns a
    # tuple, which should fail here, naming the Jacobian, not as an
    # AttributeError.
    shape = getattr(J, "shape", None)
    if shape != (ij.shape[0], 2, 2):
        got = type(J).__name__ if shape is None else tuple(shape)
        raise ValueError(
            "jacobian_fn must return an array of shape (K, 2, 2) for K points, "
            f"got {got}"
        )
    return J


def sample_jacobians(
    lat, cache, six_ij, jacobian_fn
) -> Tuple[VertexCache, ArrayLike, ArrayLike, ArrayLike, ArrayLike, int]:
    """
    ``det A`` and the criterion's sign at every triangle's six samples, each point evaluated once.

    The samples are deduplicated on their lattice keys. A key whose cache
    slot is already ``evaluated`` reads ``det`` and ``sign`` back from the
    cache. The rest go to a single :func:`call_jacobian`, or to none when
    there are none. Of those, a key with a cache slot is written back
    (:func:`cache_set_jacobian`); one with none -- a ``max_level`` midpoint,
    whose odd coordinate is never cached -- is used here and dropped, as its
    raytrace is. So across a build, as with raytraced points, no lattice
    point is evaluated twice, and every triangle sharing a sample reads one
    value there, which the critical band relies on.

    Parameters
    ----------
    lat: Lattice
    cache: VertexCache
    six_ij: ArrayLike
        Lattice coordinates of each triangle's ``theta_1, theta_2, theta_3,
        m_1, m_2, m_3``, shape ``(G, 6, 2)`` int64.
    jacobian_fn: Callable[[ArrayLike, ArrayLike], ArrayLike]
        ``jacobian_fn(x, y) -> (K, 2, 2)``.

    Returns
    -------
    cache: VertexCache
        With the new values of every cached key recorded.
    keys: ArrayLike
        ``(U,)`` int64, the distinct sample keys, ascending.
    index: ArrayLike
        ``(G, 6)`` int64 index into ``keys`` of each triangle's samples.
    det: ArrayLike
        ``(U,)`` float64 ``det A``: the cache's wherever it has one, so a
        value seeded from a frozen mesh wins, and the new one elsewhere.
    sign: ArrayLike
        ``(U,)`` int64, :func:`jacobian_signs` of each.
    n_called: int
        Points passed to ``jacobian_fn``.

    Raises
    ------
    ValueError
        As :func:`call_jacobian`.
    """
    n = six_ij.shape[0]
    if n == 0:
        return (
            cache,
            backend.zeros((0,), dtype=backend.int64),
            backend.zeros((0, 6), dtype=backend.int64),
            backend.zeros((0,), dtype=backend.float64),
            backend.zeros((0,), dtype=backend.int64),
            0,
        )
    keys, index = backend.unique(
        lattice_key(lat, six_ij).reshape(-1), return_inverse=True
    )
    slot = cache_lookup(cache, keys)
    cached = slot >= 0
    if cache_size(cache):
        at = backend.where(cached, slot, backend.zeros_like(slot))
        done = cached & cache.evaluated[at]
        held = cached & cache.has_det[at]
        det = backend.where(held, cache.det[at], backend.nan)
        sign = backend.where(done, cache.sign[at], backend.zeros_like(slot))
    else:
        done = held = backend.zeros(keys.shape, dtype=backend.bool)
        det = backend.zeros(keys.shape, dtype=backend.float64) + backend.nan
        sign = backend.zeros_like(slot)
    todo = backend.flatnonzero(~done)
    n_called = int(todo.shape[0])
    if n_called:
        J = call_jacobian(lat, lattice_ij_from_key(lat, keys[todo]), jacobian_fn)
        new_det, new_sign = jacobian_det(J), jacobian_signs(J)
        # `det` and `sign` are fresh arrays from `where`/`zeros`, so filling
        # them in place touches no caller's array.
        det = backend.fill_at_indices(
            det, todo, backend.where(held[todo], det[todo], new_det)
        )
        sign = backend.fill_at_indices(sign, todo, new_sign)
        own = backend.flatnonzero(cached[todo])
        cache = cache_set_jacobian(cache, slot[todo][own], new_det[own], new_sign[own])
    return cache, keys, index.reshape(n, 6), det, sign, n_called
