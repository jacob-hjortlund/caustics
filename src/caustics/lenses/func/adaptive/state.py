"""
The refinement's working state: vertex cache, active-vertex set, leaf store.

Each is a ``NamedTuple`` of backend arrays -- the active set a bool array over
vertex-cache slots -- with free functions that return a new value rather than
mutate their input. All of it lives on the build's ambient device
(:func:`_ambient`).
"""

from typing import NamedTuple, Tuple

from ....backend_obj import ArrayLike, backend

__all__ = (
    "VertexCache",
    "empty_cache",
    "cache_size",
    "cache_lookup",
    "cache_missing",
    "cache_insert",
    "empty_active",
    "active_add_slots",
    "active_contains_slots",
    "active_contains",
    "LeafStore",
    "empty_store",
    "store_add",
    "store_remove",
    "store_compact",
)


class VertexCache(NamedTuple):
    """
    Lattice key to slot, with the source-plane image of every evaluated point.

    Lookup is ``backend.searchsorted`` against a sorted key array rather than
    a Python dict, so a whole level's worth of points resolves in one
    vectorized call. Slots are assigned monotonically in order of first
    evaluation and never move.

    ``ij`` and ``beta`` hold the lattice coordinates and source-plane image of
    every evaluated point, shape ``(N, 2)`` and indexed by slot. ``keys`` and
    ``slots`` are the sorted-key index: ``keys`` is ascending, and
    ``slots[i]`` is the slot of ``keys[i]``.
    """

    keys: ArrayLike
    slots: ArrayLike
    ij: ArrayLike
    beta: ArrayLike


def empty_cache(device=None) -> VertexCache:
    """An empty :class:`VertexCache`."""
    return VertexCache(
        keys=backend.empty((0,), dtype=backend.int64, device=device),
        slots=backend.empty((0,), dtype=backend.int64, device=device),
        ij=backend.empty((0, 2), dtype=backend.int64, device=device),
        beta=backend.empty((0, 2), dtype=backend.float64, device=device),
    )


def cache_size(cache) -> int:
    """Number of vertices held in the cache."""
    return int(cache.ij.shape[0])


def cache_lookup(cache, keys) -> ArrayLike:
    """Slot of each key, or ``-1`` where absent."""
    if cache.keys.shape[0] == 0:
        return backend.zeros_like(keys) - 1
    n = cache.keys.shape[0]
    pos = backend.clamp(backend.searchsorted(cache.keys, keys), 0, n - 1)
    return backend.where(cache.keys[pos] == keys, cache.slots[pos], -1)


def cache_missing(cache, keys) -> ArrayLike:
    """Unique keys not yet evaluated, ascending."""
    uniq = backend.unique(keys)
    return uniq[cache_lookup(cache, uniq) < 0]


def cache_insert(cache, keys, ij, beta) -> Tuple[VertexCache, ArrayLike]:
    """
    Assign slots to new keys. ``keys`` must be unique, absent, and sorted.

    The key index is merged rather than re-sorted. Both sides are already
    sorted -- ``keys`` comes from :func:`cache_missing`, which returns
    ``backend.unique`` output, and ``cache.keys`` is maintained sorted -- and
    no key appears on both sides, so ``searchsorted`` plus a running offset
    gives each new key its position in the merged array outright. The result
    is identical to sorting the concatenation, because with all keys distinct
    the sorted order is unique.

    Returns
    -------
    VertexCache
        The updated cache.
    ArrayLike
        The slots assigned to ``keys``, in the order given.
    """
    start = cache.ij.shape[0]
    k = int(keys.shape[0])
    slots = backend.arange(start, start + k, dtype=backend.int64)

    new_ij = backend.concatenate([cache.ij, ij], dim=0)
    new_beta = backend.concatenate([cache.beta, beta], dim=0)

    total = cache.keys.shape[0] + k
    dest = backend.searchsorted(cache.keys, keys) + backend.arange(
        k, dtype=backend.int64
    )
    new_keys = backend.fill_at_indices(
        backend.empty((total,), dtype=backend.int64), dest, keys
    )
    new_slots = backend.fill_at_indices(
        backend.empty((total,), dtype=backend.int64), dest, slots
    )
    # The complement of `dest`: where the pre-existing keys/slots land in the
    # merged array. Computed as a boolean mask, then converted to indices with
    # `flatnonzero` so the merge only ever calls `fill_at_indices` with
    # integer positions.
    stay = backend.fill_at_indices(
        backend.ones((total,), dtype=backend.bool), dest, False
    )
    old_dest = backend.flatnonzero(stay)
    new_keys = backend.fill_at_indices(new_keys, old_dest, cache.keys)
    new_slots = backend.fill_at_indices(new_slots, old_dest, cache.slots)

    return (
        VertexCache(keys=new_keys, slots=new_slots, ij=new_ij, beta=new_beta),
        slots,
    )


def empty_active(device=None) -> ArrayLike:
    """
    Lattice points that are currently vertices of some triangle in the mesh.

    Stored as one flag per **vertex-cache slot**, not as a sorted key set.
    Every active key is by construction a vertex of some triangle, so it has
    already been evaluated and is already in the cache -- a second sorted
    structure would duplicate the cache's own key index, and keeping it
    sorted cost an ``np.union1d`` over the whole active set on every
    insertion, which was the single largest term in the build.

    Separate from the vertex cache, which also holds midpoints of
    tested-but-never-split triangles. Only ever grows, since a parent's
    vertices are inherited by all its children.
    """
    return backend.zeros((0,), dtype=backend.bool, device=device)


def active_add_slots(active, n_slots, slots) -> ArrayLike:
    """
    Activate vertex-cache slots.

    Takes slots rather than keys because every caller already holds them:
    re-keying a triangle's vertices only to look them up again is exactly the
    work this function exists to avoid.
    """
    # Trip-wire for the `active subset of cache` invariant. A -1 slot -- what
    # `cache_lookup` returns for an absent key -- would negative-index into
    # the last cache entry and activate the wrong vertex, and the mesh would
    # come out unbalanced rather than raising.
    #
    # `raise AssertionError` rather than a bare `assert`: `python -O` strips
    # bare asserts, and this guards against silent geometric corruption.
    if not bool(backend.all(slots >= 0)):
        raise AssertionError("cannot activate an uncached vertex")
    # Concatenate unconditionally, even when `extra` is 0, so this always
    # scatters into a fresh array and never the caller's own. `torch.cat`/
    # `jnp.concatenate` both always allocate, unlike `fill_at_indices` on its
    # own: torch mutates its first argument in place and returns it, so
    # scattering directly into `active` on the no-growth path would silently
    # clobber the caller's array under torch while leaving it untouched under
    # jax -- the exact backend-divergence `cache_insert` avoids by always
    # writing into a fresh `backend.empty`/`backend.ones` buffer.
    extra = max(n_slots - active.shape[0], 0)
    active = backend.concatenate(
        [active, backend.zeros((extra,), dtype=backend.bool)], dim=0
    )
    return backend.fill_at_indices(active, slots, True)


def active_contains_slots(active, slots) -> ArrayLike:
    """True where the slot is an active vertex. ``-1`` reads as False."""
    if active.shape[0] == 0:
        return backend.zeros(slots.shape, dtype=backend.bool)
    present = slots >= 0
    return present & active[backend.where(present, slots, 0)]


def active_contains(active, cache, keys) -> ArrayLike:
    """
    True where the key is an active vertex.

    A key absent from the cache was never evaluated, so it cannot be a
    triangle vertex and cannot be active -- ``cache_lookup`` returns ``-1``
    and :func:`active_contains_slots` reads that as False.
    """
    return active_contains_slots(active, cache_lookup(cache, keys))


class LeafStore(NamedTuple):
    """
    Terminal triangles, keyed by row index with a validity flag.

    Not append-only: a triangle marked converged at level ``d`` can be
    removed and replaced by descendants several levels later, when a distant
    refinement cascades back to it. Hence the flag and the single compaction
    at the end (:func:`store_compact`), rather than streaming into a flat
    array as we go.

    ``v`` holds the three vertex-cache slots of each leaf, shape ``(N, 3)``.
    ``level``, ``cls``, ``status`` and ``valid`` are per-row, shape ``(N,)``.
    ``status`` is a ``backend.int64`` bitmask of the ``LEAF_*`` flags above.
    """

    v: ArrayLike
    level: ArrayLike
    cls: ArrayLike
    status: ArrayLike
    valid: ArrayLike


def empty_store(device=None) -> LeafStore:
    """An empty :class:`LeafStore`."""
    return LeafStore(
        v=backend.empty((0, 3), dtype=backend.int64, device=device),
        level=backend.empty((0,), dtype=backend.int64, device=device),
        cls=backend.empty((0,), dtype=backend.int64, device=device),
        status=backend.empty((0,), dtype=backend.int64, device=device),
        valid=backend.empty((0,), dtype=backend.bool, device=device),
    )


def _broadcast_row_field(value, n_rows) -> ArrayLike:
    """A per-row array as-is, or a Python scalar broadcast to ``n_rows``."""
    if hasattr(value, "shape"):
        return value
    return backend.zeros((n_rows,), dtype=backend.int64) + value


def store_add(store, v, level, cls, status) -> Tuple[LeafStore, ArrayLike]:
    """
    Append triangles, returning their row indices.

    ``level``, ``cls`` and ``status`` may each be a Python scalar or a
    per-row array; a scalar is broadcast to the row count before
    concatenating.

    Returns
    -------
    LeafStore
        The updated store.
    ArrayLike
        The row indices assigned to ``v``, in the order given.
    """
    start = store.v.shape[0]
    k = v.shape[0]
    rows = backend.arange(start, start + k, dtype=backend.int64)

    new_store = LeafStore(
        v=backend.concatenate([store.v, v], dim=0),
        level=backend.concatenate([store.level, _broadcast_row_field(level, k)], dim=0),
        cls=backend.concatenate([store.cls, _broadcast_row_field(cls, k)], dim=0),
        status=backend.concatenate(
            [store.status, _broadcast_row_field(status, k)], dim=0
        ),
        valid=backend.concatenate(
            [store.valid, backend.ones((k,), dtype=backend.bool)], dim=0
        ),
    )
    return new_store, rows


def store_remove(store, rows) -> LeafStore:
    """
    Mark rows invalid, returning a NEW store; ``store`` itself is untouched.

    Scatters into a copy of ``store.valid`` rather than ``store.valid``
    itself: ``backend.fill_at_indices`` mutates its first argument in place
    and returns it under torch, so writing straight into ``store.valid``
    would silently clobber the input store's own array under torch while
    leaving it (and every alias of it a caller might still hold) untouched
    under jax -- the same backend divergence :func:`cache_insert` and
    :func:`active_add_slots` avoid by always writing into a fresh buffer.
    """
    valid = backend.fill_at_indices(backend.copy(store.valid), rows, False)
    return LeafStore(
        v=store.v, level=store.level, cls=store.cls, status=store.status, valid=valid
    )


def store_compact(store) -> Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
    """The surviving rows of ``v``, ``level``, ``cls`` and ``status``."""
    keep = backend.flatnonzero(store.valid)
    return store.v[keep], store.level[keep], store.cls[keep], store.status[keep]


def _ambient(array) -> ArrayLike:
    """``array`` on the build's ambient device, where :func:`refine` makes its arrays."""
    return backend.to(array, device=backend.device(backend.zeros((0,))))
