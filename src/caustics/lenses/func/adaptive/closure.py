"""
Canonical leaf order, and the conforming closure of a balanced mesh.
"""

from typing import Tuple

from ....backend_obj import ArrayLike, backend
from .geometry import _CHILD_VERTEX_INDEX_TABLE, min_angle
from .state import active_contains_slots, cache_lookup
from .lattice import lattice_key, lattice_xy, midpoint_ij

__all__ = (
    "canonical_order",
    "close",
)


def canonical_order(lat, cache, v) -> ArrayLike:
    """
    Deterministic leaf ordering, independent of the order the cascade produced.

    Sorts by the row-wise-sorted triple of vertex lattice keys, which is unique per
    triangle since a triangle is determined by its vertex set. This makes
    byte-identical output a property of the data rather than of control flow.

    Parameters
    ----------
    lat: Lattice
    cache: VertexCache
    v: ArrayLike
        Leaf vertex slots, shape ``(n, 3)`` int64.

    Returns
    -------
    ArrayLike
        Int64 permutation of ``0 .. n - 1`` that sorts ``v`` into canonical order.
    """
    keys = backend.sort(lattice_key(lat, cache.ij[v]), dim=1)
    # `backend.lexsort` follows NumPy's convention: the LAST key given is the
    # PRIMARY sort key, so the row's smallest key (`keys[:, 0]`) goes last.
    return backend.lexsort([keys[:, 2], keys[:, 1], keys[:, 0]])


def close(
    lat, cache, active, v, level, status
) -> Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
    """
    Make the balanced mesh conforming, using the pre-closure active-vertex set.

    Fixed pattern table, all vertices already cached:

    - 1 hanging node at ``m_i``: bisect from the opposite vertex into two triangles.
    - 2 hanging nodes: emit the corner triangle at the vertex opposite the whole
      edge, then split the remaining quadrilateral along whichever of its two
      diagonals maximizes the minimum angle.
    - 3 hanging nodes: the canonical red split, free.

    Parameters
    ----------
    lat: Lattice
    cache: VertexCache
    active: ArrayLike
        Pre-closure active-vertex set.
    v: ArrayLike
        Pre-closure leaf vertex slots in canonical order, shape ``(L0, 3)``.
    level, status: ArrayLike
        Shape ``(L0,)``, inherited by the emitted triangles.

    Returns
    -------
    leaves: ArrayLike
        ``(L, 3)`` vertex slots, positively oriented.
    origin: ArrayLike
        ``(L,)`` index into ``v``. Non-decreasing, so each origin's terminal
        triangles are contiguous and grouping is a slice.
    out_level, out_status: ArrayLike
        ``(L,)``, inherited from the origin.
    """
    v_ij = cache.ij[v]
    mid_ij = midpoint_ij(v_ij)
    mid_keys = lattice_key(lat, mid_ij)
    m = cache_lookup(cache, mid_keys)
    # No exactness gate. The lattice is one level finer than max_level, so
    # `midpoint_ij` is exact at every level and `mid_keys` always names the
    # true midpoint. At max_level that midpoint has an odd coordinate, is
    # traced transiently by `refine` and never cached, so `cache_lookup`
    # returns -1 and `active_contains_slots` reads it as False. Even were it
    # cached it could not be *active*: an active slot is by definition a
    # triangle vertex, and every triangle vertex has even coordinates. So no
    # max_level leaf has a hanging node -- the invariant `refine` relies on --
    # and the count == 0 pass-through below is still every max_level leaf's
    # only route through.
    #
    # `m` is already `cache_lookup(cache, mid_keys)`; asking by slot skips a
    # second searchsorted over the same keys.
    hanging = active_contains_slots(active, m)  # (L0, 3)
    # `v_ij` and the midpoint coordinates are dead from here: the pattern
    # tables below work in slots, and `geom` re-gathers from the cache.
    del v_ij, mid_ij, mid_keys
    count = backend.to(backend.sum(hanging, dim=1), dtype=backend.int64)

    # `count` is always one of {0, 1, 2, 3} -- three possibly-hanging edge
    # midpoints -- so `count + 1` is the same map as
    # `np.choose(count, [1, 2, 3, 4])`: 0 -> 1, 1 -> 2, 2 -> 3, 3 -> 4.
    n_children = count + 1
    offsets = backend.concatenate(
        [backend.zeros((1,), dtype=backend.int64), backend.cumsum(n_children, dim=0)],
        dim=0,
    )
    total = int(backend.to_numpy(offsets[-1]))
    leaves = backend.empty((total, 3), dtype=backend.int64)
    origin = backend.repeat(
        backend.arange(v.shape[0], dtype=backend.int64), n_children, axis=0
    )

    def geom(slots):
        return lattice_xy(lat, cache.ij[slots])

    sel = backend.flatnonzero(count == 0)
    leaves = backend.fill_at_indices(leaves, offsets[sel], v[sel])

    for i in range(3):
        j, k = (i + 1) % 3, (i + 2) % 3
        sel = backend.flatnonzero((count == 1) & hanging[:, i])
        if sel.shape[0] == 0:
            continue
        o = offsets[sel]
        leaves = backend.fill_at_indices(
            leaves, o, backend.stack((v[sel, i], v[sel, j], m[sel, i]), dim=1)
        )
        leaves = backend.fill_at_indices(
            leaves, o + 1, backend.stack((v[sel, i], m[sel, i], v[sel, k]), dim=1)
        )

    for c in range(3):
        a, b = (c + 1) % 3, (c + 2) % 3
        sel = backend.flatnonzero((count == 2) & ~hanging[:, c])
        if sel.shape[0] == 0:
            continue
        o = offsets[sel]
        # Corner triangle at theta_c: exactly red-split child C_c, so positively
        # oriented by construction.
        leaves = backend.fill_at_indices(
            leaves, o, backend.stack((v[sel, c], m[sel, b], m[sel, a]), dim=1)
        )
        a1 = backend.stack((v[sel, a], v[sel, b], m[sel, a]), dim=1)
        a2 = backend.stack((v[sel, a], m[sel, a], m[sel, b]), dim=1)
        b1 = backend.stack((v[sel, a], v[sel, b], m[sel, b]), dim=1)
        b2 = backend.stack((v[sel, b], m[sel, a], m[sel, b]), dim=1)
        score_a = backend.minimum(min_angle(geom(a1)), min_angle(geom(a2)))
        score_b = backend.minimum(min_angle(geom(b1)), min_angle(geom(b2)))
        use_b = score_b > score_a  # ties take candidate A, deterministically
        leaves = backend.fill_at_indices(
            leaves, o + 1, backend.where(use_b[:, None], b1, a1)
        )
        leaves = backend.fill_at_indices(
            leaves, o + 2, backend.where(use_b[:, None], b2, a2)
        )

    sel = backend.flatnonzero(count == 3)
    if sel.shape[0]:
        o = offsets[sel]
        six = backend.concatenate([v[sel], m[sel]], dim=1)  # (k, 6)
        # Raw concatenate + fancy-index gather, deliberately not a call to
        # `red_split`: this is the only path that reaches the count == 3
        # branch, and going through `red_split` would give it zero coverage
        # of its own from this branch.
        kids = six[:, _CHILD_VERTEX_INDEX_TABLE]  # (k, 4, 3)
        for t in range(4):
            leaves = backend.fill_at_indices(leaves, o + t, kids[:, t, :])

    return leaves, origin, level[origin], status[origin]
