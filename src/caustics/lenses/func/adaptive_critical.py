"""
Critical curves and caustics, traced through an adaptive mesh's critical band.

:func:`~caustics.lenses.func.adaptive.build_adaptive_mesh` keeps, as
``mesh.critical_band``, every ``max_level`` leaf ``det A`` changes sign across,
with ``det A`` and the image at its six samples. Those six samples are the
corners of the leaf's four red-split children, so ``det A`` at them defines a
piecewise-linear field on the children, and this module traces its zero set:
the critical curves, and -- interpolating the images the same way -- their
caustics. Nothing here calls the lens.

Every crossing lies on a child edge whose two ends have opposite classes, so
where ``det A`` is continuous the true critical curve crosses the same edge:
each lens-plane point is within one child edge, half a leaf edge, of the
curve.
"""

import math
from typing import Tuple

from ...backend_obj import ArrayLike, backend

__all__ = ("chain_order",)


def chain_order(succ) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
    """
    Order nodes along the paths and cycles of a successor array.

    Vectorized list ranking by pointer jumping: ``ceil(log2(K))`` rounds of
    gathers, with no Python loop over nodes.

    1. Every node follows its successor, a tail following itself, while
       taking the minimum node id seen. Once ``2**rounds >= K``, a node on a
       path points at its tail, which has no successor, while a node on a
       cycle still points into the cycle, whose minimum it now holds.
    2. Every path starts at its node without a predecessor. Every cycle is
       cut at its smallest node, which becomes its start.
    3. Wyllie's list ranking over predecessors then gives each node its start
       and its distance from it.

    The predecessor array is filled from successors that are unique, so no
    scatter ever writes a repeated index -- where torch keeps the last write
    and jax an arbitrary one.

    Parameters
    ----------
    succ: ArrayLike
        ``(K,)`` int64 successor of each node, ``-1`` for none. No node may
        have two predecessors.

    Returns
    -------
    order: ArrayLike
        ``(K,)`` int64 permutation listing the nodes curve by curve, each in
        travel order.
    offsets: ArrayLike
        ``(C + 1,)`` int64 CSR offsets into ``order``.
    closed: ArrayLike
        ``(C,)`` bool, True where the curve is a cycle.
    """
    k = succ.shape[0]
    device = backend.device(succ)
    int64 = backend.int64
    if k == 0:
        return (
            backend.zeros((0,), dtype=int64, device=device),
            backend.zeros((1,), dtype=int64, device=device),
            backend.zeros((0,), dtype=backend.bool, device=device),
        )
    node = backend.arange(k, dtype=int64, device=device)
    has_next = succ >= 0
    pred = backend.fill_at_indices(
        backend.zeros((k,), dtype=int64, device=device) - 1,
        succ[has_next],
        node[has_next],
    )
    rounds = math.ceil(math.log2(max(k, 2)))

    nxt = backend.where(has_next, succ, node)
    low = node
    for _ in range(rounds):
        low = backend.minimum(low, low[nxt])
        nxt = nxt[nxt]
    on_cycle = has_next[nxt]

    start = (pred < 0) | (on_cycle & (low == node))
    pred = backend.where(start, -1, pred)

    back = backend.where(pred >= 0, pred, node)
    dist = backend.long(pred >= 0)
    for _ in range(rounds):
        dist = dist + dist[back]
        back = back[back]

    # `lexsort`'s last key is primary: group by start node, then travel order.
    order = backend.lexsort([dist, back])
    starts = backend.flatnonzero(start)
    counts = backend.bincount(back, minlength=k)[starts]
    offsets = backend.concatenate(
        (
            backend.zeros((1,), dtype=int64, device=device),
            backend.cumsum(counts, dim=0),
        ),
        dim=0,
    )
    return order, offsets, on_cycle[starts]
