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
from typing import NamedTuple, Tuple

from ...backend_obj import ArrayLike, backend
from .adaptive import CHILD_VERTEX_INDICES

__all__ = (
    "CriticalCurves",
    "child_segments",
    "crossing_points",
    "chain_order",
    "trace_band",
    "mesh_critical_curves",
)


# `CHILD_VERTEX_INDICES` as one backend int64 array, so `child_segments`
# gathers all four children of every leaf with one fancy-indexing op, as
# `red_split` does.
_CHILD_TABLE = backend.as_array(CHILD_VERTEX_INDICES, dtype=backend.int64)


class CriticalCurves(NamedTuple):
    """
    Critical curves and their caustics, one ordered polyline per curve, CSR.

    Curve ``c`` is rows ``offsets[c]:offsets[c + 1]`` of ``lens`` and
    ``source``. Travel keeps ``det A > 0`` on the left -- which runs a
    tangential curve clockwise and a radial one counter-clockwise on a typical
    lens. A curve the fov or missing data cuts is open; ``closed`` marks the
    loops, whose last point joins back to their first. Curves are ordered by
    their lowest-indexed crossing, deterministically but with no physical
    meaning, and a loop starts there.

    When ``det A`` is exactly zero at a sample, several crossings sit on
    that sample -- which happens whenever a curve runs through lattice
    points; an SIS with ``Rein = 1`` centred on the origin hits ``(1, 0)``,
    ``(-1, 0)``, ``(0, 1)`` and ``(0, -1)`` exactly. Consecutive points can
    then coincide, so some segments of ``lens`` and ``source`` have zero
    length; they are kept, not removed, so anything computing tangents,
    normals or arc length from consecutive differences must guard against a
    zero-length segment. Where ``det A`` only touches zero at one sample
    without changing sign -- an isolated degenerate critical point -- the
    zero-is-positive rule yields a closed curve of zero extent, every point
    of it at that one sample.

    Parameters
    ----------
    lens: ArrayLike
        Critical-curve points, shape ``(P, 2)``, at the mesh dtype.

        *Unit: arcsec*
    source: ArrayLike
        The matching caustic points, shape ``(P, 2)``, at the mesh dtype.

        *Unit: arcsec*
    offsets: ArrayLike
        ``(C + 1,)`` int64 CSR offsets, ``offsets[0] == 0``.
    closed: ArrayLike
        ``(C,)`` bool, True where the curve is a loop.
    """

    lens: ArrayLike
    source: ArrayLike
    offsets: ArrayLike
    closed: ArrayLike


def _sorted_pair(a, b) -> ArrayLike:
    """``(min, max)`` of two index arrays, shape ``(K,) -> (K, 2)``."""
    return backend.stack((backend.minimum(a, b), backend.maximum(a, b)), dim=-1)


def child_segments(samples, det) -> Tuple[ArrayLike, ArrayLike]:
    """
    The oriented zero-crossing segment of ``det A`` on each red-split child.

    A sample's class is ``det >= 0``, so an exact zero counts as positive --
    a symbolic perturbation that gives every sample a strict side and so
    every child a well-defined topology. A child whose three corners are not
    all of one class has exactly one odd corner ``i``, the lone positive or
    the lone negative, and with ``(i, j, k)`` cyclic the curve crosses its
    edges ``(i, j)`` and ``(k, i)``.

    The segment is oriented to keep ``det A > 0`` on its left: from ``(i, j)``
    to ``(k, i)`` when the odd corner is positive, and the reverse when it is
    negative. Children share the parent's positive orientation, and a child
    edge is crossed in opposite directions by the two children on either side
    of it, so under this rule the segment that ends on a shared edge is always
    met by one that starts there -- which is what makes the crossings chain.

    Parameters
    ----------
    samples: ArrayLike
        ``(F, 6)`` int64, ``CriticalBand.samples``.
    det: ArrayLike
        ``(S,)`` float64, ``CriticalBand.det``.

    Returns
    -------
    start: ArrayLike
        ``(K,)`` segments' starting edges, shape ``(K, 2)`` int64 sample
        pairs, smaller index first.
    end: ArrayLike
        Their ending edges, shape ``(K, 2)``, in the same form.
    """
    kids = samples[:, _CHILD_TABLE].reshape(-1, 3)
    positive = det[kids] >= 0
    n_positive = backend.sum(backend.long(positive), dim=1)
    mixed = backend.flatnonzero((n_positive == 1) | (n_positive == 2))
    kids, positive = kids[mixed], positive[mixed]
    odd_positive = n_positive[mixed] == 1
    # The odd corner is the one True entry of `positive` where one corner is
    # positive, and the one False entry where two are.
    odd_mask = backend.where(backend.unsqueeze(odd_positive, -1), positive, ~positive)
    odd = backend.argmax(backend.long(odd_mask), 1)
    rows = backend.arange(kids.shape[0], dtype=backend.int64)
    p_i = kids[rows, odd]
    p_j = kids[rows, (odd + 1) % 3]
    p_k = kids[rows, (odd + 2) % 3]
    e_ij = _sorted_pair(p_i, p_j)
    e_ki = _sorted_pair(p_k, p_i)
    flip = backend.unsqueeze(odd_positive, -1)
    return backend.where(flip, e_ij, e_ki), backend.where(flip, e_ki, e_ij)


def crossing_points(edges, lens, source, det) -> Tuple[ArrayLike, ArrayLike]:
    """
    Where ``det A`` crosses zero on each edge, in both planes.

    The zero of the linear interpolant from ``p = edges[:, 0]`` to ``q =
    edges[:, 1]``: ``t = det[p] / (det[p] - det[q])``. The two ends have
    opposite classes, so the denominator is never zero, and ``t == 0``
    exactly when ``det[p] == 0`` -- the crossing then sits on ``p`` itself.
    The caustic point applies the same ``t`` to the images, which is the
    mesh's own piecewise-linear model of the lens map on that edge.

    Computed in float64 and returned at the dtype of ``lens`` and ``source``.

    Parameters
    ----------
    edges: ArrayLike
        ``(N, 2)`` int64 sample pairs.
    lens, source: ArrayLike
        ``(S, 2)``, ``CriticalBand.lens`` and ``CriticalBand.source``.

        *Unit: arcsec*
    det: ArrayLike
        ``(S,)`` float64, ``CriticalBand.det``.

    Returns
    -------
    lens_points: ArrayLike
        ``(N, 2)``, critical-curve points.

        *Unit: arcsec*
    source_points: ArrayLike
        ``(N, 2)``, the matching caustic points.

        *Unit: arcsec*
    """
    p, q = edges[:, 0], edges[:, 1]
    t = backend.unsqueeze(det[p] / (det[p] - det[q]), -1)
    points = []
    for plane in (lens, source):
        x = backend.to(plane, dtype=backend.float64)
        points.append(backend.to(x[p] + t * (x[q] - x[p]), dtype=plane.dtype))
    return points[0], points[1]


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


def _no_curves(band) -> CriticalCurves:
    """The :class:`CriticalCurves` of a band with no crossing."""
    device = backend.device(band.det)
    return CriticalCurves(
        lens=backend.zeros((0, 2), dtype=band.lens.dtype, device=device),
        source=backend.zeros((0, 2), dtype=band.source.dtype, device=device),
        offsets=backend.zeros((1,), dtype=backend.int64, device=device),
        closed=backend.zeros((0,), dtype=backend.bool, device=device),
    )


def trace_band(band) -> CriticalCurves:
    """
    Trace the zero set of ``det A`` through a :class:`CriticalBand`.

    Each distinct crossing edge is one node, keyed by its sample pair; each
    child's segment links its starting node to its ending one; and
    :func:`chain_order` orders the result. Each node's point is computed once,
    so the two children sharing an edge agree on it exactly.

    Parameters
    ----------
    band: CriticalBand

    Returns
    -------
    CriticalCurves

    Raises
    ------
    AssertionError
        "a critical-curve crossing has two successors" if a crossing has two
        successors, or "a critical-curve crossing has two predecessors" if
        one has two predecessors. On a conforming, positively oriented band
        with one ``det`` per sample neither can happen, so this guards
        against silent corruption rather than a reachable input.
    """
    start, end = child_segments(band.samples, band.det)
    k = start.shape[0]
    if k == 0:
        return _no_curves(band)

    s = band.lens.shape[0]
    keys = backend.concatenate(
        (start[:, 0] * s + start[:, 1], end[:, 0] * s + end[:, 1]), dim=0
    )
    nodes, inverse = backend.unique(keys, return_inverse=True)
    frm, to = inverse[:k], inverse[k:]
    n = nodes.shape[0]
    # `raise AssertionError` rather than a bare `assert`, as elsewhere in the
    # adaptive kernels: `python -O` strips bare asserts. Two separate guards,
    # so each reports which side of the chain actually broke.
    if bool(backend.any(backend.bincount(frm, minlength=n) > 1)):
        raise AssertionError("a critical-curve crossing has two successors")
    if bool(backend.any(backend.bincount(to, minlength=n) > 1)):
        raise AssertionError("a critical-curve crossing has two predecessors")

    device = backend.device(band.det)
    succ = backend.fill_at_indices(
        backend.zeros((n,), dtype=backend.int64, device=device) - 1, frm, to
    )
    order, offsets, closed = chain_order(succ)
    ordered = nodes[order]
    edges = backend.stack((ordered // s, ordered % s), dim=-1)
    lens_points, source_points = crossing_points(
        edges, band.lens, band.source, band.det
    )
    return CriticalCurves(
        lens=lens_points, source=source_points, offsets=offsets, closed=closed
    )


def mesh_critical_curves(mesh) -> CriticalCurves:
    """
    Critical curves and caustics of the lens an adaptive mesh was built from.

    A pure function of ``mesh.critical_band``: no raytrace and no Jacobian
    call. Each lens-plane point lies on a child edge whose ends straddle the
    curve, so where ``det A`` is continuous it is within one child edge of
    the true critical curve -- at most ``mesh.min_img_sep / 2``, a quarter of
    the ``min_img_sep`` passed to the build, unless ``max_depth`` bound, when
    it is half the actual ``max_level`` leaf edge. Each caustic point is the
    image interpolated along the same edge; for exact images, raytrace
    ``lens`` directly.

    A curve ends where the band does: at the fov boundary, or next to a leaf
    whose samples or Jacobian are non-finite, or, in principle, next to a
    coarser converged leaf. Pseudo-caustics of lenses with a singular centre
    are not zero sets of ``det A`` and are not traced.

    Where ``det A`` is exactly zero at a sample -- as it is at ``(1, 0)``,
    ``(-1, 0)``, ``(0, 1)`` and ``(0, -1)`` for an SIS with ``Rein = 1``
    centred on the origin, for instance -- several crossings can sit on that
    sample, so consecutive points of the returned curves can coincide
    exactly and some segments have zero length. They are kept, not removed;
    see :class:`CriticalCurves` for the guard this implies for callers
    computing tangents, normals or arc length. An isolated degenerate
    critical point -- where ``det A`` touches zero at one sample without
    changing sign -- comes back as a closed curve of zero extent, every
    point of it at that one sample.

    Parameters
    ----------
    mesh: AdaptiveMesh

    Returns
    -------
    CriticalCurves
    """
    return trace_band(mesh.critical_band)
