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

A mesh built with ``centres`` also holds holes around them
(:class:`~caustics.lenses.func.adaptive.CentreHoles`): the lens map can jump
at a lens centre, so the curves are cut at each hole's circle and re-joined
along the stored hole curve (:func:`join_at_holes`). That too is a function
of the mesh alone.
"""

import math
from typing import NamedTuple, Tuple

from ...backend_obj import ArrayLike, backend
from .adaptive import CHILD_VERTEX_INDICES, CentreHoles, empty_holes

__all__ = (
    "CriticalCurves",
    "child_segments",
    "crossing_points",
    "chain_order",
    "trace_band",
    "join_at_holes",
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

    On a mesh built with ``centres``, a curve that reaches a hole does not run
    into it: it follows the hole's circle, clockwise, to the next curve leaving
    it, so every loop bounds a ``det A > 0`` region with the holes cut out, and
    its caustic follows the hole curve -- the pseudo-caustic at an isothermal
    centre -- instead of cutting straight across it. Those points carry their
    hole's index in ``hole``. A hole curve is a pseudo-caustic only where its
    ``holes.growth`` is about 0; see
    :class:`~caustics.lenses.func.adaptive.CentreHoles`.

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
    hole: ArrayLike
        ``(P,)`` int64: -1 where the point is a crossing of ``det A = 0``,
        otherwise the index into ``holes`` of the hole whose circle the curve
        follows there. Such a point lies on that circle, not on a critical
        curve, and its ``source`` lies on the hole curve.
    holes: CentreHoles
        The holes the curves were cut and re-joined at: the mesh's own, or
        an empty one.
    """

    lens: ArrayLike
    source: ArrayLike
    offsets: ArrayLike
    closed: ArrayLike
    hole: ArrayLike
    holes: CentreHoles


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
        hole=backend.zeros((0,), dtype=backend.int64, device=device),
        holes=empty_holes(device),
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
        ``hole`` is -1 at every point and ``holes`` is empty: tracing knows
        nothing of holes; :func:`join_at_holes` applies them.

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
        lens=lens_points,
        source=source_points,
        offsets=offsets,
        closed=closed,
        hole=backend.zeros((lens_points.shape[0],), dtype=backend.int64, device=device)
        - 1,
        holes=empty_holes(device),
    )


def _turn(angle) -> ArrayLike:
    """``angle`` wrapped into ``[0, 2 pi)``."""
    two_pi = 2.0 * math.pi
    return angle - two_pi * backend.floor(angle / two_pi)


def join_at_holes(curves, holes) -> CriticalCurves:
    """
    Cut traced curves at hole circles and re-join them along the hole curves.

    Inside a hole the lens map can jump: the image of the hole's boundary
    circle is a whole curve, the hole curve, so a traced curve's points
    there -- interpolated across the jump -- mean nothing. They are replaced:

    1. Every point strictly inside a hole's disk is dropped, which cuts each
       curve into segments; a curve with no point left is dropped whole. A
       lone point between two points of the same hole counts as inside it
       too.
    2. A segment that starts right after a dropped point departs from that
       hole's circle; one that ends right before a dropped point arrives at
       it. Other curve ends -- the fov, or a band gap outside every hole --
       stay ends. Disks need only not overlap, so a traced step can go from
       a point of one hole straight to a point of another. That step is a
       *bridge*: a segment with no point of its own, which departs the
       first hole, at the angle of the step's point past its circle, and
       arrives at the second, at the angle of the step's point before its
       circle.
    3. Around each circle the ends, sorted by angle, alternate between
       arriving and departing. Each arriving end is joined to the next end
       clockwise, always a departing one: an arriving curve keeps
       ``det A > 0`` on its left, which is clockwise of it, so the join runs
       through a ``det A > 0`` wedge. The join inserts the hole's stored
       samples strictly inside that clockwise interval, their ``lens`` on the
       circle and their ``source`` on the hole curve, with ``hole`` set to the
       hole's index. A hole whose ends do not alternate is left unjoined, and
       its curves stay open. A hole the fov cuts may be left unjoined, or may
       be joined across the part of its circle outside the fov, where
       branches the mesh never traced can end. Its curves are therefore
       reliable only once the fov contains the whole hole, which
       :func:`~caustics.lenses.func.adaptive.extend_adaptive_mesh` can
       arrange.
    4. The segments are chained through their joins by :func:`chain_order`,
       numbered by their first traced point, and the bridges after them, so
       the curves keep :func:`trace_band`'s order. A curve of bridges alone
       that its joins leave without a point is dropped.

    Every returned loop is then the boundary of a ``det A > 0`` region with
    the holes cut out, whatever pairing the tracer made inside a hole, and no
    caustic runs straight across a hole curve. A curve through a centre on a
    lattice point, whose band has a gap there, comes out closed.

    A traced segment that clips a disk with neither end inside it is not
    seen; the error is within a leaf edge of the hole.

    Parameters
    ----------
    curves: CriticalCurves
        As :func:`trace_band` returns them.
    holes: CentreHoles

    Returns
    -------
    CriticalCurves
        With ``holes`` set to ``holes``.

    Raises
    ------
    AssertionError
        "a segment has two predecessors" if two arriving ends were joined to
        one departing end, which alternation rules out; it guards against
        silent corruption rather than a reachable input.
    """
    n_points = curves.lens.shape[0]
    n_holes = holes.centres.shape[0]
    if n_points == 0 or n_holes == 0:
        return curves._replace(holes=holes)
    device = backend.device(curves.lens)
    int64, f64 = backend.int64, backend.float64
    lens = backend.to(curves.lens, dtype=f64)
    centres = backend.to(holes.centres, dtype=f64)

    # 1. Tag every point strictly inside a hole's disk; disks are disjoint.
    inside = backend.norm(
        backend.unsqueeze(lens, 1) - backend.unsqueeze(centres, 0), dim=-1
    ) < backend.unsqueeze(holes.radius, 0)
    tag = backend.where(
        backend.any(inside, dim=1), backend.argmax(backend.long(inside), 1), -1
    )

    # Per-point curve bookkeeping, shared by the retag below and the rotation
    # that follows it.
    counts = curves.offsets[1:] - curves.offsets[:-1]
    curve = backend.repeat(
        backend.arange(counts.shape[0], dtype=int64, device=device), counts, axis=0
    )
    first = curves.offsets[:-1][curve]
    last = curves.offsets[1:][curve] - 1
    point = backend.arange(n_points, dtype=int64, device=device)
    curve_closed = curves.closed[curve]
    prev = backend.where(point == first, last, point - 1)
    nxt = backend.where(point == last, first, point + 1)

    # 1b. The tracer's zigzag along a circle can leave a lone crossing point
    # just outside the disk, within a leaf edge of it; untagged, its two ends
    # would sit at one angle and break the alternation below. Count it as
    # inside the hole too: an open curve's first or last point, missing one
    # of the two neighbours this needs, is never retagged this way.
    tagged = tag >= 0
    has_prev = (point != first) | curve_closed
    has_next = (point != last) | curve_closed
    lone = ~tagged & has_prev & has_next & (tag[prev] >= 0) & (tag[prev] == tag[nxt])
    tag = backend.where(lone, tag[prev], tag)
    tagged = tag >= 0

    # 2. Rotate each closed curve that enters a hole to start right after a
    # tagged run, so that no run of untagged points wraps round its end.
    after_run = backend.flatnonzero(~tagged & tagged[prev] & curve_closed)
    start = backend.copy(curves.offsets[:-1])
    if after_run.shape[0]:
        run_curve = curve[after_run]
        lead = backend.concatenate(
            (
                backend.ones((1,), dtype=backend.bool, device=device),
                run_curve[1:] != run_curve[:-1],
            ),
            dim=0,
        )
        start = backend.fill_at_indices(start, run_curve[lead], after_run[lead])
    perm = first + (start[curve] - first + point - first) % counts[curve]
    tag = tag[perm]
    tagged = tag >= 0

    # 3. Segments: the maximal runs of untagged points of each curve. Found in
    # rotated order, they are already numbered by their first traced point: a
    # closed curve now starts at its first point after a tagged run, and every
    # segment of it starts after one, so none starts before that point.
    is_first = ~tagged & ((point == first) | tagged[backend.clamp(point - 1, 0, None)])
    is_last = ~tagged & (
        (point == last) | tagged[backend.clamp(point + 1, None, n_points - 1)]
    )
    seg_first = backend.flatnonzero(is_first)
    seg_last = backend.flatnonzero(is_last)
    if seg_first.shape[0] == 0:
        return CriticalCurves(
            lens=curves.lens[:0],
            source=curves.source[:0],
            offsets=backend.zeros((1,), dtype=int64, device=device),
            closed=curves.closed[:0],
            hole=curves.hole[:0],
            holes=holes,
        )

    # 3b. Bridges. Disks need only not overlap, so a traced step can go from
    # a point of one hole straight to a point of another. That step is a
    # segment with no traced point, which departs the first hole and arrives
    # at the second: it runs from the step's second point back to its first,
    # so its departure takes the angle of the point past the first circle, its
    # arrival that of the point before the second, and it assembles to nothing
    # but its join's arc. Only a curve with a segment has bridges, and after
    # the rotation none of them crosses a curve's wrap. Bridges are numbered
    # after every real segment, in order along the curves, so a cycle holding
    # a real segment still starts at one.
    tag_next = tag[backend.clamp(point + 1, None, n_points - 1)]
    has_segment = backend.bincount(curve[seg_first], minlength=counts.shape[0]) > 0
    bridge = backend.flatnonzero(
        (point != last)
        & tagged
        & (tag_next >= 0)
        & (tag_next != tag)
        & has_segment[curve]
    )
    seg_first = backend.concatenate((seg_first, bridge + 1), dim=0)
    seg_last = backend.concatenate((seg_last, bridge), dim=0)
    n_seg = seg_first.shape[0]
    seg_closed = curves.closed[curve[seg_first]]
    starts_curve = seg_first == first[seg_first]
    ends_curve = seg_last == last[seg_last]
    before = backend.where(starts_curve, last[seg_first], seg_first - 1)
    after = backend.where(ends_curve, first[seg_last], seg_last + 1)
    departs = backend.where(starts_curve & ~seg_closed, -1, tag[before])
    arrives = backend.where(ends_curve & ~seg_closed, -1, tag[after])
    # A closed curve that never enters a hole is one segment, its own successor.
    whole = seg_closed & starts_curve & ends_curve
    succ = backend.where(whole, backend.arange(n_seg, dtype=int64, device=device), -1)

    # 4. On each circle, join every arriving end to the next end clockwise.
    lens_rot = lens[perm]
    hole_offsets = backend.to_numpy(holes.offsets).tolist()
    arcs = {}
    for h in range(n_holes):
        arr = backend.flatnonzero(arrives == h)
        dep = backend.flatnonzero(departs == h)
        m = arr.shape[0] + dep.shape[0]
        if m == 0:
            continue
        seg = backend.concatenate((arr, dep), dim=0)
        arriving = backend.concatenate(
            (
                backend.ones((arr.shape[0],), dtype=backend.bool, device=device),
                backend.zeros((dep.shape[0],), dtype=backend.bool, device=device),
            ),
            dim=0,
        )
        rel = lens_rot[backend.concatenate((seg_last[arr], seg_first[dep]), dim=0)]
        rel = rel - centres[h]
        phi = _turn(backend.arctan2(rel[:, 1], rel[:, 0]))
        o = backend.argsort(phi)
        seg, arriving, phi = seg[o], arriving[o], phi[o]
        if m % 2 or bool(backend.any(arriving == backend.roll(arriving, 1, 0))):
            continue
        k_arr = backend.flatnonzero(arriving)
        k_dep = (k_arr - 1) % m
        succ = backend.fill_at_indices(succ, seg[k_arr], seg[k_dep])
        lo, hi = hole_offsets[h], hole_offsets[h + 1]
        angle = holes.angle[lo:hi]
        for ka, kd in zip(
            backend.to_numpy(k_arr).tolist(), backend.to_numpy(k_dep).tolist()
        ):
            span = _turn(phi[ka] - phi[kd])
            delta = _turn(phi[ka] - angle)
            pick = backend.flatnonzero((delta > 0) & (delta < span))
            pick = pick[backend.argsort(delta[pick])]
            arcs[int(backend.to_numpy(seg[ka]))] = lo + pick
    # `raise AssertionError` rather than a bare `assert`, as elsewhere here:
    # `python -O` strips bare asserts.
    joined = backend.flatnonzero(succ >= 0)
    if bool(backend.any(backend.bincount(succ[joined], minlength=n_seg) > 1)):
        raise AssertionError("a segment has two predecessors")

    # 5. Chain the segments through their joins into curves.
    order, seg_offsets, closed = chain_order(succ)

    # 6. Assemble: every segment's points, then its join's arc.
    pool_lens = backend.concatenate(
        (curves.lens[perm], backend.to(holes.lens, dtype=curves.lens.dtype)), dim=0
    )
    pool_source = backend.concatenate(
        (curves.source[perm], backend.to(holes.source, dtype=curves.source.dtype)),
        dim=0,
    )
    samples = holes.offsets[1:] - holes.offsets[:-1]
    pool_hole = backend.concatenate(
        (
            backend.zeros((n_points,), dtype=int64, device=device) - 1,
            backend.repeat(
                backend.arange(n_holes, dtype=int64, device=device), samples, axis=0
            ),
        ),
        dim=0,
    )
    first_host = backend.to_numpy(seg_first).tolist()
    last_host = backend.to_numpy(seg_last).tolist()
    pieces, sizes = [], []
    for s in backend.to_numpy(order).tolist():
        pieces.append(
            backend.arange(first_host[s], last_host[s] + 1, dtype=int64, device=device)
        )
        size = last_host[s] + 1 - first_host[s]
        if s in arcs:
            pieces.append(n_points + arcs[s])
            size += int(arcs[s].shape[0])
        sizes.append(size)
    gather = backend.concatenate(pieces, dim=0)
    csum = backend.concatenate(
        (
            backend.zeros((1,), dtype=int64, device=device),
            backend.cumsum(backend.as_array(sizes, dtype=int64, device=device), dim=0),
        ),
        dim=0,
    )
    offsets = csum[seg_offsets]
    # A curve of bridges alone, whose joins inserted no sample, has no point:
    # it is dropped, as a curve wholly inside a hole is. Every other curve
    # holds a segment's traced points or an arc's samples.
    kept = backend.flatnonzero(offsets[1:] > offsets[:-1])
    if kept.shape[0] < closed.shape[0]:
        offsets = backend.concatenate((offsets[:1], offsets[1:][kept]), dim=0)
        closed = closed[kept]
    return CriticalCurves(
        lens=pool_lens[gather],
        source=pool_source[gather],
        offsets=offsets,
        closed=closed,
        hole=pool_hole[gather],
        holes=holes,
    )


def mesh_critical_curves(mesh) -> CriticalCurves:
    """
    Critical curves and caustics of the lens an adaptive mesh was built from.

    A pure function of ``mesh.critical_band`` and ``mesh.holes``: no raytrace
    and no Jacobian call. Each lens-plane point lies on a child edge whose
    ends straddle the curve, so where ``det A`` is continuous it is within
    one child edge of the true critical curve -- at most
    ``mesh.min_img_sep / 2``, a quarter of the ``min_img_sep`` passed to the
    build, unless ``max_depth`` bound, when it is half the actual
    ``max_level`` leaf edge. Each caustic point is the image interpolated
    along the same edge; for exact images, raytrace ``lens`` directly.

    A curve ends where the band does: at the fov boundary, or next to a leaf
    whose samples or Jacobian are non-finite, or, in principle, next to a
    coarser converged leaf -- unless that happens inside a hole, where the
    curve is joined instead.

    A mesh built with ``centres`` has holes around them, and the traced
    curves are cut at each hole's circle and re-joined along the stored hole
    curve by :func:`join_at_holes`; the hole curves -- the pseudo-caustics,
    where ``holes.growth`` is about 0 -- come back as ``holes``. Without
    them, a curve through a singular centre, where the lens map jumps, gets
    a caustic that cuts straight across the pseudo-caustic, and one through
    a centre on a lattice point comes out open. Either way this function
    makes no lens call.

    A curve the fov cuts can be closed by growing the mesh with
    :func:`~caustics.lenses.func.adaptive.extend_adaptive_mesh`, which reuses
    every lens evaluation already made.

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
        With ``holes`` set to ``mesh.holes``.
    """
    curves = trace_band(mesh.critical_band)
    if mesh.holes.centres.shape[0] == 0:
        return curves._replace(holes=mesh.holes)
    return join_at_holes(curves, mesh.holes)
