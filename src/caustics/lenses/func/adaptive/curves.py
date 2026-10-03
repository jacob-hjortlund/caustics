"""
Critical curves and caustics, traced through a lens mesh's critical band.

:func:`~caustics.lenses.func.adaptive.build_lens_mesh` keeps, as
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

A mesh built with ``centers`` also holds holes around them
(:class:`~caustics.lenses.func.adaptive.CenterHoles`): the lens map can jump
at a lens center, so the curves are cut at each hole's circle and re-joined
along the stored hole curve (:func:`join_at_holes`). That too is a function
of the mesh alone.
"""

import math
from typing import NamedTuple

from ....backend_obj import ArrayLike, backend
from .geometry import _CHILD_VERTEX_INDEX_TABLE, csr_offsets


class CriticalCurvesAndCaustics(NamedTuple):
    """
    Critical curves and caustics, one ordered polyline per curve, CSR.

    Curve ``c`` is rows ``offsets[c]:offsets[c + 1]`` of ``lens``, ``source``
    and ``hole``. Most points are crossings of ``det A = 0``: a critical-curve
    point in ``lens``, its caustic point in ``source``. On a mesh with holes,
    a curve that reaches a hole follows the hole's circle clockwise to the
    next curve leaving it (:func:`join_at_holes`); those points lie on the
    circle in ``lens`` and on the hole curve in ``source``, which is the
    pseudo-caustic where the mesh's ``holes.pseudo_caustic`` is True.

    Travel keeps ``det A > 0`` on the left. A curve the fov or missing data
    cuts is open; ``closed`` marks the loops, whose last point joins back to
    their first. The order of the curves, and where a loop starts, is
    deterministic but has no physical meaning. Where ``det A`` is exactly
    zero at a sample, as when a curve runs through lattice points,
    consecutive points can coincide: zero-length segments are kept.

    Parameters
    ----------
    lens: ArrayLike
        ``(P, 2)`` float64 critical-curve or hole-circle points.

        *Unit: arcsec*
    source: ArrayLike
        ``(P, 2)`` float64 caustic or hole-curve points.

        *Unit: arcsec*
    offsets: ArrayLike
        ``(C + 1,)`` int64 CSR offsets, ``offsets[0] == 0``.
    closed: ArrayLike
        ``(C,)`` bool, True where the curve is a loop.
    hole: ArrayLike
        ``(P,)`` int64: -1 at a crossing of ``det A = 0``, otherwise the
        index into the mesh's ``holes`` of the hole whose circle the point
        lies on.
    """

    lens: ArrayLike
    source: ArrayLike
    offsets: ArrayLike
    closed: ArrayLike
    hole: ArrayLike


def _sorted_pair(a, b):
    """``(min, max)`` of two index arrays, shape ``(K,) -> (K, 2)``."""
    return backend.stack((backend.minimum(a, b), backend.maximum(a, b)), dim=-1)


def triangle_segments(tri, positive):
    """
    The oriented crossing segment of each triangle whose corners are not all of one class.

    A triangle with mixed classes has exactly one odd corner ``i``, the lone
    positive or the lone negative, and with ``(i, j, k)`` cyclic the boundary
    between the classes crosses its edges ``(i, j)`` and ``(k, i)``. The
    segment keeps the positive side on its left: from ``(i, j)`` to ``(k, i)``
    when the odd corner is positive, and the reverse when it is negative. On
    positively oriented triangles sharing an edge, the segment ending on it
    is always met by one starting there, which is what makes the segments
    chain (:func:`chain_segments`).

    Parameters
    ----------
    tri: ArrayLike
        ``(K, 3)`` int64 sample indices, positively oriented.
    positive: ArrayLike
        ``(S,)`` bool class of each sample.

    Returns
    -------
    start: ArrayLike
        ``(M,)`` segments' starting edges, shape ``(M, 2)`` int64 sample
        pairs, smaller index first.
    end: ArrayLike
        Their ending edges, shape ``(M, 2)``, in the same form.
    """
    pos = positive[tri]
    n_positive = backend.sum(backend.long(pos), dim=1)
    mixed = backend.flatnonzero((n_positive == 1) | (n_positive == 2))
    tri, pos = tri[mixed], pos[mixed]
    odd_positive = n_positive[mixed] == 1
    # The odd corner is the one True entry of `pos` where one corner is
    # positive, and the one False entry where two are.
    odd_mask = backend.where(backend.unsqueeze(odd_positive, -1), pos, ~pos)
    odd = backend.argmax(backend.long(odd_mask), 1)
    rows = backend.arange(tri.shape[0], dtype=backend.int64)
    p_i = tri[rows, odd]
    p_j = tri[rows, (odd + 1) % 3]
    p_k = tri[rows, (odd + 2) % 3]
    e_ij = _sorted_pair(p_i, p_j)
    e_ki = _sorted_pair(p_k, p_i)
    flip = backend.unsqueeze(odd_positive, -1)
    return backend.where(flip, e_ij, e_ki), backend.where(flip, e_ki, e_ij)


def child_segments(samples, det):
    """
    The oriented zero-crossing segment of ``det A`` on each red-split child.

    :func:`triangle_segments` on the four children of every band leaf. A
    sample's class is ``det >= 0``: an exact zero counts as positive, which
    gives every child a definite crossing.

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
    kids = samples[:, _CHILD_VERTEX_INDEX_TABLE].reshape(-1, 3)
    return triangle_segments(kids, det >= 0)


def edge_zeros(edges, field, planes):
    """
    Where a piecewise-linear ``field`` crosses zero on each edge, in every plane.

    The zero of the linear interpolant from ``p = edges[:, 0]`` to
    ``q = edges[:, 1]``: ``t = field[p] / (field[p] - field[q])``, applied to
    each plane's positions. The ends must have opposite classes, so the
    denominator is never zero, and ``t == 0`` exactly when ``field[p] == 0``.
    Computed in float64 and returned at each plane's dtype.

    Parameters
    ----------
    edges: ArrayLike
        ``(N, 2)`` int64 sample pairs.
    field: ArrayLike
        ``(S,)`` the field at each sample.
    planes: Tuple[ArrayLike, ...]
        ``(S, 2)`` positions of the samples, one array per plane.

        *Unit: arcsec*

    Returns
    -------
    Tuple[ArrayLike, ...]
        ``(N, 2)`` zero of each edge, one array per plane.

        *Unit: arcsec*
    """
    p, q = edges[:, 0], edges[:, 1]
    t = backend.unsqueeze(field[p] / (field[p] - field[q]), -1)
    points = []
    for plane in planes:
        x = backend.to(plane, dtype=backend.float64)
        points.append(backend.to(x[p] + t * (x[q] - x[p]), dtype=plane.dtype))
    return tuple(points)


def chain_order(succ):
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

    Successors are unique, so the scatter filling predecessors never repeats
    an index.

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
    return order, csr_offsets(counts), on_cycle[starts]


def chain_segments(start, end, n_samples):
    """
    Chain oriented segments, each from one sample-pair edge to another, into curves.

    Each distinct edge is one node, keyed by its sample pair; each segment
    links its starting node to its ending one; :func:`chain_order` orders
    the result. Shared by :func:`trace_band` and
    :func:`~caustics.lenses.func.adaptive.regions.magnified_regions`.

    Parameters
    ----------
    start, end: ArrayLike
        ``(K, 2)`` int64 sample pairs, smaller index first, as
        :func:`triangle_segments` returns them.
    n_samples: int
        Number of samples, bounding every index, for the node keys.

    Returns
    -------
    edges: ArrayLike
        ``(N, 2)`` int64 node edges, curve by curve in travel order.
    offsets: ArrayLike
        ``(C + 1,)`` int64 CSR offsets into ``edges``.
    closed: ArrayLike
        ``(C,)`` bool, True where the curve is a loop.
    """
    int64 = backend.int64
    device = backend.device(start)
    k = start.shape[0]
    if k == 0:
        return (
            backend.zeros((0, 2), dtype=int64, device=device),
            backend.zeros((1,), dtype=int64, device=device),
            backend.zeros((0,), dtype=backend.bool, device=device),
        )
    s = n_samples
    keys = backend.concatenate(
        (start[:, 0] * s + start[:, 1], end[:, 0] * s + end[:, 1]), dim=0
    )
    nodes, inverse = backend.unique(keys, return_inverse=True)
    frm, to = inverse[:k], inverse[k:]
    n = nodes.shape[0]
    succ = backend.fill_at_indices(
        backend.zeros((n,), dtype=int64, device=device) - 1, frm, to
    )
    order, offsets, closed = chain_order(succ)
    ordered = nodes[order]
    return backend.stack((ordered // s, ordered % s), dim=-1), offsets, closed


def _no_curves(band):
    """The :class:`CriticalCurvesAndCaustics` of a band with no crossing."""
    device = backend.device(band.det)
    return CriticalCurvesAndCaustics(
        lens=backend.zeros((0, 2), dtype=band.lens.dtype, device=device),
        source=backend.zeros((0, 2), dtype=band.source.dtype, device=device),
        offsets=backend.zeros((1,), dtype=backend.int64, device=device),
        closed=backend.zeros((0,), dtype=backend.bool, device=device),
        hole=backend.zeros((0,), dtype=backend.int64, device=device),
    )


def trace_band(band):
    """
    Trace the zero set of ``det A`` through a :class:`CriticalBand`.

    Each child's segment (:func:`child_segments`) is chained into curves by
    :func:`chain_segments`, each distinct crossing edge one node, keyed by
    its sample pair. Each node's point is computed once, so the two children
    sharing an edge agree on it exactly.

    Parameters
    ----------
    band: CriticalBand

    Returns
    -------
    CriticalCurvesAndCaustics
        ``hole`` is -1 at every point: tracing knows nothing of holes;
        :func:`join_at_holes` applies them.
    """
    start, end = child_segments(band.samples, band.det)
    if start.shape[0] == 0:
        return _no_curves(band)
    edges, offsets, closed = chain_segments(start, end, band.lens.shape[0])
    lens_points, source_points = edge_zeros(edges, band.det, (band.lens, band.source))
    device = backend.device(band.det)
    return CriticalCurvesAndCaustics(
        lens=lens_points,
        source=source_points,
        offsets=offsets,
        closed=closed,
        hole=backend.zeros((lens_points.shape[0],), dtype=backend.int64, device=device)
        - 1,
    )


def _turn(angle):
    """``angle`` wrapped into ``[0, 2 pi)``."""
    two_pi = 2.0 * math.pi
    return angle - two_pi * backend.floor(angle / two_pi)


def join_at_holes(curves, holes):
    """
    Cut traced curves at hole circles and re-join them along the hole curves.

    Inside a hole the lens map can jump, so a traced curve's points there
    mean nothing. They are replaced:

    1. Every point strictly inside a hole's disk is dropped, and so is a lone
       point between two points of the same hole. This cuts each curve into
       segments.
    2. A segment that starts right after a dropped point departs from that
       hole's circle; one that ends right before a dropped point arrives at
       it. A traced step from one hole straight to another is a *bridge*: a
       segment with no point of its own, departing the first hole and
       arriving at the second.
    3. Where the ends around a circle, sorted by angle, alternate between
       arriving and departing, each arriving end is joined to the next end
       clockwise -- the side ``det A > 0`` is on -- through the hole's stored
       samples strictly between them. Otherwise the hole is left unjoined.
    4. The segments are chained through their joins (:func:`chain_order`).
       Curves that never reach a hole keep :func:`trace_band`'s order; a
       curve left without a point is dropped.

    Every loop then bounds a ``det A > 0`` region with the holes cut out, and
    no caustic cuts across a hole curve, except at a hole whose ends do not
    alternate; at a hole the fov cuts, until
    :func:`~caustics.lenses.func.adaptive.extend_lens_mesh` grows the fov
    over it; where ``max_depth`` left leaves larger than a hole; and where a
    traced segment clips a disk with neither end inside it.

    Parameters
    ----------
    curves: CriticalCurvesAndCaustics
        As :func:`trace_band` returns them.
    holes: CenterHoles

    Returns
    -------
    CriticalCurvesAndCaustics
    """
    n_points = curves.lens.shape[0]
    n_holes = holes.centers.shape[0]
    if n_points == 0 or n_holes == 0:
        return curves
    device = backend.device(curves.lens)
    int64, f64 = backend.int64, backend.float64
    lens = backend.to(curves.lens, dtype=f64)
    centers = backend.to(holes.centers, dtype=f64)

    # 1. Tag every point strictly inside a hole's disk; disks are disjoint.
    inside = backend.norm(
        backend.unsqueeze(lens, 1) - backend.unsqueeze(centers, 0), dim=-1
    ) < backend.unsqueeze(holes.radius, 0)
    tag = backend.where(
        backend.any(inside, dim=1), backend.argmax(backend.long(inside), 1), -1
    )

    # Per-point curve bookkeeping.
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

    # 1b. A lone untagged point between two points of one hole, left by the
    # tracer's zigzag along its circle, counts as inside it.
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

    # 3. Segments: the maximal runs of untagged points of each curve, already
    # numbered by their first traced point in rotated order.
    is_first = ~tagged & ((point == first) | tagged[backend.clamp(point - 1, 0, None)])
    is_last = ~tagged & (
        (point == last) | tagged[backend.clamp(point + 1, None, n_points - 1)]
    )
    seg_first = backend.flatnonzero(is_first)
    seg_last = backend.flatnonzero(is_last)
    if seg_first.shape[0] == 0:
        return CriticalCurvesAndCaustics(
            lens=curves.lens[:0],
            source=curves.source[:0],
            offsets=backend.zeros((1,), dtype=int64, device=device),
            closed=curves.closed[:0],
            hole=curves.hole[:0],
        )

    # 3b. Bridges: a step from one hole straight to another is a segment from
    # the step's second point back to its first, with no point of its own.
    # They are numbered after every real segment, so a cycle holding a real
    # segment still starts at one.
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
        rel = rel - centers[h]
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
    csum = csr_offsets(backend.as_array(sizes, dtype=int64, device=device))
    offsets = csum[seg_offsets]
    # A curve of bridges alone whose joins inserted no sample has no point.
    kept = backend.flatnonzero(offsets[1:] > offsets[:-1])
    if kept.shape[0] < closed.shape[0]:
        offsets = backend.concatenate((offsets[:1], offsets[1:][kept]), dim=0)
        closed = closed[kept]
    return CriticalCurvesAndCaustics(
        lens=pool_lens[gather],
        source=pool_source[gather],
        offsets=offsets,
        closed=closed,
        hole=pool_hole[gather],
    )


def critical_curves_and_caustics(mesh):
    """
    Critical curves and caustics of the lens a mesh was built from.

    The zero set of ``det A`` traced through the critical band's red-split
    children, each crossing interpolated linearly on a child edge whose ends
    straddle it -- within half a finest leaf edge of the true curve -- and its
    caustic point interpolated the same way. On a mesh with holes the curves
    are cut at each hole's circle and re-joined along its hole curve
    (:func:`join_at_holes`). A curve ends where the band does: at the fov
    boundary or next to a non-finite leaf. Consecutive points can coincide
    where ``det A`` is exactly zero at a sample; such zero-length segments
    are kept. No lens call is made.

    Parameters
    ----------
    mesh: LensMesh

    Returns
    -------
    CriticalCurvesAndCaustics
    """
    curves = trace_band(mesh.critical_band)
    if mesh.holes.centers.shape[0] == 0:
        return curves
    return join_at_holes(curves, mesh.holes)
