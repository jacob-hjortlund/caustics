"""
Holes cut around lens centers, and the images of their boundary circles.

The lens map can jump at a lens center, so no curve of ``det A = 0``
describes what happens to images there. A hole of radius ``min_img_sep`` is
left out of the lens plane instead, and the image of its boundary circle, the
hole curve, is traced (:func:`sample_holes`).
:mod:`~caustics.lenses.func.adaptive.curves` re-joins critical curves along
it.
"""

import math
from typing import NamedTuple
from warnings import warn

from ....backend_obj import ArrayLike, backend
from .geometry import build_device, to_device
from .refine import sample_points

__all__ = (
    "CenterHoles",
    "empty_holes",
    "merge_centers",
    "HOLE_INITIAL_SAMPLES",
    "HOLE_MAX_SAMPLES",
    "HOLE_GROWTH_SAMPLES",
    "hole_circle",
    "sample_holes",
)


class CenterHoles(NamedTuple):
    """
    Small disks cut around lens centers, and the images of their boundaries.

    The lens map can jump at a lens center -- an isothermal profile's
    deflection depends only on the direction from its center -- so no curve
    of ``det A = 0`` describes what happens to images there. A hole of radius
    ``radius`` around each center is left out of the lens plane instead, and
    the image of its boundary circle, the *hole curve*, is kept. At an
    isothermal center the hole curve is the pseudo-caustic to within about
    ``radius``; at a point mass, or a cusp steeper than isothermal, it is a
    huge loop; at a regular point, or a profile whose deflection vanishes at
    its center, it is a speck. ``growth`` tells these apart: the hole curve's
    size scales as ``radius**growth`` there, so ``growth`` is 0 for a
    pseudo-caustic, negative for a loop that runs off to infinity as the hole
    shrinks, and positive for a speck -- exactly ``1 - t`` for a power law of
    slope ``t``.

    ``growth`` is measured, not known, so it is never exactly 0:
    ``growth_err`` bounds its error, and ``pseudo_caustic`` marks the holes
    whose ``growth`` is 0 within it. That needs no tolerance, but it resolves
    only so much: a slope ``t`` within about ``growth_err`` of 1 -- of order
    ``(radius / size)**2``, where ``size`` is the hole curve's, so 2e-6 for
    a 0.005" hole in an SIS of 1" Einstein radius ``b`` -- reads as
    isothermal. Its hole curve then differs from the isothermal one by a
    fraction of about ``|t - 1| * |log(radius / b)|`` of its size.

    Samples are stored CSR per hole: hole ``h`` is rows
    ``offsets[h]:offsets[h + 1]`` of ``angle``, ``lens`` and ``source``, in
    strictly ascending ``angle``.

    Parameters
    ----------
    centers: ArrayLike
        ``(H, 2)`` hole centers after merging, at the mesh dtype.

        *Unit: arcsec*
    radius: ArrayLike
        ``(H,)`` float64 hole radii: the mesh's ``min_img_sep``, plus the
        spread of the centers a hole merged.

        *Unit: arcsec*
    offsets: ArrayLike
        ``(H + 1,)`` int64 CSR offsets, ``offsets[0] == 0``.
    angle: ArrayLike
        ``(P,)`` float64 sample angles in ``[0, 2 pi)``.

        *Unit: radians*
    lens: ArrayLike
        ``(P, 2)`` circle points ``center + radius * (cos, sin)(angle)``, at
        the mesh dtype.

        *Unit: arcsec*
    source: ArrayLike
        ``(P, 2)`` their images -- the hole curve -- at the mesh dtype.

        *Unit: arcsec*
    growth: ArrayLike
        ``(H,)`` float64 log-slope of the hole curve's size against the hole
        radius as the hole shrinks to nothing, extrapolated from circles
        down to ``radius / 64`` (:func:`sample_holes`). NaN when the circles
        map to a single point.
    growth_err: ArrayLike
        ``(H,)`` float64 bound on the error of ``growth``; NaN where
        ``growth`` is.
    pseudo_caustic: ArrayLike
        ``(H,)`` bool, True where ``|growth| <= growth_err``: where the hole
        curve is the pseudo-caustic, to within about ``radius``. False where
        ``growth`` is NaN.
    """

    centers: ArrayLike
    radius: ArrayLike
    offsets: ArrayLike
    angle: ArrayLike
    lens: ArrayLike
    source: ArrayLike
    growth: ArrayLike
    growth_err: ArrayLike
    pseudo_caustic: ArrayLike


def empty_holes(device=None):
    """A :class:`CenterHoles` with no hole."""
    return CenterHoles(
        centers=backend.zeros((0, 2), dtype=backend.float64, device=device),
        radius=backend.zeros((0,), dtype=backend.float64, device=device),
        offsets=backend.zeros((1,), dtype=backend.int64, device=device),
        angle=backend.zeros((0,), dtype=backend.float64, device=device),
        lens=backend.zeros((0, 2), dtype=backend.float64, device=device),
        source=backend.zeros((0, 2), dtype=backend.float64, device=device),
        growth=backend.zeros((0,), dtype=backend.float64, device=device),
        growth_err=backend.zeros((0,), dtype=backend.float64, device=device),
        pseudo_caustic=backend.zeros((0,), dtype=backend.bool, device=device),
    )


def _components(link):
    """
    Connected-component label of every node of a symmetric adjacency matrix.

    Min-label propagation, as in :func:`dedup_block_group`: each node takes
    the smallest label among itself and its neighbours until nothing
    changes, so every node ends labelled with the smallest index in its
    component.
    """
    n = link.shape[0]
    label = backend.arange(n, dtype=backend.int64)
    reach = link | backend.eye(n, dtype=backend.bool)
    while True:
        new = backend.min(backend.where(reach, backend.unsqueeze(label, 0), n), dim=1)
        if bool(backend.all(new == label)):
            return label
        label = new


def merge_centers(centers, min_img_sep):
    """
    Merge lens centers into holes whose disks do not overlap.

    Centers closer than ``2 * min_img_sep`` share a hole. A hole sits at its
    members' mean, with radius ``min_img_sep`` plus the members' largest
    distance from that mean, so a lone center gets exactly ``min_img_sep``.
    A merged hole is larger than ``min_img_sep``, so it can reach another
    hole's disk; merging repeats until no two disks overlap. The centers are
    put in lexicographic order first and the holes are returned in it, so
    the result -- sums included -- depends on the set of centers, not on the
    order they were given in.

    Parameters
    ----------
    centers: Optional[ArrayLike]
        ``(S, 2)`` lens-plane positions, any array-like. ``None`` or an empty
        array gives no hole.

        *Unit: arcsec*
    min_img_sep: float
        The mesh's own tolerance, already halved.

        *Unit: arcsec*

    Returns
    -------
    centers: ArrayLike
        ``(H, 2)`` float64 hole centers, in lexicographic ``(x, y)`` order.

        *Unit: arcsec*
    radius: ArrayLike
        ``(H,)`` float64 hole radii.

        *Unit: arcsec*
    """
    f64 = backend.float64
    if centers is None:
        return backend.zeros((0, 2), dtype=f64), backend.zeros((0,), dtype=f64)
    c = to_device(backend.as_array(centers, dtype=f64), build_device())
    if c.reshape(-1).shape[0] == 0:
        return backend.zeros((0, 2), dtype=f64), backend.zeros((0,), dtype=f64)
    c = c[backend.lexsort([c[:, 1], c[:, 0]])]
    link = (
        backend.norm(backend.unsqueeze(c, 1) - backend.unsqueeze(c, 0), dim=-1)
        < 2.0 * min_img_sep
    )
    while True:
        _, member = backend.unique(_components(link), return_inverse=True)
        k = int(backend.to_numpy(backend.max(member))) + 1
        onehot = backend.unsqueeze(
            backend.arange(k, dtype=backend.int64), 1
        ) == backend.unsqueeze(member, 0)
        weight = backend.to(onehot, dtype=f64)
        mean = (weight @ c) / backend.unsqueeze(backend.sum(weight, dim=1), 1)
        spread = backend.norm(
            backend.unsqueeze(c, 0) - backend.unsqueeze(mean, 1), dim=-1
        )
        radius = min_img_sep + backend.max(backend.where(onehot, spread, 0.0), dim=1)
        gap = backend.norm(
            backend.unsqueeze(mean, 1) - backend.unsqueeze(mean, 0), dim=-1
        )
        overlap = (
            gap < backend.unsqueeze(radius, 1) + backend.unsqueeze(radius, 0)
        ) & ~backend.eye(k, dtype=backend.bool)
        if not bool(backend.any(overlap)):
            break
        link = link | overlap[member][:, member]
    order = backend.lexsort([mean[:, 1], mean[:, 0]])
    return mean[order], radius[order]


# Hole sampling: the first even pass, the refinement cap, and the even
# samples on each of the circles the growth exponent is measured on.
HOLE_INITIAL_SAMPLES = 256
HOLE_MAX_SAMPLES = 1 << 16
HOLE_GROWTH_SAMPLES = 1024


def hole_circle(centers, radius, hole, angle):
    """
    Points ``centers[hole] + radius[hole] * (cos, sin)(angle)``, shape ``(K, 2)``.

    *Unit: arcsec*
    """
    c, r = centers[hole], radius[hole]
    return backend.stack(
        (c[:, 0] + r * backend.cos(angle), c[:, 1] + r * backend.sin(angle)), dim=-1
    )


def _trace_hole_points(trace, centers, radius, hole, angle, batch_size, scale=1.0):
    """
    Images of points on the hole circles scaled by ``scale``, shape ``(K, 2)``.

    Raises
    ------
    ValueError
        If any image is non-finite, naming the first such hole's center and
        the radius of the circle traced.
    """
    source = sample_points(
        hole_circle(centers, radius * scale, hole, angle), trace, batch_size
    )
    bad = backend.flatnonzero(~backend.all(backend.isfinite(source), dim=1))
    if bad.shape[0]:
        h = int(backend.to_numpy(hole[bad[0]]))
        x, y = backend.to_numpy(centers[h]).tolist()
        r = float(backend.to_numpy(radius[h])) * scale
        raise ValueError(
            f"the lens is not finite on the hole circle of radius {r:g} around "
            f"({x:g}, {y:g}); a hole's boundary must lie where the lens is finite"
        )
    return source


def _extent(points):
    """Diagonal of each row's bounding box, ``(H, K, 2) -> (H,)``."""
    return backend.norm(backend.max(points, dim=1) - backend.min(points, dim=1), dim=-1)


def sample_holes(trace, centers, radius, min_img_sep, batch_size):
    """
    Trace every hole's boundary circle and its image, the hole curve.

    Each circle starts with :data:`HOLE_INITIAL_SAMPLES` evenly spaced angles.
    Every angular interval whose source-plane chord exceeds ``min_img_sep``
    -- the wrap-around one included -- is then bisected, round after round,
    one ``trace`` call per round for all holes together, until no chord
    exceeds ``min_img_sep`` or a hole would pass :data:`HOLE_MAX_SAMPLES`
    samples. A hole stopped by the cap stays exact at its samples but coarser
    between them, and is warned about; only enormous hole curves, such as a
    point mass's, reach it.

    ``growth`` comes from the hole curve's size -- its bounding box's
    diagonal -- on :data:`HOLE_GROWTH_SAMPLES` even samples at ``radius``
    and at ``radius / 4``, ``/ 16`` and ``/ 64``. The slope between two
    neighbouring circles, ``g(r) = log(size(r) / size(r / 4)) / log(4)``,
    is biased by a term linear in ``r``: the lens map's smooth part --
    the identity, and every other lens's deflection -- stretches the hole
    curve by that much. At an isothermal center it is all of ``g``, of
    order ``radius / size``. Richardson extrapolation cancels it:
    ``R(r) = (4 g(r / 4) - g(r)) / 3``, and ``growth = R(r / 4)``. What
    is left falls as ``r**p`` -- ``p = 2`` at an isothermal center of a
    lens map smooth everywhere else -- so ``growth_err = |R(r) - R(r / 4)|``
    is ``4**p - 1`` times that error, a bound for any ``p >= 1/2``. No
    Jacobian is called.

    Parameters
    ----------
    trace: Callable[[ArrayLike], ArrayLike]
        From :func:`make_raytrace`.
    centers: ArrayLike
        ``(H, 2)`` float64 hole centers, from :func:`merge_centers`.

        *Unit: arcsec*
    radius: ArrayLike
        ``(H,)`` float64 hole radii, from :func:`merge_centers`.

        *Unit: arcsec*
    min_img_sep: float
        Largest source-plane chord left between neighbouring samples.

        *Unit: arcsec*
    batch_size: Optional[int]
        Forwarded to :func:`trace_points`.

    Returns
    -------
    CenterHoles
        Every array float64 but ``offsets``, on the ambient device.

    Raises
    ------
    ValueError
        If the lens is not finite somewhere on a circle at ``radius``,
        ``radius / 4``, ``/ 16`` or ``/ 64``.

    Warns
    -----
    UserWarning
        For each hole whose refinement stopped at :data:`HOLE_MAX_SAMPLES`.
    """
    n_holes = centers.shape[0]
    if n_holes == 0:
        return empty_holes()
    int64, f64 = backend.int64, backend.float64
    two_pi = 2.0 * math.pi
    ids = backend.arange(n_holes, dtype=int64)

    hole = backend.repeat(ids, HOLE_INITIAL_SAMPLES, axis=0)
    step = backend.arange(n_holes * HOLE_INITIAL_SAMPLES, dtype=int64)
    angle = backend.to(step % HOLE_INITIAL_SAMPLES, dtype=f64) * (
        two_pi / HOLE_INITIAL_SAMPLES
    )
    source = _trace_hole_points(trace, centers, radius, hole, angle, batch_size)
    capped = backend.zeros((n_holes,), dtype=backend.bool)
    while True:
        counts = backend.bincount(hole, minlength=n_holes)
        starts = backend.cumsum(counts, dim=0) - counts
        i = backend.arange(hole.shape[0], dtype=int64)
        wrap = i == starts[hole] + counts[hole] - 1
        nxt = backend.where(wrap, starts[hole], i + 1)
        chord = backend.norm(source[nxt] - source, dim=-1)
        bad = backend.flatnonzero(chord > min_img_sep)
        wanted = backend.bincount(hole[bad], minlength=n_holes)
        capped = capped | ((wanted > 0) & (counts + wanted > HOLE_MAX_SAMPLES))
        rows = bad[backend.flatnonzero(~capped[hole[bad]])]
        if rows.shape[0] == 0:
            break
        upper = backend.where(wrap[rows], angle[nxt[rows]] + two_pi, angle[nxt[rows]])
        mid = 0.5 * (angle[rows] + upper)
        mid = backend.where(mid >= two_pi, mid - two_pi, mid)
        new_hole = hole[rows]
        new_source = _trace_hole_points(
            trace, centers, radius, new_hole, mid, batch_size
        )
        hole = backend.concatenate((hole, new_hole), dim=0)
        angle = backend.concatenate((angle, mid), dim=0)
        source = backend.concatenate((source, new_source), dim=0)
        order = backend.lexsort([angle, hole])
        hole, angle, source = hole[order], angle[order], source[order]
    for h in backend.to_numpy(backend.flatnonzero(capped)).tolist():
        x, y = backend.to_numpy(centers[h]).tolist()
        n = int(backend.to_numpy(backend.bincount(hole, minlength=n_holes)[h]))
        warn(
            f"the hole curve around ({x:g}, {y:g}) stopped at {n} samples "
            f"(cap {HOLE_MAX_SAMPLES}) before every chord fell below "
            f"min_img_sep={min_img_sep:g}; it is exact at its samples but "
            "coarser than min_img_sep between them"
        )

    g_hole = backend.repeat(ids, HOLE_GROWTH_SAMPLES, axis=0)
    g_step = backend.arange(n_holes * HOLE_GROWTH_SAMPLES, dtype=int64)
    g_angle = backend.to(g_step % HOLE_GROWTH_SAMPLES, dtype=f64) * (
        two_pi / HOLE_GROWTH_SAMPLES
    )
    shape = (n_holes, HOLE_GROWTH_SAMPLES, 2)
    size = [
        _extent(
            _trace_hole_points(
                trace, centers, radius, g_hole, g_angle, batch_size, 0.25**k
            ).reshape(shape)
        )
        for k in range(4)
    ]
    # `step[k] / log(4)` is the slope `g` between circles k and k + 1, and
    # `R(r / 4) - R(r)` folds to one sum of steps. Both scale by a constant
    # rather than dividing by one: jax divides an array of more than one
    # element by a constant as a multiply by its reciprocal, so a quotient
    # would depend on how many holes share the call.
    step = [backend.log(size[k] / size[k + 1]) for k in range(3)]
    scale = 1.0 / (3.0 * math.log(4.0))
    growth = (4.0 * step[2] - step[1]) * scale
    growth_err = backend.abs(5.0 * step[1] - step[0] - 4.0 * step[2]) * scale

    counts = backend.bincount(hole, minlength=n_holes)
    offsets = backend.concatenate(
        (backend.zeros((1,), dtype=int64), backend.cumsum(counts, dim=0)), dim=0
    )
    return CenterHoles(
        centers=centers,
        radius=radius,
        offsets=offsets,
        angle=angle,
        lens=hole_circle(centers, radius, hole, angle),
        source=source,
        growth=growth,
        growth_err=growth_err,
        pseudo_caustic=backend.abs(growth) <= growth_err,
    )
