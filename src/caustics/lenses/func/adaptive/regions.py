"""
Regions of total magnification at least a threshold, from a magnification mesh.

:func:`magnified_regions` traces their boundaries, :func:`magnified_area`
measures them and :func:`in_magnified_region` tells which points lie in them.
All three read one field at the vertices of a
:class:`~caustics.lenses.func.adaptive.source_mesh.MagnificationMesh`,
``g = u(mu_min) - u(mu)`` with ``u = 1 / (1 + mu)`` (:func:`region_field`),
under one tie rule -- ``g >= 0``, that is ``mu >= mu_min``, is inside, as
``det A >= 0`` is positive in
:func:`~caustics.lenses.func.adaptive.curves.child_segments` -- on the closed
leaves not marked ``incomplete``, so the curves bound exactly the area
measured and the points classified inside. ``u`` is bounded and sends the
critical band's ``+inf`` to 0, so nothing special-cases infinity.
"""

from typing import NamedTuple, Tuple

from ....backend_obj import ArrayLike, backend
from .geometry import shape_matrix
from .lattice import lattice_on_boundary
from .query import index_hits
from .curves import chain_segments, edge_zeros, triangle_segments

__all__ = (
    "MagnifiedRegions",
    "region_field",
    "magnified_regions",
    "magnified_area",
    "in_magnified_region",
)


class MagnifiedRegions(NamedTuple):
    """
    Boundaries of ``mu_tot >= mu_min``, one ordered polyline per curve, CSR.

    Curve ``c`` is rows ``offsets[c]:offsets[c + 1]`` of ``source``. Travel
    keeps the inside on the left, so outer boundaries run counter-clockwise
    and holes clockwise. ``closed`` marks the loops, whose last point joins
    back to their first; a curve that reaches an ``incomplete`` leaf or the
    edge of the window is open. A vertex exactly at ``mu_min`` puts a
    crossing on that vertex, so consecutive points can coincide and some
    segments have zero length; they are kept, as in
    :class:`~caustics.lenses.func.adaptive.curves.CriticalCurvesAndCaustics`,
    so anything computing tangents or arc length must guard against them.

    Within an image sheet the sampled field is continuous (see
    :func:`~caustics.lenses.func.adaptive.magnification.mesh_total_magnification`),
    so a boundary is one curve wherever ``mu`` crosses ``mu_min`` once. On an
    SIS lens mesh at ``min_img_sep = 0.005``, the disk ``mu_tot >= 4`` comes
    back as a single loop within 4e-4 arcsec of the exact circle. A cored
    isothermal lens's ``mu_tot >= 6`` comes back as exactly its three circles.
    Area-ratio magnifications had given the same disk 48 extra islands and
    holes. A leaf read by its area ratio, the fallback of
    :func:`~caustics.lenses.func.adaptive.magnification.hit_magnification`,
    can still put a small excursion beside a boundary.

    Parameters
    ----------
    source: ArrayLike
        ``(P, 2)`` float64 boundary points.

        *Unit: arcsec*
    offsets: ArrayLike
        ``(C + 1,)`` int64 CSR offsets, ``offsets[0] == 0``.
    closed: ArrayLike
        ``(C,)`` bool.
    mu_min: float
        The threshold traced.
    """

    source: ArrayLike
    offsets: ArrayLike
    closed: ArrayLike
    mu_min: float


def region_field(mag, mu_min) -> ArrayLike:
    """
    ``g = 1 / (1 + mu_min) - 1 / (1 + mu)`` at every vertex: inside where ``g >= 0``.

    Parameters
    ----------
    mag: MagnificationMesh
    mu_min: float

    Returns
    -------
    ArrayLike
        ``(V,)`` float64.
    """
    return 1.0 / (1.0 + float(mu_min)) - 1.0 / (1.0 + mag.mu)


def magnified_regions(mag, mu_min) -> MagnifiedRegions:
    """
    Boundaries of the source-plane regions where ``mu_tot >= mu_min``.

    Marching triangles on the closed leaves of ``mag`` not marked
    ``incomplete`` (:func:`~caustics.lenses.func.adaptive.curves.triangle_segments`
    on :func:`region_field`), crossings at the zero of the linear
    interpolant on each edge, keyed by the edge's vertex pair and chained by
    :func:`~caustics.lenses.func.adaptive.curves.chain_segments`. The mesh
    is conforming, so the two leaves on either side of an edge see the same
    crossing there. For a threshold ``mag`` targeted, every boundary point
    is within ``mag.src_tol`` of where the sampled field crosses it; see
    :func:`~caustics.lenses.func.adaptive.source_mesh.build_magnification_mesh`.
    Where the sampled field crosses ``mu_min`` more than once near a
    boundary, the result holds small islands and holes as well as the main
    boundary; see :class:`MagnifiedRegions`.

    Parameters
    ----------
    mag: MagnificationMesh
    mu_min: float

    Returns
    -------
    MagnifiedRegions
    """
    g = region_field(mag, mu_min)
    leaves = mag.leaves[backend.flatnonzero(~mag.incomplete)]
    start, end = triangle_segments(leaves, g >= 0)
    edges, offsets, closed = chain_segments(
        start, end, mag.vertices.shape[0], "region-boundary crossing"
    )
    (points,) = edge_zeros(edges, g, (mag.vertices,))
    return MagnifiedRegions(
        source=points, offsets=offsets, closed=closed, mu_min=float(mu_min)
    )


def _inside_area(mag, g, area, complete, boundary) -> Tuple[ArrayLike, ArrayLike]:
    """The area where the interpolated ``g >= 0`` over complete leaves, and whether it is whole."""
    corners = g[mag.leaves]
    inside = corners >= 0
    n_in = backend.sum(backend.long(inside), dim=1)
    odd_inside = n_in == 1
    odd = backend.argmax(
        backend.long(backend.where(backend.unsqueeze(odd_inside, -1), inside, ~inside)),
        1,
    )
    rows = backend.arange(
        corners.shape[0], dtype=backend.int64, device=backend.device(corners)
    )
    g_i = corners[rows, odd]
    g_j = corners[rows, (odd + 1) % 3]
    g_k = corners[rows, (odd + 2) % 3]
    # The corner at the odd vertex, cut off by the segment between the two
    # crossings; on a leaf of one class this is 0/0, masked below.
    corner = (g_i / (g_i - g_j)) * (g_i / (g_i - g_k)) * area
    mixed = (n_in == 1) | (n_in == 2)
    zero = backend.zeros_like(area)
    part = backend.where(n_in == 3, area, zero)
    part = backend.where(mixed, backend.where(odd_inside, corner, area - corner), part)
    total = backend.sum(backend.where(complete, part, zero))
    reaches = backend.any(inside[backend.flatnonzero(~complete)]) | backend.any(
        (g >= 0) & boundary
    )
    return total, ~reaches


def magnified_area(mag, mu_min) -> Tuple[ArrayLike, ArrayLike]:
    """
    Source-plane area where ``mu_tot >= mu_min``, and whether the region is whole.

    Per leaf, without tracing: a leaf with every vertex inside counts its
    area ``A``; a mixed leaf with odd vertex ``i`` and crossings at
    ``t_ij``, ``t_ik`` along its edges from ``i`` counts ``t_ij * t_ik * A``
    when ``i`` is inside and ``A - t_ij * t_ik * A`` when it is not. That is
    exactly the region :func:`magnified_regions` bounds, so when every loop
    is closed it equals the loops' signed shoelace sum to rounding. Leaves
    marked ``incomplete`` count nothing.

    Parameters
    ----------
    mag: MagnificationMesh
    mu_min: float or ArrayLike
        One threshold, or a 1-D array of them: one vectorized pass each
        gives the whole ``sigma(mu_min)``.

    Returns
    -------
    area: ArrayLike
        float64 of ``mu_min``'s shape, a 0-d array for a scalar.

        *Unit: arcsec^2*
    complete: ArrayLike
        bool of ``mu_min``'s shape: False where an inside vertex lies on an
        ``incomplete`` leaf or on the window's boundary -- where the region
        runs into missing data and its curves come back open.
    """
    thresholds = backend.as_array(mu_min, dtype=backend.float64)
    P = shape_matrix(mag.vertices[mag.leaves])
    area = 0.5 * (P[:, 0, 0] * P[:, 1, 1] - P[:, 0, 1] * P[:, 1, 0])
    complete = ~mag.incomplete
    boundary = lattice_on_boundary(mag.lattice, mag.vertices_ij)
    areas, flags = [], []
    for t in backend.to_numpy(thresholds).reshape(-1).tolist():
        a, c = _inside_area(mag, region_field(mag, t), area, complete, boundary)
        areas.append(a)
        flags.append(c)
    shape = tuple(thresholds.shape)
    return backend.stack(areas).reshape(shape), backend.stack(flags).reshape(shape)


def in_magnified_region(mag, mu_min, beta) -> Tuple[ArrayLike, ArrayLike]:
    """
    Whether each source-plane point lies where ``mu_tot >= mu_min``.

    ``mag.index`` finds the leaf containing each point; leaves do not
    overlap, and on a shared edge both neighbours interpolate the same
    value, so the first is used. Inside means the barycentric interpolation
    of :func:`region_field` is ``>= 0``: exactly the side of the traced
    segment the point falls on.

    Parameters
    ----------
    mag: MagnificationMesh
    mu_min: float
    beta: ArrayLike
        ``(B, 2)`` source-plane points.

        *Unit: arcsec*

    Returns
    -------
    inside: ArrayLike
        ``(B,)`` bool, False wherever ``complete`` is.
    complete: ArrayLike
        ``(B,)`` bool, False for a point in an ``incomplete`` leaf or outside
        the window, where the answer is not known.
    """
    beta = backend.as_array(beta, dtype=backend.float64, device=mag.device)
    b = beta.shape[0]
    inside = backend.zeros((b,), dtype=backend.bool, device=mag.device)
    complete = backend.zeros((b,), dtype=backend.bool, device=mag.device)
    qidx, tri, w = index_hits(mag.index, mag.vertices, mag.leaves, beta)
    if qidx.shape[0] == 0:
        return inside, complete
    first = backend.flatnonzero(
        backend.concatenate(
            (
                backend.ones((1,), dtype=backend.bool, device=mag.device),
                qidx[1:] != qidx[:-1],
            ),
            dim=0,
        )
    )
    q, t, w = qidx[first], tri[first], w[first]
    bary = w / backend.unsqueeze(backend.sum(w, dim=1), -1)
    value = backend.sum(bary * region_field(mag, mu_min)[mag.leaves[t]], dim=1)
    known = ~mag.incomplete[t]
    complete = backend.fill_at_indices(complete, q, known)
    inside = backend.fill_at_indices(inside, q, known & (value >= 0))
    return inside, complete
