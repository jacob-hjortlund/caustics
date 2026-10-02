"""
Regions of total magnification at least a threshold, from a magnification mesh.

:func:`magnified_regions` traces their boundaries. All of this module reads
one field at the vertices of a
:class:`~caustics.lenses.func.adaptive.source_mesh.MagnificationMesh`,
``g = u(mu_min) - u(mu)`` with ``u = 1 / (1 + mu)`` (:func:`region_field`),
under one tie rule -- ``g >= 0``, that is ``mu >= mu_min``, is inside, as
``det A >= 0`` is positive in
:func:`~caustics.lenses.func.adaptive.curves.child_segments` -- on the closed
leaves not marked ``incomplete``, so the curves bound exactly the area
measured and the points classified inside. ``u`` is bounded and sends the
critical band's ``+inf`` to 0, so nothing special-cases infinity.
"""

from typing import NamedTuple

from ....backend_obj import ArrayLike, backend
from .curves import chain_segments, edge_zeros, triangle_segments

__all__ = (
    "MagnifiedRegions",
    "region_field",
    "magnified_regions",
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
