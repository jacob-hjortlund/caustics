"""
The adaptive lens-plane mesh.

An adaptively refined triangulation of the lens plane, built once and queried
from the source plane many times. It also keeps what tracing the critical
curves and caustics needs, so that needs no further lens call.

This namespace holds the public API. Everything else lives in the modules
below, listed lowest layer first: each imports only from the modules listed
before it.

Substrate -- no lens call and no mesh:

- :mod:`.geometry` -- triangle and affine maths shared by build and query.
- :mod:`.state` -- the refinement's vertex cache, active-vertex set and leaf
  store.
- :mod:`.lattice` -- the dyadic integer lattice and triangles on it.

Building -- :func:`build_adaptive_mesh` runs four stages, seed, refine, balance
and freeze, on backend float64 arrays; keeping the build in one numerical
world is what makes the ``NaN`` semantics of
:func:`~.geometry.sigma_min_2x2` and :func:`~.criterion.converged_from_deviation`
verifiable:

- :mod:`.sampling` -- raytrace and Jacobian evaluation on lattice points.
- :mod:`.criterion` -- the refinement criterion and the ``LEAF_*`` flags.
- :mod:`.band` -- the critical band, kept from the ``max_level`` pass.
- :mod:`.holes` -- holes around lens centres and their hole curves.
- :mod:`.refinement` -- refine and balance.
- :mod:`.closure` -- canonical order and conforming closure.
- :mod:`.mesh` -- :class:`AdaptiveMesh`, its spatial index, and freeze.

Using a mesh:

- :mod:`.query` -- :func:`mesh_query` and :func:`mesh_seeds`.
- :mod:`.images` -- :func:`mesh_forward_raytrace`.
- :mod:`.curves` -- :func:`mesh_critical_curves_and_caustics`.
- :mod:`.magnification` -- :func:`mesh_total_magnification`, and the sheet
  edges across which the image count changes.
- :mod:`.source_mesh` -- :func:`build_magnification_mesh`, an adaptive
  source-plane mesh sampled with the total magnification.
- :mod:`.regions` -- :func:`magnified_regions`, :func:`magnified_area` and
  :func:`in_magnified_region`, from a :class:`MagnificationMesh`.

Entry points:

- :mod:`.build` -- :func:`build_adaptive_mesh`, :func:`extend_adaptive_mesh`
  and :func:`build_closed_adaptive_mesh`.
"""

from .geometry import (
    child_matrix_tables,
    contains,
    sanitize_bary,
    shape_matrix,
    sigma_min_2x2,
    triangle_weights,
)
from .criterion import (
    LEAF_APPROX_PARITY_UNRESOLVED,
    LEAF_CONVERGED,
    LEAF_CONVERGENCE_FAILED,
    LEAF_JACOBIAN_NONFINITE,
    LEAF_JACOBIAN_PARITY_UNRESOLVED,
    LEAF_RAYTRACE_NONFINITE,
    converged_from_deviation,
    evaluate_criterion,
    midpoint_deviation,
)
from .band import CriticalBand
from .holes import CentreHoles
from .index import MeshIndex
from .mesh import AdaptiveMesh
from .query import mesh_query, mesh_seeds
from .images import mesh_forward_raytrace
from .curves import CriticalCurvesAndCaustics, mesh_critical_curves_and_caustics
from .magnification import mesh_total_magnification
from .source_mesh import MagnificationMesh, build_magnification_mesh
from .regions import (
    MagnifiedRegions,
    in_magnified_region,
    magnified_area,
    magnified_regions,
)
from .build import (
    build_adaptive_mesh,
    build_closed_adaptive_mesh,
    extend_adaptive_mesh,
)

__all__ = (
    "build_adaptive_mesh",
    "extend_adaptive_mesh",
    "build_closed_adaptive_mesh",
    "mesh_query",
    "mesh_seeds",
    "mesh_forward_raytrace",
    "mesh_critical_curves_and_caustics",
    "mesh_total_magnification",
    "build_magnification_mesh",
    "magnified_regions",
    "magnified_area",
    "in_magnified_region",
    "AdaptiveMesh",
    "MeshIndex",
    "CriticalBand",
    "CentreHoles",
    "CriticalCurvesAndCaustics",
    "MagnificationMesh",
    "MagnifiedRegions",
    "LEAF_CONVERGED",
    "LEAF_CONVERGENCE_FAILED",
    "LEAF_APPROX_PARITY_UNRESOLVED",
    "LEAF_JACOBIAN_PARITY_UNRESOLVED",
    "LEAF_RAYTRACE_NONFINITE",
    "LEAF_JACOBIAN_NONFINITE",
    "child_matrix_tables",
    "contains",
    "converged_from_deviation",
    "evaluate_criterion",
    "midpoint_deviation",
    "sanitize_bary",
    "shape_matrix",
    "sigma_min_2x2",
    "triangle_weights",
)
