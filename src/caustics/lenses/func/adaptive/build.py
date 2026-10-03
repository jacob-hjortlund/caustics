"""
Entry points that make a mesh: build, extend, and build until closed.

:func:`build_adaptive_mesh` refines every level-0 cell, balances and freezes.
:func:`extend_adaptive_mesh` runs the same stages from a built mesh, refining
only the ring of cells it adds. :func:`build_closed_adaptive_mesh` builds,
then extends until the fov cuts no critical curve.
"""

import math
import operator
from typing import Tuple
from warnings import warn

from ....backend_obj import ArrayLike, backend
from .geometry import child_matrix_tables
from .state import LeafStore, VertexCache, _ambient
from .lattice import (
    check_lattice_keys,
    depth_floor,
    extend_lattice,
    lattice_key,
    make_lattice,
    ring_triangles,
)
from .sampling import make_raytrace, trace_keys
from .criterion import LEAF_CONVERGED
from .band import CriticalBand, empty_band
from .holes import merge_centres, sample_holes
from .refinement import balance, refine
from .mesh import AdaptiveMesh, freeze
from .curves import CriticalCurvesAndCaustics, mesh_critical_curves_and_caustics

__all__ = (
    "validate_build_args",
    "warn_depth_limited",
    "warn_cancellation_floor",
    "build_adaptive_mesh",
    "seed_from_mesh",
    "extend_adaptive_mesh",
    "build_closed_adaptive_mesh",
)


def validate_build_args(
    fov, init_res, min_img_sep, max_depth, requested_min_img_sep=None
) -> None:
    """
    Reject impossible parameters, including a lattice that would overflow int64.

    ``min_img_sep`` is the value every check and the depth computation use.
    ``requested_min_img_sep`` names only the positivity message, so a caller who
    halved it (``build_adaptive_mesh``) is told the number they actually
    supplied rather than the halved one. Defaults to ``min_img_sep`` for a
    direct call, where the two coincide.
    """
    if requested_min_img_sep is None:
        requested_min_img_sep = min_img_sep
    if not fov > 0:
        raise ValueError(f"fov must be positive, got {fov}")
    if init_res < 1:
        raise ValueError(f"init_res must be at least 1, got {init_res}")
    if not min_img_sep > 0:
        raise ValueError(f"min_img_sep must be positive, got {requested_min_img_sep}")
    if max_depth < 0:
        raise ValueError(f"max_depth must be non-negative, got {max_depth}")
    max_level = min(max_depth, depth_floor(fov / init_res, min_img_sep))
    check_lattice_keys(
        init_res, max_level, "Raise min_img_sep, lower max_depth, or lower init_res."
    )


def warn_depth_limited(fov, init_res, min_img_sep, d_floor, max_level) -> None:
    """
    Warn when ``max_depth`` stopped refinement short of the size floor.

    Shared by :func:`build_adaptive_mesh` and :func:`extend_adaptive_mesh`, so
    an extension warns exactly as a fresh build of its fov would. Refinement
    is depth-limited exactly when ``d_floor > max_level``, and ``max_level``
    is then ``max_depth``, so the message names ``max_depth`` without being
    given it. ``min_img_sep`` is the halved tolerance; the caller's request
    is ``2 * min_img_sep`` exactly, since halving is exact.
    """
    if d_floor <= max_level:
        return
    l_max_final = float(math.sqrt(2.0) * fov / (init_res * 2**max_level))
    warn(
        f"Adaptive mesh is depth-limited: max_depth={max_level} is below "
        f"d_floor={d_floor}, the depth required to reach "
        f"min_img_sep={2 * min_img_sep:g} arcsec (refined internally "
        f"to {min_img_sep:g}). Refinement stops at level {max_level}, "
        f"where the maximum leaf edge is {l_max_final:.3g} arcsec. Set "
        f"max_depth >= {d_floor} to restore the size-floor guarantee, or "
        f"raise init_res / min_img_sep."
    )


def warn_cancellation_floor(raytrace_fn, fov, min_img_sep) -> None:
    """
    Warn when the dtype ``raytrace`` returned cannot resolve ``min_img_sep``.

    The midpoint deviation compares ``O(fov)`` quantities, so its roundoff
    floor is ``sqrt(8 * eps * fov)`` (see :func:`make_raytrace`). ``fov`` is
    the fov being meshed, so an extension checks the larger one.
    ``min_img_sep`` is the halved tolerance, as for
    :func:`warn_depth_limited`.
    """
    # `.info` is attached dynamically by `make_raytrace` (documented on its
    # own `# type: ignore[attr-defined]` there); mypy has no way to see it on
    # the `Callable[[ArrayLike], ArrayLike]` return annotation.
    raytrace_dtype = raytrace_fn.info["dtype"]  # type: ignore[attr-defined]
    eps = float(backend.finfo(raytrace_dtype).eps)
    cancellation_floor = float(math.sqrt(8.0 * eps * fov))
    if cancellation_floor > min_img_sep:
        warn(
            f"raytrace returned {raytrace_dtype}, whose "
            f"cancellation floor sqrt(8*eps*fov) = {cancellation_floor:.3g} "
            f"arcsec exceeds min_img_sep={2 * min_img_sep:g} (refined "
            f"internally to {min_img_sep:g}). Below that scale the midpoint "
            "deviation cancels to zero, which the criterion reads as "
            "'affine' and converges. Supply a raytrace that preserves "
            "float64."
        )


def _grow(
    raytrace_fn,
    jacobian_fn,
    lat,
    tables,
    *,
    roots,
    seed,
    seed_band,
    fov,
    init_res,
    h0,
    min_img_sep,
    d_floor,
    max_level,
    device,
    dtype,
    raytrace_batch_size,
    index_cells,
    holes=None,
) -> AdaptiveMesh:
    """
    Refine ``roots`` over ``seed``, balance, and freeze: the stages both builds share.

    :func:`build_adaptive_mesh` passes every cell and no seed;
    :func:`extend_adaptive_mesh` passes the ring and the old mesh. Running
    both through here is what makes an extension a fresh build of the larger
    fov, rather than a second algorithm that happens to agree with one.
    """
    cache, active, store, _counters, band = refine(
        raytrace_fn,
        jacobian_fn,
        lat,
        init_res,
        h0,
        min_img_sep,
        max_level,
        tables,
        raytrace_batch_size,
        roots=roots,
        seed=seed,
    )
    warn_cancellation_floor(raytrace_fn, fov, min_img_sep)
    store, cache, active, _forced = balance(
        store,
        cache,
        lat,
        active,
        max_level,
        tables[2],
        raytrace_fn,
        raytrace_batch_size,
    )
    return freeze(
        lat,
        cache,
        active,
        store,
        band,
        seed_band,
        jacobian_fn=jacobian_fn,
        fov=fov,
        init_res=init_res,
        min_img_sep=min_img_sep,
        d_floor=d_floor,
        max_level=max_level,
        dtype=dtype,
        device=device,
        index_cells=index_cells,
        holes=holes,
    )


def build_adaptive_mesh(
    raytrace,
    jacobian,
    fov,
    init_res,
    min_img_sep,
    max_depth=25,
    *,
    x0=0.0,
    y0=0.0,
    device=None,
    dtype=None,
    raytrace_batch_size=None,
    index_cells=None,
    centres=None,
) -> AdaptiveMesh:
    """
    Build an adaptively refined triangular mesh of the lens plane.

    The mesh is built once and reused across many queries; it does not depend
    on any query point. The build runs four stages, all on ``backend``
    arrays: :func:`refine` over every level-0 cell, :func:`balance`, and
    :func:`freeze` -- canonical ordering and closure, the critical band,
    freeze-time invalidation and the spatial index. :func:`extend_adaptive_mesh`
    runs the same stages from a built mesh, seeded by :func:`seed_from_mesh`,
    which is what makes an extension a fresh build of the larger fov.

    Parameters
    ----------
    raytrace: Callable[[ArrayLike, ArrayLike], Tuple[ArrayLike, ArrayLike]]
        ``raytrace(x, y) -> (bx, by)`` on 1-D arrays of shape ``(N,)``,
        mapping lens-plane to source-plane coordinates; for a caustics lens,
        ``lens.raytrace``. It goes through :func:`make_raytrace`: float64 in
        and out, on ``device``, chunked by ``raytrace_batch_size``.
    jacobian: Callable[[ArrayLike, ArrayLike], ArrayLike]
        ``jacobian(x, y) -> A`` on 1-D arrays of shape ``(N,)``, returning
        the ``(N, 2, 2)`` Jacobian ``d(beta) / d(theta)`` of the map
        ``raytrace`` traces; for a caustics lens,
        ``lens.jacobian_lens_equation``. It decides
        :func:`jacobian_parity_ok`, which the criterion requires before it
        converges any leaf, and gives ``AdaptiveMesh.vertices_det``. It is
        called directly on the build's own float64 coordinates, at most once
        per lattice point: every value it returns is kept in the vertex
        cache.
    fov: float
        Side length of the square lens-plane domain.

        *Unit: arcsec*
    init_res: int
        Number of **cells** per axis, giving ``2 * init_res**2`` level-0
        triangles. Differs from ``forward_raytrace``'s ``divisions``, which
        counts ``linspace`` *points* and yields ``(n - 1)**2`` cells.

        This also carries the completeness obligation: the criterion samples
        six points per triangle, so structure below the level-0 scale is
        invisible to it. ``init_res`` must already resolve the smallest
        curvature scale in the lens.
    min_img_sep: float
        Requested lens-plane tolerance. The parity-condemned band at
        ``max_level`` is model-dependent, so the build refines to
        ``min_img_sep / 2`` internally to narrow it; this is not a universal
        numerical bound on the band width. The
        halved value is the one stored on the returned :class:`AdaptiveMesh`,
        not the value passed in.

        *Unit: arcsec*
    max_depth: int
        Hard cap on refinement level. Refinement runs to
        ``min(max_depth, d_floor)``; a warning is raised if ``max_depth``
        binds.
    x0, y0: float
        Centre of the domain.

        *Unit: arcsec*
    device: Optional
        Device for the coordinates handed to ``raytrace`` and for the frozen
        mesh.
    dtype: Optional
        Frozen-mesh dtype, a **backend** dtype such as ``backend.float32`` --
        not a NumPy one. Defaults to ``backend.float64``, the build dtype,
        and is stored on the mesh exactly as given.
    raytrace_batch_size: Optional[int]
        Splits each per-level ``raytrace`` call for memory. Forwarded to
        :func:`refine`.
    index_cells: Optional[int]
        Spatial-index cells along the longer axis of the source-plane
        bounding box. Forwarded to :func:`build_index`.
    centres: Optional[ArrayLike]
        Lens-plane positions of the lens's centres, shape ``(S, 2)``: every
        point where the lens map may jump, such as the centre of an SIE or
        SIS without a core, or a point mass. Each gets a hole of radius
        ``min_img_sep`` -- the halved value -- and centres closer than twice
        that share one (:func:`merge_centres`). The image of each hole's
        boundary is traced and stored as ``AdaptiveMesh.holes``
        (:func:`sample_holes`), with no Jacobian call and at least 4352
        raytraces per hole: 256 initial samples and four growth circles of
        1024. A hole that reaches the cap costs up to
        :data:`HOLE_MAX_SAMPLES` + 4096 raytraces and stores up to
        :data:`HOLE_MAX_SAMPLES` samples. A ``UserWarning`` is raised for
        each hole whose refinement is capped at :data:`HOLE_MAX_SAMPLES`
        samples before every chord falls below ``min_img_sep``; such a hole
        curve is exact at its samples but coarser than ``min_img_sep``
        between them, and only an enormous hole curve, such as a point
        mass's, reaches the cap.

        :func:`~caustics.lenses.func.adaptive.curves.mesh_critical_curves_and_caustics`
        cuts every hole, singular centre or not, out of the curves it traces
        and re-joins them along the hole curves: a critical curve lying
        wholly inside a hole is dropped, with only the hole curve standing in
        for it, and anything that relies on the curves cannot see what lies
        inside a hole. Pass every centre where the lens map may jump. The
        re-joining assumes the mesh reaches its size floor: when
        ``max_depth`` binds, the leaves around a centre can be larger than
        its hole, and curves through it can stay open or keep chords.
        ``None`` stores no hole.

        *Unit: arcsec*

    Returns
    -------
    AdaptiveMesh
    """

    # The parity-condemned band at max_level is model-dependent. Refining to
    # half the requested separation narrows it, without implying a universal
    # bound on its width. Halved once, here, before any use, so every stage
    # below -- validation, the depth floor, max_level, both warnings, refine,
    # and the value stored on the returned mesh -- sees this one halved value
    # and never the caller's original. `requested_min_img_sep` is kept so
    # `validate_build_args` can name what the caller actually passed; the
    # warning helpers recover it as `2 * min_img_sep`, exact since halving is.
    requested_min_img_sep = min_img_sep
    min_img_sep = min_img_sep / 2
    validate_build_args(fov, init_res, min_img_sep, max_depth, requested_min_img_sep)
    d_floor = depth_floor(fov / init_res, min_img_sep)
    max_level = min(int(max_depth), d_floor)
    warn_depth_limited(fov, init_res, min_img_sep, d_floor, max_level)

    tables = child_matrix_tables()
    lat = make_lattice(fov, x0, y0, init_res, max_level + 1)
    raytrace_fn = make_raytrace(raytrace, device)
    hole_centres, hole_radius = merge_centres(centres, min_img_sep)
    holes = sample_holes(
        raytrace_fn, hole_centres, hole_radius, min_img_sep, raytrace_batch_size
    )
    return _grow(
        raytrace_fn,
        jacobian,
        lat,
        tables,
        roots=None,
        seed=None,
        seed_band=empty_band(),
        fov=fov,
        init_res=init_res,
        h0=fov / init_res,
        min_img_sep=min_img_sep,
        d_floor=d_floor,
        max_level=max_level,
        device=device,
        dtype=dtype,
        raytrace_batch_size=raytrace_batch_size,
        index_cells=index_cells,
        holes=holes,
    )


def seed_from_mesh(
    mesh, lat, k, raytrace_fn, batch_size
) -> Tuple[VertexCache, ArrayLike, LeafStore, CriticalBand]:
    """
    Rebuild :func:`refine`'s state from a frozen mesh, on its lattice grown by ``k``.

    A frozen mesh keeps every leaf vertex with its lattice coordinates, and
    every pre-closure leaf with its level, status and orientation class --
    everything :func:`refine`, :func:`balance` and :func:`freeze` read, but
    the midpoints of leaves that never split. :func:`force_split` evaluates
    those on demand, for the few leaves an extension forces.

    - The cache holds the mesh's vertices at their shifted coordinates, slot
      ``i`` being mesh vertex ``i``. The vertices are in lattice-key order and
      the shift keeps it, so the key index is already sorted. ``beta`` is
      ``vertices_source`` at float64. For a mesh not stored at float64 the
      vertices on its outer boundary are raytraced again: they are the only
      old points the ring's criterion reads, and it must read float64 values
      straight from ``raytrace``, as :func:`make_raytrace` explains.
    - ``det`` is ``vertices_det``, with ``has_det`` set and ``evaluated``
      clear. The ring's criterion evaluates the Jacobian again at the old
      vertices it samples, all on the old boundary, for their sign, and each
      keeps its ``det``. Being float64 at every mesh dtype, it needs no
      re-evaluation for a float32 mesh.
    - Every vertex is active: a mesh's vertices are exactly its leaves'.
    - The store has one row per origin, in ``origin_leaves`` order, with the
      level and status of its first leaf: ``leaf_origin`` is non-decreasing
      and every origin has a leaf, so ``searchsorted`` finds it with no
      scatter. Below ``max_level`` the only flag a status can carry is the
      freeze-time ``LEAF_RAYTRACE_NONFINITE``, which :func:`freeze` derives
      again, so those rows are reset to ``LEAF_CONVERGED`` -- which is also
      what a forced child must inherit, as in a fresh build.
    - The band is the mesh's own, its ``leaves`` mapped to store rows, with
      ``lens`` and ``source`` at float64.

    Parameters
    ----------
    mesh: AdaptiveMesh
    lat: Lattice
        ``extend_lattice(mesh.lattice, k)``, ``lo`` on the ambient device.
    k: int
        Level-0 cells added on each side.
    raytrace_fn: Callable[[ArrayLike], ArrayLike]
        From :func:`make_raytrace`, for the boundary of a non-float64 mesh.
    batch_size: Optional[int]
        Forwarded to :func:`trace_keys`.

    Returns
    -------
    cache: VertexCache
    active: ArrayLike
    store: LeafStore
    band: CriticalBand
        ``leaves`` indexing ``store``'s rows.
    """
    pad = int(k) << lat.level
    ij = _ambient(mesh.vertices_ij) + pad
    n_vertices = ij.shape[0]
    beta = backend.to(_ambient(mesh.vertices_source), dtype=backend.float64)
    if mesh.vertices_source.dtype != backend.float64:
        old = ij - pad
        n_old = lat.n - 2 * pad
        edge = backend.flatnonzero(backend.any((old == 0) | (old == n_old), dim=1))
        if edge.shape[0]:
            beta = backend.fill_at_indices(
                backend.copy(beta),
                edge,
                trace_keys(lat, ij[edge], raytrace_fn, batch_size),
            )
    cache = VertexCache(
        keys=lattice_key(lat, ij),
        slots=backend.arange(n_vertices, dtype=backend.int64),
        ij=ij,
        beta=beta,
        det=backend.to(_ambient(mesh.vertices_det), dtype=backend.float64),
        sign=backend.zeros((n_vertices,), dtype=backend.int64),
        has_det=backend.ones((n_vertices,), dtype=backend.bool),
        evaluated=backend.zeros((n_vertices,), dtype=backend.bool),
    )
    active = backend.ones((n_vertices,), dtype=backend.bool)

    n_origins = mesh.origin_leaves.shape[0]
    leaf_origin = _ambient(mesh.leaf_origin)
    first = backend.searchsorted(
        leaf_origin, backend.arange(n_origins, dtype=backend.int64)
    )
    level = _ambient(mesh.leaf_level)[first]
    status = backend.where(
        level == mesh.max_level, _ambient(mesh.leaf_status)[first], LEAF_CONVERGED
    )
    store = LeafStore(
        v=_ambient(mesh.origin_leaves),
        level=level,
        cls=_ambient(mesh.origin_cls),
        status=status,
        valid=backend.ones((n_origins,), dtype=backend.bool),
    )

    old_band = mesh.critical_band
    band = CriticalBand(
        leaves=leaf_origin[_ambient(old_band.leaves)],
        samples=_ambient(old_band.samples),
        lens=backend.to(_ambient(old_band.lens), dtype=backend.float64),
        source=backend.to(_ambient(old_band.source), dtype=backend.float64),
        det=_ambient(old_band.det),
    )
    return cache, active, store, band


def _extension_cells(mesh, fov) -> int:
    """
    Level-0 cells per side that grow ``mesh`` to at least ``fov``.

    The smallest ``k`` with ``mesh.fov + 2 * k * h0 >= fov``, less a
    tolerance of ``1e-9`` cells, so an fov already on a cell boundary is not
    pushed out a further ring by one ulp of rounding.

    Raises
    ------
    ValueError
        If ``fov`` is not finite, or is smaller than ``mesh.fov``.
    """
    fov = float(fov)
    if not math.isfinite(fov):
        raise ValueError(f"fov must be finite, got {fov}")
    h0 = mesh.lattice.scale * (1 << mesh.lattice.level)
    cells = (fov - mesh.fov) / (2.0 * h0)
    if cells < -1e-9:
        raise ValueError(
            f"fov={fov:g} is smaller than the mesh's fov={mesh.fov:g}; an "
            "adaptive mesh can only grow"
        )
    return max(0, math.ceil(cells - 1e-9))


def extend_adaptive_mesh(
    mesh, raytrace, jacobian, fov, *, raytrace_batch_size=None, index_cells=None
) -> AdaptiveMesh:
    """
    Grow an adaptive mesh to a larger fov about the same centre, reusing its refinement.

    For a mesh whose critical curves the fov cuts open -- see
    :func:`~caustics.lenses.func.adaptive.curves.mesh_critical_curves_and_caustics` --
    this adds whole level-0 cells around it instead of rebuilding. The result
    is bit for bit the mesh :func:`build_adaptive_mesh` produces on the same
    lattice: both run the same seed, refine, balance and freeze stages, a
    fresh build seeding from nothing and refining every cell, this seeding
    from ``mesh`` and refining only the new ring. The refinement criterion
    reads only a triangle's own samples, so every verdict inside the old
    domain still holds; only the 2:1 balance reaches across the seam, and
    :func:`balance` settles it to the same coarsest balanced refinement a
    fresh build's cascade reaches.

    A fresh :func:`build_adaptive_mesh` at the new fov computes its
    lattice's corner and spacing from that fov. Those match this mesh's --
    anchored at the original build, see :func:`extend_lattice` -- exactly
    when the arithmetic is exact, as with a dyadic fov, ``init_res`` and
    centre. Otherwise positions differ in the last bit, and the visible
    effect is in closure: :func:`close` picks the diagonal of a leaf with two
    hanging nodes by comparing minimum angles computed from those positions,
    the two candidates often tie, and the rounding decides. On a cored
    isothermal lens extended from fov 3 to 4.2, the refinement itself -- every
    pre-closure leaf, with its level, status and class -- matched the fresh
    build's, while roughly 0.5-1% of the closed leaves were split along the
    other diagonal, which reaches ``leaves``, ``leaf_area2`` and ``index``.
    Critical-band leaves have no hanging node, so critical curves are
    unaffected. A verdict flipped by the rounding is possible in principle
    but was not observed.

    Both paragraphs assume the lens gives the same value at a point whatever
    batch it arrives in -- the assumption :func:`trace_keys` makes for
    ``batch_size``. The extension traces its points in other batches than a
    fresh build does, so with a lens whose vectorized kernels differ in the
    last bit between batches it differs from a fresh build much as two fresh
    builds with different ``raytrace_batch_size`` differ from each other. On
    an EPL ``SinglePlane`` extended from fov 2.5 to 5, the leaves, statuses
    and index came out identical and 13 of about 640,000 image coordinates
    differed by at most two ulp; two fresh builds there differed in 11. That
    is a measurement, not a guarantee: a last-bit difference in an image can
    flip a verdict at a threshold or move a bounding box across an index
    cell. The same holds for the Jacobian: a vertex's ``det A``, and a point's
    sign, are whatever the call that first evaluated the point returned.

    Lens calls: the ring's own refinement, exactly what a fresh build spends
    there; the midpoints inside old leaves the balance force-splits,
    raytraced, with their Jacobians evaluated at freeze; and, for a mesh not
    stored at float64, the vertices on its outer boundary, raytraced again so
    the ring's criterion reads float64 values. Jacobian calls strictly inside
    the old domain land only in old leaves the balance splits. The ring's
    criterion evaluates the Jacobian again at the old vertices on the seam,
    for their sign, and each keeps its ``det A``. Closure, ordering and
    indexing still run over the whole mesh, without a lens call. The holes
    are carried over from ``mesh`` unchanged and cost no lens call.

    Parameters
    ----------
    mesh: AdaptiveMesh
        The mesh to grow. It is not modified.
    raytrace, jacobian:
        The pair ``mesh`` was built from, with the contract of
        :func:`build_adaptive_mesh`. Nothing can check that it is the same
        pair: the mesh deliberately stores no lens.
    fov: float
        Requested side length, about the mesh's own centre. Rounded up to the
        smallest ``mesh.fov + 2 * k * h0`` reaching it, ``h0`` being the
        level-0 cell size, within ``1e-9`` cells; the fov meshed is the
        result's ``fov``.

        *Unit: arcsec*
    raytrace_batch_size: Optional[int]
        As for :func:`build_adaptive_mesh`.
    index_cells: Optional[int]
        As for :func:`build_adaptive_mesh`. Not stored on the mesh, so pass
        the build's value again to match a fresh build.

    Returns
    -------
    AdaptiveMesh
        ``mesh`` itself when ``fov`` needs no new cell. Otherwise a new mesh
        with ``mesh``'s ``min_img_sep``, ``holes``, ``max_level``, ``d_floor``,
        ``dtype``, ``device`` and centre, and ``init_res`` grown by ``2 * k``.

    Raises
    ------
    ValueError
        If ``fov`` is not finite or is smaller than ``mesh.fov``, or if the
        grown lattice would overflow int64 keys.

    Warns
    -----
    UserWarning
        Exactly as :func:`build_adaptive_mesh` of the grown fov would: when
        ``max_depth`` limited the build, and when ``raytrace`` returns a
        dtype whose cancellation floor at the grown fov exceeds
        ``min_img_sep``.
    """
    k = _extension_cells(mesh, fov)
    if k == 0:
        return mesh
    h0 = mesh.lattice.scale * (1 << mesh.lattice.level)
    init_res = mesh.init_res + 2 * k
    fov = mesh.fov + 2 * k * h0
    check_lattice_keys(
        init_res,
        mesh.max_level,
        "Extend by less, or rebuild with a larger min_img_sep or a lower max_depth.",
    )
    warn_depth_limited(fov, init_res, mesh.min_img_sep, mesh.d_floor, mesh.max_level)

    tables = child_matrix_tables()
    lat = extend_lattice(mesh.lattice, k)._replace(lo=_ambient(mesh.lattice.lo))
    raytrace_fn = make_raytrace(raytrace, mesh.device)
    cache, active, store, seed_band = seed_from_mesh(
        mesh, lat, k, raytrace_fn, raytrace_batch_size
    )
    return _grow(
        raytrace_fn,
        jacobian,
        lat,
        tables,
        roots=ring_triangles(init_res, k, lat.level, tables[4]),
        seed=(cache, active, store),
        seed_band=seed_band,
        fov=fov,
        init_res=init_res,
        h0=h0,
        min_img_sep=mesh.min_img_sep,
        d_floor=mesh.d_floor,
        max_level=mesh.max_level,
        device=mesh.device,
        dtype=mesh.dtype,
        raytrace_batch_size=raytrace_batch_size,
        index_cells=index_cells,
        holes=mesh.holes,
    )


def _hole_rings(fov, init_res, x0, y0, centres, min_img_sep) -> Tuple[int, int]:
    """
    Level-0 cells per side that put every centre's hole strictly inside the fov.

    The holes are :func:`merge_centres`'s, at the halved ``min_img_sep`` the
    build gives it, so a merged hole reaches as far as its own radius. The
    smallest ``k`` with ``fov / 2 + k * h0`` beyond every hole's reach --
    its centre's larger offset from ``(x0, y0)``, plus its radius.

    Returns
    -------
    k: int
        Zero when every hole already lies strictly inside.
    outside: int
        How many holes did not.
    """
    hole_centres, radius = merge_centres(centres, min_img_sep)
    if hole_centres.shape[0] == 0:
        return 0, 0
    centre = backend.as_array([x0, y0], dtype=backend.float64)
    reach = backend.max(backend.abs(hole_centres - centre), dim=1) + radius
    outside = int(backend.to_numpy(backend.sum(reach >= fov / 2)))
    if outside == 0:
        return 0, 0
    excess = float(backend.to_numpy(backend.max(reach))) - fov / 2
    return math.floor(excess / (fov / init_res)) + 1, outside


def _curves_cut_by_fov(mesh, curves) -> int:
    """
    How many open curves have an end on ``mesh``'s fov boundary.

    Exact, with no tolerance: every vertex and band sample is placed by
    :func:`lattice_xy` and cast to the mesh dtype alike, so the boundary is
    the vertices' own extreme coordinates, and a crossing on a boundary child
    edge, both of whose samples share that coordinate, reproduces it bit for
    bit (:func:`~caustics.lenses.func.adaptive.curves.crossing_points`). An
    end at a band gap inside the fov does not count.
    """
    open_curves = backend.flatnonzero(~curves.closed)
    n_open = open_curves.shape[0]
    if n_open == 0:
        return 0
    ends = backend.concatenate(
        (curves.offsets[open_curves], curves.offsets[open_curves + 1] - 1)
    )
    xy = curves.lens[ends]
    lo = backend.min(mesh.vertices_lens, dim=0)
    hi = backend.max(mesh.vertices_lens, dim=0)
    on_boundary = backend.any((xy == lo) | (xy == hi), dim=1)
    cut = on_boundary[:n_open] | on_boundary[n_open:]
    return int(backend.to_numpy(backend.sum(cut)))


def build_closed_adaptive_mesh(
    raytrace,
    jacobian,
    fov,
    init_res,
    min_img_sep,
    max_depth=25,
    *,
    growth=1.5,
    max_iters=10,
    x0=0.0,
    y0=0.0,
    device=None,
    dtype=None,
    raytrace_batch_size=None,
    index_cells=None,
    centres=None,
) -> Tuple[AdaptiveMesh, CriticalCurvesAndCaustics]:
    """
    Build an adaptive mesh, then grow its fov until it cuts no critical curve.

    Three stages:

    1. When a hole around one of ``centres`` does not lie strictly inside
       ``fov`` -- its centre outside the fov, or within its own radius of the
       edge -- the fov is widened first, by the fewest whole level-0 cells
       per side that take every hole inside, keeping the level-0 cell size
       ``fov / init_res``, and a ``UserWarning`` names the fov and
       ``init_res`` built instead. A hole the fov cuts leaves the curves
       joined there unreliable (see
       :func:`~caustics.lenses.func.adaptive.curves.join_at_holes`), and
       the fov only grows, so from here on every hole stays inside.
    2. :func:`build_adaptive_mesh` builds the mesh, and
       :func:`~caustics.lenses.func.adaptive.curves.mesh_critical_curves_and_caustics`
       traces its curves.
    3. While an open curve has an end on the fov boundary, and fewer than
       ``max_iters`` extensions have run, :func:`extend_adaptive_mesh` grows
       the mesh to ``growth`` times its fov -- rounded up to whole level-0
       cells, so each step adds at least one per side -- and the curves are
       traced again. Each extension reuses every lens evaluation already
       made, so the result is exactly the mesh the build and that chain of
       extensions give.

    Only the fov is ever grown for. A curve can end inside the fov too --
    next to a leaf whose samples or Jacobian are non-finite, at a hole whose
    ends do not alternate, or where ``max_depth`` bound refinement -- and no
    growth closes it, so such a curve stops no loop and triggers none; the
    returned ``curves`` still mark it open. Nor can a critical curve that
    never reaches the fov be found: the loop grows only towards curves the
    fov already cuts.

    Growth compounds: ``max_iters`` steps of ``growth`` can reach
    ``growth**max_iters`` times the fov -- about 58 for the defaults -- and
    the mesh's level-0 cells with its square. A lattice too large for int64
    keys raises :func:`extend_adaptive_mesh`'s ``ValueError``.

    Parameters
    ----------
    raytrace, jacobian, fov, init_res, min_img_sep, max_depth, x0, y0, device,
    dtype, raytrace_batch_size, index_cells, centres:
        As for :func:`build_adaptive_mesh`, which receives all of them, with
        ``fov`` and ``init_res`` widened over the holes as stage 1 says.
        ``raytrace``, ``jacobian``, ``raytrace_batch_size`` and
        ``index_cells`` reach every extension too.
    growth: float
        Factor each extension asks to multiply the fov by, before rounding
        up to whole cells. Finite and greater than 1.
    max_iters: int
        Most extensions to run, at least zero. Widening over the holes is not
        one of them. Zero builds once and only checks.

    Returns
    -------
    mesh: AdaptiveMesh
        The last mesh built or extended.
    curves: CriticalCurvesAndCaustics
        ``mesh_critical_curves_and_caustics(mesh)``, as the last check traced it.

    Raises
    ------
    ValueError
        Before any lens call, if ``growth`` is not finite and greater than 1,
        if ``max_iters`` is not a non-negative integer, or if the build's own
        arguments are invalid; later, as :func:`build_adaptive_mesh` and
        :func:`extend_adaptive_mesh` raise.

    Warns
    -----
    UserWarning
        When the fov is widened over the holes, and when the fov still cuts
        a curve after ``max_iters`` extensions -- the last mesh is returned
        anyway, every lens evaluation kept. Also whatever
        :func:`build_adaptive_mesh` and :func:`extend_adaptive_mesh` warn.
    """
    growth = float(growth)
    if not (math.isfinite(growth) and growth > 1.0):
        raise ValueError(f"growth must be finite and greater than 1, got {growth}")
    try:
        iters = operator.index(max_iters)
    except TypeError:
        raise ValueError(
            f"max_iters must be a non-negative integer, got {max_iters!r}"
        ) from None
    if iters < 0:
        raise ValueError(f"max_iters must be a non-negative integer, got {iters}")
    # The build halves min_img_sep, and gives its holes that halved radius;
    # the holes are widened over, and the arguments checked, at the same value.
    half_sep = min_img_sep / 2
    validate_build_args(fov, init_res, half_sep, max_depth, min_img_sep)

    k, outside = _hole_rings(fov, init_res, x0, y0, centres, half_sep)
    if k:
        widened = fov + 2 * k * (fov / init_res)
        warn(
            f"{outside} hole(s) around centres reach outside fov={fov:g}; "
            f"building at fov={widened:g}, init_res={init_res + 2 * k} so that "
            "every hole lies inside it.",
            UserWarning,
            stacklevel=2,
        )
        fov, init_res = widened, init_res + 2 * k

    mesh = build_adaptive_mesh(
        raytrace,
        jacobian,
        fov,
        init_res,
        min_img_sep,
        max_depth,
        x0=x0,
        y0=y0,
        device=device,
        dtype=dtype,
        raytrace_batch_size=raytrace_batch_size,
        index_cells=index_cells,
        centres=centres,
    )
    curves = mesh_critical_curves_and_caustics(mesh)
    cut = _curves_cut_by_fov(mesh, curves)
    for _ in range(iters):
        if cut == 0:
            break
        mesh = extend_adaptive_mesh(
            mesh,
            raytrace,
            jacobian,
            growth * mesh.fov,
            raytrace_batch_size=raytrace_batch_size,
            index_cells=index_cells,
        )
        curves = mesh_critical_curves_and_caustics(mesh)
        cut = _curves_cut_by_fov(mesh, curves)
    if cut:
        warn(
            f"The fov still cuts {cut} critical curve(s) after {iters} "
            f"extension(s), at fov={mesh.fov:g}; raise max_iters or growth "
            "to grow it further.",
            UserWarning,
            stacklevel=2,
        )
    return mesh, curves
