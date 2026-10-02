"""
Level-synchronous refinement and the 2:1 balance.

:func:`refine` tests every active triangle of a level in one batch and splits
the failures. Its per-level cascade, and :func:`balance` after it, force-split
every leaf left two or more levels coarser than a neighbour.
"""

from typing import Tuple

from ....backend_obj import ArrayLike, backend
from .geometry import _CHILD_VERTEX_INDEX_TABLE
from .state import (
    LeafStore,
    VertexCache,
    active_add_slots,
    active_contains,
    cache_lookup,
    cache_size,
    empty_active,
    empty_cache,
    empty_store,
    store_add,
    store_remove,
)
from .lattice import (
    initial_triangles,
    lattice_ij_from_key,
    lattice_key,
    lattice_xy,
    midpoint_ij,
)
from .sampling import evaluate, sample_jacobians, trace_keys
from .criterion import (
    LEAF_CONVERGED,
    LEAF_RAYTRACE_NONFINITE,
    apply_jacobian_signs,
    approximate_criterion,
    evaluate_criterion,
    jacobian_rows,
)
from .band import band_from_samples, empty_band

__all__ = (
    "red_split",
    "find_unbalanced",
    "force_split",
    "refine",
    "balance",
)


def red_split(v, m, cls, compose) -> Tuple[ArrayLike, ArrayLike]:
    """
    Split into the four canonical children, triangle-major.

    Parameters
    ----------
    v: ArrayLike
        Triangle vertex slots, shape ``(n, 3)`` int64.
    m: ArrayLike
        Edge-midpoint slots ``m1, m2, m3``, shape ``(n, 3)`` int64.
    cls: ArrayLike
        Orientation class of each triangle, shape ``(n,)`` int64.
    compose: ArrayLike
        ``COMPOSE`` from :func:`child_matrix_tables`, shape ``(6, 4)``.

    Returns
    -------
    child_v: ArrayLike
        ``(4n, 3)`` vertex slots, ordered as all four children of triangle 0,
        then triangle 1, and so on.
    child_cls: ArrayLike
        ``(4n,)`` orientation classes, in the same order.
    """
    six = backend.concatenate([v, m], dim=1)  # (n, 6)
    child_v = six[:, _CHILD_VERTEX_INDEX_TABLE].reshape(-1, 3)
    child_cls = compose[cls].reshape(-1)
    return child_v, child_cls


def find_unbalanced(store, cache, lat, active, max_level, frontier_level) -> ArrayLike:
    """
    Rows of ``store`` carrying an active quarter point on some edge.

    Two independent level bounds apply. ``level <= frontier_level - 2`` is an
    optimization: only triangles at least two levels coarser than the
    frontier can have been invalidated by it, so the scan skips most of the
    store. ``level <= max_level - 2`` is the bound above which no neighbour
    can be two levels finer, since ``max_level`` is the finest level there is.

    Parameters
    ----------
    store: LeafStore
        Terminal triangles; only rows with ``valid`` set are scanned.
    cache: VertexCache
        Supplies the lattice coordinates of each row's vertices.
    lat: Lattice
        Used to key the quarter points.
    active: ArrayLike
        Pre-existing active-vertex set; membership decides a violation.
    max_level: int
        Finest allowed level.
    frontier_level: int
        Level of the finest triangles created in the current round.

    Returns
    -------
    ArrayLike
        Int64 indices into ``store``'s rows.
    """
    # `frontier_level <= max_level` holds for every call the level loop
    # makes, so the first term always binds and the second is unreachable
    # defensive code today. Keep the min(): `max_level - 2` is the level
    # above which no neighbour can be two levels finer, since `max_level` is
    # the finest level there is. (It used to double as an integrality
    # requirement for the quarter-point arithmetic below; on the widened
    # lattice quarter points stay lattice points down to `max_level - 1`, so
    # that role is gone and only the balance argument remains.)
    bound = min(frontier_level - 2, max_level - 2)
    cand = backend.flatnonzero(store.valid & (store.level <= bound))
    if cand.shape[0] == 0:
        return cand
    ij = cache.ij[store.v[cand]]  # (n, 3, 2)
    # One edge at a time, accumulating into a single `(n,)` mask, rather than
    # building the whole `(cand, 6)` key array the way `edge_quarter_keys`
    # does: the batched form held 153 MB at a million leaves -- the largest
    # transient in the level loop. The disjunction is over the same six keys
    # `edge_quarter_keys` returns; only the association changes.
    hit = backend.zeros((cand.shape[0],), dtype=backend.bool)
    for e in range(3):
        a = ij[:, e, :]
        b = ij[:, (e + 1) % 3, :]
        delta = (b - a) // 4
        for quarter in (a + delta, b - delta):
            hit = hit | active_contains(active, cache, lattice_key(lat, quarter))
    return cand[hit]


def force_split(
    store, cache, lat, active, violators, compose, raytrace_fn, batch_size
) -> Tuple[LeafStore, VertexCache, ArrayLike, ArrayLike, ArrayLike]:
    """
    Split balance violators into auto-converged children.

    The body of every balance cascade, shared by :func:`refine`'s per-level
    one and :func:`balance`'s fixed point. A forced child is auto-converged
    -- steps 3-7 are skipped so the cascade cannot re-enter the split
    machinery from inside itself -- and has no flag of its own: it inherits
    its parent's status. A violator is bounded to ``level <= max_level - 2``
    by :func:`find_unbalanced`, and only ``max_level`` rows carry a stored
    flag, so that status is always ``LEAF_CONVERGED``, transitively. A forced
    child can still reach freeze with a non-finite vertex;
    :func:`invalidate_nonfinite_origins` is left to catch that.

    The violators' midpoints are evaluated here when not yet cached. In a
    fresh build they always are -- every tested triangle cached its
    midpoints, and a forced child's are drained from ``deferred`` at the top
    of the next level, before any later cascade can re-force it -- so
    :func:`evaluate` finds nothing missing and makes no raytrace call. A leaf
    seeded from a frozen mesh has only its vertices cached, so an extension
    that forces one traces its midpoints here.

    Parameters
    ----------
    store: LeafStore
    cache: VertexCache
    lat: Lattice
    active: ArrayLike
    violators: ArrayLike
        Store rows to split, from :func:`find_unbalanced`.
    compose: ArrayLike
        ``COMPOSE`` from :func:`child_matrix_tables`.
    raytrace_fn: Callable[[ArrayLike], ArrayLike]
        From :func:`make_raytrace`, for midpoints not yet cached.
    batch_size: Optional[int]
        Forwarded to :func:`evaluate`.

    Returns
    -------
    store: LeafStore
        ``violators`` removed and their children added.
    cache: VertexCache
        With every violator midpoint evaluated.
    active: ArrayLike
        With the children's vertices activated.
    kid_mid: ArrayLike
        ``(12k,)`` int64 lattice keys of the children's edge midpoints, for
        the caller to defer or drop.
    kid_level: ArrayLike
        ``(4k,)`` int64 level of each child.

    Raises
    ------
    AssertionError
        "cascade hit an unevaluated midpoint" if a midpoint is uncached even
        after evaluation, where a ``-1`` slot would silently negative-index
        ``cache.ij`` into wrong geometry.
    """
    vv = store.v[violators]
    mid_keys = lattice_key(lat, midpoint_ij(cache.ij[vv]))
    cache = evaluate(cache, lat, mid_keys.reshape(-1), raytrace_fn, batch_size)
    vm = cache_lookup(cache, mid_keys)
    # `raise AssertionError` rather than a bare `assert`: `python -O` strips
    # bare asserts, and this one guards against silent geometric corruption.
    if not bool(backend.all(vm >= 0)):
        raise AssertionError("cascade hit an unevaluated midpoint")
    kid_v, kid_cls = red_split(vv, vm, store.cls[violators], compose)
    kid_level = backend.repeat(store.level[violators] + 1, 4, axis=0)
    kid_status = backend.repeat(store.status[violators], 4, axis=0)
    store = store_remove(store, violators)
    store, _ = store_add(store, kid_v, kid_level, kid_cls, kid_status)
    active = active_add_slots(active, cache_size(cache), kid_v.reshape(-1))
    kid_mid = lattice_key(lat, midpoint_ij(cache.ij[kid_v])).reshape(-1)
    return store, cache, active, kid_mid, kid_level


def refine(
    raytrace_fn,
    jacobian_fn,
    lat,
    init_res,
    h0,
    min_img_sep,
    max_level,
    tables,
    batch_size,
    *,
    roots=None,
    seed=None,
):
    """
    Level-synchronous refinement.

    Processes the whole active set one level at a time: gathers all unique new
    points for that level, calls ``raytrace`` once on the batch, applies the
    criterion vectorized, then partitions into converged and to-split. No
    Python-level recursion over individual triangles, no per-triangle ``raytrace``.

    **The criterion converges no triangle without the Jacobian's agreement, at
    any level.** :func:`evaluate_criterion` evaluates ``jacobian_fn`` on every
    triangle that passes all its other tests, and one whose six samples
    disagree on ``sign(det A)`` -- or where some ``A`` is non-finite or
    singular -- splits instead of converging. So no leaf the criterion
    converged has a critical curve running between its samples, while the
    Jacobian cost stays proportional to the triangles about to converge rather
    than to every triangle tested. A balance-cascade child is the exception:
    it inherits its parent's verdict without a Jacobian check of its own, and
    its three edge midpoints are points the parent's check never saw. Below
    ``max_level``, unlike raytraced points, Jacobian evaluations are not
    deduplicated: every checked triangle evaluates its own six points, shared
    vertices included. The per-level status bitmask is discarded below
    ``max_level``, since every failure there is a reason to split rather than
    a verdict.

    At ``max_level`` nothing splits, so there is no cascade, so no force-split.
    The midpoints are still evaluated: traced once, deduplicated on their
    lattice keys, and still never added to the vertex cache -- they are the
    only points with an odd coordinate, so they could never collide with a
    cache entry regardless. Their images are kept only where the critical
    band needs them. The full criterion still runs on them, with the
    Jacobian forced on every triangle whose six samples are finite, since
    here ``status`` is the leaf's final record: every test the leaf fails is
    OR-ed into it, and anything but ``LEAF_CONVERGED`` keeps the leaf out of
    the spatial index. A triangle that straddles a fold, in particular,
    contains it at a scale no further split can resolve. That forced pass
    reads every vertex a level above already evaluated from the cache and
    evaluates the rest, the never-cached midpoints among them, once each
    (:func:`sample_jacobians`), and the same values build the
    :class:`CriticalBand` (:func:`band_from_samples`) -- the leaves ``det A``
    changes sign across, with ``det A`` and the image at their samples,
    midpoint images included, that would otherwise be dropped. A triangle
    with a non-finite sample skips the criterion and is stored with
    ``LEAF_RAYTRACE_NONFINITE`` alone.

    The ``max_level`` midpoint pass is the single largest batch in the build.
    For a mesh refining uniformly to ``max_level`` on an ``N x N`` cell grid it
    adds the ``3 * N**2 + 2 * N`` edge midpoints to the ``(N + 1)**2``
    vertices, which together are exactly the ``(2 * N + 1)**2`` points of the
    widened lattice -- so a full-depth build now evaluates every lattice point
    exactly once, where it used to evaluate only the even sublattice. Roughly
    ``4x`` the ``raytrace`` calls in that worst case, and less on a genuinely
    adaptive mesh. ``counters["max_level_midpoints"]`` is the measured cost.

    **Non-finite triangles split unconditionally.** A non-finite sample point is
    maximal ignorance about a triangle, so it triggers refinement like every other
    unresolved condition rather than terminating it -- the criterion simply cannot
    be evaluated there, which is why the split carries no verdict. A red split
    hands the bad vertex to exactly one of the four children, so the other three
    re-enter the criterion normally and the singularity ends up ringed by a band of
    ``LEAF_RAYTRACE_NONFINITE`` leaves at the smallest allowed size instead of a
    hexagon of ``fov / init_res``. Terminating on the spot instead would put the
    hole at whatever level the triangle was first sampled, and such leaves are
    excluded from the spatial index -- so on a singular model that hole is exactly
    the region where the mesh is most needed. The cost is
    ``counters["nonfinite_splits"]``: six triangles per level for a point
    singularity, but ``O(area * 4**max_level)`` should a ``raytrace`` return
    non-finite values over a whole region.

    Parameters
    ----------
    raytrace_fn: Callable[[ArrayLike], ArrayLike]
        From :func:`make_raytrace`.
    jacobian_fn: Callable[[ArrayLike, ArrayLike], ArrayLike]
        ``jacobian_fn(x, y) -> (K, 2, 2)``, the Jacobian of the map
        ``raytrace_fn`` traces, forwarded to :func:`evaluate_criterion`. Called
        directly on the lattice's float64 lens-plane coordinates: not through
        :func:`make_raytrace`, and not chunked by ``batch_size``.
    lat: Lattice
    init_res: int
        Level-0 grid resolution.
    h0: float
        Level-0 cell size ``fov / init_res``.

        *Unit: arcsec*
    min_img_sep: float
        Lens-plane tolerance.

        *Unit: arcsec*
    max_level: int
        Finest allowed level.
    tables:
        ``(M, G, COMPOSE, PINV0, ROOT_CLASS)`` from :func:`child_matrix_tables`.
    batch_size: Optional[int]
        Forwarded to :func:`evaluate` and :func:`trace_keys`.
    roots: Optional[Tuple[ArrayLike, ArrayLike]]
        Level-0 triangles to refine, ``(ij, cls)`` as
        :func:`initial_triangles` returns them. ``None`` refines every cell of
        the ``init_res x init_res`` grid; an extension passes only its ring,
        from :func:`ring_triangles`.
    seed: Optional[Tuple[VertexCache, ArrayLike, LeafStore]]
        ``(cache, active, store)`` to continue from, as
        :func:`seed_from_mesh` rebuilds them; ``None`` starts empty. Seeded
        rows take part in the balance cascade like any other, but only
        ``roots`` and their descendants are tested. The cascade's frontier
        bound assumes the leaves were balanced before each level, which a
        seed need not be, so a seeded run must be followed by
        :func:`balance`.

    Returns
    -------
    cache: VertexCache
    active: ArrayLike
        The active-vertex set.
    store: LeafStore
        Every row below ``max_level`` is ``LEAF_CONVERGED``; ``max_level`` rows
        carry the full status bitmask.
    counters: dict[str, int]
        ``converged_level0``: leaves converged at level 0.
        ``parity_splits``: level ``< max_level`` triangles split on parity --
        child parity, or Jacobian parity for a triangle that would otherwise
        have converged, a non-finite or singular Jacobian included.
        ``parity_invalid``: ``max_level`` triangles failing child parity or
        Jacobian parity, the Jacobian evaluated on all of them.
        ``deviation_splits``: triangles passing child parity but split by the
        deviation test. The Jacobian is never evaluated on these.
        ``sigma_zero``: triangles seen with ``s == 0`` (parity-ok or not).
        ``nonfinite_splits``: level ``< max_level`` triangles split because some
        sample was non-finite.
        ``max_level_midpoints``: unique ``max_level`` midpoints traced for the
        parity test.
        ``jacobian_points``: points passed to ``jacobian_fn``.
        ``forced``: children produced by the balance cascade.
        ``cascade_rounds``: balance-cascade rounds run over the whole build.
    band: CriticalBand
        The critical band, with ``leaves`` indexing ``store``'s rows. Empty
        when the loop ends before ``max_level``.
    """
    M, G, COMPOSE, PINV0, ROOT_CLASS = tables
    if seed is None:
        cache, active, store = empty_cache(), empty_active(), empty_store()
    else:
        cache, active, store = seed
    counters = {
        "converged_level0": 0,
        "parity_splits": 0,
        "parity_invalid": 0,
        "deviation_splits": 0,
        "sigma_zero": 0,
        "nonfinite_splits": 0,
        "max_level_midpoints": 0,
        "jacobian_points": 0,
        "forced": 0,
        "cascade_rounds": 0,
    }

    if roots is None:
        active_ij, active_cls = initial_triangles(init_res, lat.level, ROOT_CLASS)
    else:
        active_ij, active_cls = roots
    deferred = backend.empty((0,), dtype=backend.int64)
    band = empty_band()

    for level in range(max_level + 1):
        vert_keys = lattice_key(lat, active_ij)  # (n, 3)
        need = [vert_keys.reshape(-1)]
        if level < max_level:
            mid_ij = midpoint_ij(active_ij)
            need.append(lattice_key(lat, mid_ij).reshape(-1))
            need.append(deferred)
            deferred = backend.empty((0,), dtype=backend.int64)
        cache = evaluate(
            cache, lat, backend.concatenate(need, dim=0), raytrace_fn, batch_size
        )

        v = cache_lookup(cache, vert_keys)
        active = active_add_slots(active, cache_size(cache), v.reshape(-1))
        beta_v = cache.beta[v]
        finite_v = backend.all(backend.isfinite(beta_v), dim=(1, 2))

        if level == max_level:
            # has_converged = backend.zeros((active_ij.shape[0],), dtype=backend.bool)
            # finite_samples = backend.zeros((active_ij.shape[0],), dtype=backend.bool)
            status = (
                backend.zeros((active_ij.shape[0],), dtype=backend.int64)
                + LEAF_RAYTRACE_NONFINITE
            )
            rows = backend.flatnonzero(finite_v)

            if rows.shape[0]:
                # Trace unique midpoints without adding them to the vertex cache.
                # Flattened before `backend.unique` rather than leaning on a
                # shape-preserving `return_inverse`, so the reshapes below are
                # explicit and this does not depend on the backend's own
                # convention.
                keys = lattice_key(lat, midpoint_ij(active_ij[rows])).reshape(-1)
                uniq, inv = backend.unique(keys, return_inverse=True)
                inv = inv.reshape(-1)
                mid_beta = trace_keys(
                    lat, lattice_ij_from_key(lat, uniq), raytrace_fn, batch_size
                )
                beta_m = mid_beta[inv].reshape(-1, 3, 2)
                counters["max_level_midpoints"] = int(uniq.shape[0])

                # Only evaluate the criterion where all six samples are finite.
                finite_m = backend.all(backend.isfinite(beta_m), dim=(1, 2))
                good_rows = rows[finite_m]

                if good_rows.shape[0]:
                    good_ij = active_ij[good_rows]
                    six_ij = backend.concatenate((good_ij, midpoint_ij(good_ij)), dim=1)
                    approx_status, child_ok, s = approximate_criterion(
                        beta_v[good_rows],
                        beta_m[finite_m],
                        active_cls[good_rows],
                        level,
                        h0,
                        min_img_sep,
                        PINV0,
                        COMPOSE,
                    )
                    # Forced, so every good row is tested. One Jacobian call at
                    # most, over the samples not yet evaluated -- the midpoints,
                    # never cached, among them -- so that shared samples carry
                    # one `det A` and sign, which the critical band needs.
                    tested = jacobian_rows(approx_status, True)
                    cache, sample_keys, sample_index, det, sign, called = (
                        sample_jacobians(lat, cache, six_ij, jacobian_fn)
                    )
                    counters["jacobian_points"] += called
                    _, parity_ok, criterion_status = apply_jacobian_signs(
                        approx_status, child_ok, tested, sign[sample_index][tested]
                    )

                    status = backend.fill_at_indices(
                        status, good_rows, criterion_status
                    )
                    counters["parity_invalid"] = int(
                        backend.to_numpy(backend.sum(~parity_ok))
                    )
                    counters["sigma_zero"] += int(backend.to_numpy(backend.sum(s == 0)))

                    band = band_from_samples(
                        lat, cache, sample_keys, sample_index, det, uniq, mid_beta
                    )
                    band = band._replace(leaves=good_rows[band.leaves])

            has_converged = status == LEAF_CONVERGED

            store, max_rows = store_add(store, v, level, active_cls, status)
            band = band._replace(leaves=max_rows[band.leaves])

            if level == 0:
                counters["converged_level0"] = int(
                    backend.to_numpy(backend.sum(has_converged))
                )

            break

        m = cache_lookup(cache, lattice_key(lat, mid_ij))
        beta_m = cache.beta[m]
        good = finite_v & backend.all(backend.isfinite(beta_m), dim=(1, 2))
        counters["nonfinite_splits"] += int(backend.to_numpy(backend.sum(~good)))

        rows = backend.flatnonzero(good)

        theta_v = lattice_xy(lat, active_ij[rows])
        theta_m = lattice_xy(lat, mid_ij[rows])

        keep, parity_ok, s, _status = evaluate_criterion(
            jacobian_fn,
            theta_v,
            theta_m,
            beta_v[rows],
            beta_m[rows],
            active_cls[rows],
            level,
            h0,
            min_img_sep,
            PINV0,
            COMPOSE,
        )
        counters["parity_splits"] += int(backend.to_numpy(backend.sum(~parity_ok)))
        counters["deviation_splits"] += int(
            backend.to_numpy(backend.sum(parity_ok & ~keep))
        )
        counters["sigma_zero"] += int(backend.to_numpy(backend.sum(s == 0)))

        # A non-finite triangle joins the criterion's failures in `pending`
        # rather than terminating: the criterion cannot be evaluated on it, so
        # the split is unconditional. Sorted, so which reason condemned a
        # triangle never reaches the child ordering.
        done = rows[keep]
        pending = backend.sort(
            backend.concatenate((backend.flatnonzero(~good), rows[~keep]), dim=0)
        )
        store, _ = store_add(store, v[done], level, active_cls[done], LEAF_CONVERGED)
        if level == 0:
            counters["converged_level0"] = int(done.shape[0])

        child_v, child_cls = red_split(
            v[pending], m[pending], active_cls[pending], COMPOSE
        )
        child_ij = cache.ij[child_v]
        active = active_add_slots(active, cache_size(cache), child_v.reshape(-1))

        # Balance cascade. The children above are already registered as active
        # vertices, which is what makes the quarter-point test able to see them --
        # the split must precede the cascade, not follow it.
        frontier_level = level + 1
        while True:
            violators = find_unbalanced(
                store, cache, lat, active, max_level, frontier_level
            )
            if violators.shape[0] == 0:
                break
            counters["cascade_rounds"] += 1
            store, cache, active, kid_mid, kid_level = force_split(
                store, cache, lat, active, violators, COMPOSE, raytrace_fn, batch_size
            )
            counters["forced"] += int(kid_level.shape[0])
            # Forced children are produced after this level's raytrace call has
            # gone out, and land at levels the loop will never revisit. Queue
            # their midpoints and drain at the top of the next level, so the
            # one-batch-per-level structure survives. A forced child cannot be
            # re-forced within the same cascade, because the frontier only moves
            # coarser -- so the deferral is never more than one level deep.
            deferred = backend.concatenate((deferred, kid_mid), dim=0)
            # The MAXIMUM kid level, not the minimum: a kid at level L invalidates
            # neighbours at level <= L-2, so the minimum would skip violators.
            # The maximum strictly decreases each round, which terminates the loop.
            frontier_level = int(backend.to_numpy(backend.max(kid_level)))

        active_ij, active_cls = child_ij, child_cls
        if active_ij.shape[0] == 0:
            break

    return cache, active, store, counters, band


def balance(
    store, cache, lat, active, max_level, compose, raytrace_fn, batch_size
) -> Tuple[LeafStore, VertexCache, ArrayLike, int]:
    """
    Force-split until no leaf is unbalanced: the coarsest 2:1-balanced refinement.

    :func:`refine`'s cascade scans only rows at ``level <= frontier_level -
    2``, which is complete only when the leaves were balanced before each
    level. That holds in a fresh build, whose new vertices only ever appear
    at the frontier, and fails in an extension, whose ring starts at level 0
    beside old leaves as deep as ``max_level``. This scans every valid row at
    ``level <= max_level - 2`` instead, splits what it finds with
    :func:`force_split`, and repeats until a scan finds nothing.

    Every row split has an active quarter point, and active vertices are only
    ever added, so every balanced refinement containing the current leaves
    splits that row too. The fixed point is therefore the coarsest balanced
    refinement of the leaves it started from -- the unique mesh a fresh
    build's incremental cascade reaches -- whatever order rows are met in.
    Rounds are bounded by the largest level gap between neighbours, at most
    about ``max_level``, each one scan of the store. After a fresh
    :func:`refine` the first scan finds nothing. ``max_level`` rows are never
    split, so every critical-band row stays a leaf.

    Parameters
    ----------
    store: LeafStore
    cache: VertexCache
    lat: Lattice
    active: ArrayLike
    max_level: int
    compose: ArrayLike
        ``COMPOSE`` from :func:`child_matrix_tables`.
    raytrace_fn: Callable[[ArrayLike], ArrayLike]
        Forwarded to :func:`force_split`, for midpoints not yet cached.
    batch_size: Optional[int]
        Forwarded to :func:`force_split`.

    Returns
    -------
    store: LeafStore
    cache: VertexCache
    active: ArrayLike
    forced: int
        Children created; zero when the leaves were already balanced.
    """
    forced = 0
    while True:
        violators = find_unbalanced(store, cache, lat, active, max_level, max_level)
        if violators.shape[0] == 0:
            return store, cache, active, forced
        store, cache, active, _, kid_level = force_split(
            store, cache, lat, active, violators, compose, raytrace_fn, batch_size
        )
        forced += int(kid_level.shape[0])
