"""
Adaptively refined triangular mesh of the lens plane, queryable from the source plane.

``forward_raytrace`` rebuilds a mesh on every query and can miss images when the
initial grid is coarse enough that a mapped triangle is not locally affine. This
module builds a mesh **once**, using a refinement criterion that depends only on the
lens map and not on any query point, then answers many source-plane queries against
the frozen result.

The build is a host-side NumPy float64 algorithm whose only array-API contact is the
``raytrace`` callback; the frozen mesh and ``Mesh.query`` are backend-dispatched.

Two layers sit on the frozen mesh. :meth:`Mesh.query` and its accessors return
*candidate regions* and Newton seeds; :meth:`Mesh.forward_raytrace` and
:meth:`Mesh.multiplicity_map` go on to return images. Their ``method``
argument chooses how: ``"rootfind"`` refines every seed to machine precision,
``"dedup"`` deduplicates the seeds as they stand and never calls ``raytrace``
at all.

The distinction matters and is not cosmetic. **Candidate count is not image
multiplicity** -- a point on a shared edge returns both leaves, and near-critical
leaves overlap -- so anything that needs a count has to go through the root-finding
layer, which is what :meth:`Mesh.multiplicity_map` is.

Non-goals, by design: no autodiff, no jit, no vmap. :meth:`Mesh.query` has
data-dependent output shapes, so the mesh is structurally unjittable rather than
merely undocumented for it.
"""

from dataclasses import dataclass
from enum import IntEnum
from math import ceil, log2
from typing import Any, Callable, Optional, Tuple
from warnings import warn

import numpy as np

from ..backend_obj import ArrayLike, backend
from ..utils import batch_lm, meshgrid
from .func.adaptive import (
    CHILD_VERTEX_INDICES,
    ROOT_SHAPES,
    child_matrix_tables,
    contains,
    evaluate_criterion,
    sanitize_bary,
    shape_matrix,
    triangle_weights,
)

__all__ = ["LeafStatus", "BuildStats", "Mesh", "build_adaptive_mesh"]

_MAX_KEY = 2**63 - 1

_METHODS = ("rootfind", "dedup")


def _check_method(method) -> None:
    """Reject an unrecognised image-finding method, naming the alternatives."""
    if method not in _METHODS:
        raise ValueError(f"method must be one of {_METHODS}, got {method!r}")


class LeafStatus(IntEnum):
    """Why a terminal leaf stopped refining.

    ``FORCED`` is distinct from ``CONVERGED`` because a forced child carries no
    criterion evidence at all -- that is exactly what auto-converging decides -- so
    a caller auditing coverage must be able to tell them apart. Closure triangles
    have no status of their own; they inherit their origin's.

    ``INVALID`` is not a refinement outcome the criterion can reach: a non-finite
    triangle is split unconditionally, so ``INVALID`` arises only at ``max_level``,
    where no split is left, plus the freeze-time propagation in
    :func:`_invalidate_nonfinite_origins`. That bounds the coverage hole around a
    singularity by the ``max_level`` leaf size rather than by ``fov / init_res``.
    """

    CONVERGED = 0
    SIZE_FLOOR = 1
    FORCED = 2
    INVALID = 3


def _depth_floor(fov, init_res, min_img_sep) -> int:
    """
    Level at which the longest leaf edge first falls to ``min_img_sep``.

    Red refinement makes every child similar to its parent with ratio 1/2, so
    ``l_max`` is a function of level alone and the size floor is a depth computable
    up front. ``sqrt(2) * fov / init_res`` is the level-0 hypotenuse.
    """
    l_max0 = np.sqrt(2.0) * fov / init_res
    if l_max0 <= min_img_sep:
        return 0
    return int(ceil(log2(l_max0 / min_img_sep)))


def _validate_build_args(fov, init_res, min_img_sep, max_depth) -> None:
    """Reject impossible parameters, including a lattice that would overflow int64."""
    if not fov > 0:
        raise ValueError(f"fov must be positive, got {fov}")
    if init_res < 1:
        raise ValueError(f"init_res must be at least 1, got {init_res}")
    if not min_img_sep > 0:
        raise ValueError(f"min_img_sep must be positive, got {min_img_sep}")
    if max_depth < 0:
        raise ValueError(f"max_depth must be non-negative, got {max_depth}")
    max_level = min(max_depth, _depth_floor(fov, init_res, min_img_sep))
    n = init_res * (1 << max_level)
    if (n + 1) ** 2 >= _MAX_KEY:
        raise ValueError(
            f"lattice too fine to key in int64: init_res={init_res} at level "
            f"{max_level} needs {n + 1} points per axis. Raise min_img_sep, "
            f"lower max_depth, or lower init_res."
        )


class _Lattice:
    """
    Dyadic integer lattice over the square domain, at the finest allowed level.

    Every mesh vertex is an integer pair, so midpoints are exact integer averages
    -- no float hashing, no rounding tolerance. The lens-plane position is a pure
    function of the integer pair, so two triangles sharing a vertex compute
    bit-identical coordinates. That is the root of the exact-negation property that
    stops a query falling through the seam between adjacent leaves.
    """

    def __init__(self, fov, x0, y0, init_res, max_level):
        self.n = int(init_res) * (1 << int(max_level))
        self.stride = self.n + 1
        self.scale = float(fov) / self.n
        self.lo = np.array([x0 - fov / 2.0, y0 - fov / 2.0], dtype=np.float64)

    def key(self, ij):
        """Lattice key of integer coordinates, shape ``(..., 2) -> (...)``."""
        return ij[..., 0] * self.stride + ij[..., 1]

    def ij_from_key(self, key):
        """Inverse of :meth:`key`, shape ``(...) -> (..., 2)``."""
        return np.stack((key // self.stride, key % self.stride), axis=-1)

    def xy(self, ij):
        """Lens-plane position, shape ``(..., 2) -> (..., 2)``.

        *Unit: arcsec*
        """
        return self.lo + ij.astype(np.float64) * self.scale

    def on_boundary(self, ij):
        """True where the point lies on the edge of the domain."""
        return (
            (ij[..., 0] == 0)
            | (ij[..., 0] == self.n)
            | (ij[..., 1] == 0)
            | (ij[..., 1] == self.n)
        )


class _VertexCache:
    """
    Lattice key to slot, with the source-plane image of every evaluated point.

    Lookup is ``np.searchsorted`` against a sorted key array rather than a Python
    dict, so a whole level's worth of points resolves in one vectorized call. Slots
    are assigned monotonically in order of first evaluation and never move.
    """

    def __init__(self):
        self._keys = np.empty(0, dtype=np.int64)
        self._slots = np.empty(0, dtype=np.int64)
        self.ij = np.empty((0, 2), dtype=np.int64)
        self.beta = np.empty((0, 2), dtype=np.float64)

    def __len__(self):
        return self.ij.shape[0]

    def lookup(self, keys):
        """Slot of each key, or ``-1`` where absent."""
        if self._keys.size == 0:
            return np.full(np.shape(keys), -1, dtype=np.int64)
        pos = np.clip(np.searchsorted(self._keys, keys), 0, self._keys.size - 1)
        return np.where(self._keys[pos] == keys, self._slots[pos], -1)

    def missing(self, keys):
        """Unique keys not yet evaluated, ascending."""
        uniq = np.unique(keys)
        return uniq[self.lookup(uniq) < 0]

    def insert(self, keys, ij, beta):
        """Assign slots to new keys. ``keys`` must be unique and absent."""
        start = len(self)
        slots = np.arange(start, start + keys.size, dtype=np.int64)
        self.ij = np.concatenate([self.ij, ij])
        self.beta = np.concatenate([self.beta, beta])
        merged_k = np.concatenate([self._keys, keys])
        merged_s = np.concatenate([self._slots, slots])
        order = np.argsort(merged_k, kind="stable")
        self._keys = merged_k[order]
        self._slots = merged_s[order]
        return slots


class _ActiveKeys:
    """
    Lattice points that are currently vertices of some triangle in the mesh.

    Stored as one flag per **vertex-cache slot**, not as a sorted key set.
    Every active key is by construction a vertex of some triangle, so it has
    already been evaluated and is already in the cache -- a second sorted
    structure duplicated the cache's own key index, and keeping it sorted cost
    an ``np.union1d`` over the whole active set on every insertion, which was
    the single largest term in the build.

    Separate from the vertex cache, which also holds midpoints of
    tested-but-never-split triangles. Only ever grows, since a parent's
    vertices are inherited by all its children.
    """

    def __init__(self, cache):
        self._cache = cache
        self._flags = np.zeros(0, dtype=bool)

    def _grow(self):
        n = len(self._cache)
        if self._flags.size < n:
            flags = np.zeros(n, dtype=bool)
            flags[: self._flags.size] = self._flags
            self._flags = flags

    def add_slots(self, slots):
        """
        Activate vertex-cache slots.

        Takes slots rather than keys because every caller already holds them:
        re-keying a triangle's vertices only to look them up again is exactly
        the work this class exists to avoid.
        """
        slots = np.asarray(slots, dtype=np.int64)
        # Trip-wire for the `active subset of cache` invariant. A -1 slot --
        # what `_VertexCache.lookup` returns for an absent key -- would
        # negative-index into the last cache entry and activate the wrong
        # vertex, and the mesh would come out unbalanced rather than raising.
        #
        # `raise AssertionError` rather than a bare `assert`: `python -O`
        # strips bare asserts, and this guards against silent geometric
        # corruption. `_refine` guards its cascade the same way.
        if slots.size and not (slots >= 0).all():
            raise AssertionError("cannot activate an uncached vertex")
        self._grow()
        self._flags[slots] = True

    def contains_slots(self, slots):
        """True where the slot is an active vertex. ``-1`` reads as False."""
        if self._flags.size == 0:
            return np.zeros(np.shape(slots), dtype=bool)
        self._grow()
        present = slots >= 0
        return present & self._flags[np.where(present, slots, 0)]

    def contains(self, keys):
        """
        True where the key is an active vertex.

        A key absent from the cache was never evaluated, so it cannot be a
        triangle vertex and cannot be active -- ``lookup`` returns ``-1`` and
        :meth:`contains_slots` reads that as False.
        """
        self._grow()
        return self.contains_slots(self._cache.lookup(keys))


class _LeafStore:
    """
    Terminal triangles, keyed by row index with a validity flag.

    Not append-only: a triangle marked converged at level ``d`` can be removed and
    replaced by descendants several levels later, when a distant refinement cascades
    back to it. Hence the flag and the single compaction at the end, rather than
    streaming into a flat array as we go.
    """

    def __init__(self):
        self.v = np.empty((0, 3), dtype=np.int64)
        self.level = np.empty(0, dtype=np.int64)
        self.cls = np.empty(0, dtype=np.int64)
        self.status = np.empty(0, dtype=np.int8)
        self.valid = np.empty(0, dtype=bool)

    def add(self, v, level, cls, status):
        """Append triangles, returning their row indices."""
        start = self.v.shape[0]
        k = v.shape[0]
        self.v = np.concatenate([self.v, v])
        self.level = np.concatenate(
            [self.level, np.broadcast_to(np.asarray(level, dtype=np.int64), (k,))]
        )
        self.cls = np.concatenate([self.cls, np.asarray(cls, dtype=np.int64)])
        self.status = np.concatenate(
            [self.status, np.broadcast_to(np.asarray(status, dtype=np.int8), (k,))]
        )
        self.valid = np.concatenate([self.valid, np.ones(k, dtype=bool)])
        return np.arange(start, start + k, dtype=np.int64)

    def remove(self, rows):
        self.valid[rows] = False

    def compact(self):
        keep = np.flatnonzero(self.valid)
        return self.v[keep], self.level[keep], self.cls[keep], self.status[keep]


def _initial_triangles(init_res, max_level, root_class):
    """
    Level-0 triangles: two per cell, split on the ``(0,0)-(1,1)`` diagonal.

    Both are emitted positively oriented. ``func/base.py`` builds its pair with
    *opposite* handedness; this fixes that so orientation is globally consistent and
    downstream degree or winding-number arguments stay available.

    Returns
    -------
    ij: ndarray
        ``(2 * init_res**2, 3, 2)`` int64 lattice coordinates.
    cls: ndarray
        ``(2 * init_res**2,)`` int64 orientation classes.
    """
    step = 1 << int(max_level)
    i, j = np.meshgrid(np.arange(init_res), np.arange(init_res), indexing="ij")
    base = np.stack((i.ravel(), j.ravel()), axis=-1).astype(np.int64) * step
    blocks, classes = [], []
    for s, shape in enumerate(ROOT_SHAPES):
        offs = np.asarray(shape, dtype=np.int64) * step
        blocks.append(base[:, None, :] + offs[None, :, :])
        classes.append(np.full(base.shape[0], root_class[s], dtype=np.int64))
    return np.concatenate(blocks, axis=0), np.concatenate(classes)


def _midpoint_ij(ij):
    """
    Edge midpoints ``m1, m2, m3``, with ``m_i`` opposite ``theta_i``.

    Exact integer averaging: at any level below ``max_level`` the coordinate sums
    are even by construction.
    """
    return np.stack(
        (
            (ij[:, 1] + ij[:, 2]) // 2,
            (ij[:, 2] + ij[:, 0]) // 2,
            (ij[:, 0] + ij[:, 1]) // 2,
        ),
        axis=1,
    )


def _red_split(v, m, cls, compose):
    """
    Split into the four canonical children, triangle-major.

    Returns ``(4n, 3)`` vertex slots and ``(4n,)`` classes, ordered as all four
    children of triangle 0, then triangle 1, and so on.
    """
    six = np.concatenate([v, m], axis=1)  # (n, 6)
    idx = np.asarray(CHILD_VERTEX_INDICES)  # (4, 3)
    return six[:, idx].reshape(-1, 3), compose[cls].reshape(-1)


def _make_raytrace_np(raytrace, device):
    """
    Wrap a backend ``raytrace(x, y) -> (bx, by)`` as a host-side ``(N,2) -> (N,2)``.

    Coordinates go out as float64. The criterion is a difference of ``O(fov)``
    quantities, so its roundoff floor is ``eps * fov`` and it is meaningless below
    ``h ~ sqrt(8 * eps * fov)``. For ``fov = 5`` that is ``2e-3`` arcsec in float32
    -- comparable to a typical ``min_img_sep`` -- and below it the deviation cancels
    to exactly zero, which the test reads as "perfectly affine" and converges. That
    is the fail-open direction and no clamp fixes it, so the build is always float64.
    """
    info = {"done": False, "dtype": np.dtype(np.float64)}

    def call(xy):
        x = backend.as_array(xy[:, 0], dtype=backend.float64, device=device)
        y = backend.as_array(xy[:, 1], dtype=backend.float64, device=device)
        out = raytrace(x, y)
        if not info["done"]:
            if not isinstance(out, tuple) or len(out) != 2:
                raise ValueError(
                    "raytrace must return a 2-tuple (bx, by) of arrays with shape "
                    f"(N,); got {type(out).__name__}"
                )
            info["done"] = True
        bx = backend.to_numpy(out[0]).reshape(-1)
        by = backend.to_numpy(out[1]).reshape(-1)
        info["dtype"] = bx.dtype
        if bx.shape[0] != xy.shape[0] or by.shape[0] != xy.shape[0]:
            raise ValueError(
                f"raytrace returned {bx.shape[0]} points for {xy.shape[0]} inputs; "
                "it must be shape-preserving on 1-D input"
            )
        return np.stack((bx, by), axis=-1).astype(np.float64)

    call.info = info
    return call


def _evaluate(cache, lattice, keys, raytrace_np, batch_size):
    """Evaluate every not-yet-cached key, in one logical batch per call."""
    todo = cache.missing(keys)
    if todo.size == 0:
        return
    ij = lattice.ij_from_key(todo)
    xy = lattice.xy(ij)
    if batch_size is None or xy.shape[0] <= batch_size:
        beta = raytrace_np(xy)
    else:
        n_chunks = int(ceil(xy.shape[0] / batch_size))
        beta = np.concatenate(
            [raytrace_np(chunk) for chunk in np.array_split(xy, n_chunks)]
        )
    cache.insert(todo, ij, beta)


def _edge_quarter_keys(lattice, ij):
    """
    Keys of both quarter points on each of the three edges.

    A neighbour across an edge that is two or more levels finer has one of these as
    a vertex, so six hash lookups decide balance for a triangle -- no
    edge-to-triangle adjacency table and no ancestry walk. Both quarter points are
    checked because the neighbour across an edge can itself be non-uniform.

    The caller must only pass triangles at level ``<= max_level - 2``, where the
    edge vectors are divisible by four and the quarter points are lattice points.

    Parameters
    ----------
    ij: ndarray
        Lattice coordinates, shape ``(n, 3, 2)``.

    Returns
    -------
    ndarray
        Shape ``(n, 6)`` int64.
    """
    a = ij[:, [0, 1, 2], :]
    b = ij[:, [1, 2, 0], :]
    delta = (b - a) // 4
    return np.concatenate((lattice.key(a + delta), lattice.key(b - delta)), axis=1)


def _find_unbalanced(store, cache, lattice, active, max_level, frontier_level):
    """
    Rows of ``store`` carrying an active quarter point on some edge.

    Two independent level bounds apply. ``level <= frontier_level - 2`` is an
    optimization: only triangles at least two levels coarser than the frontier can
    have been invalidated by it, so the scan skips most of the store.
    ``level <= max_level - 2`` is an integrality requirement: below it the quarter
    points are not lattice points and no finer neighbour can exist.

    Parameters
    ----------
    store: _LeafStore
        Terminal triangles; only rows with ``valid`` set are scanned.
    cache: _VertexCache
        Supplies the lattice coordinates of each row's vertices.
    lattice: _Lattice
        Used to key the quarter points.
    active: _ActiveKeys
        Pre-existing active-vertex set; membership decides a violation.
    max_level: int
        Finest allowed level.
    frontier_level: int
        Level of the finest triangles created in the current round.

    Returns
    -------
    ndarray
        Int64 indices into ``store``'s rows.
    """
    # ``frontier_level <= max_level`` holds for every call `_refine` makes, so the
    # first term always binds and the second is unreachable defensive code today.
    # Keep the min(): it is what makes the integrality precondition of
    # `_edge_quarter_keys` a property of this function rather than of its caller.
    bound = min(frontier_level - 2, max_level - 2)
    cand = np.flatnonzero(store.valid & (store.level <= bound))
    if cand.size == 0:
        return cand
    keys = _edge_quarter_keys(lattice, cache.ij[store.v[cand]])
    return cand[active.contains(keys).any(axis=1)]


@dataclass
class _Refinement:
    """Output of the level loop, before closure and freezing."""

    cache: _VertexCache
    active: _ActiveKeys
    store: _LeafStore
    counters: dict


def _refine(
    raytrace_np, lattice, init_res, h0, min_img_sep, max_level, tables, batch_size
):
    """
    Level-synchronous refinement.

    Processes the whole active set one level at a time: gathers all unique new
    points for that level, calls ``raytrace`` once on the batch, applies the
    criterion vectorized, then partitions into converged and to-split. No
    Python-level recursion over individual triangles, no per-triangle ``raytrace``.

    At ``max_level`` nothing splits, so there is no cascade, so no force-split, so
    no leaf ever needs its midpoints -- and closure needs none either, because a
    hanging node is a *vertex* of the finer neighbour. The evaluation set therefore
    collapses to the vertices, saving the single largest batch in the build. The
    non-finite check still runs there, on the vertices alone; without it a leaf with
    a ``NaN`` vertex would enter the spatial index and swallow every query in its
    cell.

    **Non-finite triangles split unconditionally.** A non-finite sample point is
    maximal ignorance about a triangle, so it triggers refinement like every other
    unresolved condition rather than terminating it -- the criterion simply cannot
    be evaluated there, which is why the split carries no verdict. A red split
    hands the bad vertex to exactly one of the four children, so the other three
    re-enter the criterion normally and the singularity ends up ringed by a band of
    ``INVALID`` leaves at the smallest allowed size instead of a hexagon of
    ``fov / init_res``. Terminating on the spot instead would put the hole at
    whatever level the triangle was first sampled, and ``INVALID`` leaves are
    excluded from the spatial index -- so on a singular model that hole is exactly
    the region where the mesh is most needed. The cost is
    ``counters["nonfinite_splits"]``: six triangles per level for a point
    singularity, but ``O(area * 4**max_level)`` should a ``raytrace`` return
    non-finite values over a whole region.
    """
    M, G, COMPOSE, PINV0, ROOT_CLASS = tables
    cache = _VertexCache()
    active = _ActiveKeys(cache)
    store = _LeafStore()
    counters = {
        "converged_level0": 0,
        "parity_splits": 0,
        "deviation_splits": 0,
        "sigma_zero": 0,
        "nonfinite_splits": 0,
        "forced": 0,
        "cascade_rounds": 0,
    }

    active_ij, active_cls = _initial_triangles(init_res, max_level, ROOT_CLASS)
    deferred = np.empty(0, dtype=np.int64)

    for level in range(max_level + 1):
        vert_keys = lattice.key(active_ij)  # (n, 3)
        need = [vert_keys.reshape(-1)]
        if level < max_level:
            mid_ij = _midpoint_ij(active_ij)
            need.append(lattice.key(mid_ij).reshape(-1))
            need.append(deferred)
            deferred = np.empty(0, dtype=np.int64)
        _evaluate(cache, lattice, np.concatenate(need), raytrace_np, batch_size)

        v = cache.lookup(vert_keys)
        active.add_slots(v.reshape(-1))
        beta_v = cache.beta[v]
        finite_v = np.isfinite(beta_v).all(axis=(1, 2))

        if level == max_level:
            store.add(v[~finite_v], level, active_cls[~finite_v], LeafStatus.INVALID)
            store.add(v[finite_v], level, active_cls[finite_v], LeafStatus.SIZE_FLOOR)
            break

        m = cache.lookup(lattice.key(mid_ij))
        beta_m = cache.beta[m]
        good = finite_v & np.isfinite(beta_m).all(axis=(1, 2))
        counters["nonfinite_splits"] += int((~good).sum())

        rows = np.flatnonzero(good)
        keep, parity_ok, s = evaluate_criterion(
            beta_v[rows],
            beta_m[rows],
            active_cls[rows],
            level,
            h0,
            min_img_sep,
            PINV0,
            COMPOSE,
        )
        counters["parity_splits"] += int((~parity_ok).sum())
        counters["deviation_splits"] += int((parity_ok & ~keep).sum())
        counters["sigma_zero"] += int((s == 0).sum())

        # A non-finite triangle joins the criterion's failures in `pending`
        # rather than terminating: the criterion cannot be evaluated on it, so
        # the split is unconditional. Sorted, so which reason condemned a
        # triangle never reaches the child ordering.
        done = rows[keep]
        pending = np.sort(np.concatenate((np.flatnonzero(~good), rows[~keep])))
        store.add(v[done], level, active_cls[done], LeafStatus.CONVERGED)
        if level == 0:
            counters["converged_level0"] = int(done.size)

        child_v, child_cls = _red_split(
            v[pending], m[pending], active_cls[pending], COMPOSE
        )
        child_ij = cache.ij[child_v]
        active.add_slots(child_v.reshape(-1))

        # Balance cascade. The children above are already registered as active
        # vertices, which is what makes the quarter-point test able to see them --
        # the split must precede the cascade, not follow it.
        frontier_level = level + 1
        while True:
            violators = _find_unbalanced(
                store, cache, lattice, active, max_level, frontier_level
            )
            if violators.size == 0:
                break
            counters["cascade_rounds"] += 1
            store.remove(violators)
            vv = store.v[violators]
            vij = cache.ij[vv]
            vm = cache.lookup(lattice.key(_midpoint_ij(vij)))
            # Trip-wire for the re-forcing invariant. A violator's midpoints are
            # normally already cached, but a forced child re-forced within the
            # same cascade would still have its midpoints sitting in `deferred`,
            # and a -1 slot here would silently negative-index `cache.ij` into
            # wrong geometry rather than raising. See spec section 2.3.
            #
            # `raise AssertionError` rather than a bare `assert`: `python -O`
            # strips bare asserts, and this one guards against silent geometric
            # corruption, not just a debugging convenience. `func/adaptive.py`
            # uses `raise AssertionError` for the same class of guard.
            if not (vm >= 0).all():
                raise AssertionError("cascade hit an unevaluated midpoint")
            kid_v, kid_cls = _red_split(vv, vm, store.cls[violators], COMPOSE)
            kid_level = np.repeat(store.level[violators] + 1, 4)
            # A forced child is auto-converged: steps 3-7 are skipped so the
            # cascade cannot re-enter the split machinery from inside itself.
            #
            # FORCED unconditionally, with no INVALID arm to inherit: a violator
            # is bounded to `level <= max_level - 2` by `_find_unbalanced`, and
            # the only INVALID leaves in the store sit at `max_level` -- the
            # branch above adds them and breaks out of the loop before any
            # cascade runs. So no violator is ever INVALID. A forced child can
            # still reach freeze with a non-finite vertex, via a deferred
            # midpoint the criterion never saw; that is what
            # `_invalidate_nonfinite_origins` is for.
            store.add(kid_v, kid_level, kid_cls, LeafStatus.FORCED)
            counters["forced"] += int(kid_v.shape[0])
            kid_ij = cache.ij[kid_v]
            active.add_slots(kid_v.reshape(-1))
            # Forced children are produced after this level's raytrace call has
            # gone out, and land at levels the loop will never revisit. Queue
            # their midpoints and drain at the top of the next level, so the
            # one-batch-per-level structure survives. A forced child cannot be
            # re-forced within the same cascade, because the frontier only moves
            # coarser -- so the deferral is never more than one level deep.
            deferred = np.concatenate(
                (deferred, lattice.key(_midpoint_ij(kid_ij)).reshape(-1))
            )
            # The MAXIMUM kid level, not the minimum: a kid at level L invalidates
            # neighbours at level <= L-2, so the minimum would skip violators.
            # The maximum strictly decreases each round, which terminates the loop.
            frontier_level = int(kid_level.max())

        active_ij, active_cls = child_ij, child_cls
        if active_ij.shape[0] == 0:
            break

    return _Refinement(cache=cache, active=active, store=store, counters=counters)


def _canonical_order(lattice, cache, v):
    """
    Deterministic leaf ordering, independent of the order the cascade produced.

    Sorts by the row-wise-sorted triple of vertex lattice keys, which is unique per
    triangle since a triangle is determined by its vertex set. This makes
    byte-identical output a property of the data rather than of control flow.
    """
    keys = np.sort(lattice.key(cache.ij[v]), axis=1)
    return np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))


def _min_angle(tri):
    """
    Smallest interior angle of each triangle, in radians.

    Parameters
    ----------
    tri: ndarray
        Shape ``(n, 3, 2)``.

    Returns
    -------
    ndarray
        Shape ``(n,)``. Degenerate triangles give ``0.0`` rather than ``NaN``.
    """
    a = np.linalg.norm(tri[:, 2] - tri[:, 1], axis=-1)
    b = np.linalg.norm(tri[:, 0] - tri[:, 2], axis=-1)
    c = np.linalg.norm(tri[:, 1] - tri[:, 0], axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        cosines = np.stack(
            (
                (b * b + c * c - a * a) / (2 * b * c),
                (c * c + a * a - b * b) / (2 * c * a),
                (a * a + b * b - c * c) / (2 * a * b),
            ),
            axis=1,
        )
    angles = np.arccos(np.clip(cosines, -1.0, 1.0))
    return np.nan_to_num(angles.min(axis=1), nan=0.0)


def _close(lattice, cache, active, v, level, status):
    """
    Make the balanced mesh conforming, using the pre-closure active-vertex set.

    Fixed pattern table, all vertices already cached:

    - 1 hanging node at ``m_i``: bisect from the opposite vertex into two triangles.
    - 2 hanging nodes: emit the corner triangle at the vertex opposite the whole
      edge, then split the remaining quadrilateral along whichever of its two
      diagonals maximizes the minimum angle.
    - 3 hanging nodes: the canonical red split, free.

    Parameters
    ----------
    v: ndarray
        Pre-closure leaf vertex slots in canonical order, shape ``(L0, 3)``.
    level, status: ndarray
        Shape ``(L0,)``, inherited by the emitted triangles.

    Returns
    -------
    leaves: ndarray
        ``(L, 3)`` vertex slots, positively oriented.
    origin: ndarray
        ``(L,)`` index into ``v``. Non-decreasing, so each origin's terminal
        triangles are contiguous and grouping is a slice.
    out_level, out_status: ndarray
        ``(L,)``, inherited from the origin.
    """
    v_ij = cache.ij[v]
    mid_ij = _midpoint_ij(v_ij)
    mid_keys = lattice.key(mid_ij)
    m = cache.lookup(mid_keys)
    # A max_level leaf's edges are one lattice unit long, so their true midpoints
    # are not lattice points at all -- `_midpoint_ij`'s floor division silently
    # collapses onto one of that same edge's own (trivially active) endpoints
    # instead. Gating on exact reconstruction sends those leaves through as
    # count == 0, matching `_refine`'s invariant that no leaf at max_level has a
    # hanging node; below max_level the sums are always even by construction (see
    # `_midpoint_ij`), so the gate has no effect there.
    #
    # That covers the gate being a no-op *below* max_level; the other half is that
    # it always *fires* -- is never merely usually True -- *at* max_level, i.e.
    # `exact` is exactly equivalent to `level < max_level`. Proof: at max_level
    # every leaf is a unimodular lattice triangle, |det(edge matrix)| == 1. If any
    # edge's two endpoints shared parity in both coordinates, that edge vector
    # would be all-even; using it as one column of an edge matrix built from the
    # triangle's other two edges would then force the determinant to be even (or,
    # were a second edge also all-even, divisible by four) -- both impossible when
    # |det| == 1. So no edge of a max_level leaf can ever have matching endpoint
    # parity, `exact` is False on every edge, and the count == 0 pass-through
    # above is that leaf's only route through -- not an artifact of this fixture.
    exact = ((v_ij[:, [1, 2, 0]] + v_ij[:, [2, 0, 1]]) % 2 == 0).all(axis=-1)
    # `m` is already `cache.lookup(mid_keys)`; asking by slot skips a second
    # searchsorted over the same keys.
    hanging = exact & active.contains_slots(m)  # (L0, 3)
    count = hanging.sum(axis=1)

    n_children = np.choose(count, [1, 2, 3, 4])
    offsets = np.concatenate(([0], np.cumsum(n_children)))
    leaves = np.empty((int(offsets[-1]), 3), dtype=np.int64)
    origin = np.repeat(np.arange(v.shape[0], dtype=np.int64), n_children)

    def geom(slots):
        return lattice.xy(cache.ij[slots])

    sel = np.flatnonzero(count == 0)
    leaves[offsets[sel]] = v[sel]

    for i in range(3):
        j, k = (i + 1) % 3, (i + 2) % 3
        sel = np.flatnonzero((count == 1) & hanging[:, i])
        if sel.size == 0:
            continue
        o = offsets[sel]
        leaves[o] = np.stack((v[sel, i], v[sel, j], m[sel, i]), axis=1)
        leaves[o + 1] = np.stack((v[sel, i], m[sel, i], v[sel, k]), axis=1)

    for c in range(3):
        a, b = (c + 1) % 3, (c + 2) % 3
        sel = np.flatnonzero((count == 2) & ~hanging[:, c])
        if sel.size == 0:
            continue
        o = offsets[sel]
        # Corner triangle at theta_c: exactly red-split child C_c, so positively
        # oriented by construction.
        leaves[o] = np.stack((v[sel, c], m[sel, b], m[sel, a]), axis=1)
        a1 = np.stack((v[sel, a], v[sel, b], m[sel, a]), axis=1)
        a2 = np.stack((v[sel, a], m[sel, a], m[sel, b]), axis=1)
        b1 = np.stack((v[sel, a], v[sel, b], m[sel, b]), axis=1)
        b2 = np.stack((v[sel, b], m[sel, a], m[sel, b]), axis=1)
        score_a = np.minimum(_min_angle(geom(a1)), _min_angle(geom(a2)))
        score_b = np.minimum(_min_angle(geom(b1)), _min_angle(geom(b2)))
        use_b = score_b > score_a  # ties take candidate A, deterministically
        leaves[o + 1] = np.where(use_b[:, None], b1, a1)
        leaves[o + 2] = np.where(use_b[:, None], b2, a2)

    sel = np.flatnonzero(count == 3)
    if sel.size:
        o = offsets[sel]
        idx = np.asarray(CHILD_VERTEX_INDICES)
        kids = np.concatenate([v[sel], m[sel]], axis=1)[:, idx]  # (k, 4, 3)
        for t in range(4):
            leaves[o + t] = kids[:, t, :]

    return leaves, origin, level[origin], status[origin]


@dataclass(frozen=True)
class BuildStats:
    """
    Diagnostics from a mesh build. All leaf counts are **pre-closure** except
    ``n_leaves``, since closure only re-tiles existing leaves.

    ``n_converged_at_level_0`` is the diagnostic for the module's known blind spot:
    the criterion samples six points per triangle and cannot see structure below
    that scale, so completeness is conditional on ``init_res`` already resolving the
    smallest curvature scale in the lens. A large level-0 convergence fraction means
    the mesh never looked below ``init_res`` anywhere and that precondition went
    untested.

    ``n_converged_at_level_0`` is an EVENT count, not a leaf count, and is the one
    exception to the "all leaf counts are pre-closure" framing above: it counts
    triangles that *passed step 7* at level 0, not triangles that are still
    level-0 leaves by the time refinement finishes. The balance cascade can later
    force-split a level-0 triangle that passed step 7 (to satisfy a coarser
    neighbour's 2:1 balance against a finer one elsewhere), which removes it from
    ``leaves_by_level[0]`` while leaving it counted here -- so
    ``n_converged_at_level_0`` can exceed ``leaves_by_level[0]``. That is correct
    and intentional: a force-split level-0 leaf still received no criterion
    evidence below level 0 (its children are ``FORCED``, not re-evaluated), so the
    event count is the more informative blind-spot diagnostic, not a bug to be
    reconciled against the leaf count.

    A triangle failing both parity and deviation is counted in ``n_parity_splits``;
    ``n_deviation_splits`` counts only among parity-passers.

    ``n_nonfinite_splits`` is disjoint from both: a triangle with a non-finite
    sample point never reaches the criterion at all, and splitting it is a decision
    made in the absence of evidence rather than because of it. It is also the cost
    diagnostic for that decision -- a point singularity contributes six per level,
    so a count growing like ``4**level`` means ``raytrace`` is returning non-finite
    values over an area and the descent is quadrupling inside it.
    """

    d_floor: int
    max_level: int
    depth_limited: bool
    l_max_final: float
    n_converged: int
    n_size_floor: int
    n_forced: int
    n_invalid: int
    n_converged_at_level_0: int
    n_parity_splits: int
    n_deviation_splits: int
    n_nonfinite_splits: int
    leaves_by_level: Tuple[int, ...]
    n_nonfinite_vertices: int
    n_sigma_min_exactly_zero: int
    n_boundary_leaves_at_floor: int
    n_vertices: int
    n_leaves_pre_closure: int
    n_leaves: int
    n_closure_by_pattern: Tuple[int, int, int]
    raytrace_dtype: str
    cancellation_floor: float


def _numpy_dtype(dtype):
    """NumPy equivalent of a backend float dtype; float64 when unspecified."""
    if dtype is None:
        return np.dtype(np.float64)
    return backend.to_numpy(backend.zeros((), dtype=dtype)).dtype


def _invalidate_nonfinite_origins(vs, leaves, origin, pre_status):
    """
    Re-check finiteness at freeze and propagate invalidity through the origin.

    A ``FORCED`` leaf inherits its vertices from a parent whose midpoints were
    never finiteness-tested, and a closure triangle can pick up a midpoint no
    criterion ever saw, so a non-finite vertex can reach freeze on a leaf not
    already marked ``INVALID``. Without this it would enter the spatial index and
    swallow every query in its cell.

    Invalidity is propagated UP to the origin and then back down, rather than
    applied to the leaf alone: the termination-reason counts are pre-closure, so
    marking only the leaf would leave ``n_converged + n_size_floor + n_forced +
    n_invalid`` disagreeing with ``n_leaves_pre_closure``. It is also the
    conservative direction -- if one triangle of a region has a bad vertex, the
    region is not trustworthy.

    Factored out of :func:`build_adaptive_mesh` so it can be exercised directly:
    every vertex reaching a full build has already been finiteness-checked by
    :func:`_refine`, except on a narrow cascade path no available fixture
    reaches, so inline this logic would be untestable.

    Returns
    -------
    ndarray
        ``pre_status`` with every origin owning a non-finite leaf set to
        ``INVALID``.
    """
    leaf_finite = np.isfinite(vs[leaves]).all(axis=(1, 2))
    origin_bad = np.zeros(pre_status.shape[0], dtype=bool)
    np.logical_or.at(origin_bad, origin, ~leaf_finite)
    return np.where(origin_bad, np.int8(LeafStatus.INVALID), pre_status)


def _build_index(vs, leaves, valid_rows, index_cells):
    """
    Uniform-grid CSR index over source-plane axis-aligned bounding boxes.

    One cell lookup per query is complete: ``beta`` lies in the triangle, which lies
    in its AABB, which is covered by the cells the leaf registered in, so ``beta``'s
    own cell always contains any leaf containing ``beta``. No neighbour search is
    needed. Per-cell lists are stored ascending, which gives ``query`` its sorted
    CSR blocks with no sort at query time.

    Deliberately float64 regardless of the mesh's ``dtype``: build-side and
    query-side cell arithmetic (``Mesh.query``'s ``(chunk - lo) / cell``) must agree
    by construction, and forcing this to the mesh's own dtype would let the two
    sides round independently right at cell boundaries -- worse, not cleaner.
    """
    tri = vs[leaves[valid_rows]].astype(np.float64)
    if tri.shape[0] == 0:
        lo = np.zeros(2)
        return (
            lo,
            np.ones(2),
            1,
            1,
            np.zeros(2, np.int64),
            np.empty(0, np.int64),
            np.ones(2),
        )
    flat = tri.reshape(-1, 2)
    lo, hi = flat.min(axis=0), flat.max(axis=0)
    span = np.where(hi > lo, hi - lo, 1.0)  # a degenerate axis becomes one cell
    if index_cells is None:
        c = np.sqrt(span[0] * span[1] / tri.shape[0])
    else:
        c = float(span.max()) / int(index_cells)
    c = max(float(c), np.finfo(np.float64).tiny)
    nx = max(1, int(ceil(span[0] / c)))
    ny = max(1, int(ceil(span[1] / c)))
    cell = span / np.array([nx, ny], dtype=np.float64)

    upper = np.array([nx - 1, ny - 1], dtype=np.int64)
    i0 = np.clip(((tri.min(axis=1) - lo) / cell).astype(np.int64), 0, upper)
    i1 = np.clip(((tri.max(axis=1) - lo) / cell).astype(np.int64), 0, upper)
    tall = i1[:, 1] - i0[:, 1] + 1
    counts = (i1[:, 0] - i0[:, 0] + 1) * tall
    owner = np.repeat(np.arange(counts.size), counts)
    within = np.arange(int(counts.sum())) - np.repeat(
        np.cumsum(counts) - counts, counts
    )
    cell_id = (i0[owner, 0] + within // tall[owner]) * ny + (
        i0[owner, 1] + within % tall[owner]
    )
    leaf_id = valid_rows[owner]
    order = np.lexsort((leaf_id, cell_id))  # ascending leaf id within each cell
    cell_leaves = leaf_id[order]
    cell_offsets = np.searchsorted(cell_id[order], np.arange(nx * ny + 1))
    return lo, cell, nx, ny, cell_offsets.astype(np.int64), cell_leaves, hi


def _dedup_block_group(points, rows, n_blocks, m, tol):
    """
    Connected-component representatives for blocks of exactly ``m`` points.

    Every slot is real, so this carries none of the padding machinery a
    ragged formulation needs: no validity mask, no clipped gather, and the
    "no label" sentinel is ``m`` rather than a global maximum. Grouping the
    caller's blocks by count and calling this once per distinct count is what
    keeps the ``(n_blocks, m, m)`` intermediate proportional to
    ``sum_c B_c * c**2`` instead of ``B * max(c)**2``.

    Parameters
    ----------
    points: ArrayLike
        The caller's full point array, shape ``(K, 2)``.

        *Unit: arcsec*

    rows: ndarray
        ``(n_blocks * m,)`` int64 indices into ``points``, block-major.
    n_blocks, m: int
        Block count and the common per-block point count.
    tol: float
        Separation below which two points are the same image.

        *Unit: arcsec*

    Returns
    -------
    ArrayLike
        ``(n_blocks * m,)`` bool, in the order of ``rows``.
    """
    device = backend.device(points)
    int64 = backend.module.int64
    p = points[backend.as_array(rows, dtype=int64, device=device)]
    p = p.reshape(n_blocks, m, 2)

    delta = backend.unsqueeze(p, 2) - backend.unsqueeze(p, 1)
    # Squared distances against a squared tolerance: no sqrt, and the
    # comparison is exact on the diagonal, so every point is its own
    # neighbour and the label update below is a true minimum over the closed
    # neighbourhood.
    adjacent = backend.long(backend.sum(delta * delta, dim=-1) < tol * tol)

    # Min-label propagation. `m` is the sentinel for "no label": it exceeds
    # every real slot index, so it never wins a minimum against a neighbour.
    index = backend.unsqueeze(backend.arange(m, dtype=int64, device=device), 0)
    labels = index + backend.zeros((n_blocks, m), dtype=int64, device=device)
    for _ in range(m):
        neighbour = adjacent * backend.unsqueeze(labels, 1) + (1 - adjacent) * m
        updated = backend.min(neighbour, dim=2)
        if bool(backend.to_numpy(backend.all(updated == labels))):
            break
        labels = updated

    # Each component now carries the lowest slot index it contains, and that
    # slot is its own label -- so the fixed points are exactly one per
    # component.
    return (labels == index).reshape(-1)


def _dedup_representatives(points, counts, tol):
    """
    One representative per cluster of near-coincident points, within each block.

    Clusters are the **connected components** of the ``distance < tol`` graph, not
    the greedy clusters :func:`~caustics.lenses.func.base.remove_duplicate_points`
    produces. The difference is order dependence: for three collinear points
    spaced ``0.9 * tol`` apart, greedy returns two representatives in one input
    order and one in another, so the image count would depend on the order
    :meth:`Mesh.query` happened to emit candidates in. Components are a function
    of the point set alone, which is what makes a multiplicity map reproducible.

    Adjacency is strict ``<``, so a pair separated by exactly ``tol`` stays
    distinct. That matches the build contract, where ``min_img_sep`` is a size
    floor the mesh resolves *to* rather than a scale it merges away.

    Vectorized by grouping blocks that share a count and running each group at
    its own width, because the greedy loop is one Python iteration per point --
    fine for the handful of images of a single source, hopeless for the
    ``nx * ny`` blocks of a multiplicity map. Blocks of zero or one point never
    reach the kernel; their answer is already known. The cost is a
    ``(B_c, c, c)`` intermediate per distinct count ``c``, which is why callers
    may still want to chunk over query points when a single block is enormous.

    Parameters
    ----------
    points: ArrayLike
        Shape ``(K, 2)``, laid out block-major: block ``b`` occupies the
        ``counts[b]`` rows following those of blocks ``0 .. b - 1``.

        *Unit: arcsec*

    counts: ndarray
        Shape ``(B,)`` int, with ``counts.sum() == K``. Zero-length blocks are
        allowed.
    tol: float
        Separation below which two points are the same image.

        *Unit: arcsec*

    Returns
    -------
    ArrayLike
        ``(K,)`` bool, True on exactly one point per cluster.
    """
    counts = np.asarray(counts, dtype=np.int64)
    total = int(counts.sum())
    device = backend.device(points)
    if total == 0:
        return backend.as_array(np.zeros(0, dtype=bool), device=device)

    starts = np.cumsum(counts) - counts
    row_groups, keep_groups = [], []

    # A block of one point is its own representative and a block of none
    # contributes nothing, so neither reaches the clustering kernel at all.
    # On a multiplicity map those are the large majority of blocks -- 88% at
    # the reference configuration -- and skipping them is the single biggest
    # reduction in what the kernel has to hold.
    singles = np.flatnonzero(counts == 1)
    if singles.size:
        row_groups.append(starts[singles])
        keep_groups.append(
            backend.as_array(np.ones(singles.size, dtype=bool), device=device)
        )

    # The rest are grouped by *equal* count so each group runs at its own M.
    # Padding every block to the global maximum is what made the intermediate
    # `B * max(c)**2` and put a fine multiplicity map out of memory.
    for m in np.unique(counts[counts > 1]):
        blocks = np.flatnonzero(counts == m)
        rows = (
            starts[blocks][:, None] + np.arange(int(m), dtype=np.int64)[None, :]
        ).reshape(-1)
        row_groups.append(rows)
        keep_groups.append(_dedup_block_group(points, rows, blocks.size, int(m), tol))

    # Restore block-major order by inverse permutation rather than a scatter:
    # torch keeps the last write on duplicate indices and jax accumulates, so
    # a gather is the only form that means the same thing on both backends.
    # Every row belongs to exactly one group, so `perm` is a permutation.
    perm = np.concatenate(row_groups)
    inverse = np.empty(total, dtype=np.int64)
    inverse[perm] = np.arange(total, dtype=np.int64)
    stacked = backend.concatenate(keep_groups, dim=0)
    return stacked[backend.as_array(inverse, dtype=backend.module.int64, device=device)]


class Mesh:
    """
    A frozen adaptive mesh of the lens plane, queryable from the source plane.

    One topology, two embeddings. Vertex index ``v`` is shared across both planes;
    there is no separate source-plane triangle table, so the correspondence is
    structural rather than an invariant kept in sync.

    ``INVALID`` leaves remain in ``leaves``, ``leaf_status`` and the conformity
    relation but are never registered in the spatial index, so :meth:`query` cannot
    return them. That is a genuine coverage hole in the lens plane -- but a
    ``min_img_sep``-scale one, not an ``init_res``-scale one: a non-finite triangle
    refines rather than terminating, so it can only come to rest at ``max_level``.
    ``stats.n_invalid`` sizes the hole and ``stats.n_nonfinite_splits`` the descent
    that shrank it.

    ``min_img_sep`` is stored because it is the mesh's own defining tolerance, in
    both of its build roles and again as the dedup radius in
    :meth:`forward_raytrace`. ``raytrace`` deliberately is **not** stored and is
    passed per call: a callable carries no identity the mesh could check, so
    holding one would imply a guarantee that it matches the build when nothing can
    enforce it.
    """

    def __init__(
        self,
        vertices_lens,
        vertices_source,
        leaves,
        leaf_area2,
        leaf_origin,
        leaf_status,
        leaf_level,
        origin_leaves,
        index,
        stats,
        d_floor,
        max_level,
        min_img_sep,
        dtype,
        device,
    ):
        self.vertices_lens = vertices_lens
        self.vertices_source = vertices_source
        self.leaves = leaves
        self.leaf_area2 = leaf_area2
        self.leaf_origin = leaf_origin
        self.leaf_status = leaf_status
        self.leaf_level = leaf_level
        self.origin_leaves = origin_leaves
        self.stats = stats
        self.d_floor = d_floor
        self.max_level = max_level
        self.min_img_sep = min_img_sep
        self.dtype = dtype
        self.device = device
        (
            self._index_lo,
            self._index_cell,
            self._nx,
            self._ny,
            self._cell_offsets,
            self._cell_leaves,
            self._index_hi,
        ) = index

    def _empty_result(self, n_queries):
        int64 = backend.module.int64
        zeros = backend.zeros((n_queries + 1,), dtype=int64, device=self.device)
        return (
            backend.zeros((0,), dtype=int64, device=self.device),
            zeros,
            backend.zeros((0, 3), dtype=self.vertices_source.dtype, device=self.device),
        )

    def _as_beta(self, beta):
        """Coerce query points to the mesh's dtype and device, shape ``(B, 2)``."""
        beta = backend.as_array(
            beta, dtype=self.vertices_source.dtype, device=self.device
        )
        if len(beta.shape) != 2 or beta.shape[1] != 2:
            raise ValueError(
                f"beta must have shape (B, 2), got {tuple(beta.shape)}. A single "
                "point must be passed as shape (1, 2)."
            )
        return beta

    def query(self, beta, batch_size: Optional[int] = None):
        """
        Terminal leaves whose source-plane image contains each query point.

        Returns candidate regions, not images. A point on a shared edge returns
        both leaves -- zeros count as inside, which is what guarantees no query
        falls through a seam -- and near-critical leaves overlap, so **candidate
        count is not image multiplicity**.

        Parameters
        ----------
        beta: ArrayLike
            Source-plane query points, shape ``(B, 2)`` strictly. A bare ``(2,)``
            raises rather than being promoted, so output shapes are never ambiguous.

            *Unit: arcsec*

        batch_size: Optional[int]
            Chunk size over query points. ``None`` processes all at once. Results
            are byte-identical for every value; this only bounds peak memory, which
            spikes for chunks landing near a caustic.

        Returns
        -------
        leaf_indices: ArrayLike
            ``(K,)`` indices into ``leaves``, strictly ascending within each block.
        offsets: ArrayLike
            ``(B + 1,)`` CSR offsets, ``offsets[0] == 0`` and ``offsets[B] == K``.
        bary: ArrayLike
            ``(K, 3)`` barycentric coordinates of ``beta`` in the source-plane image
            of the hit triangle, guaranteed to lie in the simplex.
        """
        beta = self._as_beta(beta)
        n = beta.shape[0]
        if n == 0:
            return self._empty_result(0)

        int64 = backend.module.int64
        step = n if batch_size is None else max(1, int(batch_size))
        idx_parts, bary_parts, count_parts = [], [], []

        for lo in range(0, n, step):
            chunk = beta[lo : lo + step]
            b = chunk.shape[0]
            u = backend.long(backend.floor((chunk - self._index_lo) / self._index_cell))
            # Containment is a coordinate test against the stored exact `hi`, not a
            # cell-index test on `u`. `cell = span / [nx, ny]`, so a point sitting
            # exactly on the upper bbox edge (x == hi_x) gives u_x == nx, which
            # fails `u_x < nx` even though `_build_index` clips leaf registration
            # to column nx - 1 -- i.e. a leaf whose AABB reaches `hi` *is* indexed,
            # in the very column `u_x < nx` rejects. Recomputing `hi` here as
            # `lo + cell * [nx, ny]` would reintroduce the same fragility, since
            # `(span / n) * n` need not equal `span` to the ulp; the exact `hi`
            # from the build is used instead. A leaf containing a `beta` with
            # x == hi has its `i1_x` clipped to `nx - 1`, the column the clamp
            # below selects, so this coordinate test is provably complete.
            inside = (
                (chunk[:, 0] >= self._index_lo[0])
                & (chunk[:, 0] <= self._index_hi[0])
                & (chunk[:, 1] >= self._index_lo[1])
                & (chunk[:, 1] <= self._index_hi[1])
            )
            # Clipped only to keep the gather in range; `inside` forces an empty
            # block for out-of-bbox points.
            cell = backend.clamp(u[:, 0], 0, self._nx - 1) * self._ny + backend.clamp(
                u[:, 1], 0, self._ny - 1
            )
            start = self._cell_offsets[cell]
            count = backend.where(
                inside, self._cell_offsets[cell + 1] - start, backend.zeros_like(start)
            )
            total = int(backend.to_numpy(backend.sum(count)))
            if total == 0:
                count_parts.append(np.zeros(b, dtype=np.int64))
                continue

            qidx = backend.repeat(
                backend.arange(b, dtype=int64, device=self.device), count, axis=0
            )
            base = backend.cumsum(count) - count
            within = backend.arange(
                total, dtype=int64, device=self.device
            ) - backend.repeat(base, count, axis=0)
            cand = self._cell_leaves[start[qidx] + within]

            w = triangle_weights(self.vertices_source[self.leaves[cand]], chunk[qidx])
            hit = contains(w)

            # Per-query hit counts by cumsum differences. Never add_at_indices:
            # torch keeps the last write on duplicate indices, jax accumulates.
            csum = backend.concatenate(
                (
                    backend.zeros((1,), dtype=int64, device=self.device),
                    backend.cumsum(backend.long(hit)),
                ),
                dim=0,
            )
            count_parts.append(backend.to_numpy(csum[base + count] - csum[base]))
            idx_parts.append(cand[hit])
            # Gather after masking, not before: `cand` is the pre-containment
            # candidate list, so `leaf_area2[cand]` is a full-length temporary
            # thrown away by the mask on the very next operation.
            bary_parts.append(sanitize_bary(w[hit], self.leaf_area2[cand[hit]]))

        counts = np.concatenate(count_parts)
        offsets = backend.as_array(
            np.concatenate(([0], np.cumsum(counts))).astype(np.int64),
            dtype=int64,
            device=self.device,
        )
        if not idx_parts:
            empty = self._empty_result(n)
            return empty[0], offsets, empty[2]
        return (
            backend.concatenate(idx_parts, dim=0),
            offsets,
            backend.concatenate(bary_parts, dim=0),
        )

    @staticmethod
    def _check_call(beta, leaf_indices):
        if (beta is None) == (leaf_indices is None):
            raise ValueError(
                "pass exactly one of `beta` (query and gather) or `leaf_indices` "
                "(gather only)"
            )

    def triangles_lens(self, beta=None, batch_size=None, *, leaf_indices=None):
        """
        Lens-plane vertices of hit leaves, shape ``(K, 3, 2)``.

        With ``beta`` returns ``(triangles, offsets)``; with ``leaf_indices``
        returns ``triangles`` alone.

        *Unit: arcsec*
        """
        self._check_call(beta, leaf_indices)
        if leaf_indices is not None:
            return self.vertices_lens[self.leaves[leaf_indices]]
        idx, offsets, _ = self.query(beta, batch_size=batch_size)
        return self.vertices_lens[self.leaves[idx]], offsets

    def triangles_source(self, beta=None, batch_size=None, *, leaf_indices=None):
        """
        Source-plane vertices of hit leaves, shape ``(K, 3, 2)``.

        *Unit: arcsec*
        """
        self._check_call(beta, leaf_indices)
        if leaf_indices is not None:
            return self.vertices_source[self.leaves[leaf_indices]]
        idx, offsets, _ = self.query(beta, batch_size=batch_size)
        return self.vertices_source[self.leaves[idx]], offsets

    def seeds(self, beta=None, batch_size=None, *, leaf_indices=None, bary=None):
        """
        Lens-plane preimage under each hit leaf's own affine map, shape ``(K, 2)``.

        That map is exactly the one step 7 bounds, so the result is a Newton seed
        accurate to ``min_img_sep`` by construction. Because ``bary`` is guaranteed
        to lie in the simplex, the seed always lies inside the leaf.

        *Unit: arcsec*
        """
        self._check_call(beta, leaf_indices)
        if leaf_indices is not None:
            if bary is None:
                raise ValueError("`bary` is required alongside `leaf_indices`")
            return self._seed(leaf_indices, bary)
        idx, offsets, computed = self.query(beta, batch_size=batch_size)
        return self._seed(idx, computed), offsets

    def _seed(self, leaf_indices, bary):
        tri = self.vertices_lens[self.leaves[leaf_indices]]
        return backend.sum(tri * backend.unsqueeze(bary, -1), dim=1)

    def _forward_chunk(self, chunk, raytrace, method, tol, lm_kwargs, want_images):
        """
        Images and per-source counts for one chunk of query points.

        ``want_images`` exists for :meth:`multiplicity_map`, which needs only
        the counts: gathering the representatives it is about to discard costs
        a full ``(K, 2)`` array per chunk, and with a small ``batch_size``
        those accumulate across every chunk of the map.

        Returns
        -------
        images: ArrayLike or None
            ``(K, 2)`` lens-plane positions, or ``None`` when the chunk found
            none or ``want_images`` is False.
        counts: ndarray
            ``(b,)`` int64 host-side image multiplicity.
        """
        b = chunk.shape[0]
        none = (None, np.zeros(b, dtype=np.int64))

        idx, offsets, bary = self.query(chunk)
        seed = self.seeds(leaf_indices=idx, bary=bary)
        if seed.shape[0] == 0:
            return none
        off_np = backend.to_numpy(offsets)

        if method == "dedup":
            # No residual filter and no displacement filter. Both exist to
            # reject a root that *wandered* away from its seed -- see the Notes
            # on `forward_raytrace`. A seed cannot wander: `bary` lies in the
            # simplex, so the seed lies inside its leaf, and that leaf's
            # source-plane image contains `beta`. Every seed is therefore
            # already an approximate image, and filtering would be testing a
            # property the construction guarantees.
            survivors, kept = seed, off_np[1:] - off_np[:-1]
        else:
            to_source = self._to_source(raytrace)
            # One target per seed, so a source with several candidate leaves
            # root-finds each of them against its own beta.
            spans = offsets[1:] - offsets[:-1]
            target = backend.repeat(chunk, spans, axis=0)
            root, _, _ = batch_lm(seed, target, to_source, **lm_kwargs)

            converged = backend.sum((to_source(root) - target) ** 2, dim=-1) < tol * tol
            # See the note above on why containment and the ball are OR-ed.
            tri = self.vertices_lens[self.leaves[idx]]
            near = contains(triangle_weights(tri, root)) | (
                backend.sum((root - seed) ** 2, dim=-1) <= self.min_img_sep**2
            )
            keep = converged & near

            # Block bookkeeping in NumPy: the counts are host-side integers the
            # dedup and the caller both need, and this module is already a
            # host-driven build, so a device round trip buys nothing.
            keep_np = backend.to_numpy(keep).astype(np.int64)
            csum = np.concatenate(([0], np.cumsum(keep_np)))
            kept = csum[off_np[1:]] - csum[off_np[:-1]]
            if kept.sum() == 0:
                return none
            survivors = root[keep]

        unique = _dedup_representatives(survivors, kept, self.min_img_sep)
        unique_np = backend.to_numpy(unique).astype(np.int64)
        kept_off = np.concatenate(([0], np.cumsum(kept)))
        csum = np.concatenate(([0], np.cumsum(unique_np)))
        counts = csum[kept_off[1:]] - csum[kept_off[:-1]]
        return (survivors[unique] if want_images else None), counts

    def _image_chunks(
        self,
        beta,
        raytrace,
        batch_size,
        method,
        residual_tol,
        lm_kwargs,
        want_images,
    ):
        """Yield :meth:`_forward_chunk`'s ``(images, counts)`` per chunk."""
        tol = self.min_img_sep if residual_tol is None else float(residual_tol)
        lm_kwargs = {} if lm_kwargs is None else dict(lm_kwargs)
        n = beta.shape[0]
        step = n if batch_size is None else max(1, int(batch_size))
        for lo in range(0, n, step):
            yield self._forward_chunk(
                beta[lo : lo + step], raytrace, method, tol, lm_kwargs, want_images
            )

    @staticmethod
    def _to_source(raytrace):
        """Wrap ``raytrace(x, y)`` as a ``(..., 2) -> (..., 2)`` map."""

        def to_source(xy):
            return backend.stack(raytrace(xy[..., 0], xy[..., 1]), dim=-1)

        return to_source

    def forward_raytrace(
        self,
        beta,
        raytrace: Callable[[ArrayLike, ArrayLike], Tuple[ArrayLike, ArrayLike]],
        batch_size: Optional[int] = None,
        *,
        method: str = "rootfind",
        residual_tol: Optional[float] = None,
        lm_kwargs: Optional[dict] = None,
    ):
        """
        Image-plane positions of every image of each source-plane point.

        :meth:`seeds` supplies a Newton seed per candidate leaf, accurate to
        ``min_img_sep`` by construction; Levenberg-Marquardt refines each seed to a
        root of the lens equation, unconverged roots are discarded, and the
        survivors are deduplicated at ``min_img_sep``.

        Parameters
        ----------
        beta: ArrayLike
            Source-plane points, shape ``(B, 2)`` strictly. A single point must be
            passed as ``(1, 2)``.

            *Unit: arcsec*

        raytrace: Callable
            **Must be the same callable this mesh was built from**, called as
            ``raytrace(x, y) -> (bx, by)``. The seeds handed to the root finder are
            preimages under *this* mesh's leaves, so a different lens would be
            root-found from meaningless starting points -- silently, since the
            residual filter would simply reject most of them and return too few
            images rather than raising. This cannot be checked: a callable carries
            no identity the mesh could have recorded at build time.
        batch_size: Optional[int]
            Chunk size over source points. Bounds peak memory for the whole
            pipeline, not just :meth:`query` -- the root finder holds ``(K, 2)``
            states and the dedup a ``(B, M, M)`` adjacency. Results are identical
            for every value.
        method: str
            ``"rootfind"`` (default) refines every seed with
            Levenberg-Marquardt and returns machine-precision image positions.
            ``"dedup"`` skips the root finder entirely and deduplicates the
            seeds, which are already accurate to ``min_img_sep`` by
            construction. It **never calls** ``raytrace``, which is why it is
            roughly two orders of magnitude faster; ``raytrace``,
            ``residual_tol`` and ``lm_kwargs`` are accepted and ignored.

            Positions from ``"dedup"`` are accurate to ``min_img_sep``, not to
            machine precision. Counts agree with ``"rootfind"`` except within
            about ``min_img_sep`` of a caustic -- measured at 12 pixels in
            24656 on an EPL-plus-shear lens. Use ``"rootfind"`` when the
            position itself matters, ``"dedup"`` when the count does.
        residual_tol: Optional[float]
            Source-plane tolerance on ``|raytrace(x) - beta|`` for accepting a
            root. Defaults to ``min_img_sep``.

            *Unit: arcsec*

        lm_kwargs: Optional[dict]
            Extra keyword arguments for :func:`~caustics.utils.batch_lm`, e.g.
            ``max_iter``.

        Returns
        -------
        images: ArrayLike
            ``(K, 2)`` lens-plane image positions, laid out block-major: the
            ``counts[b]`` images of source ``b`` follow those of sources
            ``0 .. b - 1``.

            *Unit: arcsec*

        counts: ArrayLike
            ``(B,)`` int64 image multiplicity of each source point.

        Notes
        -----
        Two filters decide that a root is an image, and both are needed.

        The **residual** test alone is weak near a fold caustic, where the lens map
        is quadratic: a point sitting well over ``min_img_sep`` from the true image
        in the lens plane can still have a small source-plane residual, so it
        survives the residual test, escapes the dedup, and inflates the count
        exactly where multiplicity structure matters most.

        The **displacement** test closes that hole using a guarantee the mesh
        already makes -- the seed lies inside its leaf and is accurate to
        ``min_img_sep`` -- so a root that left its own neighbourhood is not the root
        its seed was pointing at. It is a disjunction rather than plain containment
        because a leaf at the size floor is itself only about ``min_img_sep``
        across, so a genuine root near a leaf edge can legitimately land just
        outside it; requiring containment alone would drop real images.

        Neither filter applies under ``method="dedup"``. Both reject a root
        that *wandered* -- the residual test catches a solve that converged to
        nothing, the displacement test one that converged to a different
        image. A seed cannot wander: it lies inside its own leaf, whose
        source-plane image contains ``beta``. Filtering it would test a
        property the construction already guarantees.

        Root finding runs in the dtype of the frozen mesh, so a mesh built with
        ``dtype=backend.float32`` caps the achievable accuracy near the
        ``stats.cancellation_floor`` the build already warns about.
        """
        _check_method(method)
        beta = self._as_beta(beta)
        n = beta.shape[0]
        int64 = backend.module.int64

        def as_counts(counts):
            return backend.as_array(
                np.asarray(counts, dtype=np.int64), dtype=int64, device=self.device
            )

        def no_images():
            return backend.zeros(
                (0, 2), dtype=self.vertices_lens.dtype, device=self.device
            )

        if n == 0:
            return no_images(), as_counts(np.empty(0))

        image_parts, count_parts = [], []
        for images, counts in self._image_chunks(
            beta, raytrace, batch_size, method, residual_tol, lm_kwargs, True
        ):
            count_parts.append(counts)
            if images is not None:
                image_parts.append(images)

        counts = as_counts(np.concatenate(count_parts))
        if not image_parts:
            return no_images(), counts
        return backend.concatenate(image_parts, dim=0), counts

    def multiplicity_map(
        self,
        raytrace: Callable[[ArrayLike, ArrayLike], Tuple[ArrayLike, ArrayLike]],
        pixelscale: float,
        nx: Optional[int] = None,
        ny: Optional[int] = None,
        *,
        x0: Optional[float] = None,
        y0: Optional[float] = None,
        method: str = "rootfind",
        batch_size: Optional[int] = None,
        residual_tol: Optional[float] = None,
        lm_kwargs: Optional[dict] = None,
    ):
        """
        Image multiplicity on a regular grid of source-plane positions.

        A :meth:`forward_raytrace` per pixel, reduced to its image count. The
        caustics are where the count changes.

        Parameters
        ----------
        raytrace: Callable
            The same callable this mesh was built from -- see
            :meth:`forward_raytrace`.
        method: str
            ``"rootfind"`` (default) or ``"dedup"``; see
            :meth:`forward_raytrace`. ``"dedup"`` is roughly two orders of
            magnitude faster here and is usually the right choice for a map,
            which needs counts rather than positions.
        pixelscale: float
            Side length of a source-plane pixel. Pixels are square, and this is
            the knob to compare against ``min_img_sep``: much below it the map
            resolves structure the mesh itself cannot, and much above it the
            multiplicity jumps at the caustics alias.

            *Unit: arcsec*

        nx, ny: Optional[int]
            Pixel counts. Each defaults to covering the source-plane bounding box
            of the indexed leaves, ``ceil(span / pixelscale)`` -- so the default
            field of view is the extent of the source-plane mesh, and square
            pixels give it a slight overhang.
        x0, y0: Optional[float]
            Grid centre, defaulting to the centre of that same bounding box.

            *Unit: arcsec*

        batch_size: Optional[int]
            Chunk size over pixels, forwarded to :meth:`forward_raytrace`. This
            matters more here than anywhere else in the module: the default
            ``None`` root-finds every pixel of the map in one batch.
        residual_tol, lm_kwargs
            Forwarded to :meth:`forward_raytrace`.

        Returns
        -------
        multiplicity: ArrayLike
            ``(ny, nx)`` int64 image count, with ``multiplicity[j, i]`` the pixel
            at ``(x_i, y_j)``.
        extent: Tuple[float, float, float, float]
            ``(x_min, x_max, y_min, y_max)`` outer pixel edges, ready for
            ``imshow(multiplicity, origin="lower", extent=extent)``. Worth
            returning even though it is derivable, because with ``nx``, ``ny``,
            ``x0`` and ``y0`` defaulted the caller does not know them.

        Notes
        -----
        Under ``method="rootfind"`` the cost is ``nx * ny`` root-finding solves
        over the mesh's mean candidate count, so it grows quadratically in
        ``1 / pixelscale`` and the solver dominates everything else -- 99% of
        the runtime at ``pixelscale=1e-2`` on a typical mesh. Under
        ``method="dedup"`` there is no solver, and the cost is the spatial
        query alone.

        Multiplicity here is the count of *distinct converged roots*, which is why
        it needed :meth:`forward_raytrace` rather than :meth:`query`: candidate
        count is not multiplicity, since a query on a shared edge returns both
        leaves and near-critical leaves overlap.
        """
        _check_method(method)
        if not pixelscale > 0:
            raise ValueError(f"pixelscale must be positive, got {pixelscale}")
        lo = backend.to_numpy(self._index_lo)
        hi = backend.to_numpy(self._index_hi)
        nx = max(1, int(ceil((hi[0] - lo[0]) / pixelscale))) if nx is None else int(nx)
        ny = max(1, int(ceil((hi[1] - lo[1]) / pixelscale))) if ny is None else int(ny)
        cx = float((lo[0] + hi[0]) / 2) if x0 is None else float(x0)
        cy = float((lo[1] + hi[1]) / 2) if y0 is None else float(y0)

        # `utils.meshgrid` already produces pixel *centres*, zero-centred, with
        # `indexing="xy"` -- so it gives (ny, nx) directly and only needs shifting
        # onto the requested centre. Sampling centres rather than a `linspace`
        # over the bounding box also keeps every query off the bbox edge, which is
        # the degenerate case `query`'s containment test has to special-case.
        gx, gy = meshgrid(
            pixelscale,
            nx,
            ny,
            device=self.device,
            dtype=self.vertices_source.dtype,
        )
        beta = backend.stack((gx + cx, gy + cy), dim=-1).reshape(-1, 2)
        # Counts only: `want_images=False` stops the per-chunk representatives
        # being gathered at all. The map discards them on the next line, and at
        # a fine `pixelscale` that array is hundreds of MB -- accumulated
        # across every chunk when `batch_size` is set.
        counts = backend.as_array(
            np.concatenate(
                [
                    part
                    for _, part in self._image_chunks(
                        beta,
                        raytrace,
                        batch_size,
                        method,
                        residual_tol,
                        lm_kwargs,
                        False,
                    )
                ]
            ),
            dtype=backend.module.int64,
            device=self.device,
        )
        extent = (
            cx - pixelscale * nx / 2,
            cx + pixelscale * nx / 2,
            cy - pixelscale * ny / 2,
            cy + pixelscale * ny / 2,
        )
        return counts.reshape(ny, nx), extent


def build_adaptive_mesh(
    raytrace: Callable[[ArrayLike, ArrayLike], Tuple[ArrayLike, ArrayLike]],
    fov: float,
    init_res: int,
    min_img_sep: float,
    max_depth: int = 25,
    *,
    x0: float = 0.0,
    y0: float = 0.0,
    device: Optional[Any] = None,
    dtype: Optional[Any] = None,
    raytrace_batch_size: Optional[int] = None,
    index_cells: Optional[int] = None,
) -> Mesh:
    """
    Build an adaptively refined triangular mesh of the lens plane.

    The mesh is built once and reused across many queries; it does not depend on any
    query point.

    Parameters
    ----------
    raytrace: Callable
        Maps lens-plane to source-plane coordinates, called as
        ``raytrace(x, y) -> (bx, by)`` on 1-D arrays of shape ``(N,)``.
    fov: float
        Side length of the square lens-plane domain.

        *Unit: arcsec*

    init_res: int
        Number of **cells** per axis, giving ``2 * init_res**2`` level-0 triangles.
        Note this differs from ``forward_raytrace``'s ``divisions``, which counts
        ``linspace`` *points* and yields ``(n - 1)**2`` cells.

        This is also the parameter carrying the completeness obligation: the
        criterion samples six points per triangle, so structure below the level-0
        scale is invisible to it. ``init_res`` must already resolve the smallest
        curvature scale in the lens; ``stats.n_converged_at_level_0`` is the check.
    min_img_sep: float
        Lens-plane tolerance, in two roles: the size floor ``l_max <= min_img_sep``
        and the step-7 threshold. No converged leaf hides an image pair separated by
        more than ``min_img_sep / 4``.

        *Unit: arcsec*

    max_depth: int
        Hard cap on refinement level. Refinement runs to
        ``min(max_depth, d_floor)``; a warning is raised if ``max_depth`` binds.
    x0, y0: float
        Centre of the domain.

        *Unit: arcsec*

    device: Optional
        Device for the coordinates handed to ``raytrace`` and for the frozen mesh.
    dtype: Optional
        Frozen-mesh dtype. Defaults to float64, the build dtype. Pass
        ``backend.float32`` to halve query memory.
    raytrace_batch_size: Optional[int]
        Splits each per-level ``raytrace`` call for memory. One logical batch per
        level is preserved. This bounds only the size of each ``raytrace`` call,
        **not** ``_VertexCache.beta``, which retains the source-plane image of
        every point ever evaluated for the whole build: a converged leaf's
        midpoints cannot be pruned, since the balance cascade may force-split
        that leaf later and need them. A caller setting this to bound peak
        memory should budget for the whole vertex cache, not just one level.
    index_cells: Optional[int]
        Spatial-index cells along the longer axis of the source-plane bounding box.

    Returns
    -------
    Mesh
    """
    _validate_build_args(fov, init_res, min_img_sep, max_depth)
    d_floor = _depth_floor(fov, init_res, min_img_sep)
    max_level = min(int(max_depth), d_floor)
    l_max_final = float(np.sqrt(2.0) * fov / (init_res * 2**max_level))
    if d_floor > max_depth:
        warn(
            f"Adaptive mesh is depth-limited: max_depth={max_depth} is below "
            f"d_floor={d_floor}, the depth required to reach "
            f"min_img_sep={min_img_sep:g} arcsec. Refinement stops at level "
            f"{max_level}, where the maximum leaf edge is {l_max_final:.3g} arcsec. "
            f"Set max_depth >= {d_floor} to restore the size-floor guarantee, or "
            f"raise init_res / min_img_sep."
        )

    tables = child_matrix_tables()
    lattice = _Lattice(fov, x0, y0, init_res, max_level)
    raytrace_np = _make_raytrace_np(raytrace, device)
    ref = _refine(
        raytrace_np,
        lattice,
        init_res,
        fov / init_res,
        min_img_sep,
        max_level,
        tables,
        raytrace_batch_size,
    )

    eps = float(np.finfo(raytrace_np.info["dtype"]).eps)
    cancellation_floor = float(np.sqrt(8.0 * eps * fov))
    if cancellation_floor > min_img_sep:
        warn(
            f"raytrace returned {raytrace_np.info['dtype']}, whose cancellation "
            f"floor sqrt(8*eps*fov) = {cancellation_floor:.3g} arcsec exceeds "
            f"min_img_sep={min_img_sep:g}. Below that scale the midpoint deviation "
            "cancels to zero, which the criterion reads as 'affine' and converges. "
            "Supply a raytrace that preserves float64."
        )

    pre_v, pre_level, pre_cls, pre_status = ref.store.compact()
    order = _canonical_order(lattice, ref.cache, pre_v)
    pre_v, pre_level, pre_status = pre_v[order], pre_level[order], pre_status[order]
    leaf_v, origin, leaf_level, leaf_status = _close(
        lattice, ref.cache, ref.active, pre_v, pre_level, pre_status
    )

    # Compaction: sorting by lattice key makes vertex order a function of the
    # geometry alone and gives row-major locality for query-time gathers.
    used = np.unique(np.concatenate([leaf_v.reshape(-1), pre_v.reshape(-1)]))
    used = used[np.argsort(lattice.key(ref.cache.ij[used]), kind="stable")]
    remap = np.zeros(len(ref.cache), dtype=np.int64)
    remap[used] = np.arange(used.size, dtype=np.int64)
    leaves_np = remap[leaf_v]
    origin_leaves_np = remap[pre_v]

    np_dtype = _numpy_dtype(dtype)
    vl = lattice.xy(ref.cache.ij[used]).astype(np_dtype)
    vs = ref.cache.beta[used].astype(np_dtype)

    # Re-check finiteness at freeze. A FORCED leaf inherits vertices from a parent
    # whose midpoints were never finiteness-tested, and a closure triangle can pick
    # up a midpoint that no criterion ever saw, so a non-finite vertex can reach
    # here on a leaf not already marked INVALID. Without this it would enter the
    # index and swallow every query in its cell.
    #
    # Invalidity is propagated UP to the origin and then back down, rather than
    # being applied to the leaves alone: the termination-reason counts are
    # pre-closure, so marking only the leaf would leave `n_converged + n_size_floor
    # + n_forced + n_invalid` disagreeing with `n_leaves_pre_closure`. It is also
    # the conservative direction -- if one triangle of a region has a bad vertex,
    # the region is not trustworthy.
    pre_status = _invalidate_nonfinite_origins(vs, leaves_np, origin, pre_status)
    leaf_status = pre_status[origin]

    # INVALID leaves may carry inf/nan vertices, and their `leaf_area2` is never
    # consumed -- they are excluded from the index below. Suppressing here keeps a
    # build over a lens with a non-finite region from spraying numpy
    # RuntimeWarnings at the caller; it changes no value. Same pattern as
    # `_min_angle`.
    with np.errstate(invalid="ignore"):
        P = shape_matrix(vs[leaves_np])
        leaf_area2 = P[:, 0, 0] * P[:, 1, 1] - P[:, 0, 1] * P[:, 1, 0]
    valid_rows = np.flatnonzero(leaf_status != LeafStatus.INVALID)
    index = _build_index(vs, leaves_np, valid_rows, index_cells)

    group_sizes = np.bincount(origin, minlength=pre_v.shape[0])
    boundary = lattice.on_boundary(ref.cache.ij[pre_v]).any(axis=1)
    stats = BuildStats(
        d_floor=d_floor,
        max_level=max_level,
        depth_limited=d_floor > max_depth,
        l_max_final=l_max_final,
        n_converged=int((pre_status == LeafStatus.CONVERGED).sum()),
        n_size_floor=int((pre_status == LeafStatus.SIZE_FLOOR).sum()),
        n_forced=int((pre_status == LeafStatus.FORCED).sum()),
        n_invalid=int((pre_status == LeafStatus.INVALID).sum()),
        n_converged_at_level_0=int(ref.counters["converged_level0"]),
        n_parity_splits=int(ref.counters["parity_splits"]),
        n_deviation_splits=int(ref.counters["deviation_splits"]),
        n_nonfinite_splits=int(ref.counters["nonfinite_splits"]),
        leaves_by_level=tuple(
            int(n) for n in np.bincount(pre_level, minlength=max_level + 1)
        ),
        n_nonfinite_vertices=int((~np.isfinite(ref.cache.beta).all(axis=1)).sum()),
        n_sigma_min_exactly_zero=int(ref.counters["sigma_zero"]),
        n_boundary_leaves_at_floor=int(
            (boundary & (pre_status == LeafStatus.SIZE_FLOOR)).sum()
        ),
        n_vertices=int(used.size),
        n_leaves_pre_closure=int(pre_v.shape[0]),
        n_leaves=int(leaves_np.shape[0]),
        n_closure_by_pattern=(
            int((group_sizes == 2).sum()),
            int((group_sizes == 3).sum()),
            int((group_sizes == 4).sum()),
        ),
        raytrace_dtype=str(raytrace_np.info["dtype"]),
        cancellation_floor=cancellation_floor,
    )

    def to_backend(array, integer=False):
        if integer:
            return backend.as_array(array, dtype=backend.module.int64, device=device)
        return backend.as_array(array, device=device)

    return Mesh(
        vertices_lens=to_backend(vl),
        vertices_source=to_backend(vs),
        leaves=to_backend(leaves_np, integer=True),
        leaf_area2=to_backend(leaf_area2),
        leaf_origin=to_backend(origin, integer=True),
        leaf_status=to_backend(leaf_status.astype(np.int64), integer=True),
        leaf_level=to_backend(leaf_level, integer=True),
        origin_leaves=to_backend(origin_leaves_np, integer=True),
        index=(
            to_backend(index[0]),
            to_backend(index[1]),
            index[2],
            index[3],
            to_backend(index[4], integer=True),
            to_backend(index[5], integer=True),
            to_backend(index[6]),
        ),
        stats=stats,
        d_floor=d_floor,
        max_level=max_level,
        min_img_sep=float(min_img_sep),
        dtype=np_dtype,
        device=device,
    )
