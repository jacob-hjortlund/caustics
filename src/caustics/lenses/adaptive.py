"""
Adaptively refined triangular mesh of the lens plane, queryable from the source plane.

``forward_raytrace`` rebuilds a mesh on every query and can miss images when the
initial grid is coarse enough that a mapped triangle is not locally affine. This
module builds a mesh **once**, using a refinement criterion that depends only on the
lens map and not on any query point, then answers many source-plane queries against
the frozen result.

The build is a host-side NumPy float64 algorithm whose only array-API contact is the
``raytrace`` callback; the frozen mesh and ``Mesh.query`` are backend-dispatched.

Non-goals, by design: no root finding, no image deduplication
(``forward_raytrace_rootfind`` composes with :meth:`Mesh.seeds` for that), no
autodiff, no jit, no vmap. Candidate count is **not** image multiplicity -- a point
on a shared edge returns both leaves, and near-critical leaves overlap.
"""

from enum import IntEnum
from math import ceil, log2

import numpy as np

from .func.adaptive import CHILD_VERTEX_INDICES, ROOT_SHAPES

__all__ = ["LeafStatus"]

_MAX_KEY = 2**63 - 1


class LeafStatus(IntEnum):
    """Why a terminal leaf stopped refining.

    ``FORCED`` is distinct from ``CONVERGED`` because a forced child carries no
    criterion evidence at all -- that is exactly what auto-converging decides -- so
    a caller auditing coverage must be able to tell them apart. Closure triangles
    have no status of their own; they inherit their origin's.
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

    Separate from the vertex cache, which also holds midpoints of
    tested-but-never-split triangles. Only ever grows, since a parent's vertices are
    inherited by all its children.
    """

    def __init__(self):
        self._keys = np.empty(0, dtype=np.int64)

    def add(self, keys):
        self._keys = np.union1d(self._keys, np.asarray(keys, dtype=np.int64))

    def contains(self, keys):
        if self._keys.size == 0:
            return np.zeros(np.shape(keys), dtype=bool)
        pos = np.clip(np.searchsorted(self._keys, keys), 0, self._keys.size - 1)
        return self._keys[pos] == keys


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
            [self.level, np.full(k, int(level), dtype=np.int64)]
        )
        self.cls = np.concatenate([self.cls, np.asarray(cls, dtype=np.int64)])
        self.status = np.concatenate(
            [self.status, np.full(k, int(status), dtype=np.int8)]
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
