"""
The critical band: the ``max_level`` leaves ``det A`` changes sign across.

Built from the Jacobians of the build's own ``max_level`` pass
(:func:`band_from_samples`), carried across an extension
(:func:`merge_bands`), and mapped onto the frozen leaves
(:func:`band_leaf_index`). :mod:`~caustics.lenses.func.adaptive.curves`
traces it.
"""

from typing import NamedTuple

from ....backend_obj import ArrayLike, backend
from .state import cache_lookup
from .lattice import lattice_ij_from_key, lattice_key, lattice_xy, midpoint_ij

__all__ = (
    "CriticalBand",
    "empty_band",
    "band_from_samples",
    "band_leaf_index",
    "band_sample_keys",
    "merge_bands",
    "in_band",
    "build_band",
)


class CriticalBand(NamedTuple):
    """
    The ``max_level`` leaves ``det A`` changes sign across, with ``det A`` at their samples.

    A leaf is in the band when the determinants at its six samples -- the
    three vertices and the three edge midpoints, the corners of its four
    red-split children -- are all finite and not all of one class, where a
    sample's class is ``det >= 0``: an exact zero counts as positive. Every
    ``LEAF_JACOBIAN_PARITY_UNRESOLVED`` leaf is in the band. So is a leaf
    flagged ``LEAF_JACOBIAN_NONFINITE`` only because some sample's
    determinant is exactly zero, when zero counting as positive leaves its
    classes mixed -- which is what keeps a critical curve through lattice
    points from vanishing. The one exception is rounding, not overflow: the
    sign test behind the flags reads a row-scaled ``det A``, while this band
    stores the raw ``det A``, and the two can disagree whenever ``det A`` is
    within a few ulps of zero, even with O(1) entries. Measured on 1e6
    near-singular matrices, 154,070 had a raw determinant ``>= 0`` paired
    with a scaled determinant ``< 0``, and 48,349 had a nonzero raw
    determinant paired with a scaled determinant of exactly zero. So,
    rarely, a flagged leaf can be missing from the band, or a band leaf can
    carry neither flag. Tracing is unaffected, because the band is
    consistent with itself.

    Samples are deduplicated by lattice key: a sample shared by several band
    leaves is one row, with one ``det``, so every leaf sharing it agrees on
    its class. That consistency is what lets
    :func:`~caustics.lenses.func.adaptive.curves.mesh_critical_curves_and_caustics`
    chain the crossings of neighbouring leaves.

    Parameters
    ----------
    leaves: ArrayLike
        Shape ``(F,)`` int64 index into ``AdaptiveMesh.leaves``. As
        :func:`refine` returns it, an index into the :class:`LeafStore`'s
        rows instead; :func:`freeze` maps it onto the mesh and sorts the
        rows by it, so row order depends on the mesh alone.
    samples: ArrayLike
        Shape ``(F, 6)`` int64 index into ``lens``, ``source`` and ``det``:
        each leaf's ``theta_1, theta_2, theta_3, m_1, m_2, m_3``, the vertices
        in the leaf's own order and ``m_i`` opposite ``theta_i``.
        Samples are numbered in ascending lattice-key order.
    lens: ArrayLike
        Lens-plane position of each sample, shape ``(S, 2)``, at the mesh
        dtype.

        *Unit: arcsec*
    source: ArrayLike
        Source-plane image of each sample, shape ``(S, 2)``, at the mesh
        dtype.

        *Unit: arcsec*
    det: ArrayLike
        ``det A`` at each sample, shape ``(S,)``, always float64.
    """

    leaves: ArrayLike
    samples: ArrayLike
    lens: ArrayLike
    source: ArrayLike
    det: ArrayLike


def empty_band(device=None) -> CriticalBand:
    """A :class:`CriticalBand` with no leaves."""
    return CriticalBand(
        leaves=backend.zeros((0,), dtype=backend.int64, device=device),
        samples=backend.zeros((0, 6), dtype=backend.int64, device=device),
        lens=backend.zeros((0, 2), dtype=backend.float64, device=device),
        source=backend.zeros((0, 2), dtype=backend.float64, device=device),
        det=backend.zeros((0,), dtype=backend.float64, device=device),
    )


def band_from_samples(lat, cache, keys, index, det, mid_keys, mid_beta) -> CriticalBand:
    """
    The :class:`CriticalBand` among triangles whose six samples are evaluated.

    Parameters
    ----------
    lat: Lattice
    cache: VertexCache
        Supplies each vertex sample's image.
    keys, index, det: ArrayLike
        From :func:`sample_jacobians`: ``det`` is float64, one value per key.
    mid_keys: ArrayLike
        ``(M,)`` int64 keys of the traced ``max_level`` midpoints, ascending.
    mid_beta: ArrayLike
        ``(M, 2)`` float64, their images.

        *Unit: arcsec*

    Returns
    -------
    CriticalBand
        With ``leaves`` indexing the rows of ``index``, and positions and
        images at float64.

    Raises
    ------
    AssertionError
        If a band sample is neither a cached vertex nor a traced midpoint.
    """
    det6 = det[index]
    positive = det6 >= 0
    in_band = (
        backend.all(backend.isfinite(det6), dim=1)
        & backend.any(positive, dim=1)
        & backend.any(~positive, dim=1)
    )
    rows = backend.flatnonzero(in_band)
    if rows.shape[0] == 0:
        return empty_band()

    used, samples = backend.unique(index[rows].reshape(-1), return_inverse=True)
    point_keys = keys[used]
    # A vertex key is always cached and a max_level midpoint key never is --
    # it has an odd coordinate -- so cache membership alone tells them apart.
    slot = cache_lookup(cache, point_keys)
    vertex = slot >= 0
    at = backend.clamp(
        backend.searchsorted(mid_keys, point_keys), 0, mid_keys.shape[0] - 1
    )
    # `raise AssertionError` rather than a bare `assert`, as in
    # `active_add_slots`: `python -O` strips bare asserts, and a sample
    # matched to the wrong image would silently misplace the caustic.
    if not bool(backend.all(vertex | (mid_keys[at] == point_keys))):
        raise AssertionError("band sample is neither a cached vertex nor a midpoint")
    source = backend.where(
        backend.unsqueeze(vertex, -1),
        cache.beta[backend.where(vertex, slot, 0)],
        mid_beta[at],
    )
    return CriticalBand(
        leaves=rows,
        samples=samples.reshape(-1, 6),
        lens=lattice_xy(lat, lattice_ij_from_key(lat, point_keys)),
        source=source,
        det=det[used],
    )


def band_leaf_index(valid, order, origin, rows) -> ArrayLike:
    """
    Frozen-mesh leaf of each ``max_level`` :class:`LeafStore` row.

    A ``max_level`` row is never invalidated -- :func:`refine` breaks right
    after adding it, before any cascade -- so its compact position is the
    count of valid rows before it. :func:`canonical_order` then permutes the
    compact rows, and :func:`close` emits exactly one leaf per ``max_level``
    origin, since none has a hanging node; ``origin`` is non-decreasing, so
    ``searchsorted`` finds it.

    Parameters
    ----------
    valid: ArrayLike
        ``LeafStore.valid``, shape ``(N,)`` bool.
    order: ArrayLike
        The :func:`canonical_order` permutation of the compact rows.
    origin: ArrayLike
        From :func:`close`, shape ``(L,)``, non-decreasing.
    rows: ArrayLike
        ``(F,)`` int64 store rows, each a ``max_level`` leaf.

    Returns
    -------
    ArrayLike
        ``(F,)`` int64 index into the closed leaves.
    """
    compact = backend.cumsum(backend.long(valid), dim=0)[rows] - 1
    canonical = backend.argsort(order)[compact]
    return backend.searchsorted(origin, canonical)


def band_sample_keys(lat, cache, store, rows) -> ArrayLike:
    """
    Lattice keys of the six samples of each store row, shape ``(F, 6)``.

    In :class:`CriticalBand` order, ``theta_1, theta_2, theta_3, m_1, m_2,
    m_3``: the vertices through the vertex cache, the midpoints exact on the
    lattice through :func:`midpoint_ij`.
    """
    ij = cache.ij[store.v[rows]]
    return lattice_key(lat, backend.concatenate((ij, midpoint_ij(ij)), dim=1))


def merge_bands(lat, cache, store, first, second) -> CriticalBand:
    """
    One :class:`CriticalBand` from two whose ``leaves`` are disjoint store rows.

    A band's samples are its distinct sample keys in ascending order -- true
    of every band :func:`band_from_samples` returns and of every band this
    returns -- so each band's own keys follow from its rows alone. The merged
    band has one sample per distinct key over both, ascending again. A key
    both hold -- a sample on the seam between an old mesh and the ring an
    extension adds -- takes ``first``'s ``det`` and ``source``, so every
    sample has one value, which is what
    :func:`~caustics.lenses.func.adaptive.curves.trace_band` chains on.
    ``lens`` is recomputed from the keys by :func:`lattice_xy`.

    Every scatter writes distinct indices: all of ``first``'s positions, and
    only those of ``second``'s that ``first`` lacks.

    Parameters
    ----------
    lat: Lattice
    cache: VertexCache
    store: LeafStore
    first, second: CriticalBand
        ``leaves`` index ``store``'s rows; ``source`` and ``det`` float64.

    Returns
    -------
    CriticalBand
        ``leaves`` is ``first.leaves`` then ``second.leaves``, still store
        rows; ``lens`` and ``source`` float64.

    Raises
    ------
    AssertionError
        If a band's rows do not hold exactly its samples.
    """
    if first.leaves.shape[0] + second.leaves.shape[0] == 0:
        return empty_band()
    k1 = band_sample_keys(lat, cache, store, first.leaves)
    k2 = band_sample_keys(lat, cache, store, second.leaves)
    own1 = backend.unique(k1.reshape(-1))
    own2 = backend.unique(k2.reshape(-1))
    # `raise AssertionError` rather than a bare `assert`: `python -O` strips
    # bare asserts, and a band read off the wrong keys would pair samples
    # with another point's values.
    if own1.shape[0] != first.det.shape[0] or own2.shape[0] != second.det.shape[0]:
        raise AssertionError("a band's rows do not hold exactly its samples")
    keys, samples = backend.unique(
        backend.concatenate((k1, k2), dim=0).reshape(-1), return_inverse=True
    )
    at1 = backend.searchsorted(keys, own1)
    at2 = backend.searchsorted(keys, own2)
    if own1.shape[0]:
        pos = backend.clamp(backend.searchsorted(own1, own2), 0, own1.shape[0] - 1)
        only2 = backend.flatnonzero(own1[pos] != own2)
    else:
        only2 = backend.arange(own2.shape[0], dtype=backend.int64)
    n = keys.shape[0]
    det = backend.zeros((n,), dtype=backend.float64)
    det = backend.fill_at_indices(det, at2[only2], second.det[only2])
    det = backend.fill_at_indices(det, at1, first.det)
    source = backend.zeros((n, 2), dtype=backend.float64)
    source = backend.fill_at_indices(source, at2[only2], second.source[only2])
    source = backend.fill_at_indices(source, at1, first.source)
    return CriticalBand(
        leaves=backend.concatenate((first.leaves, second.leaves), dim=0),
        samples=samples.reshape(-1, 6),
        lens=lattice_xy(lat, lattice_ij_from_key(lat, keys)),
        source=source,
        det=det,
    )


def in_band(det6):
    """
    True where a leaf's six ``det A`` are finite and not all of one class.

    A sample's class is ``det >= 0``: an exact zero counts as positive, which
    keeps a critical curve through lattice points from vanishing.
    """
    positive = det6 >= 0
    return (
        backend.all(backend.isfinite(det6), dim=1)
        & backend.any(positive, dim=1)
        & backend.any(~positive, dim=1)
    )


def build_band(lat, leaves, keys6, table_keys, table_values):
    """
    The :class:`CriticalBand` of ``leaves``, its sample values read from a table.

    Parameters
    ----------
    lat: Lattice
    leaves: ArrayLike
        ``(F,)`` int64 index of each band leaf into ``origin_leaves``.
    keys6: ArrayLike
        ``(F, 6)`` int64 lattice keys of each leaf's ``theta_1, theta_2,
        theta_3, m_1, m_2, m_3``.
    table_keys: ArrayLike
        ``(T,)`` int64 ascending keys, holding every key of ``keys6``.
    table_values: ArrayLike
        ``(T, 3)`` float64 ``(bx, by, det A)`` at ``table_keys``.

    Returns
    -------
    CriticalBand
        With samples numbered in ascending key order.
    """
    keys, samples = backend.unique(keys6.reshape(-1), return_inverse=True)
    values = table_values[backend.searchsorted(table_keys, keys)]
    return CriticalBand(
        leaves=leaves,
        samples=samples.reshape(-1, 6),
        lens=lattice_xy(lat, lattice_ij_from_key(lat, keys)),
        source=values[:, :2],
        det=values[:, 2],
    )
