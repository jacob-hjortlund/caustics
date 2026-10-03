import numpy as np
import pytest

from caustics.backend_obj import backend
from caustics.lenses.func import adaptive
from caustics.lenses.func.adaptive import criterion
from caustics.lenses.func.adaptive import (
    LEAF_APPROX_PARITY_UNRESOLVED,
    LEAF_CONVERGED,
    LEAF_CONVERGENCE_FAILED,
    LEAF_JACOBIAN_NONFINITE,
    LEAF_JACOBIAN_PARITY_UNRESOLVED,
    LEAF_RAYTRACE_NONFINITE,
    build_adaptive_mesh,
    child_matrix_tables,
)
from caustics.lenses.func.adaptive.lattice import (
    make_lattice,
)
from caustics.lenses.func.adaptive.refinement import red_split
from caustics.lenses.func.adaptive.sampling import evaluate, make_raytrace, trace_keys
from caustics.lenses.func.adaptive.state import (
    active_add_slots,
    active_contains,
    active_contains_slots,
    cache_insert,
    cache_lookup,
    cache_missing,
    cache_set_jacobian,
    cache_size,
    empty_active,
    empty_cache,
    empty_store,
    store_add,
    store_compact,
    store_remove,
)


def _i64(x):
    return backend.as_array(np.asarray(x, dtype=np.int64), dtype=backend.int64)


def _f64(x):
    return backend.as_array(np.asarray(x, dtype=np.float64), dtype=backend.float64)


def _bool(x):
    return backend.as_array(np.asarray(x, dtype=bool))


def test_cache_lookup_missing_insert():
    cache = empty_cache()
    assert backend.to_numpy(cache_lookup(cache, _i64([5, 9]))).tolist() == [-1, -1]

    todo = cache_missing(cache, _i64([9, 5, 9]))
    assert backend.to_numpy(todo).tolist() == [5, 9]

    cache, slots = cache_insert(
        cache, todo, _i64([[0, 5], [1, 2]]), _f64([[0.0, 1.0], [2.0, 3.0]])
    )
    assert backend.to_numpy(slots).tolist() == [0, 1]
    assert backend.to_numpy(cache_lookup(cache, _i64([9, 5, 7]))).tolist() == [
        1,
        0,
        -1,
    ]
    assert cache_size(cache) == 2


def test_cache_keys_stay_sorted_across_interleaved_inserts():
    rng = np.random.default_rng(7)
    cache = empty_cache()
    seen = []
    for _ in range(12):
        batch = rng.integers(0, 200, 9)
        todo = cache_missing(cache, _i64(batch))
        n = int(backend.to_numpy(todo).size)
        if n == 0:
            continue
        cache, _ = cache_insert(
            cache, todo, _i64(np.zeros((n, 2))), _f64(np.zeros((n, 2)))
        )
        seen.extend(backend.to_numpy(todo).tolist())
        keys = backend.to_numpy(cache.keys)
        assert (np.diff(keys) > 0).all(), "cache keys must stay strictly ascending"
    assert sorted(set(seen)) == backend.to_numpy(cache.keys).tolist()


def test_cache_slots_are_assigned_in_insertion_order():
    cache = empty_cache()
    cache, slots_a = cache_insert(
        cache, _i64([10, 20]), _i64(np.zeros((2, 2))), _f64(np.zeros((2, 2)))
    )
    cache, slots_b = cache_insert(
        cache, _i64([5, 15]), _i64(np.zeros((2, 2))), _f64(np.zeros((2, 2)))
    )
    assert backend.to_numpy(slots_a).tolist() == [0, 1]
    assert backend.to_numpy(slots_b).tolist() == [2, 3]
    # Slot order is insertion order; key order is sorted. They differ.
    assert backend.to_numpy(cache.keys).tolist() == [5, 10, 15, 20]
    assert backend.to_numpy(cache.slots).tolist() == [2, 0, 3, 1]


def test_inserted_points_carry_no_jacobian():
    cache, _ = cache_insert(
        empty_cache(), _i64([3, 8]), _i64(np.zeros((2, 2))), _f64(np.zeros((2, 2)))
    )
    assert np.isnan(backend.to_numpy(cache.det)).all()
    assert backend.to_numpy(cache.sign).tolist() == [0, 0]
    assert not backend.to_numpy(cache.has_det).any()
    assert not backend.to_numpy(cache.evaluated).any()


def test_cache_set_jacobian_records_every_sign_and_only_a_first_det():
    """Slot 1 already carries a det, as a vertex seeded from a frozen mesh
    does: it gains a sign and keeps its det. The input cache is untouched,
    under torch too."""
    cache, _ = cache_insert(
        empty_cache(), _i64([3, 8, 9]), _i64(np.zeros((3, 2))), _f64(np.zeros((3, 2)))
    )
    cache = cache._replace(
        det=_f64([np.nan, 7.0, np.nan]), has_det=_bool([False, True, False])
    )
    got = cache_set_jacobian(cache, _i64([1, 2]), _f64([-1.0, 2.0]), _i64([-1, 1]))
    assert np.array_equal(backend.to_numpy(got.det), [np.nan, 7.0, 2.0], equal_nan=True)
    assert backend.to_numpy(got.sign).tolist() == [0, -1, 1]
    assert backend.to_numpy(got.has_det).tolist() == [False, True, True]
    assert backend.to_numpy(got.evaluated).tolist() == [False, True, True]
    assert np.array_equal(
        backend.to_numpy(cache.det), [np.nan, 7.0, np.nan], equal_nan=True
    )
    assert backend.to_numpy(cache.sign).tolist() == [0, 0, 0]
    assert backend.to_numpy(cache.has_det).tolist() == [False, True, False]
    assert backend.to_numpy(cache.evaluated).tolist() == [False, False, False]


def test_active_membership_and_negative_slots():
    cache = empty_cache()
    cache, slots = cache_insert(
        cache, _i64([3, 8]), _i64(np.zeros((2, 2))), _f64(np.zeros((2, 2)))
    )
    active = empty_active()
    active = active_add_slots(active, cache_size(cache), _i64([0]))
    assert backend.to_numpy(
        active_contains_slots(active, _i64([0, 1, -1]))
    ).tolist() == [True, False, False]
    assert backend.to_numpy(
        active_contains(active, cache, _i64([3, 8, 99]))
    ).tolist() == [True, False, False]


def test_active_add_slots_rejects_an_uncached_slot():
    active = empty_active()
    with pytest.raises(AssertionError, match="cannot activate an uncached vertex"):
        active_add_slots(active, 2, _i64([-1, 0]))


def test_active_add_slots_does_not_mutate_its_input_on_the_no_growth_path():
    """Regression test: torch's `fill_at_indices` mutates its argument in place
    and returns it, so scattering straight into `active` when no growth is
    needed would silently clobber the caller's own array under torch while
    leaving it untouched under jax. `before` aliases the array returned by the
    first call; the second call must not be able to reach through that alias.
    """
    active = active_add_slots(empty_active(), 2, _i64([0]))
    before = active
    after = active_add_slots(active, 2, _i64([1]))  # same n_slots: no growth
    assert backend.to_numpy(before).tolist() == [True, False]
    assert backend.to_numpy(after).tolist() == [True, True]


def test_cache_insert_keeps_ij_and_beta_aligned_with_their_slot_across_many_merges():
    """Coverage restored from the deleted `test_vertex_cache_survives_many_
    small_inserts`: each key's ij/beta row must stay aligned with its own
    slot, not just its key position, across many merge-inserts.

    Descending keys force a merge at the front of ``cache.keys`` on every
    insert, the worst case for the searchsorted-merge in :func:`cache_insert`.
    """
    cache = empty_cache()
    probe = []
    for k in range(40):
        key = 1000 - k
        cache, slot = cache_insert(
            cache, _i64([key]), _i64([[key, key]]), _f64([[key, key]]) * 0.5
        )
        assert backend.to_numpy(slot).tolist() == [k]
        probe.append(key)

    probe = np.array(probe, dtype=np.int64)
    assert cache_size(cache) == 40
    assert backend.to_numpy(cache_lookup(cache, _i64(probe))).tolist() == list(
        range(40)
    )
    assert np.array_equal(backend.to_numpy(cache.ij)[:, 0], probe)
    assert np.allclose(
        backend.to_numpy(cache.beta)[:, 0], probe.astype(np.float64) * 0.5
    )


def test_store_add_remove_compact():
    store = empty_store()
    store, rows = store_add(
        store, _i64([[0, 1, 2], [3, 4, 5]]), 1, _i64([0, 1]), LEAF_CONVERGED
    )
    assert backend.to_numpy(rows).tolist() == [0, 1]

    store, rows2 = store_add(
        store, _i64([[6, 7, 8]]), 2, _i64([2]), LEAF_CONVERGENCE_FAILED
    )
    assert backend.to_numpy(rows2).tolist() == [2]

    store = store_remove(store, _i64([0]))
    v, level, cls, status = store_compact(store)
    assert backend.to_numpy(v).tolist() == [[3, 4, 5], [6, 7, 8]]
    assert backend.to_numpy(level).tolist() == [1, 2]
    assert backend.to_numpy(status).tolist() == [
        LEAF_CONVERGED,
        LEAF_CONVERGENCE_FAILED,
    ]


def test_store_add_accepts_per_row_level_and_status():
    """A per-row status is stored as given, combined flags included."""
    both = LEAF_APPROX_PARITY_UNRESOLVED | LEAF_JACOBIAN_PARITY_UNRESOLVED
    store = empty_store()
    store, _ = store_add(
        store,
        _i64([[0, 1, 2], [3, 4, 5]]),
        _i64([1, 4]),
        _i64([0, 1]),
        _i64([LEAF_CONVERGED, both]),
    )
    _, level, _, status = store_compact(store)
    assert backend.to_numpy(level).tolist() == [1, 4]
    assert backend.to_numpy(status).tolist() == [LEAF_CONVERGED, both]


def test_store_survives_many_small_adds_and_removals():
    rng = np.random.default_rng(11)
    store = empty_store()
    alive = []
    for i in range(40):
        k = int(rng.integers(1, 5))
        v = np.arange(3 * k).reshape(k, 3) + 100 * i
        store, rows = store_add(store, _i64(v), i, _i64(np.zeros(k)), 0)
        alive.extend(zip(backend.to_numpy(rows).tolist(), v.tolist()))
        if len(alive) > 3 and i % 3 == 0:
            drop = alive.pop(0)
            store = store_remove(store, _i64([drop[0]]))
    v_out, _, _, _ = store_compact(store)
    assert backend.to_numpy(v_out).tolist() == [rec[1] for rec in alive]


def test_leaf_status_constants_are_distinct_single_bit_flags():
    """Every failure flag is its own bit, so any OR of them decodes uniquely.

    ``LEAF_CONVERGED`` is zero -- the empty set of failures -- which is what
    lets ``status == LEAF_CONVERGED`` mean "no test failed" however many flags
    a failing leaf carries.
    """
    flags = [
        LEAF_CONVERGENCE_FAILED,
        LEAF_APPROX_PARITY_UNRESOLVED,
        LEAF_JACOBIAN_PARITY_UNRESOLVED,
        LEAF_RAYTRACE_NONFINITE,
        LEAF_JACOBIAN_NONFINITE,
    ]
    assert LEAF_CONVERGED == 0
    assert flags == [1, 2, 4, 8, 16]
    assert all(type(v) is int for v in [LEAF_CONVERGED, *flags])
    combos = {
        sum(f for i, f in enumerate(flags) if mask >> i & 1)
        for mask in range(2 ** len(flags))
    }
    assert len(combos) == 2 ** len(flags), "some OR of flags is ambiguous"
    for name in ("LEAF_SIZE_FLOOR", "LEAF_FORCED", "LEAF_INVALID", "LEAF_NONFINITE"):
        for namespace in (adaptive, criterion):
            assert not hasattr(namespace, name), f"{name} was retired with the bitmask"


def test_store_remove_does_not_mutate_its_input():
    """Regression test: torch's `fill_at_indices` mutates its argument in
    place and returns it, so scattering straight into `store.valid` would
    silently clobber the caller's own array under torch while leaving it
    untouched under jax. `before` aliases the array the input store holds;
    removing rows from the returned store must not be able to reach through
    that alias.
    """
    store = empty_store()
    store, _ = store_add(
        store, _i64([[0, 1, 2], [3, 4, 5]]), 0, _i64([0, 0]), LEAF_CONVERGED
    )
    before = store.valid
    after = store_remove(store, _i64([0]))
    assert backend.to_numpy(before).tolist() == [True, True]
    assert backend.to_numpy(store.valid).tolist() == [True, True]
    assert backend.to_numpy(after.valid).tolist() == [False, True]


def test_red_split_is_triangle_major_and_advances_the_class():
    _, _, compose, _, _ = child_matrix_tables()
    v = _i64([[0, 1, 2], [3, 4, 5]])
    m = _i64([[6, 7, 8], [9, 10, 11]])
    cls = _i64([0, 4])
    cv, cc = red_split(v, m, cls, compose)
    assert cv.shape == (8, 3) and cc.shape == (8,)
    cv_np = backend.to_numpy(cv)
    cc_np = backend.to_numpy(cc)
    compose_np = backend.to_numpy(compose)
    assert cv_np[0].tolist() == [0, 8, 7]  # C_1 = (theta1, m3, m2)
    assert cv_np[3].tolist() == [6, 7, 8]  # C_4 = (m1, m2, m3)
    assert cc_np[:4].tolist() == compose_np[0].tolist()
    assert cc_np[4:].tolist() == compose_np[4].tolist()


def test_make_raytrace_forces_float64_and_records_the_callback_dtype():
    seen = {}

    def rt(x, y):
        seen["dtype"] = x.dtype
        return 2.0 * x, 3.0 * y

    fn = make = make_raytrace(rt, None)
    out = fn(_f64([[1.0, 2.0], [3.0, 4.0]]))
    assert seen["dtype"] == backend.float64
    assert backend.to_numpy(out).tolist() == [[2.0, 6.0], [6.0, 12.0]]
    assert make.info["dtype"] == backend.float64


def test_make_raytrace_returns_to_the_input_device(monkeypatch):
    xy = _f64([[1.0, 2.0]])
    input_device = backend.device(xy)
    calls = []
    seen = {}
    original_to = backend.to

    def recording_to(array, *args, **kwargs):
        calls.append(kwargs.copy())
        return original_to(array, *args, **kwargs)

    def rt(x, y):
        seen["device"] = backend.device(x)
        return x, y

    monkeypatch.setattr(backend, "to", recording_to)
    out = make_raytrace(rt, input_device)(xy)
    assert seen["device"] == input_device
    assert backend.device(out) == input_device
    assert calls[-1]["device"] == input_device


def test_make_raytrace_records_float32_callback_dtype_but_returns_float64():
    def rt(x, y):
        return backend.to(x, dtype=backend.float32), backend.to(
            y, dtype=backend.float32
        )

    fn = make_raytrace(rt, None)
    out = fn(_f64([[1.0, 2.0]]))
    assert fn.info["dtype"] == backend.float32
    assert out.dtype == backend.float64


def test_make_raytrace_rejects_a_non_tuple_return():
    fn = make_raytrace(lambda x, y: x, None)
    with pytest.raises(ValueError, match="2-tuple"):
        fn(_f64([[1.0, 2.0]]))


def test_make_raytrace_rejects_a_shape_changing_callback():
    fn = make_raytrace(lambda x, y: (x[:1], y[:1]), None)
    with pytest.raises(ValueError, match="shape-preserving"):
        fn(_f64([[1.0, 2.0], [3.0, 4.0]]))


def test_trace_keys_is_bit_identical_under_chunking():
    lat = make_lattice(4.0, 0.0, 0.0, 4, 3)
    fn = make_raytrace(lambda x, y: (x * x - y, y * y + x), None)
    ij = _i64(
        np.stack(np.meshgrid(np.arange(9), np.arange(9), indexing="ij"), -1).reshape(
            -1, 2
        )
    )
    whole = backend.to_numpy(trace_keys(lat, ij, fn, None))
    for size in (1, 7, 33):
        assert (backend.to_numpy(trace_keys(lat, ij, fn, size)) == whole).all()


def test_evaluate_only_traces_uncached_points():
    calls = {"n": 0}

    def rt(x, y):
        calls["n"] += int(x.shape[0])
        return x, y

    lat = make_lattice(4.0, 0.0, 0.0, 2, 2)
    fn = make_raytrace(rt, None)
    cache = empty_cache()
    keys = _i64([3, 7, 3, 11])
    cache = evaluate(cache, lat, keys, fn, None)
    assert calls["n"] == 3
    cache = evaluate(cache, lat, keys, fn, None)
    assert calls["n"] == 3, "already-cached points must not be re-traced"


def _affine_pair():
    """A contracting affine lens map and its Jacobian."""

    def raytrace(x, y):
        return 0.5 * x, 0.5 * y

    def jacobian(x, y):
        half, zero = x * 0.0 + 0.5, x * 0.0
        return backend.stack(
            (backend.stack((half, zero), dim=-1), backend.stack((zero, half), dim=-1)),
            dim=-2,
        )

    return raytrace, jacobian


def test_build_rejects_a_swapped_pair_loudly():
    """``jacobian`` in ``raytrace``'s place returns an array, not a 2-tuple."""
    raytrace, jacobian = _affine_pair()
    with pytest.raises(ValueError, match="2-tuple"):
        build_adaptive_mesh(jacobian, raytrace, 4.0, 4, 0.1)


def test_build_rejects_raytrace_given_as_jacobian_loudly():
    """``raytrace`` in ``jacobian``'s place returns a 2-tuple, not an array."""
    raytrace, _ = _affine_pair()
    with pytest.raises(ValueError, match="jacobian_fn must return an array"):
        build_adaptive_mesh(raytrace, raytrace, 4.0, 4, 0.1)
