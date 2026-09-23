import numpy as np
import pytest

from caustics.backend_obj import backend
from caustics.lenses.func import adaptive as new


@pytest.fixture
def oracle_module():
    return pytest.importorskip(
        "caustics.lenses.old_adaptive", reason="optional frozen differential oracle"
    )


def test_lattice_key_round_trips_and_matches_the_oracle(oracle_module):
    lat = new.make_lattice(4.0, 0.0, 0.0, 2, 3)
    old = oracle_module._Lattice(4.0, 0.0, 0.0, 2, 3)
    ij = np.array([[0, 0], [1, 3], [16, 16], [5, 11]], dtype=np.int64)
    ij_b = backend.as_array(ij, dtype=backend.int64)

    key = new.lattice_key(lat, ij_b)
    assert backend.to_numpy(key).tolist() == old.key(ij).tolist()
    assert backend.to_numpy(new.lattice_ij_from_key(lat, key)).tolist() == ij.tolist()
    assert np.allclose(backend.to_numpy(new.lattice_xy(lat, ij_b)), old.xy(ij))


def test_lattice_on_boundary_matches_the_oracle(oracle_module):
    lat = new.make_lattice(4.0, 0.0, 0.0, 2, 2)
    old = oracle_module._Lattice(4.0, 0.0, 0.0, 2, 2)
    ij = np.array([[0, 4], [8, 1], [3, 3], [8, 8]], dtype=np.int64)
    got = backend.to_numpy(
        new.lattice_on_boundary(lat, backend.as_array(ij, dtype=backend.int64))
    )
    assert got.tolist() == old.on_boundary(ij).tolist()


@pytest.mark.parametrize(
    "fov,init_res,min_img_sep", [(5.0, 4, 0.1), (1.0, 1, 2.0), (10.0, 8, 0.001)]
)
def test_depth_floor_matches_the_oracle(oracle_module, fov, init_res, min_img_sep):
    assert new.depth_floor(fov, init_res, min_img_sep) == oracle_module._depth_floor(
        fov, init_res, min_img_sep
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"fov": 0.0, "init_res": 2, "min_img_sep": 0.1, "max_depth": 3},
        {"fov": 1.0, "init_res": 0, "min_img_sep": 0.1, "max_depth": 3},
        {"fov": 1.0, "init_res": 2, "min_img_sep": 0.0, "max_depth": 3},
        {"fov": 1.0, "init_res": 2, "min_img_sep": 0.1, "max_depth": -1},
    ],
)
def test_validate_build_args_rejects_bad_input(kwargs):
    with pytest.raises(ValueError):
        new.validate_build_args(**kwargs)


def test_validate_build_args_rejects_lattice_overflow():
    with pytest.raises(ValueError, match="lattice too fine"):
        new.validate_build_args(
            fov=1.0, init_res=2**20, min_img_sep=1e-12, max_depth=40
        )


def test_depth_floor_matches_the_size_criterion():
    assert new.depth_floor(5.0, 100, 10.0) == 0
    d = new.depth_floor(5.0, 100, 1e-3)
    l0 = np.sqrt(2) * 5.0 / 100
    assert l0 / 2**d <= 1e-3 < l0 / 2 ** (d - 1)


def test_lattice_key_roundtrip_and_geometry():
    lat = new.make_lattice(4.0, 0.0, 0.0, 4, 3)
    assert lat.n == 32
    ij_np = np.array([[0, 0], [32, 32], [7, 19]], dtype=np.int64)
    ij = backend.as_array(ij_np, dtype=backend.int64)
    key = new.lattice_key(lat, ij)
    assert (
        backend.to_numpy(new.lattice_ij_from_key(lat, key)).tolist() == ij_np.tolist()
    )
    assert np.allclose(backend.to_numpy(new.lattice_xy(lat, ij[0])), [-2.0, -2.0])
    assert np.allclose(backend.to_numpy(new.lattice_xy(lat, ij[1])), [2.0, 2.0])
    assert backend.to_numpy(new.lattice_on_boundary(lat, ij)).tolist() == [
        True,
        True,
        False,
    ]


def test_widening_the_lattice_does_not_move_any_vertex():
    """Bit-identical coordinates, not merely close ones.

    `scale' = fov / (2n)` equals `fl(fov / n) / 2` exactly, because binary
    floating point is scale-invariant under powers of two, and `(2 * ij) *
    scale'` then rounds the same exact real as `ij * scale`. If this ever fails,
    the widened lattice has perturbed the frozen mesh's geometry and every
    downstream bit-exactness argument in the module is void.
    """
    fov, init_res, max_level = 4.0, 4, 3
    narrow = new.make_lattice(fov, 0.0, 0.0, init_res, max_level)
    wide = new.make_lattice(fov, 0.0, 0.0, init_res, max_level + 1)
    assert wide.n == 2 * narrow.n
    assert wide.level == max_level + 1

    ij_np = np.stack(
        np.meshgrid(np.arange(narrow.n + 1), np.arange(narrow.n + 1), indexing="ij"),
        axis=-1,
    ).reshape(-1, 2)
    ij = backend.as_array(ij_np, dtype=backend.int64)
    assert np.array_equal(
        backend.to_numpy(new.lattice_xy(narrow, ij)),
        backend.to_numpy(new.lattice_xy(wide, 2 * ij)),
    )


def _i64(x):
    return backend.as_array(np.asarray(x, dtype=np.int64), dtype=backend.int64)


def _f64(x):
    return backend.as_array(np.asarray(x, dtype=np.float64), dtype=backend.float64)


def test_cache_lookup_missing_insert():
    cache = new.empty_cache()
    assert backend.to_numpy(new.cache_lookup(cache, _i64([5, 9]))).tolist() == [-1, -1]

    todo = new.cache_missing(cache, _i64([9, 5, 9]))
    assert backend.to_numpy(todo).tolist() == [5, 9]

    cache, slots = new.cache_insert(
        cache, todo, _i64([[0, 5], [1, 2]]), _f64([[0.0, 1.0], [2.0, 3.0]])
    )
    assert backend.to_numpy(slots).tolist() == [0, 1]
    assert backend.to_numpy(new.cache_lookup(cache, _i64([9, 5, 7]))).tolist() == [
        1,
        0,
        -1,
    ]
    assert new.cache_size(cache) == 2


def test_cache_keys_stay_sorted_across_interleaved_inserts():
    rng = np.random.default_rng(7)
    cache = new.empty_cache()
    seen = []
    for _ in range(12):
        batch = rng.integers(0, 200, 9)
        todo = new.cache_missing(cache, _i64(batch))
        n = int(backend.to_numpy(todo).size)
        if n == 0:
            continue
        cache, _ = new.cache_insert(
            cache, todo, _i64(np.zeros((n, 2))), _f64(np.zeros((n, 2)))
        )
        seen.extend(backend.to_numpy(todo).tolist())
        keys = backend.to_numpy(cache.keys)
        assert (np.diff(keys) > 0).all(), "cache keys must stay strictly ascending"
    assert sorted(set(seen)) == backend.to_numpy(cache.keys).tolist()


def test_cache_slots_are_assigned_in_insertion_order():
    cache = new.empty_cache()
    cache, slots_a = new.cache_insert(
        cache, _i64([10, 20]), _i64(np.zeros((2, 2))), _f64(np.zeros((2, 2)))
    )
    cache, slots_b = new.cache_insert(
        cache, _i64([5, 15]), _i64(np.zeros((2, 2))), _f64(np.zeros((2, 2)))
    )
    assert backend.to_numpy(slots_a).tolist() == [0, 1]
    assert backend.to_numpy(slots_b).tolist() == [2, 3]
    # Slot order is insertion order; key order is sorted. They differ.
    assert backend.to_numpy(cache.keys).tolist() == [5, 10, 15, 20]
    assert backend.to_numpy(cache.slots).tolist() == [2, 0, 3, 1]


def test_active_membership_and_negative_slots():
    cache = new.empty_cache()
    cache, slots = new.cache_insert(
        cache, _i64([3, 8]), _i64(np.zeros((2, 2))), _f64(np.zeros((2, 2)))
    )
    active = new.empty_active()
    active = new.active_add_slots(active, new.cache_size(cache), _i64([0]))
    assert backend.to_numpy(
        new.active_contains_slots(active, _i64([0, 1, -1]))
    ).tolist() == [True, False, False]
    assert backend.to_numpy(
        new.active_contains(active, cache, _i64([3, 8, 99]))
    ).tolist() == [True, False, False]


def test_active_add_slots_rejects_an_uncached_slot():
    active = new.empty_active()
    with pytest.raises(AssertionError, match="cannot activate an uncached vertex"):
        new.active_add_slots(active, 2, _i64([-1, 0]))


def test_active_add_slots_does_not_mutate_its_input_on_the_no_growth_path():
    """Regression test: torch's `fill_at_indices` mutates its argument in place
    and returns it, so scattering straight into `active` when no growth is
    needed would silently clobber the caller's own array under torch while
    leaving it untouched under jax. `before` aliases the array returned by the
    first call; the second call must not be able to reach through that alias.
    """
    active = new.active_add_slots(new.empty_active(), 2, _i64([0]))
    before = active
    after = new.active_add_slots(active, 2, _i64([1]))  # same n_slots: no growth
    assert backend.to_numpy(before).tolist() == [True, False]
    assert backend.to_numpy(after).tolist() == [True, True]


def test_cache_insert_keeps_ij_and_beta_aligned_with_their_slot_across_many_merges():
    """Coverage restored from the deleted `test_vertex_cache_survives_many_
    small_inserts`: each key's ij/beta row must stay aligned with its own
    slot, not just its key position, across many merge-inserts.

    Descending keys force a merge at the front of ``cache.keys`` on every
    insert, the worst case for the searchsorted-merge in :func:`cache_insert`.
    """
    cache = new.empty_cache()
    probe = []
    for k in range(40):
        key = 1000 - k
        cache, slot = new.cache_insert(
            cache, _i64([key]), _i64([[key, key]]), _f64([[key, key]]) * 0.5
        )
        assert backend.to_numpy(slot).tolist() == [k]
        probe.append(key)

    probe = np.array(probe, dtype=np.int64)
    assert new.cache_size(cache) == 40
    assert backend.to_numpy(new.cache_lookup(cache, _i64(probe))).tolist() == list(
        range(40)
    )
    assert np.array_equal(backend.to_numpy(cache.ij)[:, 0], probe)
    assert np.allclose(
        backend.to_numpy(cache.beta)[:, 0], probe.astype(np.float64) * 0.5
    )


def test_store_add_remove_compact():
    store = new.empty_store()
    store, rows = new.store_add(
        store, _i64([[0, 1, 2], [3, 4, 5]]), 1, _i64([0, 1]), new.LEAF_CONVERGED
    )
    assert backend.to_numpy(rows).tolist() == [0, 1]

    store, rows2 = new.store_add(
        store, _i64([[6, 7, 8]]), 2, _i64([2]), new.LEAF_CONVERGENCE_FAILED
    )
    assert backend.to_numpy(rows2).tolist() == [2]

    store = new.store_remove(store, _i64([0]))
    v, level, cls, status = new.store_compact(store)
    assert backend.to_numpy(v).tolist() == [[3, 4, 5], [6, 7, 8]]
    assert backend.to_numpy(level).tolist() == [1, 2]
    assert backend.to_numpy(status).tolist() == [
        new.LEAF_CONVERGED,
        new.LEAF_CONVERGENCE_FAILED,
    ]


def test_store_add_accepts_per_row_level_and_status():
    """A per-row status is stored as given, combined flags included."""
    both = new.LEAF_APPROX_PARITY_UNRESOLVED | new.LEAF_JACOBIAN_PARITY_UNRESOLVED
    store = new.empty_store()
    store, _ = new.store_add(
        store,
        _i64([[0, 1, 2], [3, 4, 5]]),
        _i64([1, 4]),
        _i64([0, 1]),
        _i64([new.LEAF_CONVERGED, both]),
    )
    _, level, _, status = new.store_compact(store)
    assert backend.to_numpy(level).tolist() == [1, 4]
    assert backend.to_numpy(status).tolist() == [new.LEAF_CONVERGED, both]


def test_store_survives_many_small_adds_and_removals():
    rng = np.random.default_rng(11)
    store = new.empty_store()
    alive = []
    for i in range(40):
        k = int(rng.integers(1, 5))
        v = np.arange(3 * k).reshape(k, 3) + 100 * i
        store, rows = new.store_add(store, _i64(v), i, _i64(np.zeros(k)), 0)
        alive.extend(zip(backend.to_numpy(rows).tolist(), v.tolist()))
        if len(alive) > 3 and i % 3 == 0:
            drop = alive.pop(0)
            store = new.store_remove(store, _i64([drop[0]]))
    v_out, _, _, _ = new.store_compact(store)
    assert backend.to_numpy(v_out).tolist() == [rec[1] for rec in alive]


def test_leaf_status_constants_are_distinct_single_bit_flags():
    """Every failure flag is its own bit, so any OR of them decodes uniquely.

    ``LEAF_CONVERGED`` is zero -- the empty set of failures -- which is what
    lets ``status == LEAF_CONVERGED`` mean "no test failed" however many flags
    a failing leaf carries.
    """
    flags = [
        new.LEAF_CONVERGENCE_FAILED,
        new.LEAF_APPROX_PARITY_UNRESOLVED,
        new.LEAF_JACOBIAN_PARITY_UNRESOLVED,
        new.LEAF_RAYTRACE_NONFINITE,
        new.LEAF_JACOBIAN_NONFINITE,
    ]
    assert new.LEAF_CONVERGED == 0
    assert flags == [1, 2, 4, 8, 16]
    assert all(type(v) is int for v in [new.LEAF_CONVERGED, *flags])
    combos = {
        sum(f for i, f in enumerate(flags) if mask >> i & 1)
        for mask in range(2 ** len(flags))
    }
    assert len(combos) == 2 ** len(flags), "some OR of flags is ambiguous"
    for name in ("LEAF_SIZE_FLOOR", "LEAF_FORCED", "LEAF_INVALID", "LEAF_NONFINITE"):
        assert not hasattr(new, name), f"{name} was retired with the bitmask"


def test_store_remove_does_not_mutate_its_input():
    """Regression test: torch's `fill_at_indices` mutates its argument in
    place and returns it, so scattering straight into `store.valid` would
    silently clobber the caller's own array under torch while leaving it
    untouched under jax. `before` aliases the array the input store holds;
    removing rows from the returned store must not be able to reach through
    that alias.
    """
    store = new.empty_store()
    store, _ = new.store_add(
        store, _i64([[0, 1, 2], [3, 4, 5]]), 0, _i64([0, 0]), new.LEAF_CONVERGED
    )
    before = store.valid
    after = new.store_remove(store, _i64([0]))
    assert backend.to_numpy(before).tolist() == [True, True]
    assert backend.to_numpy(store.valid).tolist() == [True, True]
    assert backend.to_numpy(after.valid).tolist() == [False, True]


def test_initial_triangles_match_the_oracle(oracle_module):
    _, _, _, _, root_class = new.child_matrix_tables()
    ij, cls = new.initial_triangles(3, 2, root_class)
    want_ij, want_cls = oracle_module._initial_triangles(
        3, 2, oracle_module.child_matrix_tables()[4]
    )
    assert backend.to_numpy(ij).tolist() == want_ij.tolist()
    assert backend.to_numpy(cls).tolist() == want_cls.tolist()


def test_midpoint_ij_matches_the_oracle_and_is_exact(oracle_module):
    ij = np.array([[[0, 0], [4, 0], [0, 4]], [[2, 2], [6, 2], [2, 6]]], dtype=np.int64)
    got = backend.to_numpy(new.midpoint_ij(_i64(ij)))
    assert got.tolist() == oracle_module._midpoint_ij(ij).tolist()
    # opposite-vertex convention: m_i bisects the edge opposite theta_i
    assert got[0, 0].tolist() == [2, 2]


def test_red_split_matches_the_oracle(oracle_module):
    _, _, compose, _, _ = new.child_matrix_tables()
    v = _i64([[0, 1, 2], [3, 4, 5]])
    m = _i64([[6, 7, 8], [9, 10, 11]])
    cls = _i64([0, 3])
    child_v, child_cls = new.red_split(v, m, cls, compose)
    want_v, want_cls = oracle_module._red_split(
        np.array([[0, 1, 2], [3, 4, 5]]),
        np.array([[6, 7, 8], [9, 10, 11]]),
        np.array([0, 3]),
        oracle_module.child_matrix_tables()[2],
    )
    assert backend.to_numpy(child_v).tolist() == want_v.tolist()
    assert backend.to_numpy(child_cls).tolist() == want_cls.tolist()


def test_edge_quarter_keys_match_the_oracle(oracle_module):
    lat = new.make_lattice(4.0, 0.0, 0.0, 2, 3)
    old = oracle_module._Lattice(4.0, 0.0, 0.0, 2, 3)
    ij = np.array([[[0, 0], [8, 0], [0, 8]]], dtype=np.int64)
    got = backend.to_numpy(new.edge_quarter_keys(lat, _i64(ij)))
    assert got.tolist() == oracle_module._edge_quarter_keys(old, ij).tolist()


def test_initial_triangles_tile_the_square_and_are_positively_oriented():
    """Ported from `test_adaptive_mesh.py`: geometric properties that the
    oracle-comparison test above does not check -- full square coverage and
    a consistent positive orientation -- rather than element-wise equality
    with the oracle.
    """
    _, _, _, _, root_class = new.child_matrix_tables()
    init_res, max_level = 4, 2
    ij, cls = new.initial_triangles(init_res, max_level, root_class)
    assert ij.shape == (2 * init_res**2, 3, 2)
    P = new.shape_matrix(backend.to(ij, dtype=backend.float64))
    area = P[..., 0, 0] * P[..., 1, 1] - P[..., 0, 1] * P[..., 1, 0]
    assert bool(backend.all(area > 0))
    step = 1 << max_level
    assert np.isclose(float(backend.sum(area)) / 2, (init_res * step) ** 2)
    assert set(backend.to_numpy(cls).tolist()) == set(
        backend.to_numpy(root_class).tolist()
    )


def test_midpoints_are_exact_integers_and_opposite_their_vertex():
    ij = np.array([[[0, 0], [4, 0], [0, 4]]], dtype=np.int64)
    m = new.midpoint_ij(_i64(ij))
    assert backend.to_numpy(m)[0].tolist() == [[2, 2], [0, 2], [2, 0]]


def test_midpoints_are_exact_at_max_level_on_the_widened_lattice():
    """The reason the lattice is one level finer than max_level.

    With the lattice at max_level a triangle's edges are one unit long and
    `midpoint_ij`'s floor division collapses each "midpoint" onto one of that
    edge's own endpoints. One level finer, every edge vector is even at every
    level up to and including max_level, so the midpoints are genuine lattice
    points -- and they are exactly the points with an odd coordinate, which is
    what guarantees they can never collide with a cached vertex.
    """
    _, _, _, _, root_class = new.child_matrix_tables()
    max_level = 3
    lat = new.make_lattice(4.0, 0.0, 0.0, 2, max_level + 1)
    ij, cls = new.initial_triangles(2, lat.level, root_class)
    # Descend to max_level by taking child C_4 (the middle child) each time.
    for _ in range(max_level):
        ij = new.midpoint_ij(ij)
    assert bool(backend.all(ij % 2 == 0)), "max_level vertices are even"

    mid = new.midpoint_ij(ij)
    # Exact: the floor division threw nothing away.
    assert bool(
        backend.all((ij[:, [1, 2, 0]] + ij[:, [2, 0, 1]]) % 2 == 0)
    ), "edge endpoint sums must be even for the midpoint to be exact"
    # Every midpoint has an odd coordinate, so it is not a vertex of any level.
    assert bool(backend.all(backend.any(mid % 2 == 1, dim=-1)))


def test_red_split_is_triangle_major_and_advances_the_class():
    _, _, compose, _, _ = new.child_matrix_tables()
    v = _i64([[0, 1, 2], [3, 4, 5]])
    m = _i64([[6, 7, 8], [9, 10, 11]])
    cls = _i64([0, 4])
    cv, cc = new.red_split(v, m, cls, compose)
    assert cv.shape == (8, 3) and cc.shape == (8,)
    cv_np = backend.to_numpy(cv)
    cc_np = backend.to_numpy(cc)
    compose_np = backend.to_numpy(compose)
    assert cv_np[0].tolist() == [0, 8, 7]  # C_1 = (theta1, m3, m2)
    assert cv_np[3].tolist() == [6, 7, 8]  # C_4 = (m1, m2, m3)
    assert cc_np[:4].tolist() == compose_np[0].tolist()
    assert cc_np[4:].tolist() == compose_np[4].tolist()


def test_edge_quarter_keys_are_lattice_points_of_both_quarters():
    lat = new.make_lattice(4.0, 0.0, 0.0, 1, 4)  # n = 16
    ij = np.array([[[0, 0], [16, 0], [0, 16]]], dtype=np.int64)
    keys = new.edge_quarter_keys(lat, _i64(ij))
    ij_from_key = backend.to_numpy(new.lattice_ij_from_key(lat, keys[0]))
    got = {tuple(p) for p in ij_from_key}
    assert (4, 0) in got and (12, 0) in got  # edge (0,1)
    assert (0, 4) in got and (0, 12) in got  # edge (2,0)


def test_make_raytrace_forces_float64_and_records_the_callback_dtype():
    seen = {}

    def rt(x, y):
        seen["dtype"] = x.dtype
        return 2.0 * x, 3.0 * y

    fn = make = new.make_raytrace(rt, None)
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
    out = new.make_raytrace(rt, input_device)(xy)
    assert seen["device"] == input_device
    assert backend.device(out) == input_device
    assert calls[-1]["device"] == input_device


def test_make_raytrace_records_float32_callback_dtype_but_returns_float64():
    def rt(x, y):
        return backend.to(x, dtype=backend.float32), backend.to(
            y, dtype=backend.float32
        )

    fn = new.make_raytrace(rt, None)
    out = fn(_f64([[1.0, 2.0]]))
    assert fn.info["dtype"] == backend.float32
    assert out.dtype == backend.float64


def test_make_raytrace_rejects_a_non_tuple_return():
    fn = new.make_raytrace(lambda x, y: x, None)
    with pytest.raises(ValueError, match="2-tuple"):
        fn(_f64([[1.0, 2.0]]))


def test_make_raytrace_rejects_a_shape_changing_callback():
    fn = new.make_raytrace(lambda x, y: (x[:1], y[:1]), None)
    with pytest.raises(ValueError, match="shape-preserving"):
        fn(_f64([[1.0, 2.0], [3.0, 4.0]]))


def test_trace_keys_is_bit_identical_under_chunking():
    lat = new.make_lattice(4.0, 0.0, 0.0, 4, 3)
    fn = new.make_raytrace(lambda x, y: (x * x - y, y * y + x), None)
    ij = _i64(
        np.stack(np.meshgrid(np.arange(9), np.arange(9), indexing="ij"), -1).reshape(
            -1, 2
        )
    )
    whole = backend.to_numpy(new.trace_keys(lat, ij, fn, None))
    for size in (1, 7, 33):
        assert (backend.to_numpy(new.trace_keys(lat, ij, fn, size)) == whole).all()


def test_evaluate_only_traces_uncached_points():
    calls = {"n": 0}

    def rt(x, y):
        calls["n"] += int(x.shape[0])
        return x, y

    lat = new.make_lattice(4.0, 0.0, 0.0, 2, 2)
    fn = new.make_raytrace(rt, None)
    cache = new.empty_cache()
    keys = _i64([3, 7, 3, 11])
    cache = new.evaluate(cache, lat, keys, fn, None)
    assert calls["n"] == 3
    cache = new.evaluate(cache, lat, keys, fn, None)
    assert calls["n"] == 3, "already-cached points must not be re-traced"
