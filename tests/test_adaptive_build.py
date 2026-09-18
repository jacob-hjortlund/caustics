import numpy as np
import pytest

from caustics.backend_obj import backend
from caustics.lenses import old_adaptive as oracle
from caustics.lenses.func import adaptive as new


def test_lattice_key_round_trips_and_matches_the_oracle():
    lat = new.make_lattice(4.0, 0.0, 0.0, 2, 3)
    old = oracle._Lattice(4.0, 0.0, 0.0, 2, 3)
    ij = np.array([[0, 0], [1, 3], [16, 16], [5, 11]], dtype=np.int64)
    ij_b = backend.as_array(ij, dtype=backend.int64)

    key = new.lattice_key(lat, ij_b)
    assert backend.to_numpy(key).tolist() == old.key(ij).tolist()
    assert backend.to_numpy(new.lattice_ij_from_key(lat, key)).tolist() == ij.tolist()
    assert np.allclose(backend.to_numpy(new.lattice_xy(lat, ij_b)), old.xy(ij))


def test_lattice_on_boundary_matches_the_oracle():
    lat = new.make_lattice(4.0, 0.0, 0.0, 2, 2)
    old = oracle._Lattice(4.0, 0.0, 0.0, 2, 2)
    ij = np.array([[0, 4], [8, 1], [3, 3], [8, 8]], dtype=np.int64)
    got = backend.to_numpy(
        new.lattice_on_boundary(lat, backend.as_array(ij, dtype=backend.int64))
    )
    assert got.tolist() == old.on_boundary(ij).tolist()


@pytest.mark.parametrize(
    "fov,init_res,min_img_sep", [(5.0, 4, 0.1), (1.0, 1, 2.0), (10.0, 8, 0.001)]
)
def test_depth_floor_matches_the_oracle(fov, init_res, min_img_sep):
    assert new.depth_floor(fov, init_res, min_img_sep) == oracle._depth_floor(
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
