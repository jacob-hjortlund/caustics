import numpy as np
import pytest

from caustics.backend_obj import backend


def test_sort_is_ascending_and_stable():
    x = backend.as_array(np.array([3, 1, 2, 1], dtype=np.int64))
    assert backend.to_numpy(backend.sort(x)).tolist() == [1, 1, 2, 3]


def test_argsort_is_stable():
    x = backend.as_array(np.array([3, 1, 2, 1], dtype=np.int64))
    assert backend.to_numpy(backend.argsort(x)).tolist() == [1, 3, 2, 0]


def test_unique_returns_sorted_values_and_inverse():
    x = backend.as_array(np.array([3, 1, 2, 1], dtype=np.int64))
    values, inverse = backend.unique(x, return_inverse=True)
    assert backend.to_numpy(values).tolist() == [1, 2, 3]
    assert backend.to_numpy(inverse).reshape(-1).tolist() == [2, 0, 1, 0]


def test_unique_without_inverse_returns_values_only():
    x = backend.as_array(np.array([3, 1, 2, 1], dtype=np.int64))
    assert backend.to_numpy(backend.unique(x)).tolist() == [1, 2, 3]


def test_bincount_honours_the_minimum_length():
    x = backend.as_array(np.array([3, 1, 2, 1], dtype=np.int64))
    assert backend.to_numpy(backend.bincount(x, minlength=6)).tolist() == [
        0,
        2,
        1,
        1,
        0,
        0,
    ]


def test_bincount_of_an_empty_array_is_all_zeros():
    x = backend.as_array(np.zeros(0, dtype=np.int64), dtype=backend.int64)
    assert backend.to_numpy(backend.bincount(x, minlength=3)).tolist() == [0, 0, 0]


def test_flatnonzero_returns_ascending_indices():
    x = backend.as_array(np.array([3, 1, 2, 1], dtype=np.int64))
    assert backend.to_numpy(backend.flatnonzero(x > 1)).tolist() == [0, 2]


def test_sign_matches_numpy():
    x = backend.as_array(np.array([-2.0, 0.0, 3.0]))
    assert backend.to_numpy(backend.sign(x)).tolist() == [-1.0, 0.0, 1.0]


def test_lexsort_matches_numpy_with_the_last_key_primary():
    rng = np.random.default_rng(0)
    a, b, c = (rng.integers(0, 3, 40) for _ in range(3))
    keys = [backend.as_array(k, dtype=backend.int64) for k in (a, b, c)]
    got = backend.to_numpy(backend.lexsort(keys)).tolist()
    assert got == np.lexsort((a, b, c)).tolist()


def test_finfo_exposes_eps_for_both_float_widths():
    assert backend.finfo(backend.float64).eps == pytest.approx(2.220446049250313e-16)
    assert backend.finfo(backend.float32).eps > backend.finfo(backend.float64).eps


def test_integer_dtype_properties_round_trip():
    x = backend.zeros((3,), dtype=backend.int64)
    assert backend.to_numpy(x).dtype == np.int64
    y = backend.zeros((3,), dtype=backend.int8)
    assert backend.to_numpy(y).dtype == np.int8


def test_all_and_any_reduce_over_a_single_axis():
    x = backend.as_array(np.array([[True, True], [True, False]]))
    assert backend.to_numpy(backend.all(x, dim=1)).tolist() == [True, False]
    assert backend.to_numpy(backend.any(x, dim=1)).tolist() == [True, True]


def test_all_and_any_reduce_over_a_tuple_of_axes():
    x = backend.as_array(np.ones((2, 3, 4), dtype=bool))
    assert backend.to_numpy(backend.all(x, dim=(1, 2))).tolist() == [True, True]
    assert backend.to_numpy(backend.any(x, dim=(1, 2))).tolist() == [True, True]


def test_all_and_any_without_dim_still_reduce_everything():
    x = backend.as_array(np.array([[True, False]]))
    assert bool(backend.to_numpy(backend.all(x))) is False
    assert bool(backend.to_numpy(backend.any(x))) is True
