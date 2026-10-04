"""The backend the adaptive bookkeeping runs on, and the conversions at its boundary."""

from typing import NamedTuple

import caskade as ck
import numpy as np
import pytest

from caustics.backend_obj import backend
from caustics.lenses.func.adaptive.mesh_backend import (
    TorchMeshBackend,
    map_arrays,
    mesh_backend,
    to_mesh,
    to_user,
)


class Pair(NamedTuple):
    a: object
    b: object


def test_the_torch_mesh_backend_computes_in_torch_and_leaves_the_global_backend():
    import torch

    before = ck.backend.backend
    torch_backend = TorchMeshBackend()
    x = torch_backend.as_array(np.array([3, 1, 2]), dtype=torch_backend.int64)
    assert isinstance(x, torch.Tensor)
    assert torch_backend.to_numpy(torch_backend.sort(x)).tolist() == [1, 2, 3]
    assert ck.backend.backend == before and backend.backend == before


def test_map_arrays_reaches_arrays_in_nested_tuples_and_keeps_the_rest():
    value = Pair(a=(np.ones(2), 3.0), b=Pair(a=None, b=np.zeros(1)))
    got = map_arrays(value, lambda a: a + 1)
    assert isinstance(got, Pair) and isinstance(got.b, Pair)
    assert type(got.a) is tuple and got.a[1] == 3.0 and got.b.a is None
    np.testing.assert_array_equal(got.a[0], [2.0, 2.0])
    np.testing.assert_array_equal(got.b.b, [1.0])


def test_map_arrays_returns_the_value_itself_when_nothing_changes():
    value = Pair(a=np.ones(2), b=(1, None))
    assert map_arrays(value, lambda a: a) is value


@pytest.mark.skipif(backend.backend != "torch", reason="the torch path only")
def test_under_torch_the_bookkeeping_is_the_users_backend_and_nothing_converts():
    value = Pair(a=backend.as_array([1.0, 2.0]), b=2.5)
    assert mesh_backend is backend
    assert to_mesh(value) is value
    assert to_user(value) is value
