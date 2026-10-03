import numpy as np
import pytest

from caustics.backend_obj import backend
from caustics.lenses.func.adaptive import (
    build_adaptive_mesh,
)
from caustics.lenses.func.adaptive.sampling import make_raytrace


def _f64(x):
    return backend.as_array(np.asarray(x, dtype=np.float64), dtype=backend.float64)


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
