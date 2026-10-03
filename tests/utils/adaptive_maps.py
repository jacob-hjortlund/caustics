"""Analytic lens maps and helpers shared by the adaptive-mesh tests."""

from types import SimpleNamespace

import numpy as np

from caustics.backend_obj import backend


def to_np(x):
    return backend.to_numpy(x)


def f64(x):
    return backend.as_array(np.asarray(x, dtype=np.float64), dtype=backend.float64)


def i64(x):
    return backend.as_array(np.asarray(x, dtype=np.int64), dtype=backend.int64)


def stack_2x2(a, b, c, d):
    return backend.stack(
        (backend.stack((a, b), dim=-1), backend.stack((c, d), dim=-1)), dim=-2
    )


def lens(raytrace, jacobian):
    """A stand-in for a caustics lens."""
    return SimpleNamespace(raytrace=raytrace, jacobian_lens_equation=jacobian)
