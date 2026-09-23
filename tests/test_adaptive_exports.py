import importlib

import pytest

import caustics
import caustics.lenses as lenses
from caustics.lenses import func


@pytest.mark.parametrize("package", [caustics, lenses, func])
def test_new_symbols_are_re_exported_at_every_package_level(package):
    for name in (
        "build_adaptive_mesh",
        "mesh_query",
        "mesh_seeds",
        "mesh_forward_raytrace",
        "AdaptiveMesh",
        "MeshIndex",
        "CriticalBand",
        "CriticalCurves",
        "mesh_critical_curves",
    ):
        assert getattr(package, name) is getattr(func, name), name
        assert name in package.__all__, name


@pytest.mark.parametrize("package", [caustics, lenses, func])
def test_dropped_symbols_are_gone(package):
    for name in ("Mesh", "LeafStatus", "BuildStats"):
        assert not hasattr(package, name), name
        assert name not in package.__all__, name


def test_the_old_module_no_longer_exists():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("caustics.lenses.adaptive")


def test_leaf_status_constants_are_exported():
    for name in (
        "LEAF_CONVERGED",
        "LEAF_CONVERGENCE_FAILED",
        "LEAF_APPROX_PARITY_UNRESOLVED",
        "LEAF_JACOBIAN_PARITY_UNRESOLVED",
        "LEAF_RAYTRACE_NONFINITE",
        "LEAF_JACOBIAN_NONFINITE",
    ):
        assert getattr(func, name) is getattr(func.adaptive, name), name
        assert name in func.__all__, name
    for name in ("LEAF_SIZE_FLOOR", "LEAF_FORCED", "LEAF_INVALID", "LEAF_NONFINITE"):
        assert not hasattr(func, name), name
        assert name not in func.__all__, name
