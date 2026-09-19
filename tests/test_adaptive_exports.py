import importlib

import pytest

import caustics
from caustics.lenses import func


def test_new_symbols_are_re_exported():
    for name in (
        "build_adaptive_mesh",
        "mesh_query",
        "mesh_seeds",
        "mesh_forward_raytrace",
        "AdaptiveMesh",
        "MeshIndex",
    ):
        assert hasattr(func, name), name
        assert name in func.__all__, name


def test_build_adaptive_mesh_is_reachable_from_the_package_root():
    assert hasattr(caustics, "build_adaptive_mesh")
    assert "build_adaptive_mesh" in caustics.__all__


def test_dropped_symbols_are_gone():
    for name in ("Mesh", "LeafStatus", "BuildStats"):
        assert not hasattr(caustics, name), name
        assert name not in caustics.__all__, name


def test_the_old_module_no_longer_exists():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("caustics.lenses.adaptive")


def test_leaf_status_constants_are_exported():
    for name in (
        "LEAF_CONVERGED",
        "LEAF_SIZE_FLOOR",
        "LEAF_FORCED",
        "LEAF_INVALID",
        "LEAF_NONFINITE",
    ):
        assert hasattr(func, name), name
