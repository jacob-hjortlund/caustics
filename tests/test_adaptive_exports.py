import ast
import importlib
import pathlib

import pytest

import caustics
import caustics.lenses as lenses
from caustics.lenses import func


@pytest.mark.parametrize("package", [caustics, lenses, func])
def test_new_symbols_are_re_exported_at_every_package_level(package):
    for name in (
        "build_adaptive_mesh",
        "extend_adaptive_mesh",
        "build_closed_adaptive_mesh",
        "mesh_query",
        "mesh_seeds",
        "mesh_forward_raytrace",
        "AdaptiveMesh",
        "MeshIndex",
        "CriticalBand",
        "CentreHoles",
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


PACKAGE_DIR = pathlib.Path(func.adaptive.__file__).parent

# Each module may import only from the modules before it.
LAYERS = (
    "geometry",
    "state",
    "lattice",
    "sampling",
    "criterion",
    "band",
    "holes",
    "refinement",
    "closure",
    "mesh",
    "query",
    "images",
    "curves",
    "build",
)

CURATED = (
    "build_adaptive_mesh",
    "extend_adaptive_mesh",
    "build_closed_adaptive_mesh",
    "mesh_query",
    "mesh_seeds",
    "mesh_forward_raytrace",
    "mesh_critical_curves",
    "AdaptiveMesh",
    "MeshIndex",
    "CriticalBand",
    "CentreHoles",
    "CriticalCurves",
    "LEAF_CONVERGED",
    "LEAF_CONVERGENCE_FAILED",
    "LEAF_APPROX_PARITY_UNRESOLVED",
    "LEAF_JACOBIAN_PARITY_UNRESOLVED",
    "LEAF_RAYTRACE_NONFINITE",
    "LEAF_JACOBIAN_NONFINITE",
    "affine_from_triangles",
    "child_matrix_tables",
    "contains",
    "converged_from_deviation",
    "evaluate_criterion",
    "midpoint_deviation",
    "sanitize_bary",
    "shape_matrix",
    "sigma_min_2x2",
    "triangle_weights",
)


def _modules():
    return sorted(PACKAGE_DIR.glob("*.py"))


def test_the_package_holds_exactly_the_layered_modules():
    assert {p.stem for p in _modules()} == {"__init__", *LAYERS}


def test_modules_import_only_from_lower_layers():
    for i, name in enumerate(LAYERS):
        tree = ast.parse((PACKAGE_DIR / f"{name}.py").read_text())
        for node in tree.body:
            if isinstance(node, ast.ImportFrom) and node.level == 1:
                assert node.module in LAYERS[:i], f"{name} imports .{node.module}"


def test_no_deferred_imports():
    for path in _modules():
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.If) and "TYPE_CHECKING" in ast.unparse(node.test):
                raise AssertionError(f"{path.name}:{node.lineno}: TYPE_CHECKING block")
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for inner in ast.walk(node):
                    assert not isinstance(
                        inner, (ast.Import, ast.ImportFrom)
                    ), f"{path.name}:{inner.lineno}: import inside {node.name}"


def test_no_numpy_import_in_the_package():
    for path in _modules():
        assert "import numpy" not in path.read_text(), path.name


def test_adaptive_namespace_is_curated():
    assert sorted(func.adaptive.__all__) == sorted(CURATED)
    for name in CURATED:
        assert hasattr(func.adaptive, name), name
    for internal in ("refine", "freeze", "trace_band", "make_raytrace", "Lattice"):
        assert not hasattr(func.adaptive, internal), internal
    assert func.adaptive.refinement.refine.__module__ == (
        "caustics.lenses.func.adaptive.refinement"
    )


def test_the_old_critical_module_is_gone():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("caustics.lenses.func.adaptive_critical")
