from caustics.lenses.adaptive import (
    LeafStatus,
    BuildStats,
    Mesh,
    build_adaptive_mesh,
)


def test_public_symbols_are_re_exported():
    import caustics

    assert caustics.build_adaptive_mesh is build_adaptive_mesh
    assert caustics.Mesh is Mesh
    assert caustics.LeafStatus is LeafStatus
    assert caustics.BuildStats is BuildStats
    assert caustics.func.sigma_min_2x2 is not None
    assert caustics.func.triangle_weights is not None
