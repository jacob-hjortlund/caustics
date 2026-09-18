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
