"""Building an adaptive mesh whose fov cuts none of its critical curves.

``build_closed_adaptive_mesh`` builds, then extends by ``growth`` until no
open curve ends on the fov boundary, after first widening the fov over every
centre's hole. The fixtures use dyadic fovs, centres and cell sizes, so every
expected fov and ``init_res`` below is exact and derived by hand from the
rounding rule of ``extend_adaptive_mesh``: ``k`` whole level-0 cells a side.
"""

import warnings
from types import SimpleNamespace

import numpy as np
import pytest

from caustics.backend_obj import backend
from caustics.lenses.func.adaptive import (
    AdaptiveMesh,
    CriticalCurves,
    build_adaptive_mesh,
    build_closed_adaptive_mesh,
    extend_adaptive_mesh,
    mesh_critical_curves,
)
from caustics.lenses.func.adaptive.lattice import lattice_xy


def to_np(x):
    return backend.to_numpy(x)


# ---------------------------------------------------------------------------
# Lenses
# ---------------------------------------------------------------------------


def recording_lens(fn, jac):
    """A lens over numpy maps that records every batch ``raytrace`` is called on."""
    calls = {"raytrace": [], "jacobian": []}

    def raytrace(x, y):
        xy = np.stack([to_np(x), to_np(y)], axis=-1)
        calls["raytrace"].append(xy)
        out = fn(xy)
        return backend.as_array(out[:, 0]), backend.as_array(out[:, 1])

    def jacobian_lens_equation(x, y):
        xy = np.stack([to_np(x), to_np(y)], axis=-1)
        calls["jacobian"].append(xy)
        return backend.as_array(jac(xy), dtype=backend.float64)

    lens = SimpleNamespace(
        raytrace=raytrace, jacobian_lens_equation=jacobian_lens_equation
    )
    return lens, calls


def sie_like(p):
    """A cored isothermal sphere: tangential curve at radius ~1.18, radial at ~0.32."""
    r = np.sqrt(p[:, 0] ** 2 + p[:, 1] ** 2 + 0.05)
    return p - 1.2 * p / r[:, None]


def sie_like_jacobian(p):
    x, y = p[:, 0], p[:, 1]
    r = np.sqrt(x * x + y * y + 0.05)
    k = 1.2 / r**3
    J = np.empty((p.shape[0], 2, 2))
    J[:, 0, 0] = 1.0 - 1.2 / r + k * x * x
    J[:, 0, 1] = k * x * y
    J[:, 1, 0] = k * x * y
    J[:, 1, 1] = 1.0 - 1.2 / r + k * y * y
    return J


AFFINE = np.array([[0.7, 0.1], [-0.2, 0.9]])


def affine(p):
    return p @ AFFINE.T


def affine_jacobian(p):
    return np.tile(AFFINE, (p.shape[0], 1, 1))


def broken_where(fn, jac, where, value=np.nan):
    """``fn`` and ``jac`` with ``value`` wherever ``where(p)`` holds."""

    def broken(p):
        out = fn(p)
        out[where(p)] = value
        return out

    def broken_jacobian(p):
        J = jac(p)
        J[where(p)] = value
        return J

    return broken, broken_jacobian


def sis(c, b):
    """A singular isothermal sphere of Einstein radius ``b`` at ``c``, and its Jacobian."""
    c = np.asarray(c, dtype=np.float64)

    def fn(p):
        d = p - c
        return p - b * d / np.hypot(d[:, 0], d[:, 1])[:, None]

    def jac(p):
        d = p - c
        r = np.hypot(d[:, 0], d[:, 1])
        k = b / r**3
        J = np.empty((p.shape[0], 2, 2))
        J[:, 0, 0] = 1.0 - b / r + k * d[:, 0] ** 2
        J[:, 0, 1] = J[:, 1, 0] = k * d[:, 0] * d[:, 1]
        J[:, 1, 1] = 1.0 - b / r + k * d[:, 1] ** 2
        return J

    return fn, jac


# An SIS whose tangential curve, radius 0.6 about C_IN, lies inside
# [-1, 1]**2; its centre is off every lattice point.
C_IN = (0.3001, -0.2003)


# ---------------------------------------------------------------------------
# Comparisons
# ---------------------------------------------------------------------------


def _assert_same(a, b, name):
    if hasattr(a, "shape"):
        a, b = to_np(a), to_np(b)
        assert (a.dtype, a.shape) == (b.dtype, b.shape), name
        assert np.array_equal(a, b, equal_nan=a.dtype.kind == "f"), name
    else:
        assert a == b, name


def assert_meshes_equal(got, want):
    """Every field equal, arrays bit for bit; the lattice as the map it defines."""
    for name in AdaptiveMesh._fields:
        a, b = getattr(got, name), getattr(want, name)
        if name == "lattice":
            assert (a.level, a.n, a.stride, a.scale) == (
                b.level,
                b.n,
                b.stride,
                b.scale,
            )
            corners = backend.to(
                backend.as_array([[0, 0], [a.n, a.n]], dtype=backend.int64),
                device=backend.device(a.lo),
            )
            _assert_same(lattice_xy(a, corners), lattice_xy(b, corners), name)
        elif name in ("index", "critical_band", "holes"):
            for field in type(a)._fields:
                _assert_same(getattr(a, field), getattr(b, field), f"{name}.{field}")
        else:
            _assert_same(a, b, name)


def assert_curves_equal(got, want):
    for field in CriticalCurves._fields:
        a, b = getattr(got, field), getattr(want, field)
        if field == "holes":
            for sub in type(a)._fields:
                _assert_same(getattr(a, sub), getattr(b, sub), f"holes.{sub}")
        else:
            _assert_same(a, b, field)


def closed_build(fn, jac, fov, init_res, min_img_sep, **kw):
    """``build_closed_adaptive_mesh`` on a recording lens, with every warning kept."""
    lens, calls = recording_lens(fn, jac)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        mesh, curves = build_closed_adaptive_mesh(
            lens, fov, init_res, min_img_sep, **kw
        )
    messages = [str(w.message) for w in record]
    return mesh, curves, messages, lens, calls


def plain_build(fn, jac, fov, init_res, min_img_sep, **kw):
    lens, _ = recording_lens(fn, jac)
    return build_adaptive_mesh(lens, fov, init_res, min_img_sep, **kw), lens


# ---------------------------------------------------------------------------
# Growing until the fov cuts no curve
# ---------------------------------------------------------------------------


def test_a_curve_the_fov_cuts_is_grown_until_it_closes():
    """The tangential curve, radius ~1.18, crosses every side of the fov-2
    domain. With ``h0 = 0.125`` and ``growth = 1.05`` each step asks for
    less than a cell a side, so each adds one: fov 2 -> 2.25, still cut
    (half-width 1.125), -> 2.5, closed (half-width 1.25)."""
    kw = dict(fov=2.0, init_res=16, min_img_sep=0.05)
    mesh, curves, messages, _, _ = closed_build(
        sie_like, sie_like_jacobian, growth=1.05, **kw
    )
    assert (mesh.fov, mesh.init_res) == (2.5, 20)
    assert curves.closed.shape[0] > 0 and to_np(curves.closed).all()
    assert messages == []
    start, lens = plain_build(sie_like, sie_like_jacobian, **kw)
    assert_meshes_equal(mesh, extend_adaptive_mesh(start, lens, 2.5))
    assert_curves_equal(curves, mesh_critical_curves(mesh))


@pytest.mark.parametrize("x0, y0", [(0.5, 0.0), (-0.5, 0.0), (0.0, 0.5), (0.0, -0.5)])
def test_a_curve_cut_by_any_one_side_of_the_fov_grows_it(x0, y0):
    """Off-centre by 0.5, the fov-3 domain's near side is 1.0 from the lens
    centre, inside the tangential curve, and the other three are 1.5 or more
    from it. The default ``growth`` asks for fov 4.5, two whole cells of
    ``h0 = 0.5`` a side: fov 5, init_res 10."""
    mesh, curves, messages, _, _ = closed_build(
        sie_like, sie_like_jacobian, 3.0, 6, 0.05, x0=x0, y0=y0
    )
    assert (mesh.fov, mesh.init_res) == (5.0, 10)
    assert to_np(curves.closed).all()
    assert messages == []


def test_a_mesh_whose_fov_cuts_no_curve_is_the_plain_build():
    kw = dict(fov=3.0, init_res=6, min_img_sep=0.05)
    mesh, curves, messages, _, _ = closed_build(sie_like, sie_like_jacobian, **kw)
    assert messages == []
    want, _ = plain_build(sie_like, sie_like_jacobian, **kw)
    assert_meshes_equal(mesh, want)
    assert_curves_equal(curves, mesh_critical_curves(want))


def test_a_curve_open_inside_the_fov_does_not_grow_it():
    """A non-finite strip across the tangential curve, well inside the fov,
    leaves it open at the strip's edges. Growing cannot close it, so the fov
    stays, and nothing warns: it is not the fov that cuts it."""
    fn, jac = broken_where(
        sie_like,
        sie_like_jacobian,
        lambda p: (p[:, 0] > 0.9) & (p[:, 0] < 1.3) & (np.abs(p[:, 1]) < 0.2),
    )
    mesh, curves, messages, _, _ = closed_build(fn, jac, 3.0, 6, 0.05)
    assert not to_np(curves.closed).all()
    assert (mesh.fov, mesh.init_res) == (3.0, 6)
    assert messages == []


@pytest.mark.parametrize("max_iters, fov, init_res", [(0, 2.0, 16), (1, 2.25, 18)])
def test_running_out_of_iterations_warns_and_returns_the_last_mesh(
    max_iters, fov, init_res
):
    """The case of the first test, stopped short of fov 2.5."""
    mesh, curves, messages, _, _ = closed_build(
        sie_like,
        sie_like_jacobian,
        2.0,
        16,
        0.05,
        growth=1.05,
        max_iters=max_iters,
    )
    assert (mesh.fov, mesh.init_res) == (fov, init_res)
    assert not to_np(curves.closed).all()
    assert len(messages) == 1 and "still cuts" in messages[0]
    assert_curves_equal(curves, mesh_critical_curves(mesh))


def test_build_options_reach_the_build_and_every_extension():
    """The mesh dtype, the batch size and the index resolution must reach the
    build and the extension alike: the result equals the build extended by
    hand with the same options, and no raytrace batch passes 5."""
    kw = dict(
        fov=2.0,
        init_res=4,
        min_img_sep=0.05,
        dtype=backend.float32,
        raytrace_batch_size=5,
        index_cells=16,
    )
    mesh, _, messages, _, calls = closed_build(sie_like, sie_like_jacobian, **kw)
    assert messages == []
    assert max(len(xy) for xy in calls["raytrace"]) <= 5
    start, lens = plain_build(sie_like, sie_like_jacobian, **kw)
    want = extend_adaptive_mesh(start, lens, 3.0, raytrace_batch_size=5, index_cells=16)
    assert mesh.dtype == backend.float32
    assert_meshes_equal(mesh, want)


# ---------------------------------------------------------------------------
# Widening the initial fov over the centres' holes
# ---------------------------------------------------------------------------


def test_a_centre_outside_the_fov_widens_it_before_the_build_and_warns():
    """The hole at x = 1.8001, radius 0.025, needs a half-width above 1.8251:
    two cells of ``h0 = 0.5`` a side, fov 4 and init_res 8. That is done
    before the build, not by the loop: with ``max_iters = 0`` it still
    happens, and the result is the plain build at fov 4."""
    c_out = (1.8001, 0.2003)
    fn, jac = sis(C_IN, 0.6)
    mesh, curves, messages, _, _ = closed_build(
        fn, jac, 2.0, 4, 0.05, centres=[C_IN, c_out], max_iters=0
    )
    assert (mesh.fov, mesh.init_res) == (4.0, 8)
    assert len(messages) == 1
    assert "fov=4" in messages[0] and "init_res=8" in messages[0]
    assert to_np(curves.closed).all()
    want, _ = plain_build(fn, jac, 4.0, 8, 0.05, centres=[C_IN, c_out])
    assert_meshes_equal(mesh, want)


@pytest.mark.parametrize(
    "edge",
    [
        pytest.param([(0.9901, 0.2003)], id="centre-inside-hole-across"),
        # Closer than twice the halved 0.05, so one hole: mean x 0.9601,
        # radius 0.025 + 0.0225, reaching 1.0076. Unmerged, each would reach
        # only 0.9851.
        pytest.param([(0.9601, 0.2003), (0.9601, 0.2453)], id="merged-pair"),
    ],
)
def test_a_hole_across_the_fov_edge_widens_it(edge):
    """A hole reaching past the half-width 1 widens the fov by one cell of
    ``h0 = 0.5`` a side, to fov 3 and init_res 6, though its centre is inside."""
    fn, jac = sis(C_IN, 0.6)
    mesh, _, messages, _, _ = closed_build(fn, jac, 2.0, 4, 0.05, centres=[C_IN, *edge])
    assert (mesh.fov, mesh.init_res) == (3.0, 6)
    assert len(messages) == 1 and "init_res=6" in messages[0]


def test_holes_inside_the_fov_leave_it_alone():
    """The hole at x = 0.9601 reaches 0.9851 with the build's halved radius
    0.025: inside. With the unhalved 0.05 it would reach past 1."""
    fn, jac = sis(C_IN, 0.6)
    centres = [C_IN, (0.9601, 0.2003)]
    mesh, curves, messages, _, _ = closed_build(fn, jac, 2.0, 4, 0.05, centres=centres)
    assert messages == []
    assert to_np(curves.closed).all()
    want, _ = plain_build(fn, jac, 2.0, 4, 0.05, centres=centres)
    assert_meshes_equal(mesh, want)


# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kw, match",
    [
        (dict(growth=1.0), "growth"),
        (dict(growth=0.5), "growth"),
        (dict(growth=np.nan), "growth"),
        (dict(growth=np.inf), "growth"),
        (dict(max_iters=-1), "max_iters"),
        (dict(max_iters=1.5), "max_iters"),
        (dict(fov=-2.0), "fov"),
    ],
)
def test_bad_arguments_raise_before_any_lens_call(kw, match):
    args = dict(fov=2.0, init_res=4, min_img_sep=0.05, centres=[(3.0, 0.0)])
    args.update(kw)
    lens, calls = recording_lens(affine, affine_jacobian)
    with pytest.raises(ValueError, match=match):
        build_closed_adaptive_mesh(lens, **args)
    assert calls == {"raytrace": [], "jacobian": []}
