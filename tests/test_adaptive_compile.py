"""``compile_bookkeeping``: ``lens_status`` and the index's reached-cells step under ``torch.compile``, and the fallback to eager."""

import gc
import inspect
import json
import subprocess
import sys
import warnings
import weakref

import numpy as np
import pytest
import torch

from caustics.backend_obj import backend
from caustics.cosmology import FlatLambdaCDM
from caustics.lenses import SIE
from caustics.lenses.func.adaptive import criterion as criterion_module
from caustics.lenses.func.adaptive import index as index_module
from caustics.lenses.func.adaptive import lens_mesh as lens_mesh_module
from caustics.lenses.func.adaptive import refine as refine_module
from caustics.lenses.func.adaptive.criterion import LEAF_RAYTRACE_NONFINITE, lens_status
from caustics.lenses.func.adaptive.index import build_index, index_hits
from caustics.lenses.func.adaptive.lens_mesh import (
    build_closed_lens_mesh,
    build_lens_mesh,
    extend_lens_mesh,
    leaf_grow,
)
from caustics.lenses.func.adaptive.mesh_backend import Compiled, mesh_backend, to_mesh
from caustics.lenses.func.adaptive.region import build_closed_magnification_region

from adaptive_maps import (
    assert_hits_equal,
    assert_same,
    broken_where,
    brute_hits,
    build,
    finite_rows,
    i64,
    localised_fold,
    localised_fold_jacobian,
    numpy_lens,
    point_mass,
    sis,
    to_np,
)

# The fields a compiled build need only agree on to rounding; the index's
# float fields bound its boxes, grown by ``origin_deviation``.
ROUNDING = (
    "origin_deviation",
    "origin_sigma_min",
    "vertices_jacobian",
    "det",
    "index.lo",
    "index.hi",
    "index.fine",
)


@pytest.fixture
def fresh_compiling(monkeypatch):
    """Compiling as at the start of a process: nothing failed, no copy built, no compiled run."""
    monkeypatch.setattr(Compiled, "failed", None)
    for c in (criterion_module._COMPILED_LENS_STATUS, index_module._COMPILED_REACHED):
        monkeypatch.setattr(c, "copy", None)
        monkeypatch.setattr(c, "runs", 0)


def fake_compile(monkeypatch, make):
    """torch.compile replaced by ``make(fn)``; returns the names of the functions compiled, in order."""
    names = []

    def compile_(fn, *args, **kw):
        if args or kw != {"dynamic": True, "options": {"cpp.dynamic_threads": True}}:
            # Not an Exception, so the fallback cannot swallow it.
            pytest.fail(f"torch.compile called with {args}, {kw}")
        names.append(fn.__name__)
        return make(fn)

    monkeypatch.setattr(torch, "compile", compile_)
    return names


def compile_warnings(record):
    """The warnings in ``record`` whose message contains "torch.compile failed"."""
    return [w for w in record if "torch.compile failed" in str(w.message)]


@pytest.fixture(scope="session")
def inductor_cache(pytestconfig, tmp_path_factory):
    """
    A directory of the tests' own for Inductor's on-disk cache.

    Under pytest's cache, so that a rerun reuses what an earlier one
    compiled; a temporary one when pytest's cache is off.
    """
    cache = getattr(pytestconfig, "cache", None)
    if cache is None:
        return tmp_path_factory.mktemp("torchinductor")
    return cache.mkdir("torchinductor")


@pytest.fixture
def working_compile(inductor_cache, monkeypatch):
    """
    Skip unless on Linux with a torch.compile that compiles and runs a trivial function.

    Inductor keeps the code it compiles in :func:`inductor_cache`, in this
    process and in any the test starts, never in the cache of the user's
    own processes: those would reuse kernels compiled here from tiny
    inputs. Torch's precompiled headers, which hold no kernel, stay in its
    per-user directory.
    """
    if sys.platform != "linux":
        pytest.skip("the compiled path is tested on Linux only")
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(inductor_cache))
    try:
        torch.compile(lambda x: 2.0 * x + 1.0, dynamic=True)(torch.arange(3.0))
    except Exception as err:
        pytest.skip(f"torch.compile does not work here: {type(err).__name__}: {err}")


def cannot_compile(fn):
    """A ``make`` that cannot build a copy, as where no C++ compiler works."""
    raise RuntimeError("no compiler")


def cannot_compile_chained(fn):
    """:func:`cannot_compile`, its error raised from another, as torch's often are."""
    try:
        raise OSError("no C++ compiler found")
    except OSError as cause:
        raise RuntimeError("no compiler") from cause


class HoldingError(RuntimeError):
    """An error holding a frame and inputs, as torch's ``first_useful_frame`` and ``_torch_dynamo_tracer_output`` do."""

    def __init__(self, message, first_useful_frame, inputs):
        super().__init__(message)
        self.first_useful_frame = first_useful_frame
        self.inputs = inputs


def fails_holding_its_caller_s_frame_and_inputs(fn):
    """A copy whose error holds the frame that called the copy and the call's arguments, as torch's do beyond their traceback."""

    def copy(*args):
        # Torch's frame is that of the code that called the compiled
        # function. No local of the copy's holds the error, so that the copy
        # makes no cycle of its own.
        raise HoldingError("backend failed", sys._getframe(1), args)

    return copy


def fails_on_the_second_call(fn):
    """A copy that runs ``fn`` once, then raises, as a failed recompile would."""
    calls = []

    def copy(*args):
        calls.append(None)
        if len(calls) > 1:
            raise RuntimeError("recompile failed")
        return fn(*args)

    return copy


def recording(calls):
    """A ``make`` whose copy runs ``fn``, appending each call's arguments to ``calls``."""

    def make(fn):
        def copy(*args):
            calls.append(args)
            return fn(*args)

        return copy

    return make


def status_inputs(n):
    """``lens_status``'s arguments on ``n`` random rows: ``beta6, cls, level, h0, min_img_sep``."""
    gen = torch.Generator().manual_seed(0)
    beta6 = torch.randn(n, 6, 2, dtype=torch.float64, generator=gen)
    cls = torch.randint(0, 4, (n,), dtype=torch.int64, generator=gen)
    level = torch.randint(0, 10, (n,), dtype=torch.int64, generator=gen)
    return beta6, cls, level, 0.3125, 5e-4


def assert_outputs_equal(got, want):
    """``lens_status`` outputs equal bit for bit, output by output."""
    assert len(got) == len(want)
    for a, b in zip(got, want):
        assert_same(a, b)


def clear_of_thresholds(mesh, rel=1e-9):
    """Every finite origin's deviation is further than rel from sigma_min * min_img_sep, and every finite band det further than rel * max|det| from zero."""
    mesh = to_mesh(mesh)
    finite = (to_np(mesh.origin_status) & LEAF_RAYTRACE_NONFINITE) == 0
    deviation = to_np(mesh.origin_deviation)[finite]
    threshold = to_np(mesh.origin_sigma_min)[finite] * mesh.min_img_sep
    det = to_np(mesh.critical_band.det)
    det = det[np.isfinite(det)]
    return bool(
        np.all(np.abs(deviation - threshold) > rel * np.abs(threshold))
        and np.all(np.abs(det) > rel * np.max(np.abs(det), initial=0.0))
    )


def build_and_extend(fn, jac, **kw):
    """``build(fn, jac, 4.0, 4, 0.05)`` extended to a fov of 6, both with ``kw``."""
    mesh, lens_ = build(fn, jac, 4.0, 4, 0.05, **kw)
    return extend_lens_mesh(mesh, lens_.raytrace, 6.0, **kw)


def sie_lens():
    """The ``SIE`` of ``tests/test_adaptive_lens_mesh.py``'s ``sie_mesh``."""
    return SIE(
        name="sie",
        cosmology=FlatLambdaCDM(name="cosmo"),
        z_l=0.5,
        z_s=1.5,
        x0=0.0,
        y0=0.0,
        q=0.4,
        phi=np.pi / 5,
        Rein=1.0,
        s=1e-3,
    )


def fold_build(**kw):
    return build_lens_mesh(
        numpy_lens(localised_fold, localised_fold_jacobian).raytrace, 4.0, 4, 0.05, **kw
    )


def fold_past_the_edge(p):
    """:func:`localised_fold` turned to fold across x, on ``2.05 < x < 3.15``: affine within a fov of 4."""
    folded = localised_fold(np.stack([p[:, 1], p[:, 0] - 2.6], axis=-1))
    return np.stack([folded[:, 1], folded[:, 0]], axis=-1)


def fold_extension(**kw):
    """A build extended from a fov of 4 to 6, both with ``kw``: the extension splits old rows and refits them."""
    # build_lens_mesh calls the raytrace alone.
    raytrace = numpy_lens(fold_past_the_edge, None).raytrace
    mesh = build_lens_mesh(raytrace, 4.0, 4, 0.05, **kw)
    return extend_lens_mesh(mesh, raytrace, 6.0, **kw)


def sis_closed_build(**kw):
    raytrace = numpy_lens(*sis()).raytrace
    return build_closed_lens_mesh(raytrace, 6.0, 16, 0.02, centers=[(0.0, 0.0)], **kw)


def sis_closed_region(**kw):
    raytrace = numpy_lens(*sis()).raytrace
    return build_closed_magnification_region(
        raytrace, 6.0, 16, 0.02, 3.0, centers=[(0.0, 0.0)], **kw
    )


def sie_closed_build(**kw):
    return sie_lens().build_closed_lens_mesh(5.0, 16, 0.05, **kw)


def sie_closed_region(**kw):
    return sie_lens().build_closed_lens_mesh(5.0, 16, 0.05, mu_min=3.0, **kw)


def sie_forward_raytrace(**kw):
    return sie_lens().forward_raytrace(
        backend.as_array(0.05),
        backend.as_array(0.02),
        fov=5.0,
        init_res=16,
        min_img_sep=0.05,
        **kw,
    )


def index_inputs():
    """A point-mass mesh's index inputs: grown leaves, and leaves beside non-finite ones."""
    mesh = to_mesh(
        build_lens_mesh(
            numpy_lens(*point_mass()).raytrace, 5.0, 16, 0.05, centers=[(0.0, 0.0)]
        )
    )
    rows = i64(finite_rows(mesh))
    grow = leaf_grow(mesh)
    assert (to_np(grow)[to_np(rows)] > 0).any() and rows.shape[0] < mesh.leaves.shape[0]
    return mesh, rows, grow


def test_a_copy_that_cannot_be_built_warns_once_and_runs_eagerly(
    fresh_compiling, monkeypatch
):
    error = RuntimeError("no compiler")

    def make(fn):
        raise error

    names = fake_compile(monkeypatch, make)
    double, increment = Compiled(lambda x: 2 * x), Compiled(lambda x: x + 1)
    x = torch.arange(3.0)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        got = [double(x), double(x), increment(x)]
    for a, b in zip(got, [2 * x, 2 * x, x + 1]):
        assert torch.equal(a, b)
    assert len(compile_warnings(record)) == 1
    assert type(Compiled.failed) is RuntimeError
    assert str(Compiled.failed) == "RuntimeError: no compiler"
    assert len(names) == 1


@pytest.mark.parametrize(
    "make",
    [
        cannot_compile,
        cannot_compile_chained,
        fails_holding_its_caller_s_frame_and_inputs,
    ],
    ids=["alone", "chained", "holding a frame and the inputs"],
)
def test_a_failed_compile_keeps_no_argument_and_no_caller_s_locals_alive(
    make, fresh_compiling, monkeypatch
):
    """
    ``Compiled.failed`` keeps a description of the failure, and nothing of the exception.

    The exception's traceback, and attributes torch sets on it, hold frames,
    through each frame every caller's, each with its final locals, and the
    call's arguments: in a build, refine's state and the meshes, for the
    rest of the process. Checked with the cyclic collector off: a cycle
    through the exception would keep all of it until a full collection.
    """
    fake_compile(monkeypatch, make)
    double = Compiled(lambda x: 2 * x)
    refs = {}

    def caller():
        x = torch.arange(3.0)
        refs["argument"] = weakref.ref(x)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y = double(x)
        refs["caller's final local"] = weakref.ref(y)

    gc.disable()
    try:
        caller()
        alive = [name for name, ref in refs.items() if ref() is not None]
    finally:
        gc.enable()
    assert isinstance(Compiled.failed, RuntimeError)
    assert alive == []


def test_a_copy_that_fails_on_a_later_call_warns_once_and_runs_eagerly(
    fresh_compiling, monkeypatch
):
    fake_compile(monkeypatch, fails_on_the_second_call)
    double, increment = Compiled(lambda x: 2 * x), Compiled(lambda x: x + 1)
    x = torch.arange(3.0)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        got = [double(x), double(x), increment(x)]
    for a, b in zip(got, [2 * x, 2 * x, x + 1]):
        assert torch.equal(a, b)
    assert len(compile_warnings(record)) == 1
    assert double.runs == 1


def test_an_error_the_eager_function_raises_too_propagates_and_keeps_compiling_on(
    fresh_compiling, monkeypatch
):
    def fn(x):
        raise ValueError("bad input")

    fake_compile(monkeypatch, lambda fn: fn)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        with pytest.raises(ValueError):
            Compiled(fn)(torch.arange(3.0))
    assert not compile_warnings(record)
    assert Compiled.failed is None


@pytest.mark.parametrize(
    "make",
    [
        cannot_compile,
        cannot_compile_chained,
        fails_holding_its_caller_s_frame_and_inputs,
    ],
    ids=["alone", "chained", "holding a frame and the inputs"],
)
def test_a_failed_compile_whose_rerun_raises_too_keeps_nothing_alive(
    make, fresh_compiling, monkeypatch
):
    """
    The rerun's error propagates alone, compiling stays on, and nothing of the failed compile stays alive.

    Checked with the cyclic collector off: once the caller drops the
    rerun's error, the argument and the caller's locals are freed.
    """
    fake_compile(monkeypatch, make)

    def fn(x):
        raise ValueError("bad input")

    bad = Compiled(fn)
    refs, raised = {}, []

    def caller():
        x = torch.arange(3.0)
        local = x + 1.0
        refs["argument"] = weakref.ref(x)
        refs["caller's local"] = weakref.ref(local)
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            try:
                bad(x)
            except ValueError as err:
                # Whether it is raised alone, not the context it may hold.
                raised.append(err.__context__ is None)
        raised.append(len(compile_warnings(record)))

    gc.disable()
    try:
        caller()
        alive = [name for name, ref in refs.items() if ref() is not None]
    finally:
        gc.enable()
    assert raised == [True, 0]
    assert Compiled.failed is None
    assert alive == []


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_arrays_off_the_cpu_run_eagerly(fresh_compiling, monkeypatch):
    """The first array of the arguments decides, found through tuples as well."""
    names = fake_compile(monkeypatch, cannot_compile)
    double = Compiled(lambda x: 2 * x)
    add = Compiled(lambda pair: pair[0] + pair[1])
    x = torch.arange(3.0, device="cuda")
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        doubled, added = double(x), add((x, x))
    assert torch.equal(doubled, 2 * x) and torch.equal(added, x + x)
    assert not compile_warnings(record)
    assert double.copy is None and add.copy is None
    assert names == []


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_arrays_off_the_cpu_run_the_eager_function(fresh_compiling, monkeypatch):
    """Off the CPU, ``eager`` runs and ``fn`` does not, the first array found through tuples as well."""
    names = fake_compile(monkeypatch, cannot_compile)
    calls = []
    double = Compiled(
        lambda x: calls.append("fn") or 2 * x,
        eager=lambda x: calls.append("eager") or 2 * x,
    )
    add = Compiled(
        lambda pair: calls.append("fn") or pair[0] + pair[1],
        eager=lambda pair: calls.append("eager") or pair[0] + pair[1],
    )
    x = torch.arange(3.0, device="cuda")
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        doubled, added = double(x), add((x, x))
    assert torch.equal(doubled, 2 * x) and torch.equal(added, x + x)
    assert calls == ["eager", "eager"]
    assert not compile_warnings(record)
    assert double.copy is None and add.copy is None
    assert names == []


def test_compiled_runs_its_eager_function_wherever_it_runs_eagerly(
    fresh_compiling, monkeypatch
):
    calls = []
    fake_compile(monkeypatch, fails_on_the_second_call)
    double = Compiled(
        lambda x: calls.append("fn") or 2 * x,
        eager=lambda x: calls.append("eager") or 2 * x,
    )
    x = torch.arange(3.0)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        got = [double(x), double(x), double(x)]
    assert all(torch.equal(g, 2 * x) for g in got)
    # The copy, the failing call's rerun, then compiling off.
    assert calls == ["fn", "eager", "eager"]
    assert Compiled(len).eager is len


@pytest.mark.parametrize("n", [0, 1])
def test_lens_status_runs_fewer_than_two_rows_eagerly(n, fresh_compiling, monkeypatch):
    calls = []
    names = fake_compile(monkeypatch, recording(calls))
    args = status_inputs(n)
    got = lens_status(*args, compile_bookkeeping=True)
    assert_outputs_equal(got, lens_status(*args))
    assert names == [] and calls == []


@pytest.mark.parametrize("as_tensors", [False, True], ids=["floats", "0-d tensors"])
def test_lens_status_hands_its_compiled_copy_0d_float64_tensors(
    as_tensors, fresh_compiling, monkeypatch
):
    calls = []
    fake_compile(monkeypatch, recording(calls))
    beta6, cls, level, h0, min_img_sep = status_inputs(5)
    given = (h0, min_img_sep)
    if as_tensors:
        given = tuple(torch.tensor(v, dtype=torch.float64) for v in given)
    got = lens_status(beta6, cls, level, *given, compile_bookkeeping=True)
    (args,) = calls
    for value, want in zip(args[3:], (h0, min_img_sep)):
        assert isinstance(value, torch.Tensor)
        assert value.shape == () and value.dtype == torch.float64
        assert value.device == beta6.device and value.item() == want
    assert_outputs_equal(got, lens_status(beta6, cls, level, *given))


def test_a_failed_compile_runs_the_index_s_step_in_blocks(fresh_compiling, monkeypatch):
    mesh, rows, grow = index_inputs()
    eager = build_index(mesh.vertices_source, mesh.leaves, rows, grow[rows])
    fake_compile(monkeypatch, cannot_compile)
    monkeypatch.setattr(index_module, "REACH_BLOCK", 64)
    sizes, real = [], index_module._reached

    def counting(tx, ty, side, reach, i0x, i0y, i1y, box, dx, lo):
        sizes.append(int(box.shape[0]))
        return real(tx, ty, side, reach, i0x, i0y, i1y, box, dx, lo)

    monkeypatch.setattr(index_module, "_reached", counting)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        got = build_index(
            mesh.vertices_source,
            mesh.leaves,
            rows,
            grow[rows],
            compile_bookkeeping=True,
        )
    assert len(compile_warnings(record)) == 1
    assert len(sizes) > 1 and max(sizes) <= 64
    assert_same(got, eager)


def test_the_default_build_never_compiles(fresh_compiling, monkeypatch):
    names = fake_compile(monkeypatch, cannot_compile)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        build_and_extend(localised_fold, localised_fold_jacobian)
    assert not compile_warnings(record)
    assert names == []


@pytest.mark.parametrize(
    "make",
    [cannot_compile, fails_on_the_second_call],
    ids=["at creation", "on the second call"],
)
def test_a_build_whose_compiling_fails_warns_once_and_equals_the_eager_build(
    make, fresh_compiling, monkeypatch
):
    eager = build_and_extend(localised_fold, localised_fold_jacobian)
    fake_compile(monkeypatch, make)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        compiled = build_and_extend(
            localised_fold, localised_fold_jacobian, compile_bookkeeping=True
        )
    assert len(compile_warnings(record)) == 1
    assert_same(compiled, eager)


@pytest.mark.parametrize(
    "entry",
    [
        fold_build,
        fold_extension,
        sis_closed_build,
        sis_closed_region,
        sie_closed_build,
        sie_closed_region,
        sie_forward_raytrace,
    ],
    ids=[
        "build_lens_mesh",
        "extend_lens_mesh",
        "build_closed_lens_mesh",
        "build_closed_magnification_region",
        "Lens.build_closed_lens_mesh",
        "Lens.build_closed_lens_mesh with mu_min",
        "Lens.forward_raytrace",
    ],
)
def test_every_build_hands_compile_bookkeeping_to_lens_status_and_the_index(
    entry, fresh_compiling, monkeypatch
):
    """
    Every ``lens_status`` call, and every index build, of two rows or more goes through its compiled copy.

    The copies run ``fn`` itself, so a call that reaches ``_lens_status``
    directly, or an index build whose reached-cells step runs in blocks,
    was not handed the flag. The extension splits old rows, so it reaches
    ``_finish``'s refit of them, with two rows or more.
    """
    names = fake_compile(monkeypatch, lambda fn: fn)
    status_rows, refit_rows, blocked_rows = [], [], []
    status_fn = criterion_module._lens_status
    status_in_lens_mesh = lens_mesh_module.lens_status
    in_blocks = index_module._reached_in_blocks

    def status_from_lens_mesh(beta6, *args, **kw):
        # Called by ``_finish`` itself, it refits the old rows split.
        if sys._getframe(1).f_code is lens_mesh_module._finish.__code__:
            refit_rows.append(beta6.shape[0])
        return status_in_lens_mesh(beta6, *args, **kw)

    def direct_status(beta6, *args):
        status_rows.append(beta6.shape[0])
        return status_fn(beta6, *args)

    def blocked(tx, *args):
        # tx holds the vertices of each indexed row's triangle.
        blocked_rows.append(tx.shape[0])
        return in_blocks(tx, *args)

    monkeypatch.setattr(criterion_module, "_lens_status", direct_status)
    monkeypatch.setattr(lens_mesh_module, "lens_status", status_from_lens_mesh)
    monkeypatch.setattr(index_module, "_reached_in_blocks", blocked)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        entry(compile_bookkeeping=True)
    assert {"_lens_status", "_reached"} <= set(names)
    assert [n for n in status_rows if n >= 2] == []
    assert [n for n in blocked_rows if n >= 2] == []
    assert not compile_warnings(record)
    if entry is fold_extension:
        assert [n for n in refit_rows if n >= 2] != []


def test_nothing_in_refine_is_compiled(fresh_compiling, monkeypatch):
    names = fake_compile(monkeypatch, lambda fn: fn)
    fold_extension(compile_bookkeeping=True)
    assert set(names) == {"_lens_status", "_reached"}
    for fn in (refine_module.refine, refine_module.balance, refine_module.unbalanced):
        assert "compile_bookkeeping" not in inspect.signature(fn).parameters


@pytest.mark.parametrize(
    "maps",
    [
        (localised_fold, localised_fold_jacobian),
        broken_where(localised_fold, localised_fold_jacobian, lambda p: p[:, 0] > 1.0),
    ],
    ids=["localised fold", "non-finite where x > 1"],
)
def test_a_compiled_build_agrees_with_the_eager_build_to_rounding(
    maps, working_compile, monkeypatch
):
    monkeypatch.setattr(Compiled, "failed", None)
    eager = build_and_extend(*maps)
    assert clear_of_thresholds(eager), "the config must keep clear of the thresholds"
    status_copy = criterion_module._COMPILED_LENS_STATUS
    runs = status_copy.runs
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        compiled = build_and_extend(*maps, compile_bookkeeping=True)
    assert_same(compiled, eager, rounding=ROUNDING)
    assert not compile_warnings(record)
    assert status_copy.runs > runs


def test_a_compiled_build_extended_eagerly_agrees_with_eager_to_rounding(
    working_compile, monkeypatch
):
    monkeypatch.setattr(Compiled, "failed", None)
    eager = build_and_extend(localised_fold, localised_fold_jacobian)
    assert clear_of_thresholds(eager), "the config must keep clear of the thresholds"
    status_copy = criterion_module._COMPILED_LENS_STATUS
    runs = status_copy.runs
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        mesh, lens_ = build(
            localised_fold,
            localised_fold_jacobian,
            4.0,
            4,
            0.05,
            compile_bookkeeping=True,
        )
        mixed = extend_lens_mesh(mesh, lens_.raytrace, 6.0, compile_bookkeeping=False)
    assert_same(mixed, eager, rounding=ROUNDING)
    assert not compile_warnings(record)
    assert status_copy.runs > runs


def test_compiled_lens_status_keeps_the_images_gradient(working_compile, monkeypatch):
    monkeypatch.setattr(Compiled, "failed", None)
    beta6, cls, level, h0, min_img_sep = status_inputs(64)
    beta6.requires_grad_()
    runs = criterion_module._COMPILED_LENS_STATUS.runs
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        compiled = lens_status(
            beta6, cls, level, h0, min_img_sep, compile_bookkeeping=True
        )
    eager = lens_status(beta6, cls, level, h0, min_img_sep)
    assert not compile_warnings(record)
    assert criterion_module._COMPILED_LENS_STATUS.runs == runs + 1
    assert [o.requires_grad for o in compiled[1:]] == [True] * 3
    assert [o.requires_grad for o in eager[1:]] == [True] * 3
    np.testing.assert_array_equal(to_np(compiled[0]), to_np(eager[0]))
    for a, b in zip(compiled[1:], eager[1:]):
        np.testing.assert_allclose(to_np(a), to_np(b), rtol=1e-12, atol=1e-12)


def test_the_compiled_index_is_the_eager_one(working_compile, monkeypatch):
    monkeypatch.setattr(Compiled, "failed", None)
    mesh, rows, grow = index_inputs()
    args = (mesh.vertices_source, mesh.leaves, rows, grow[rows])
    eager = build_index(*args)
    pad = index_module.PAD
    for moved in (pad * (1 - 1e-3), pad * (1 + 1e-3)):
        monkeypatch.setattr(index_module, "PAD", moved)
        # No interval end lies within 1e-6 of a cell line.
        assert_same(build_index(*args), eager)
    monkeypatch.setattr(index_module, "PAD", pad)
    copy = index_module._COMPILED_REACHED
    runs = copy.runs
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        compiled = build_index(*args, compile_bookkeeping=True)
        one = build_index(
            mesh.vertices_source,
            mesh.leaves,
            rows[:1],
            grow[rows[:1]],
            compile_bookkeeping=True,
        )
    assert not compile_warnings(record)
    assert copy.runs == runs + 1  # the build of one row ran eagerly
    assert_same(compiled, eager)
    assert_same(
        one, build_index(mesh.vertices_source, mesh.leaves, rows[:1], grow[rows[:1]])
    )
    rng = np.random.default_rng(41)
    beta = mesh_backend.as_array(
        rng.uniform(-2, 2, (200, 2)), dtype=mesh_backend.float64
    )
    assert_hits_equal(
        index_hits(compiled, mesh.vertices_source, mesh.leaves, beta, grow),
        brute_hits(mesh.vertices_source, mesh.leaves, rows, beta, grow),
    )


# Run by the test below in a process of its own: after a real failure,
# torch leaves its frame evaluation hook installed for the rest of the
# process, and with it every caller holds its call's arguments until the
# call returns, which later tests of when memory is freed would see.
REAL_FAILURE = """
import gc, json, warnings, weakref

import torch
from torch._inductor import config as inductor_config

from caustics.lenses.func.adaptive.mesh_backend import Compiled

inductor_config.cpp.cxx = (None, "/nonexistent/g++")
# So that no cached artifact stands in for the compiler.
inductor_config.fx_graph_cache = False
affine = Compiled(lambda x: 3.0 * x - 1.0)
refs = {}


def caller():
    x = torch.arange(5.0)
    local = x + 1.0
    refs["argument"] = weakref.ref(x)
    refs["caller's local"] = weakref.ref(local)
    return affine(x), 3.0 * x - 1.0


# Off for the rest of the process: a cycle through the exception would keep
# all of it until a full collection.
gc.disable()
with warnings.catch_warnings(record=True) as record:
    warnings.simplefilter("always")
    got, eager = caller()
equal = torch.equal(got, eager)
del got, eager
print(json.dumps(dict(
    messages=[str(w.message) for w in record],
    equal=equal,
    alive=[name for name, ref in refs.items() if ref() is not None],
)))
"""


def test_a_real_compile_failure_warns_once_equals_eager_and_keeps_nothing_alive(
    working_compile,
):
    """
    ``torch.compile`` with no C++ compiler to call.

    Inductor's error holds, beside its traceback, the frame the compiled
    function was called from and the inputs it traced. Checked with the
    cyclic collector off, as in
    :func:`test_a_failed_compile_keeps_no_argument_and_no_caller_s_locals_alive`.
    """
    child = subprocess.run(
        [sys.executable, "-c", REAL_FAILURE],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert child.returncode == 0, child.stderr
    got = json.loads(child.stdout.splitlines()[-1])
    assert len([m for m in got["messages"] if "torch.compile failed" in m]) == 1
    assert got["equal"]
    assert got["alive"] == []
