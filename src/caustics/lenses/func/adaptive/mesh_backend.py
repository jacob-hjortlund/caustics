"""
The backend the mesh bookkeeping runs on, and conversions to and from the user's.

The bookkeeping is gathers, sorts and searches on arrays whose sizes depend
on the data. Torch runs them as they come; eager jax compiles each anew for
nearly every call, so under jax the bookkeeping runs on torch on the CPU.
Lens calls and root finding stay in the user's backend, and the public
functions convert at their boundary (:func:`to_mesh`, :func:`to_user`).
Under torch ``mesh_backend`` is ``backend`` itself and nothing converts.
:class:`Compiled` runs a bookkeeping function through ``torch.compile``.
"""

import importlib
from warnings import warn

import numpy as np

from ....backend_obj import Backend, backend


class TorchMeshBackend(Backend):
    """
    ``Backend``'s torch methods, without switching caskade's global backend.

    ``as_array`` also takes jax arrays, and sequences holding them, through a
    numpy copy.
    """

    def __init__(self):
        self.module = importlib.import_module("torch")
        self.setup_torch()
        self._backend = "torch"

    def _readable(self, array):
        """``array``, or a numpy copy of it where torch cannot read it."""
        if isinstance(array, (list, tuple)) or type(array).__module__.startswith("jax"):
            return np.array(array)
        return array

    def _as_array_torch(self, array, dtype=None, device=None):
        return super()._as_array_torch(self._readable(array), dtype, device)


mesh_backend = backend if backend.backend == "torch" else TorchMeshBackend()


def map_arrays(value, fn):
    """
    ``fn`` on every array of ``value``, recursively through tuples and NamedTuples.

    Anything without a ``shape`` passes through. ``value`` itself comes back
    when ``fn`` returns every array unchanged.
    """
    if isinstance(value, tuple):
        items = [map_arrays(v, fn) for v in value]
        if all(new is old for new, old in zip(items, value)):
            return value
        return type(value)(*items) if hasattr(value, "_fields") else tuple(items)
    if hasattr(value, "shape"):
        return fn(value)
    return value


def _first_array(value):
    """The first array of ``value``, searched through tuples and NamedTuples; None when it holds none."""
    if isinstance(value, tuple):
        for v in value:
            found = _first_array(v)
            if found is not None:
                return found
        return None
    return value if hasattr(value, "device") else None


class Compiled:
    """
    ``fn``, run through a copy compiled by ``torch.compile`` where it can be, and eagerly otherwise.

    The copy is ``torch.compile`` of ``fn`` with ``dynamic=True`` and
    ``options={"cpp.dynamic_threads": True}``, built on the first call that
    runs it and reused by every later one. Torch reuses compiled code for
    inputs of every size, and in later processes through its on-disk cache,
    so a loop's code may come from a compile on far smaller inputs.
    ``cpp.dynamic_threads`` makes every compiled loop run in parallel
    whatever the sizes it was compiled for, its threads chosen when it
    runs.

    A call runs ``eager`` in place of the copy when compiling is off for
    the process, or when the first array among its arguments, searched
    through tuples and NamedTuples, is not on the CPU.

    When building or calling the copy raises an ``Exception``, ``eager``
    is rerun on the same arguments. If the rerun raises too, its error
    propagates and compiling stays on, the fault not being the compiler's.
    Otherwise ``Compiled.failed`` keeps a description of the exception,
    not the exception itself: its traceback, and attributes torch sets on
    it, hold frames and the inputs it traced, and through each frame every
    caller's, with their locals. One warning is given, and every
    ``Compiled`` runs eagerly from then on in this process. ``fn`` must
    therefore write nothing in place: a failed compiled call is rerun
    eagerly on the same arguments.

    Parameters
    ----------
    fn: Callable
        Takes positional arguments alone.
    eager: Optional[Callable]
        Run in place of the copy wherever a call runs eagerly, such as
        ``fn`` in steps that bound its temporaries: on ``fn``'s arguments
        it must return what ``fn`` returns. None is ``fn`` itself.

    Attributes
    ----------
    fn: Callable
    eager: Callable
        ``eager``, or ``fn`` where it is None.
    copy: Optional[Callable]
        The compiled copy; None until it is built.
    runs: int
        Calls the compiled copy has run without error.
    failed: Optional[RuntimeError]
        A class attribute, shared by every ``Compiled``: a new exception
        whose message is the type and message of the one that turned
        compiling off for the process, holding nothing of it, so that no
        frame and no input of the failed call stays alive; None while
        compiling is on.
    """

    failed = None

    def __init__(self, fn, eager=None):
        self.fn = fn
        self.eager = fn if eager is None else eager
        self.copy = None
        self.runs = 0

    def __call__(self, *args):
        first = _first_array(args)
        if Compiled.failed is not None or (
            first is not None and first.device.type != "cpu"
        ):
            return self.eager(*args)
        try:
            if self.copy is None:
                # Looked up now, not at import, so that it can be replaced.
                self.copy = mesh_backend.module.compile(
                    self.fn, dynamic=True, options={"cpp.dynamic_threads": True}
                )
            out = self.copy(*args)
        except Exception as err:
            # Only its text outlives the handler: the exception's traceback
            # holds this frame, so a local holding it would be a cycle.
            text = f"{type(err).__name__}: {err}"
        else:
            self.runs += 1
            return out
        # Outside the handler, so that an error of the rerun is raised alone.
        out = self.eager(*args)
        Compiled.failed = RuntimeError(text)
        warn(
            f"torch.compile failed ({text}); the mesh bookkeeping runs eagerly "
            "from now on in this process.",
            stacklevel=2,
        )
        return out


if mesh_backend is backend:

    def to_mesh(value):
        """``value`` with its arrays as ``mesh_backend`` arrays: under torch, ``value``."""
        return value

    def to_user(value, device=None):
        """``value`` with its arrays as ``backend`` arrays on ``device``."""
        return map_arrays(value, lambda a: backend.to(a, device=device))

else:
    _Tensor = mesh_backend.module.Tensor

    def to_mesh(value):
        """``value`` with its arrays as ``mesh_backend`` arrays, copied."""
        # A copy: the bookkeeping writes in place, and a jax buffer is immutable.
        return map_arrays(
            value,
            lambda a: (
                a
                if isinstance(a, _Tensor)
                else mesh_backend.module.as_tensor(np.array(a))
            ),
        )

    def to_user(value, device=None):
        """``value`` with its arrays as ``backend`` arrays on ``device``, copied."""
        # Copied on the host, so no later in-place write reaches it, then put:
        # ``jnp.array`` and ``jnp.asarray`` compile a program for every new shape.
        return map_arrays(
            value,
            lambda a: (
                backend.to(a.numpy().copy(), device=device)
                if isinstance(a, _Tensor)
                else a
            ),
        )
