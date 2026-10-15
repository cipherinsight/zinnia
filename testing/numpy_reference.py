"""NumPy as the tests' reference implementation sees it.

Zinnia supports NumPy 2.2 and later, and keeps accepting the names and
argument forms that NumPy 2.2 still provides and later releases removed (see
"Removed NumPy Names" in the NumPy API documentation): ``np.in1d``,
``np.trapz``, ``np.row_stack``, inexact ``mode`` names of ``np.convolve`` and
``np.correlate``, and ``np.cross`` of 2-component vectors. A test that runs a
kernel's source under NumPy to compute the expected result gives it ``np``
from here: ``numpy`` with those names and forms computed by the current
functions, so the reference does not depend on the installed NumPy still
providing them.
"""
from types import SimpleNamespace

import numpy


def _in1d(ar1, ar2, assume_unique=False, invert=False, *, kind=None):
    """``np.in1d`` (removed in NumPy 2.4): ``np.isin`` on the flattened first argument."""
    return numpy.isin(ar1, ar2, assume_unique=assume_unique, invert=invert, kind=kind).ravel()


def _correlation_mode(mode):
    """A ``mode`` of ``np.convolve``/``np.correlate``. Before NumPy 2.3 a mode
    string was matched by its first letter in any case (``'FULL'``, ``'s'``)."""
    if isinstance(mode, str) and mode:
        return {"v": "valid", "s": "same", "f": "full"}.get(mode[0].lower(), mode)
    return mode


def _convolve(a, v, mode="full"):
    return numpy.convolve(a, v, mode=_correlation_mode(mode))


def _correlate(a, v, mode="valid"):
    return numpy.correlate(a, v, mode=_correlation_mode(mode))


def _cross(a, b, axisa=-1, axisb=-1, axisc=-1, axis=None):
    """``np.cross``. Before NumPy 2.5 a 2-component vector was a 3-component
    vector with zero z, and the cross product of two 2-component vectors was
    its z component alone."""
    if axis is not None:
        axisa = axisb = axisc = axis
    a = numpy.moveaxis(numpy.asarray(a), axisa, -1)
    b = numpy.moveaxis(numpy.asarray(b), axisb, -1)
    lengths = (a.shape[-1], b.shape[-1])
    if 2 not in lengths or not set(lengths) <= {2, 3}:
        return numpy.cross(a, b, axisc=axisc)

    def as_3d(x):
        return numpy.concatenate([x, numpy.zeros_like(x[..., :1])], axis=-1) if x.shape[-1] == 2 else x

    c = numpy.cross(as_3d(a), as_3d(b))
    return c[..., 2] if lengths == (2, 2) else numpy.moveaxis(c, -1, axisc)


REMOVED = {
    "in1d": _in1d,
    "trapz": numpy.trapezoid,
    "row_stack": numpy.vstack,
    "convolve": _convolve,
    "correlate": _correlate,
    "cross": _cross,
}


class _NumPyReference(SimpleNamespace):
    """``numpy`` with :data:`REMOVED` in place of the installed functions."""

    def __getattr__(self, name):
        return getattr(numpy, name)


np = _NumPyReference(**REMOVED)
