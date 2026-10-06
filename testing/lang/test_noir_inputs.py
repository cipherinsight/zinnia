"""Kernel inputs reach the Noir circuit as the mock reads them (npb-gaps).

Two Noir emitter defects made NPBench/Pythran programs unprovable on Noir:

- a kernel parameter named like one of the emitter's own values (`v2` in
  npbench gemver) was shadowed by it, and `main.nr` did not compile; such a
  parameter (and a Noir keyword or type name) is now renamed in `main.nr` and
  `Prover.toml` (`prove::noir::inputs::noir_param_identifier`);
- a complex input was rejected: an element of a complex array is read at a
  two-segment path (element, real/imaginary part), which the emitter refused,
  and a complex scalar had no `Prover.toml` encoding. A parameter is now one
  flat array of its scalar leaves, and a read finds its leaf by the
  parameter's layout (`prove::noir::inputs::LeafLayout`).

Each kernel's outputs equal NumPy's on the mock, and halo2-base and Noir return
exactly the mock's field elements.
"""
import numpy as np
import pytest

from testing.lang.all_backends import assert_all_backends_match_numpy

Z = np.array([[1.5 - 2.0j, -0.25 + 0.5j, 3.0 + 0.0j], [0.0 - 1.0j, 2.0 + 2.5j, -1.75 - 0.125j]])


@pytest.mark.parametrize("source, args", [
    # npbench gemver's parameter names, and names Noir reserves.
    ('def kernel(u1: Float[Array, "3"], v1: Float[Array, "3"], u2: Float[Array, "3"], v2: Float[Array, "3"], v0: Float):\n'
     '    return np.outer(u1, v1) + np.outer(u2, v2) + v0\n',
     (np.array([1.0, -2.0, 0.5]), np.array([0.25, 3.0, -1.0]), np.array([2.0, 0.0, -0.5]),
      np.array([-1.5, 1.0, 4.0]), 0.75)),
    ('def kernel(type: Integer, fn: Integer, std: Integer, param_x: Integer, u8: Integer):\n'
     '    return type * 10000 + fn * 1000 + std * 100 + param_x * 10 + u8\n',
     (1, 2, 3, 4, 5)),
])
def test_parameters_named_like_noir_names(source, args):
    assert_all_backends_match_numpy(source, *args)


@pytest.mark.parametrize("source, args", [
    # pythran crotg: complex scalars.
    ('def kernel(a: Complex, b: Complex):\n    return a * b.conjugate(), abs(a)\n',
     (1.5 - 2.0j, -0.25 + 3.0j)),
    # A multi-axis read of a complex array (pythran clip / make_decision / spectraldns).
    ('def kernel(z: Complex[Array, "2 3"]):\n    return z[1, 2] * z[0, 1] + z[1, 0]\n', (Z,)),
    ('def kernel(z: Complex[Array, "2 3"], w: Float[Array, "2 3"]):\n    return z * w\n',
     (Z, np.array([[1.0, -2.0, 0.5], [0.0, 3.0, -1.0]]))),
    # A complex array next to a complex scalar.
    ('def kernel(z: Complex[Array, "2 3"], c: Complex):\n    return z[:, 1] - c\n', (Z, 0.5 + 0.25j)),
])
def test_complex_inputs(source, args):
    assert_all_backends_match_numpy(source, *args)
