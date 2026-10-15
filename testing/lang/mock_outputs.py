"""Test helpers: run a kernel on the mock backend and read its outputs by position.

The mock backend reports each exposed public as ``public_<i|f>_<statement id>``
and returns them unordered. Sorting by statement id restores the order in which
the circuit exposed them, which is the order of the flattened return value.
Float publics are Q32 fixed point (value * 2**32).
"""
import re

import numpy as np

from testing.numpy_reference import np as numpy_reference
from zinnia import ZKCircuit

Q32 = 2**32


def numpy_argument(a):
    """An argument as the NumPy reference receives it: an array is copied (the
    kernel may write it); a Python number becomes the NumPy scalar of the
    circuit input's type (int64, float64, bool). It stays a scalar: `np.copy`
    would make a 0-d array, which NumPy takes as an advanced index
    (`m[i][j] = 0` would then write a copy)."""
    if isinstance(a, np.ndarray):
        return np.copy(a)
    if isinstance(a, bool):
        return np.bool_(a)
    if isinstance(a, int):
        return np.int64(a)
    if isinstance(a, float):
        return np.float64(a)
    return a


def compile_kernel(source: str):
    """Compile ``def kernel(...)`` source (``np`` and the Zinnia types in scope)."""
    return ZKCircuit.from_source("kernel", source).compile()


def _statement_id(entry: str) -> int:
    label = entry.partition("=")[0]
    return int(re.search(r"_(\d+)$", label).group(1))


def run_mock(program, *args):
    """Prove on the mock backend. Returns ``(satisfied, outputs)`` where
    ``outputs`` is the full output sequence in return order: the exposed
    publics in exposure order (floats decoded), with the constants the
    compiler elided from the public instance put back at their slots."""
    proof = program.prove(*args, backend="mock")
    exposed = []
    for entry in sorted(proof.public_values, key=_statement_id):
        label, _, raw = entry.partition("=")
        value = int(raw)
        exposed.append(value / Q32 if label.startswith("public_f_") else value)
    elided = program.get_elided_publics()
    slots = [None] * (len(exposed) + len(elided))
    for e in elided:
        slots[e["slot"]] = e["Int"] if "Int" in e else e["Float"]
    rest = iter(exposed)
    outputs = [v if v is not None else next(rest) for v in slots]
    return proof.proof_bytes_hex == "mock_satisfied", outputs


def numpy_kernel(source: str):
    """The same kernel source as a plain NumPy function (annotations and
    decorators such as ``@requires`` dropped), with ``np`` from
    `testing.numpy_reference`."""
    import ast

    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.arg):
            node.annotation = None
        if isinstance(node, ast.FunctionDef):
            node.returns = None
            node.decorator_list = []
    env = {"np": numpy_reference}
    exec(compile(ast.fix_missing_locations(tree), "<kernel>", "exec"), env)
    return env["kernel"]


def assert_matches_numpy(source: str, *args, atol: float = 1e-6):
    """Compile and mock-prove ``source`` on ``args``; the outputs must equal
    NumPy's result position by position (flattened, row-major)."""
    expected = np.asarray(numpy_kernel(source)(*[numpy_argument(a) for a in args]), dtype=float).ravel()
    satisfied, outputs = run_mock(compile_kernel(source), *args)
    assert satisfied, "honest witness rejected"
    np.testing.assert_allclose(np.asarray(outputs, dtype=float), expected, atol=atol, rtol=0)
    return outputs


def mock_publics(proof) -> list[int]:
    """The mock's exposed public values in exposure order (floats in Q32)."""
    return [int(entry.partition("=")[2]) for entry in sorted(proof.public_values, key=_statement_id)]


def run_mock_layout(program, *args):
    """Prove on the mock backend and decode the outputs with the compiler's
    output layout. Returns ``(satisfied, values, shapes)``: the flattened
    output elements in output order and the shape of every output part, as
    ``benchmarking.harness.outputs.decode_with_layout`` reads them."""
    from benchmarking.harness.adapters.zinnia_common import with_public_input_gaps
    from benchmarking.harness.outputs import decode_with_layout

    proof = program.prove(*args, backend="mock")
    inputs = program.get_public_inputs()
    fields = with_public_input_gaps(mock_publics(proof), inputs)
    values, shapes = decode_with_layout(fields, program.get_output_layout(), inputs)
    return proof.proof_bytes_hex == "mock_satisfied", values, shapes


def numpy_outputs(source: str, *args):
    """NumPy's outputs of ``kernel`` (its return value) as ``(values, shapes)``,
    flattened the way the harness compares them (arguments as `numpy_argument`
    makes them)."""
    from benchmarking.harness.outputs import reference_leaves

    leaves, shapes = reference_leaves(numpy_kernel(source)(*[numpy_argument(a) for a in args]))
    return [leaf.value for leaf in leaves], shapes


def numpy_in_place_outputs(source: str, names: list[str], *args):
    """NumPy's outputs of an in-place kernel: the arguments ``names`` after the call."""
    from benchmarking.harness.outputs import reference_leaves
    import inspect

    kernel = numpy_kernel(source)
    copies = [numpy_argument(a) for a in args]
    assert kernel(*copies) is None
    params = list(inspect.signature(kernel).parameters)
    leaves, shapes = reference_leaves(tuple(copies[params.index(n)] for n in names))
    return [leaf.value for leaf in leaves], shapes

