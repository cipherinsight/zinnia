"""Test helpers: run a kernel on the mock backend and read its outputs by position.

The mock backend reports each exposed public as ``public_<i|f>_<statement id>``
and returns them unordered. Sorting by statement id restores the order in which
the circuit exposed them, which is the order of the flattened return value.
Float publics are Q32 fixed point (value * 2**32).
"""
import re

import numpy as np

from zinnia import ZKCircuit

Q32 = 2**32


def compile_kernel(source: str):
    """Compile ``def kernel(...)`` source (``np`` and the Zinnia types in scope)."""
    return ZKCircuit.from_source("kernel", source).compile()


def _statement_id(entry: str) -> int:
    label = entry.partition("=")[0]
    return int(re.search(r"_(\d+)$", label).group(1))


def run_mock(program, *args):
    """Prove on the mock backend. Returns ``(satisfied, outputs)`` where
    ``outputs`` are the exposed publics in exposure order (floats decoded)."""
    proof = program.prove(*args, backend="mock")
    outputs = []
    for entry in sorted(proof.public_values, key=_statement_id):
        label, _, raw = entry.partition("=")
        value = int(raw)
        outputs.append(value / Q32 if label.startswith("public_f_") else value)
    return proof.proof_bytes_hex == "mock_satisfied", outputs


def numpy_kernel(source: str):
    """The same kernel source as a plain NumPy function (annotations dropped)."""
    import ast

    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.arg):
            node.annotation = None
        if isinstance(node, ast.FunctionDef):
            node.returns = None
    env = {"np": np}
    exec(compile(ast.fix_missing_locations(tree), "<kernel>", "exec"), env)
    return env["kernel"]


def assert_matches_numpy(source: str, *args, atol: float = 1e-6):
    """Compile and mock-prove ``source`` on ``args``; the outputs must equal
    NumPy's result position by position (flattened, row-major)."""
    expected = np.asarray(numpy_kernel(source)(*[np.copy(a) for a in args]), dtype=float).ravel()
    satisfied, outputs = run_mock(compile_kernel(source), *args)
    assert satisfied, "honest witness rejected"
    np.testing.assert_allclose(np.asarray(outputs, dtype=float), expected, atol=atol, rtol=0)
    return outputs
