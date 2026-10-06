"""Test helper: a kernel's outputs equal NumPy's on the mock, and halo2-base and
Noir prove exactly the mock's values.

The mock is compared with NumPy (values within ``atol``, shapes exactly, in
output order). The circuit backends are compared with the mock field element
by field element: a halo2-base proof that verifies, and the values Noir's
``main`` returns under ``nargo execute``.
"""
import json
import re
import shutil
from pathlib import Path

import numpy as np

from testing.lang.mock_outputs import compile_kernel, numpy_outputs, run_mock_layout

BN254_P = 0x30644E72E131A029B85045B68181585D2833E84879B9709143E1F593F0000001
NOIR_PARAMS = {"k": 0, "precision_bits": 32, "lookup_bits": 8}


def nargo_available() -> bool:
    return (Path.home() / ".nargo" / "bin" / "nargo").is_file() or shutil.which("nargo") is not None


def mock_field_publics(program, *args) -> list[int]:
    """The mock's exposed publics as field elements, in statement order."""
    proof = program.prove(*args, backend="mock")
    assert proof.proof_bytes_hex == "mock_satisfied", "honest witness rejected by the mock"
    by_statement = {}
    for entry in proof.public_values:
        label, _, raw = entry.partition("=")
        by_statement[int(re.search(r"_(\d+)$", label).group(1))] = int(raw) % BN254_P
    return [by_statement[k] for k in sorted(by_statement)]


def halo2_field_publics(program, *args) -> list[int]:
    proof = program.prove(*args, backend="halo2-base")
    assert program.verify(proof), "halo2-base proof does not verify"
    return [int.from_bytes(bytes.fromhex(h), "little") for h in proof.public_values]


def noir_field_publics(program, *args) -> list[int]:
    from zinnia._zinnia_core import execute_noir_circuit
    from zinnia.exec.input_parser import build_circuit_inputs

    witness = build_circuit_inputs(program.program_inputs, args)
    returned = execute_noir_circuit(program._compiled_ir, json.dumps(witness), {}, json.dumps(NOIR_PARAMS))
    return [int(v, 16) % BN254_P for v in json.loads(returned)]


def assert_all_backends_match_numpy(source: str, *args, atol: float = 1e-6, noir: bool = True):
    """``source`` defines ``def kernel(...)``; see the module docstring."""
    program = compile_kernel(source)
    satisfied, values, shapes = run_mock_layout(program, *args)
    assert satisfied, "honest witness rejected by the mock"
    expected_values, expected_shapes = numpy_outputs(source, *args)
    assert shapes == expected_shapes
    np.testing.assert_allclose(np.asarray(values, dtype=float), np.asarray(expected_values, dtype=float),
                               atol=atol, rtol=0)
    mock = mock_field_publics(program, *args)
    assert halo2_field_publics(program, *args) == mock, "halo2-base differs from the mock"
    if noir and nargo_available():
        assert noir_field_publics(program, *args) == mock, "Noir differs from the mock"
    return values
