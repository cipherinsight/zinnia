"""Regression tests for C07: a misspelled / absent input name in the witness
must surface a clear "missing witness value" error instead of silently
substituting zero (which used to yield a consistent all-zeros run that proved
and verified, masking the user's typo).

The preprocessor previously resolved an unknown input path with
`unwrap_or(Fp::zero())`. The fix propagates the `witness_missing` error so the
preprocess stage fails loudly. The structural-emit path below exercises ONLY
preprocess + circuit emission (no synthesizer `read_input` that would otherwise
catch the miss), so it pins the preprocessor's behaviour directly.
"""
import json

import pytest

from zinnia import zk_circuit
from zinnia.api.zk_circuit import ZKCircuit
from zinnia.exec.input_parser import build_circuit_inputs
from zinnia.compile._bridge import prove_circuit, emit_structural_circuit


def _compile_add_one():
    @zk_circuit
    def add_one(x: int):
        # The assert keeps the `x` ReadInteger live so it is not elided.
        assert x > 0
        _zinnia_result = x + 1

    return ZKCircuit.from_method(add_one).compile()


def test_misspelled_input_name_errors_on_prove():
    """A witness whose param name does not match the circuit input must raise a
    clear witness-missing error at prove time, not produce a silent proof."""
    prog = _compile_add_one()
    witness = build_circuit_inputs(prog.program_inputs, (5,))

    # Sanity: the correctly-named witness proves fine.
    artifact = prove_circuit(prog._compiled_ir, json.dumps(witness), {}, "mock", None)
    assert artifact

    # Misspell the input name.
    witness["params"][0]["name"] = "x_typo"
    with pytest.raises(Exception) as excinfo:
        prove_circuit(prog._compiled_ir, json.dumps(witness), {}, "mock", None)
    msg = str(excinfo.value)
    assert "witness" in msg.lower() and "x" in msg, (
        f"expected a missing-witness error mentioning 'x', got: {msg!r}"
    )


def test_misspelled_input_name_errors_in_preprocess_only_path():
    """`emit_structural_circuit` runs preprocess + emission with no synthesizer
    read_input fallback, so it pins the PREPROCESSOR's strictness: a missing
    input path must error rather than silently feeding zeros into emission."""
    prog = _compile_add_one()
    witness = build_circuit_inputs(prog.program_inputs, (5,))
    witness["params"][0]["name"] = "x_typo"

    with pytest.raises(Exception) as excinfo:
        emit_structural_circuit(
            prog._compiled_ir, json.dumps(witness), {}, "noir", None
        )
    msg = str(excinfo.value)
    assert "witness" in msg.lower() and "x" in msg, (
        f"expected a missing-witness error mentioning 'x', got: {msg!r}"
    )
