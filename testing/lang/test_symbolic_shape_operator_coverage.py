"""Operator coverage for symbolic/refinement-erased array shapes."""

import pytest

from zinnia import *
from zinnia.debug.exception import ZinniaException


def test_symbolic_static_elementwise_reduction_and_indexing():
    source = """
@requires(lambda: N == 2 and M == 3)
def foo(x: Float[Array, "N M"]):
    y = x + 2.0
    assert y[0, 0] == 3.0
    assert y[1, 2] == 8.0
    assert y.sum() == 33.0
"""

    proof = ZKCircuit.from_source("foo", source).prove(
        np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        backend="mock",
    )
    assert proof.proof_bytes_hex == "mock_satisfied"


def test_symbolic_static_matmul_shared_dimension():
    source = """
@requires(lambda: M == 3)
def foo(a: Float[Array, "N=2 M"], b: Float[Array, "M K=2"]):
    r = a @ b
    assert r[0][0] == 22.0
    assert r[1][1] == 64.0
"""

    a = np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b = np.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    proof = ZKCircuit.from_source("foo", source).prove(a, b, backend="mock")
    assert proof.proof_bytes_hex == "mock_satisfied"


def test_symbolic_bounded_dynamic_elementwise_reduction_and_indexing():
    source = """
def foo(x: Float[Array, "N<=4 M<=3"]):
    y = x + 1.0
    assert y[0, 0] == 2.0
    assert y[1, 1] == 5.0
    assert y.sum() == 14.0
"""

    proof = ZKCircuit.from_source("foo", source).prove(
        np.asarray([[1.0, 2.0], [3.0, 4.0]]),
        backend="mock",
    )
    assert proof.proof_bytes_hex == "mock_satisfied"


def test_symbolic_bounded_dynamic_shape_and_size_use_runtime_metadata():
    source = """
def foo(x: Float[Array, "N<=4 M<=3"]):
    assert x.shape[0] == 2
    assert x.shape[1] == 2
    assert x.size == 4
"""

    proof = ZKCircuit.from_source("foo", source).prove(
        np.asarray([[1.0, 2.0], [3.0, 4.0]]),
        backend="mock",
    )
    assert proof.proof_bytes_hex == "mock_satisfied"


def test_symbolic_bounded_dynamic_axis_reductions_use_runtime_shape():
    source = """
def foo(x: Float[Array, "N<=4 M<=3"]):
    col = x.sum(axis=0)
    row = x.sum(axis=1)
    assert col.shape[0] == 2
    assert col.size == 2
    assert col[0] == 4.0
    assert col[1] == 6.0
    assert row.shape[0] == 2
    assert row.size == 2
    assert row[0] == 3.0
    assert row[1] == 7.0
"""

    proof = ZKCircuit.from_source("foo", source).prove(
        np.asarray([[1.0, 2.0], [3.0, 4.0]]),
        backend="mock",
    )
    assert proof.proof_bytes_hex == "mock_satisfied"


def test_symbolic_bounded_dynamic_mean_uses_runtime_shape():
    source = """
def foo(x: Float[Array, "N<=4 M<=3"]):
    assert x.mean() == 2.5
    col = x.mean(axis=0)
    row = x.mean(axis=1)
    assert col.shape[0] == 2
    assert col[0] == 2.0
    assert col[1] == 3.0
    assert row.shape[0] == 2
    assert row[0] == 1.5
    assert row[1] == 3.5
"""

    proof = ZKCircuit.from_source("foo", source).prove(
        np.asarray([[1.0, 2.0], [3.0, 4.0]]),
        backend="mock",
    )
    assert proof.proof_bytes_hex == "mock_satisfied"


def test_symbolic_bounded_dynamic_shape_methods_and_unary_np_surface():
    source = """
def foo(x: Float[Array, "N<=4 M<=1"]):
    sq = x.squeeze(axis=1)
    assert sq.shape[0] == 2
    assert sq.size == 2
    sw = x.swapaxes(0, 1)
    assert sw.shape[0] == 1
    assert sw.shape[1] == 2
"""

    proof = ZKCircuit.from_source("foo", source).prove(
        np.asarray([[1.0], [3.0]]),
        backend="mock",
    )
    assert proof.proof_bytes_hex == "mock_satisfied"


def test_symbolic_bounded_dynamic_deferred_methods_have_explicit_diagnostics():
    source = """
def foo(x: Float[Array, "N<=4"]):
    y = x.var()
    assert y >= 0.0
"""

    with pytest.raises(ZinniaException, match="DynamicNDArray.var is not yet supported"):
        ZKCircuit.from_source("foo", source).compile()


def test_symbolic_bounded_dynamic_reciprocal_has_explicit_diagnostic():
    source = """
def foo(x: Float[Array, "N<=4"]):
    y = np.reciprocal(x)
    assert y[0] == 1.0
"""

    with pytest.raises(ZinniaException, match="np.reciprocal is not yet supported"):
        ZKCircuit.from_source("foo", source).compile()


@pytest.mark.parametrize("op", ["floor", "ceil", "trunc", "round"])
def test_symbolic_bounded_dynamic_rounding_ops_have_explicit_diagnostics(op):
    source = f"""
def foo(x: Float[Array, "N<=4"]):
    y = np.{op}(x)
    assert y[0] == 1.0
"""

    with pytest.raises(ZinniaException, match=fr"np\.{op} is not yet supported"):
        ZKCircuit.from_source("foo", source).compile()


def test_symbolic_bounded_dynamic_matmul_has_explicit_diagnostic():
    source = """
def foo(x: Float[Array, "N<=2 M<=2"], y: Float[Array, "M<=2 K<=2"]):
    z = x @ y
    assert z[0, 0] == 1.0
"""

    with pytest.raises(ZinniaException, match="DynamicNDArray matmul is not yet supported"):
        ZKCircuit.from_source("foo", source).compile()


def test_symbolic_bounded_dynamic_np_linalg_has_explicit_diagnostic():
    source = """
def foo(x: Float[Array, "N<=2 M<=2"], y: Float[Array, "N<=2 M<=2"]):
    z = np.linalg.verify_inverse(x, y)
    assert z[0, 0] == 1.0
"""

    with pytest.raises(ZinniaException, match=r"np\.linalg\.verify_inverse is not yet supported"):
        ZKCircuit.from_source("foo", source).compile()


@pytest.mark.parametrize("method", ["flatten", "tolist"])
def test_symbolic_bounded_dynamic_flatten_tolist_have_explicit_diagnostics(method):
    source = f"""
def foo(x: Float[Array, "N<=4 M<=3"]):
    y = x.{method}()
    assert len(y) >= 0
"""

    with pytest.raises(
        ZinniaException,
        match=fr"DynamicNDArray\.{method} is not yet supported.*max-buffer padding",
    ):
        ZKCircuit.from_source("foo", source).compile()


def test_symbolic_bounded_dynamic_rejects_per_axis_bound_violation():
    source = """
def foo(x: Float[Array, "N<=2 M<=6"]):
    assert x.sum() == 12.0
"""

    program = ZKCircuit.from_source("foo", source).compile()
    assert program.get_program_inputs()[0].dt == {
        "DynamicNDArray": {
            "dtype": "Float",
            "max_length": 12,
            "max_rank": 2,
            "max_shape": [2, 6],
        }
    }
    with pytest.raises(ZinniaException, match="Axis 0 expected at most 2"):
        program.prove(np.ones((3, 4)), backend="mock")


def test_dynamic_ndarray_metadata_assertion_is_live_in_mock_backend():
    source = """
def foo(x: Float[Array, "N<=4 M<=3"]):
    assert x.sum() == 4.0
"""

    program = ZKCircuit.from_source("foo", source).compile()
    witness = np.ones((2, 2))
    proof = program.prove(witness, backend="mock")
    assert proof.proof_bytes_hex == "mock_satisfied"

    # Corrupt only the metadata after Python witness parsing. This verifies
    # that the backend assertion is live, independent of parser validation.
    from zinnia.compile._bridge import prove_circuit
    from zinnia.exec.input_parser import build_circuit_inputs
    from zinnia.exec.proof_result import ZKProofResult
    import json

    witness_json = build_circuit_inputs(program.get_program_inputs(), (witness,))
    witness_json["params"][0]["value"]["Array"]["runtime_shape"] = [5, 2]
    proof = ZKProofResult.from_json(
        prove_circuit(
            program._compiled_ir,
            json.dumps(witness_json),
            {},
            "mock",
            None,
        )
    )
    assert proof.proof_bytes_hex.startswith("mock_unsatisfied:")
    assert "axis 0 expected 0..=4, got 5" in proof.proof_bytes_hex


def test_dynamic_ndarray_metadata_assertion_is_live_in_halo2_base_backend():
    """Code review C15: `assert_dynamic_ndarray_meta` on halo2-base was a
    no-op (`Ok(())`) at commit 4b69c08 — it constrained nothing, so a prover
    could witness a DynamicNDArray runtime shape outside its allocation
    envelope and the benchmarked backend would prove AND verify it. The mock
    backend caught this (test above); halo2-base did not.

    The fix hoists the legacy lowering into a `Synthesizer` trait default,
    which halo2-base now inherits. This test pins the contract end-to-end: an
    honest in-envelope shape proves and verifies, while a corrupted runtime
    shape (axis past its per-axis bound) produces a proof that fails to
    verify. At 4b69c08 the corrupted proof verified `True`.
    """
    import json

    from zinnia.compile._bridge import prove_circuit
    from zinnia.exec.input_parser import build_circuit_inputs

    source = """
def foo(x: Float[Array, "N<=4 M<=3"]):
    assert x.sum() == 4.0
"""
    program = ZKCircuit.from_source("foo", source).compile()
    witness = np.ones((2, 2))

    # Honest in-envelope shape proves and verifies.
    honest = program.prove(witness, backend="halo2-base")
    assert program.verify(honest) is True

    # Corrupt only the runtime-shape metadata after Python witness parsing:
    # claim axis 0 = 5, which exceeds its per-axis bound 4. The payload still
    # carries 4 ones so the value assert is satisfiable; soundness now rests
    # entirely on the meta-assert the fix restored.
    witness_json = build_circuit_inputs(program.get_program_inputs(), (witness,))
    witness_json["params"][0]["value"]["Array"]["runtime_shape"] = [5, 2]
    # The constraint is live: since C18 (always-on witness-satisfaction
    # check) halo2-base REFUSES the out-of-envelope witness at prove time —
    # the restored meta-assert lowers to an `assert` gate cell pinned to 0,
    # which the always-on check rejects before `create_proof`. This is a
    # strictly stronger contract than the prove-then-verify-false form this
    # test originally pinned (at 4b69c08 the meta-assert was a no-op and the
    # tampered artifact verified True).
    with pytest.raises(RuntimeError, match="Witness does not satisfy"):
        prove_circuit(
            program._compiled_ir,
            json.dumps(witness_json),
            {},
            "halo2-base",
            None,
        )


def test_symbolic_bounded_dynamic_slice_reshape_and_transpose():
    source = """
def foo(x: Float[Array, "N<=6"]):
    y = x.reshape((2, 3)).T
    assert y[0, 1] == 4.0
    assert y.sum() == 21.0
"""

    proof = ZKCircuit.from_source("foo", source).prove(
        np.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        backend="mock",
    )
    assert proof.proof_bytes_hex == "mock_satisfied"


def test_symbolic_bounded_dynamic_static_reshape_rejects_short_runtime_length():
    source = """
def foo(x: Float[Array, "N<=6"]):
    y = x.reshape((2, 3))
    assert y.sum() == 10.0
"""

    proof = ZKCircuit.from_source("foo", source).prove(
        np.asarray([1.0, 2.0, 3.0, 4.0]),
        backend="mock",
    )
    assert proof.proof_bytes_hex.startswith("mock_unsatisfied:")


def test_symbolic_bounded_dynamic_concat_stack_and_split():
    concat_source = """
def foo(x: Float[Array, "N<=4"], y: Float[Array, "M<=4"]):
    z = np.concatenate([x, y])
    assert z.sum() == 10.0
"""
    concat_proof = ZKCircuit.from_source("foo", concat_source).prove(
        np.asarray([1.0, 2.0]),
        np.asarray([3.0, 4.0]),
        backend="mock",
    )
    assert concat_proof.proof_bytes_hex == "mock_satisfied"

    stack_source = """
def foo(x: Float[Array, "N<=4"], y: Float[Array, "N<=4"]):
    z = np.stack([x, y])
    assert z[0, 1] == 2.0
    assert z[1, 0] == 3.0
"""
    stack_proof = ZKCircuit.from_source("foo", stack_source).prove(
        np.asarray([1.0, 2.0]),
        np.asarray([3.0, 4.0]),
        backend="mock",
    )
    assert stack_proof.proof_bytes_hex == "mock_satisfied"

    split_source = """
def foo(x: Float[Array, "N<=6"]):
    parts = np.split(x, 2)
    assert parts[0].sum() == 6.0
    assert parts[1].sum() == 15.0
"""
    split_proof = ZKCircuit.from_source("foo", split_source).prove(
        np.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        backend="mock",
    )
    assert split_proof.proof_bytes_hex == "mock_satisfied"


def test_symbolic_bool_and_int_arrays():
    int_source = """
@requires(lambda: N == 4)
def foo(x: Int[Array, "N"]):
    y = x + 2
    assert y.sum() == 18
"""
    int_proof = ZKCircuit.from_source("foo", int_source).prove(
        np.asarray([1, 2, 3, 4]),
        backend="mock",
    )
    assert int_proof.proof_bytes_hex == "mock_satisfied"

    bool_source = """
@requires(lambda: N == 3)
def foo(mask: Bool[Array, "N"]):
    assert mask.any() == True
    assert mask.all() == False
"""
    bool_proof = ZKCircuit.from_source("foo", bool_source).prove(
        np.asarray([True, False, True]),
        backend="mock",
    )
    assert bool_proof.proof_bytes_hex == "mock_satisfied"
