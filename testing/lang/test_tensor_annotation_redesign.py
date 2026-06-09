"""Literal-static slice of the jaxtyping-aligned tensor annotation surface."""

import pytest

from zinnia import *
from zinnia.compile.zinnia_compiler import ZinniaCompiler
from zinnia.debug.exception import InvalidAnnotationException, ZinniaException
from zinnia.exec.input_parser import build_circuit_inputs


SYMBOLIC_TEST_N = 2
SYMBOLIC_TEST_M = 2


def test_scalar_aliases_keep_builtin_runtime_shape():
    assert isinstance(1, Integer)
    assert isinstance(1, Int)
    assert isinstance(1.5, Float)
    assert isinstance(True, Bool)
    assert isinstance(1 + 2j, Complex)
    assert issubclass(int, Integer)
    assert issubclass(float, Float)
    assert issubclass(bool, Bool)
    assert issubclass(complex, Complex)


def test_float_array_literal_static_shape_compiles():
    @zk_circuit
    def foo(x: Float[Array, "2 2"]):
        assert x.sum() == 10.0

    assert foo(np.asarray([[1.0, 2.0], [3.0, 4.0]]))


def test_float_array_exact_symbolic_shape_concretizes_before_input_lowering():
    @zk_circuit
    def foo(x: Float[Array, "N=2 M=2"]):
        assert x.sum() == 10.0

    assert foo(np.asarray([[1.0, 2.0], [3.0, 4.0]]))


def test_exact_symbolic_shape_parser_emits_symbolic_descriptor_with_bindings():
    source = """
def foo(x: Float[Array, "N=2 M=3"]):
    assert x[0, 0] == x[0, 0]
"""

    ast_dict = ZinniaCompiler.circuit_ast_parse(source, "foo")
    input_dt = ast_dict["inputs"][0]["annotation"]["dt"]
    assert input_dt == {
        "SymbolicNDArray": {
            "shape": ["N", "M"],
            "dtype": "Float",
            "bindings": {"N": 2, "M": 3},
        }
    }


def test_unresolved_symbolic_shape_parser_emits_symbolic_descriptor():
    source = """
def foo(x: Float[Array, "N M"]):
    assert x[0, 0] == x[0, 0]
"""

    ast_dict = ZinniaCompiler.circuit_ast_parse(source, "foo")
    input_dt = ast_dict["inputs"][0]["annotation"]["dt"]
    assert input_dt == {
        "SymbolicNDArray": {"shape": ["N", "M"], "dtype": "Float", "bindings": {}}
    }


def test_exact_symbolic_binding_flows_across_later_annotations():
    source = """
def foo(a: Float[Array, "N=2 M=3"], b: Float[Array, "M K=4"]):
    assert a[0, 0] == a[0, 0]
    assert b[0, 0] == b[0, 0]
"""

    ast_dict = ZinniaCompiler.circuit_ast_parse(source, "foo")
    assert ast_dict["inputs"][1]["annotation"]["dt"] == {
        "SymbolicNDArray": {
            "shape": [3, "K"],
            "dtype": "Float",
            "bindings": {"K": 4},
        }
    }


def test_conflicting_exact_symbolic_bindings_reject():
    source = """
def foo(a: Float[Array, "N=2"], b: Float[Array, "N=3"]):
    assert a[0] == a[0]
    assert b[0] == b[0]
"""

    with pytest.raises(InvalidAnnotationException) as exc_info:
        ZinniaCompiler.circuit_ast_parse(source, "foo")
    assert "Conflicting exact shape bindings for `N`" in str(exc_info.value)


def test_int_and_bool_array_literal_static_shapes_compile():
    @zk_circuit
    def int_array(x: Int[Array, "4"]):
        assert x.sum() == 10

    @zk_circuit
    def bool_array(mask: Bool[Array, "2 2"]):
        assert mask.sum() == 2

    assert int_array(np.asarray([1, 2, 3, 4]))
    assert bool_array(np.asarray([[True, False], [False, True]]))


def test_complex_array_literal_static_shape_uses_same_parser_path():
    @zk_circuit
    def foo(x: Complex[Array, "4"]):
        assert x[0].real == x[0].real

    ZKCircuit.from_method(foo).compile()


def test_old_ndarray_syntax_still_compiles():
    @zk_circuit
    def foo(x: NDArray[Integer, 2, 2]):
        assert x.sum() == 10

    assert foo(np.asarray([[1, 2], [3, 4]]))


def test_old_ndarray_syntax_accepts_module_constant_shape_symbols():
    @zk_circuit
    def foo(x: NDArray[Float, SYMBOLIC_TEST_N, SYMBOLIC_TEST_M]):
        assert x.sum() == 10.0

    assert foo(np.asarray([[1.0, 2.0], [3.0, 4.0]]))


def test_old_and_new_array_annotations_can_mix():
    @zk_circuit
    def foo(a: NDArray[Integer, 4], b: Int[Array, "4"]):
        assert a.sum() == b.sum()

    assert foo(np.asarray([1, 2, 3, 4]), np.asarray([4, 3, 2, 1]))


@zk_circuit
def _symbolic_shape(x: Float[Array, "M N"]):
    assert x[0] == x[0]


@zk_circuit
def _unresolved_arithmetic_shape(x: Float[Array, "M+1"]):
    assert x[0] == x[0]


@pytest.mark.parametrize("fn", [_symbolic_shape])
def test_unresolved_symbolic_shapes_fail_at_backend_erasure_gate(fn):
    with pytest.raises(ZinniaException) as exc_info:
        ZKCircuit.from_method(fn).compile()
    msg = str(exc_info.value)
    assert "must be resolved to an exact value or finite bound" in msg


@pytest.mark.parametrize("fn", [_unresolved_arithmetic_shape])
def test_unresolved_arithmetic_shapes_reject(fn):
    with pytest.raises(ZinniaException) as exc_info:
        ZKCircuit.from_method(fn).compile()
    msg = str(exc_info.value)
    assert "SymbolicNDArray shape dimensions must be integers, unresolved symbols, exact bindings, or bounded symbols" in msg


def test_shape_arithmetic_expression_resolves_from_prior_exact_binding():
    @zk_circuit
    def foo(a: Float[Array, "N=2"], b: Float[Array, "N+1"]):
        assert b.sum() == 6.0

    assert foo(np.asarray([1.0, 2.0]), np.asarray([1.0, 2.0, 3.0]))


def test_shape_arithmetic_expression_parser_emits_static_dimension():
    source = """
def foo(a: Float[Array, "N=2"], b: Float[Array, "N+1"]):
    assert b[0] == b[0]
"""

    ast_dict = ZinniaCompiler.circuit_ast_parse(source, "foo")
    assert ast_dict["inputs"][1]["annotation"]["dt"] == {
        "NDArray": {"shape": [3], "dtype": "Float"}
    }


def test_shape_param_exact_fact_concretizes_static_array():
    source = """
@requires(lambda: N == 2)
def foo(x: Float[Array, "N"]):
    assert x.sum() == 3.0
"""

    program = ZKCircuit.from_source("foo", source).compile()
    assert program.get_program_inputs()[0].dt == {
        "NDArray": {"shape": [2], "dtype": "Float"}
    }
    assert program.prove(np.asarray([1.0, 2.0]), backend="mock").proof_bytes_hex == "mock_satisfied"


def test_shape_param_bound_fact_concretizes_dynarray_bound():
    source = """
@requires(lambda: 1 <= N <= 4)
def foo(x: Float[DynArray, "N"]):
    assert x[0] == x[0]
"""

    program = ZKCircuit.from_source("foo", source).compile()
    assert program.get_program_inputs()[0].dt == {
        "DynamicNDArray": {"dtype": "Float", "max_length": 4, "max_rank": 1, "max_shape": [4]}
    }
    assert program.prove(np.asarray([1.0, 2.0]), backend="mock").proof_bytes_hex == "mock_satisfied"


def test_shape_param_bound_fact_uses_smt_bound_solver():
    source = """
@requires(lambda: 1 <= N <= 512)
def foo(x: Float[DynArray, "N"]):
    assert x[0] == x[0]
"""

    program = ZKCircuit.from_source("foo", source).compile()
    assert program.get_program_inputs()[0].dt == {
        "DynamicNDArray": {"dtype": "Float", "max_length": 512, "max_rank": 1, "max_shape": [512]}
    }


def test_shape_param_exact_fact_supports_matmul_shared_dimension():
    source = """
@requires(lambda: M == 3)
def foo(a: Float[Array, "N=2 M"], b: Float[Array, "M K=2"]):
    r = a @ b
    assert r[0][0] == 22.0
    assert r[1][1] == 64.0
"""

    program = ZKCircuit.from_source("foo", source).compile()
    assert [pi.dt for pi in program.get_program_inputs()] == [
        {"NDArray": {"shape": [2, 3], "dtype": "Float"}},
        {"NDArray": {"shape": [3, 2], "dtype": "Float"}},
    ]
    a = np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b = np.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    assert program.prove(a, b, backend="mock").proof_bytes_hex == "mock_satisfied"


def test_array_bounded_symbolic_annotation_preserves_symbolic_descriptor_until_compile():
    source = """
def foo(x: Float[Array, "N<=8"]):
    assert x[0] == x[0]
"""

    ast_dict = ZinniaCompiler.circuit_ast_parse(source, "foo")
    input_dt = ast_dict["inputs"][0]["annotation"]["dt"]
    assert input_dt == {
        "SymbolicNDArray": {
            "shape": [{"symbol": "N", "upper": 8}],
            "dtype": "Float",
            "bindings": {},
        }
    }

    program = ZKCircuit.from_source("foo", source).compile()
    assert program.get_program_inputs()[0].dt == {
        "DynamicNDArray": {"dtype": "Float", "max_length": 8, "max_rank": 1, "max_shape": [8]}
    }


def test_bounded_symbolic_array_expression_uses_known_public_bound():
    source = """
def foo(a: Float[Array, "N=4"], x: Float[Array, "K<=N*2"]):
    assert x[0] == x[0]
"""

    ast_dict = ZinniaCompiler.circuit_ast_parse(source, "foo")
    input_dt = ast_dict["inputs"][1]["annotation"]["dt"]
    assert input_dt == {
        "SymbolicNDArray": {
            "shape": [{"symbol": "K", "upper": 8}],
            "dtype": "Float",
            "bindings": {},
        }
    }

    program = ZKCircuit.from_source("foo", source).compile()
    assert program.get_program_inputs()[1].dt == {
        "DynamicNDArray": {"dtype": "Float", "max_length": 8, "max_rank": 1, "max_shape": [8]}
    }


def test_bounded_dynarray_expression_uses_known_public_bound():
    source = """
def foo(a: Float[Array, "N=4"], x: Float[DynArray, "K<=N*2"]):
    assert x[0] == x[0]
"""

    ast_dict = ZinniaCompiler.circuit_ast_parse(source, "foo")
    input_dt = ast_dict["inputs"][1]["annotation"]["dt"]
    assert input_dt == {"DynamicNDArray": {"dtype": "Float", "max_length": 8, "max_rank": 1}}


def test_dynarray_bounded_annotation_compiles():
    @zk_circuit
    def foo(x: Float[DynArray, "N<=4"]):
        assert x[0] == x[0]

    ZKCircuit.from_method(foo).compile()


def test_dynarray_bounded_annotation_parser_emits_dynamic_ndarray():
    source = """
def foo(x: Float[DynArray, "N<=8"]):
    assert x[0] == x[0]
"""

    ast_dict = ZinniaCompiler.circuit_ast_parse(source, "foo")
    input_dt = ast_dict["inputs"][0]["annotation"]["dt"]
    assert input_dt == {"DynamicNDArray": {"dtype": "Float", "max_length": 8, "max_rank": 1}}


def test_dynarray_bounded_witness_uses_public_max_bound_and_runtime_shape():
    @zk_circuit
    def foo(x: Float[DynArray, "N<=4"]):
        assert x[0] == x[0]

    circuit = ZKCircuit.from_method(foo)
    circuit.compile()
    witness = build_circuit_inputs(
        circuit.program.get_program_inputs(), (np.asarray([1.0, 2.0, 3.0]),)
    )
    array = witness["params"][0]["value"]["Array"]
    assert array["shape"] == [4]
    assert array["runtime_shape"] == [3]
    assert [elem["Float"] for elem in array["elements"]] == [1.0, 2.0, 3.0, 0.0]


def test_dynarray_bounded_witness_rejects_values_above_public_bound():
    @zk_circuit
    def foo(x: Float[DynArray, "N<=4"]):
        assert x[0] == x[0]

    circuit = ZKCircuit.from_method(foo)
    circuit.compile()
    with pytest.raises(ZinniaException) as exc_info:
        build_circuit_inputs(circuit.program.get_program_inputs(), (np.asarray([1.0] * 5),))
    assert "Expected at most 4 elements" in str(exc_info.value)


def test_dynarray_annotation_without_bound_rejects():
    @zk_circuit
    def foo(x: Float[DynArray, "N"]):
        assert x[0] == x[0]

    with pytest.raises(ZinniaException) as exc_info:
        ZKCircuit.from_method(foo).compile()
    assert "must be resolved to an exact value or finite bound" in str(exc_info.value)


def test_vec_annotation_is_deferred():
    @zk_circuit
    def foo(xs: Vec[Int, 4]):
        assert xs[0] == xs[0]

    with pytest.raises(ZinniaException) as exc_info:
        ZKCircuit.from_method(foo).compile()
    assert "Vec" in str(exc_info.value)
    assert "deferred" in str(exc_info.value)
