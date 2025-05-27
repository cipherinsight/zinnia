import pytest

from zinnia import *


def test_program_compile():
    @zk_circuit
    def foo(x: Public[Integer], y: Private[Integer]):
        assert x == y

    foo_circuit = ZKCircuit.from_method(foo)
    program = foo_circuit.compile()
    assert isinstance(program, ZKCompiledProgram)


def test_program_argparse():
    @zk_external
    def external_addition(x, y) -> Integer:
        return x + y

    @zk_circuit
    def foo(x: Public[Integer], y: Private[Integer]):
        number = external_addition(x, y)
        assert number == x + y

    the_external = ZKExternalFunc.from_method(external_addition)
    foo_circuit = ZKCircuit.from_method(foo, externals=[the_external])
    program = foo_circuit.compile()
    parsed_inputs = program.argparse(3, 4)
    assert isinstance(parsed_inputs, ZKParsedInput)
    [entry_0, entry_1, entry_2] = parsed_inputs.entries
    assert entry_0.indices == (0, 0) and entry_0.value == 3
    assert entry_1.indices == (0, 1) and entry_1.value == 4
    assert entry_2.indices == (1, ) and entry_2.value == 7


def test_program_serialization():
    @zk_external
    def external_addition(x, y) -> Integer:
        return x + y

    @zk_circuit
    def foo(x: Public[Integer], y: Private[Integer]):
        number = external_addition(x, y)
        assert number == x + y

    the_external = ZKExternalFunc.from_method(external_addition)
    foo_circuit = ZKCircuit.from_method(foo, externals=[the_external])
    program = foo_circuit.compile()
    serialized = program.serialize()
    assert len(serialized) > 0 and isinstance(serialized, str)


def test_program_deserialization():
    @zk_external
    def external_addition(x, y) -> Integer:
        return x + y

    @zk_circuit
    def foo(x: Public[Integer], y: Private[Integer]):
        number = external_addition(x, y)
        assert number == x + y

    the_external = ZKExternalFunc.from_method(external_addition)
    foo_circuit = ZKCircuit.from_method(foo, externals=[the_external])
    program = foo_circuit.compile()
    serialized = program.serialize()
    assert len(serialized) > 0 and isinstance(serialized, str)
    program = ZKCompiledProgram.deserialize(serialized, external_funcs=[the_external])
    exec_ctx = program.get_execution_context()
    mock_executor = MockProgramExecutor(exec_ctx, program, ZinniaConfig())
    assert mock_executor.exec(3, 4)


def test_program_deserialization_error_external_provided():
    @zk_external
    def external_addition(x, y) -> Integer:
        return x + y

    @zk_external
    def external_multiplication(x, y) -> Integer:
        return x * y

    @zk_circuit
    def foo(x: Public[Integer], y: Private[Integer]):
        number = external_addition(x, y)
        assert number == x + y

    the_external_1 = ZKExternalFunc.from_method(external_addition)
    the_external_2 = ZKExternalFunc.from_method(external_multiplication)
    foo_circuit = ZKCircuit.from_method(foo, externals=[the_external_1])
    program = foo_circuit.compile()
    serialized = program.serialize()
    assert len(serialized) > 0 and isinstance(serialized, str)
    with pytest.raises(ZinniaException) as e:
        program = ZKCompiledProgram.deserialize(serialized, external_funcs=[the_external_1, the_external_2])
        exec_ctx = program.get_execution_context()
        mock_executor = MockProgramExecutor(exec_ctx, program, ZinniaConfig())
        assert mock_executor.exec(3, 4)
    assert "External function external_multiplication provided, but not expected" in str(e)
    with pytest.raises(ZinniaException) as e:
        program = ZKCompiledProgram.deserialize(serialized)
        exec_ctx = program.get_execution_context()
        mock_executor = MockProgramExecutor(exec_ctx, program, ZinniaConfig())
        assert mock_executor.exec(3, 4)
    assert "External function external_addition expected, but not provided" in str(e)
