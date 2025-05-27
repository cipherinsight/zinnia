import random

from zinnia import *


def test_circuit_annotator():
    # Note: `foo` will not be a `ZKCircuit` instance after being annotated by `zk_circuit`
    @zk_circuit
    def foo(x: Public[Integer], y: Private[Integer]):
        assert x == y


def test_circuit_callable_outside_circuit():
    @zk_circuit
    def foo(x: Public[Integer], y: Private[Integer]):
        assert x == y

    result = foo(1, 1)
    assert isinstance(result, ZKExecResult)
    assert result


def test_circuit_from_method_1():
    def foo(x: Public[Integer], y: Private[Integer]):
        assert x == y

    foo_circuit = ZKCircuit.from_method(foo)
    assert isinstance(foo_circuit, ZKCircuit)
    assert foo_circuit.get_name() == "foo"
    result = foo_circuit(1, 1)
    assert isinstance(result, ZKExecResult)
    assert result


def test_circuit_from_method_2():
    @zk_circuit
    def foo(x: Public[Integer], y: Private[Integer]):
        assert x == y

    foo_circuit = ZKCircuit.from_method(foo)
    assert isinstance(foo_circuit, ZKCircuit)
    assert foo_circuit.get_name() == "foo"
    result = foo_circuit(1, 1)
    assert isinstance(result, ZKExecResult)
    assert result


def test_circuit_from_source():
    source = "def foo(x: Public[Integer], y: Private[Integer]):    assert x == y"
    foo_circuit = ZKCircuit.from_source('foo', source)
    assert isinstance(foo_circuit, ZKCircuit)
    assert foo_circuit.get_name() == "foo"
    result = foo_circuit(1, 1)
    assert isinstance(result, ZKExecResult)
    assert result


def test_circuit_with_chips_1():
    @zk_chip
    def is_thirteen(x) -> Integer:
        return x == 13

    @zk_circuit
    def foo(x: Public[Integer], y: Private[Integer]):
        assert is_thirteen(x + y)

    assert foo(1, 12)


def test_circuit_with_chips_2():
    @zk_chip
    def is_thirteen(x) -> Integer:
        return x == 13

    @zk_circuit
    def foo(x: Public[Integer], y: Private[Integer]):
        assert is_thirteen(x + y)

    is_13_chip = ZKChip.from_method(is_thirteen)
    foo_circuit = ZKCircuit.from_method(foo, chips=[is_13_chip])
    assert foo_circuit(1, 12)


def test_circuit_with_externals_1():
    @zk_external
    def get_a_random_number(x, y) -> Integer:
        return random.randint(x, y)

    @zk_circuit
    def foo(x: Public[Integer], y: Private[Integer]):
        number = get_a_random_number(x, y)
        assert x <= number <= y

    for _ in range(100):
        assert foo(0, 10)


def test_circuit_with_externals_2():
    @zk_external
    def get_a_random_number(x, y) -> Integer:
        return random.randint(x, y)

    @zk_circuit
    def foo(x: Public[Integer], y: Private[Integer]):
        number = get_a_random_number(x, y)
        assert x <= number <= y

    the_external = ZKExternalFunc.from_method(get_a_random_number)
    foo_circuit = ZKCircuit.from_method(foo, externals=[the_external])
    for _ in range(100):
        assert foo_circuit(0, 10)
