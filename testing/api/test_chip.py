import pytest

from zinnia import *


def test_chip_annotator():
    @zk_chip
    def foo() -> Integer:
        return 13

    assert isinstance(foo, ZKChip)
    assert foo.get_name() == "foo"


def test_chip_not_callable_outside_circuit():
    @zk_chip
    def foo() -> Integer:
        return 13

    with pytest.raises(ZinniaException) as e:
        assert foo() == 13
    assert "is not callable outside of a circuit" in str(e)


def test_chip_from_method_1():
    def foo() -> Integer:
        return 13

    foo_chip = ZKChip.from_method(foo)
    assert isinstance(foo_chip, ZKChip)
    assert foo_chip.get_name() == "foo"


def test_chip_from_method_2():
    @zk_chip
    def foo() -> Integer:
        return 13

    foo_chip = ZKChip.from_method(foo)
    assert isinstance(foo_chip, ZKChip)
    assert foo_chip.get_name() == "foo"


def test_chip_from_source():
    source = "def foo() -> Integer:    return 13"
    foo_chip = ZKChip.from_source("foo", source)
    assert isinstance(foo_chip, ZKChip)
    assert foo_chip.get_name() == "foo"


def test_compiled_program_name_not_shadowed_by_chip():
    """Regression (C11): compiling a circuit that uses a chip must stamp the
    program with the circuit's name, not the (last) chip's name. The compiler
    used to reuse the `name` parameter as the chip-loop variable, leaking the
    chip name into ZKCompiledProgram.get_program_name()."""
    double = ZKChip.from_source("double", "def double(x) -> Integer:    return x * 2")

    @zk_circuit
    def add_doubled(a: Public[Integer], b: Private[Integer]):
        assert double(a) + double(b) == 30

    program = ZKCircuit.from_method(add_doubled, chips=[double]).compile()
    # The chip is named "double"; the circuit is "add_doubled".
    assert program.get_program_name() == "add_doubled"
