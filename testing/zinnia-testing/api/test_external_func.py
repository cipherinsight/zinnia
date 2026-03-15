import random

from zinnia import *


def test_external_annotator():
    @zk_external
    def foo() -> Integer:
        return random.randint(0, 10)

    assert isinstance(foo, ZKExternalFunc)
    assert foo.get_name() == "foo"


def test_external_callable_outside_circuit():
    @zk_external
    def foo() -> Integer:
        return random.randint(0, 10)

    assert 0 <= foo() <= 10


def test_external_from_method_1():
    def foo() -> Integer:
        return random.randint(0, 10)

    foo_external = ZKExternalFunc.from_method(foo)
    assert isinstance(foo_external, ZKExternalFunc)
    assert foo_external.get_name() == "foo"


def test_external_from_method_2():
    @zk_external
    def foo() -> Integer:
        return random.randint(0, 10)

    foo_external = ZKExternalFunc.from_method(foo)
    assert isinstance(foo_external, ZKExternalFunc)
    assert foo_external.get_name() == "foo"
