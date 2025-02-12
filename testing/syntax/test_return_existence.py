import pytest

from zinnia import *


def test_has_return_1():
    @zk_chip
    def bar(x) -> int:
        if True:
            return 0

    @zk_circuit
    def foo(x: int):
        assert bar(0) == 0
        assert bar(x) == 0

    assert foo(1)


def test_has_return_2():
    @zk_chip
    def bar(x) -> int:
        if x == 0:
            return 0

    @zk_circuit
    def foo_1(x: int):
        assert bar(0) == 0

    @zk_circuit
    def foo_2(x: int):
        assert bar(1) == 0

    @zk_circuit
    def foo_3(x: int):
        assert bar(x) == 0

    assert foo_1(1)
    with pytest.raises(ZinniaException) as e:
        foo_2(1)
    assert "Chip control ends without a return statement" in str(e.value)
    with pytest.raises(ZinniaException) as e:
        foo_3(0)
    assert "Chip control ends without a return statement" in str(e.value)
    with pytest.raises(ZinniaException) as e:
        foo_3(1)
    assert "Chip control ends without a return statement" in str(e.value)


def test_has_return_3():
    @zk_chip
    def bar(x) -> int:
        if x == 0:
            return 0
        else:
            return 1

    @zk_circuit
    def foo_1(x: int, y: int):
        assert bar(x) == y

    @zk_circuit
    def foo_2():
        assert bar(0) == 0
        assert bar(1) == 1

    assert foo_1(0, 0)
    assert foo_1(1, 1)
    assert foo_2()


def test_has_return_4():
    @zk_chip
    def bar(x) -> int:
        if x == 0:
            return 0
        return 1

    @zk_circuit
    def foo_1(x: int, y: int):
        assert bar(x) == y

    @zk_circuit
    def foo_2():
        assert bar(0) == 0
        assert bar(1) == 1

    assert foo_1(0, 0)
    assert foo_1(1, 1)
    assert foo_2()


def test_has_return_5():
    @zk_chip
    def bar() -> int:
        if False:
            return 0
        return 1

    @zk_circuit
    def foo():
        assert bar() == 1

    assert foo()


def test_has_return_6():
    @zk_chip
    def bar() -> int:
        if False:
            return 0

    @zk_circuit
    def foo():
        bar()

    with pytest.raises(ZinniaException) as e:
        assert foo()
    assert "Chip control ends without a return statement" in str(e.value)


def test_has_return_7():
    @zk_chip
    def bar() -> int:
        while True:
            return 0

    @zk_circuit
    def foo():
        assert bar() == 0

    assert foo()


def test_has_return_8():
    @zk_chip
    def bar(x) -> int:
        while True:
            if x == 0:
                return 0

    @zk_circuit
    def foo():
        assert bar(0) == 0

    assert foo()


def test_has_return_9():
    @zk_chip
    def bar(x) -> int:
        while True:
            if x == 0:
                return 0
            else:
                break

    @zk_circuit
    def foo(x: int):
        assert bar(x) == 0

    with pytest.raises(ZinniaException) as e:
        assert foo(0)
    assert "Chip control ends without a return statement" in str(e.value)


def test_has_return_10():
    @zk_chip
    def bar(x) -> int:
        for i in range(10):
            if x == i:
                return i

    @zk_circuit
    def foo(x: int):
        assert bar(x) == x


    with pytest.raises(ZinniaException) as e:
        assert foo(1)
    assert "Chip control ends without a return statement" in str(e.value)


def test_has_return_11():
    @zk_chip
    def bar(x) -> int:
        for i in range(10):
            if x == i:
                return i
        return 10

    @zk_circuit
    def foo(x: int):
        assert bar(x) == x

    for i in range(10):
        assert foo(i)


def test_has_return_12():
    @zk_chip
    def bar(x) -> int:
        for i in range(10):
            if x == i:
                return i
        else:
            return 10

    @zk_circuit
    def foo(x: int):
        assert bar(x) == x

    for i in range(11):
        assert foo(i)


def test_has_return_13():
    @zk_chip
    def bar(x) -> int:
        for i in range(10):
            break
        else:
            return 10

    @zk_circuit
    def foo(x: int):
        assert bar(x) == x

    with pytest.raises(ZinniaException) as e:
        assert foo(1)
    assert "Chip control ends without a return statement" in str(e.value)


def test_has_return_14():
    @zk_chip
    def bar(x) -> int:
        for i in range(10):
            break
        else:
            return 10
        return x

    @zk_circuit
    def foo(x: int):
        assert bar(x) == x

    for i in range(10):
        assert foo(i)


def test_has_return_15():
    @zk_chip
    def bar(x) -> int:
        for i in range(10):
            if i == 0:
                break
        else:
            return 10

    @zk_circuit
    def foo(x: int):
        assert bar(x) == x

    with pytest.raises(ZinniaException) as e:
        assert foo(0)
    assert "Chip control ends without a return statement" in str(e.value)


def test_has_return_16():
    @zk_chip
    def bar(x) -> int:
        while True:
            if x == 0:
                return 0
            else:
                return 1

    @zk_circuit
    def foo(x: int):
        assert bar(x) == x

    assert foo(0)
