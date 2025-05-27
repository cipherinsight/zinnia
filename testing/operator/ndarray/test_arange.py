from zinnia import *

def test_simple_arange_1():
    @zk_circuit
    def foo():
        ary = np.arange(10)
        assert ary.tolist() == list(range(10))

    assert foo()


def test_simple_arange_2():
    @zk_circuit
    def foo():
        ary = np.arange(0, 10, 2)
        assert ary.tolist() == list(range(0, 10, 2))

    assert foo()


def test_simple_arange_3():
    @zk_circuit
    def foo():
        ary = np.arange(0, 10, 2, float)
        assert ary.tolist() == list(float(x) for x in range(0, 10, 2))

    assert foo()


def test_simple_arange_4():
    @zk_circuit
    def foo():
        ary = np.arange(0, 1, 0.5, int)
        assert ary.tolist() == [0, 0]

    assert foo()
