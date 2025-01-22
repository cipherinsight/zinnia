from zenopy import zk_circuit, ZKCircuit


def test_create_tuple():
    """This test is to check if the tuple is created correctly"""
    @zk_circuit
    def foo():
        tup = (1, 2, 3)
        assert tup[0] == 1
        assert tup[1] == 2
        assert tup[2] == 3

    ZKCircuit.from_method(foo).compile()


def test_repeat_tuple():
    """This test is to check if the tuple is repeated correctly"""
    @zk_circuit
    def foo():
        tup = (1, 2) * 4
        assert len(tup) == 8
        assert tup[0] == tup[2] == tup[4] == tup[6] == 1
        assert tup[1] == tup[3] == tup[5] == tup[7] == 2

    ZKCircuit.from_method(foo).compile()


def test_concatenate_tuple():
    """This test is to check if the tuple is concatenated correctly"""
    @zk_circuit
    def foo():
        tup1 = (1, 2)
        tup2 = (3, 4)
        tup = tup1 + tup2
        assert len(tup) == 4
        assert tup[0] == 1
        assert tup[1] == 2
        assert tup[2] == 3
        assert tup[3] == 4

    ZKCircuit.from_method(foo).compile()


def test_tuple_comparison():
    """This test is to check if the tuple comparison is done correctly"""
    @zk_circuit
    def foo():
        tup1 = (1, 2)
        tup2 = (1, 2)
        assert tup1 == tup2 and tup1 <= tup2 <= tup1
        tup1 = (1, 2)
        tup2 = (1, 3)
        assert tup1 != tup2
        assert tup1 < tup2 and tup1 <= tup2
        tup1 = (1, 2)
        tup2 = (1, 1)
        assert tup1 != tup2
        assert tup1 > tup2 and tup1 >= tup2
        tup1 = (1, 2)
        tup2 = (1, 2, 3)
        assert tup1 != tup2
        assert tup1 < tup2 and tup1 <= tup2

    ZKCircuit.from_method(foo).compile()


def test_tuple_comparison_with_inner():
    """This test is to check if the tuple comparison with inner is done correctly"""
    @zk_circuit
    def foo():
        tup1 = (1, 2, (3, 4))
        tup2 = (1, 2, (3, 4))
        assert tup1 == tup2
        tup1 = (1, 2, (3, 4))
        tup2 = (1, 2, (3, 5))
        assert tup1 != tup2
        assert tup1 < tup2
        tup1 = (1, 2, (3, 4))
        tup2 = (1, 2, (3, 3))
        assert tup1 != tup2
        assert tup1 > tup2
        tup1 = (1, 2, (3, 4))
        tup2 = (1, 2, (3, 4, 5))
        assert tup1 != tup2
        assert tup1 < tup2

    ZKCircuit.from_method(foo).compile()
