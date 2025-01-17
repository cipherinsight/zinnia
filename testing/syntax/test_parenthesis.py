from zenopy import zk_circuit, ZKCircuit


def test_create_tuple():
    @zk_circuit
    def foo():
        tup = (1, 2, 3)
        assert tup[0] == 1
        assert tup[1] == 2
        assert tup[2] == 3

    ZKCircuit.from_method(foo, {}).compile()
