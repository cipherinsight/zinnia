from zenopy import zk_circuit, ZKCircuit


def test_create_list():
    """This test is to check if the list is created correctly"""
    @zk_circuit
    def foo():
        _list = [1, 2, 3]
        assert _list[0] == 1
        assert _list[1] == 2
        assert _list[2] == 3

    ZKCircuit.from_method(foo, {}).compile()


def test_repeat_list():
    """This test is to check if the list is repeated correctly"""
    @zk_circuit
    def foo():
        _list = [1, 2] * 4
        assert len(_list) == 8
        assert _list[0] == _list[2] == _list[4] == _list[6] == 1
        assert _list[1] == _list[3] == _list[5] == _list[7] == 2

    ZKCircuit.from_method(foo, {}).compile()

def test_concatenate_list():
    """This test is to check if the list is concatenated correctly"""
    @zk_circuit
    def foo():
        list_1 = [1, 2]
        list_2 = [3, 4]
        tup = list_1 + list_2
        assert len(tup) == 4
        assert tup[0] == 1
        assert tup[1] == 2
        assert tup[2] == 3
        assert tup[3] == 4

    ZKCircuit.from_method(foo, {}).compile()


def test_list_comparison():
    """This test is to check if the list comparison is done correctly"""
    @zk_circuit
    def foo():
        list_1 = (1, 2)
        list_2 = (1, 2)
        assert list_1 == list_2 and list_1 <= list_2 <= list_1
        list_1 = (1, 2)
        list_2 = (1, 3)
        assert list_1 != list_2
        assert list_1 < list_2 and list_1 <= list_2
        list_1 = (1, 2)
        list_2 = (1, 1)
        assert list_1 != list_2
        assert list_1 > list_2 and list_1 >= list_2
        list_1 = (1, 2)
        list_2 = (1, 2, 3)
        assert list_1 != list_2
        assert list_1 < list_2 and list_1 <= list_2

    ZKCircuit.from_method(foo, {}).compile()


def test_list_comparison_with_inner():
    """This test is to check if the list comparison with inner is done correctly"""
    @zk_circuit
    def foo():
        list_1 = (1, 2, (3, 4))
        list_2 = (1, 2, (3, 4))
        assert list_1 == list_2
        list_1 = (1, 2, (3, 4))
        list_2 = (1, 2, (3, 5))
        assert list_1 != list_2
        assert list_1 < list_2
        list_1 = (1, 2, (3, 4))
        list_2 = (1, 2, (3, 3))
        assert list_1 != list_2
        assert list_1 > list_2
        list_1 = (1, 2, (3, 4))
        list_2 = (1, 2, (3, 4, 5))
        assert list_1 != list_2
        assert list_1 < list_2

    ZKCircuit.from_method(foo, {}).compile()
