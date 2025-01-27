import pytest

from zinnia import *


def test_hash_integer():
    @zk_circuit
    def foo(x: PoseidonHashed[Integer]):
        assert x == 5

    hashed_object = PoseidonHashed(5, 0)
    assert foo(hashed_object)


def test_error_hash_float():
    @zk_circuit
    def foo(x: PoseidonHashed[Float]):
        assert x == 5.0

    with pytest.raises(ZinniaException) as e:
        hashed_object = PoseidonHashed(5.0, 0)
        assert foo(hashed_object)
    assert "Cannot perform Poseidon hash on Float type." in str(e.value)


def test_hash_ndarray():
    @zk_circuit
    def foo(x: PoseidonHashed[NDArray[Integer, 2, 2]]):
        assert x.sum() == 10

    hashed_object = PoseidonHashed([[1, 2], [3, 4]], 0)
    assert foo(hashed_object)


def test_hash_recursively():
    @zk_circuit
    def foo(x: PoseidonHashed[List[Tuple[Integer, Integer], NDArray[Integer, 2]]]):
        assert x[0][0] + x[0][1] + x[1][0] + x[1][1] == 10

    hashed_object = PoseidonHashed([(1, 2), [3, 4]], 0)
    assert foo(hashed_object)


def test_hash_as_inner_type():
    @zk_circuit
    def foo(x: Tuple[PoseidonHashed[Integer], Integer]):
        assert x[0] + x[1] == 10

    hashed_object = PoseidonHashed(2, 0)
    assert foo((hashed_object, 8))
