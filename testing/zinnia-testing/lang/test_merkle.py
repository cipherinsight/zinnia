from zinnia import *


def test_poseidon_hash_builtin_compiles():
    @zk_circuit
    def foo(x: Public[int]):
        h1 = poseidon_hash(x)
        h2 = poseidon_hash(x)
        assert h1 == h2

    ZKCircuit.from_method(foo).compile()


def test_merkle_verify_builtin_compiles_with_ndarray():
    @zk_circuit
    def foo(
        leaf: Public[int],
        root: Public[int],
        siblings: Public[NDArray[int, 2]],
        directions: Public[NDArray[bool, 2]],
    ):
        valid = merkle_verify(leaf, root, siblings, directions)
        assert valid == valid

    ZKCircuit.from_method(foo).compile()
