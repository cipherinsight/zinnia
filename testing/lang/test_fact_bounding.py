"""Bounds through chains of arithmetic on bounded values (fact review).

The interval fact of an arithmetic result is a conjunction, `lo <= v && v <=
hi` (`interval_fact_for_int_binary`); the bound scan (`derive_bounds_from_facts`)
read only top-level comparisons, so a chain of two operations lost the bound:
`np.zeros(2 * (n + 2))` under `0 <= n <= 5` was rejected ("shape element must
be a compile-time constant"), and loops over such values were unrolled to looser
bounds than the facts give.
"""
import numpy as np
import pytest

from zinnia import requires, zk_circuit
from zinnia.api.zk_circuit import ZKCircuit


def prove(fn, *args):
    proof = ZKCircuit.from_method(fn).compile().prove(*args, backend="mock")
    return proof.proof_bytes_hex == "mock_satisfied", [int(v.split("=")[1]) for v in proof.public_values]


@zk_circuit
@requires(lambda n: 0 <= n <= 5)
def sized_by_chain(n: int):
    m = n + 2
    k = m * 2 - 1
    return np.arange(k + 1).sum()


@zk_circuit
@requires(lambda n: 0 <= n <= 5)
def loop_over_chain(n: int):
    m = n + 2
    k = m * 2 - 1
    i = 0
    while i < k + 1:
        i += 1
    return i


@pytest.mark.parametrize("n", range(6))
def test_array_sized_by_a_chain_of_bounded_arithmetic(n):
    assert prove(sized_by_chain, n) == (True, [int(np.arange(2 * (n + 2)).sum())])


@pytest.mark.parametrize("n", [0, 5])
def test_loop_over_a_chain_of_bounded_arithmetic(n):
    assert prove(loop_over_chain, n) == (True, [2 * (n + 2)])


@pytest.mark.parametrize("fn", [sized_by_chain, loop_over_chain])
def test_a_value_outside_the_facts_has_no_proof(fn):
    assert not prove(fn, 6)[0]
    assert not prove(fn, -1)[0]
