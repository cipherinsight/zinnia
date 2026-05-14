"""Lang smoke test for `compiler.op-fact-group-3a-reductions-static-ensures`.

Confirms `np.all` and `np.any` compile cleanly when their boolean output
flows downstream. The Rust unit tests
(`fire_contract_all_yields_output_bool_facts`,
`fire_contract_any_yields_output_bool_facts`) cover the fact-deposition
mechanics; this lang test is the functional end-to-end check that the
ops compile and consume their bool result without regression.
"""
import numpy as np

from zinnia import zk_circuit
from zinnia.api.zk_circuit import ZKCircuit


def test_np_all_on_static_array_compiles():
    """`np.all` over a static array — output ∈ {0, 1} fact is planted at
    the reduction's return site. Smoke test only.
    """
    @zk_circuit
    def foo():
        arr = np.asarray([1, 1, 1, 1])
        _zinnia_result = np.all(arr)

    _ = ZKCircuit.from_method(foo).compile()


def test_np_any_on_static_array_compiles():
    """Dual of the `all` test for `np.any`."""
    @zk_circuit
    def foo():
        arr = np.asarray([0, 0, 1, 0])
        _zinnia_result = np.any(arr)

    _ = ZKCircuit.from_method(foo).compile()
