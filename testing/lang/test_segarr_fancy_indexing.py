"""Regression tests for static-array fancy indexing with trailing ranges."""

from zinnia import *
from zinnia.api.zk_circuit import ZKCircuit


def public_values(proof):
    ordered = []
    for item in proof.public_values:
        label, value = item.split("=")
        ordered.append((int(label.split("_")[-1]), int(float(value))))
    ordered.sort()
    return [value for _, value in ordered]


def test_static_array_fancy_index_preserves_column_axis_shape():
    @zk_circuit
    def gather_cols(a: Integer[Array, "2 3"], c: Integer[Array, "3"]):
        return a[:, c]

    proof = ZKCircuit.from_method(gather_cols).prove(
        [[1, 2, 3], [4, 5, 6]],
        [2, 0, 1],
        backend="mock",
    )
    assert proof.proof_bytes_hex == "mock_satisfied"
    assert public_values(proof) == [3, 1, 2, 6, 4, 5]


def test_static_array_fancy_index_preserves_trailing_range_axes():
    @zk_circuit
    def gather_planes(a: Integer[Array, "2 2 2"], c: Integer[Array, "2"]):
        return a[c, :, :]

    proof = ZKCircuit.from_method(gather_planes).prove(
        [
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]],
        ],
        [1, 0],
        backend="mock",
    )
    assert proof.proof_bytes_hex == "mock_satisfied"
    assert public_values(proof) == [5, 6, 7, 8, 1, 2, 3, 4]
