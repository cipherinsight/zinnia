"""Static-array coverage for np.argwhere / np.nonzero dynamic outputs."""

from collections import Counter, defaultdict

from zinnia import *
from zinnia.api.zk_circuit import ZKCircuit


def test_np_argwhere_static_1d_returns_dynamic_coordinate_cells():
    @zk_circuit
    def foo():
        coords = np.argwhere(np.asarray([0, 5, 0, 7]))
        assert coords.sum() == 4  # coordinates [1, 3]

    assert foo()


def test_np_argwhere_static_2d_returns_dynamic_coordinate_cells():
    @zk_circuit
    def foo():
        coords = np.argwhere(np.asarray([[0, 2, 3], [0, 0, 4]]))
        assert coords.sum() == 6  # rows [[0, 1], [0, 2], [1, 2]]

    assert foo()


def test_np_nonzero_static_2d_returns_per_axis_dynamic_arrays():
    @zk_circuit
    def foo():
        rows, cols = np.nonzero(np.asarray([[0, 2, 3], [0, 0, 4]]))
        assert rows.sum() == 1  # [0, 0, 1]
        assert cols.sum() == 5  # [1, 2, 2]

    assert foo()


def test_two_axis_dynamic_slice_fuses_without_fullsize_intermediate_segment():
    @zk_circuit
    def foo(A: Integer[Array, "4 4"]):
        bbox = np.argwhere(A)
        (ystart, xstart), (ystop, xstop) = bbox.min(0), bbox.max(0) + 1
        return A[ystart:ystop, xstart:xstop]

    stmts = ZKCircuit.from_method(foo).compile().get_ir_stmts()
    per_seg = defaultdict(Counter)
    for stmt in stmts:
        inst = stmt.get("ir_instance", {})
        cls = inst.get("__class__")
        if cls in ("AllocateMemoryIR", "WriteMemoryIR", "ReadMemoryIR"):
            seg = inst["ir_data"]["segment_id"]
            per_seg[seg][cls] += 1

    fullsize_segments = [
        seg
        for seg, counts in per_seg.items()
        if counts["WriteMemoryIR"] == 16 and counts["ReadMemoryIR"] == 16
    ]
    assert len(fullsize_segments) == 1


def test_two_axis_dynamic_slice_public_layout_is_top_left_packed():
    @zk_circuit
    def foo(A: Integer[Array, "4 4"]):
        bbox = np.argwhere(A)
        (ystart, xstart), (ystop, xstop) = bbox.min(0), bbox.max(0) + 1
        return A[ystart:ystop, xstart:xstop]

    proof = ZKCircuit.from_method(foo).prove(
        [
            [0, 0, 0, 0],
            [0, 5, 6, 0],
            [0, 7, 8, 0],
            [0, 0, 0, 0],
        ],
        backend="mock",
    )
    ordered = []
    for item in proof.public_values:
        label, value = item.split("=")
        ordered.append((int(label.split("_")[-1]), int(value)))
    ordered.sort()
    values = [value for _, value in ordered]
    assert values == [
        5, 6, 0, 0,
        7, 8, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
    ]
