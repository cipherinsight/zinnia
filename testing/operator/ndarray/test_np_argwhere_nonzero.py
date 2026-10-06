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

    # No 4x4 intermediate is built in memory between the two sliced axes
    # (the window gather shifts rows, then columns, as values). Full-size
    # segments are at most the source and the slice result (a
    # runtime-shaped array whose 16 cells the output contract reads back;
    # those reads are at constant addresses, so `ConstantMemoryForwarding`
    # replaces them by the written cells and only the 16 writes remain).
    fullsize_segments = [seg for seg, counts in per_seg.items() if counts["WriteMemoryIR"] == 16]
    assert 1 <= len(fullsize_segments) <= 2


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
    # The contract pins the runtime bbox shape (length, then per-axis
    # extents) ahead of the payload — a dynamic-bounds slice no longer
    # binds only the static envelope rectangle.
    assert values[:3] == [4, 2, 2]
    assert values[3:] == [
        5, 6, 0, 0,
        7, 8, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
    ]
