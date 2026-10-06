"""Index arrays: integer positions gather, boolean masks select (fix-fancy).

NumPy reads `a[idx]` for an array-valued `idx` by its dtype: a bool array is a
mask that keeps the elements where it is true; an integer array lists the
positions to gather (a negative one counts from the end, an out-of-range one is
an `IndexError`). The compiler used to guess the dtype from the cells: a runtime
integer index array of `a`'s shape was read as a mask (`b[np.argsort(a)]` proved
`[10, 20]` for NumPy's `[20, 30, 10]`), and a bool array with integer cells
(`np.ones(3, dtype=bool)` after a write) as positions.

Every kernel is compared with NumPy, position by position, on the mock backend.
The rejection tests show that a wrong output (or an out-of-range index) has no
proof, on the mock and on halo2-base; a few run on Noir (`nargo execute`).
"""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from testing.lang.mock_outputs import compile_kernel, numpy_outputs, run_mock_layout
from zinnia import ZKCircuit
from zinnia.config.zinnia_config import ZinniaConfig

A3 = np.array([3.0, 1.0, 2.0])
B3 = np.array([10, 20, 30])
I3 = np.array([2, 0, 1])
M5 = np.array([0, 3, 0, 4, 1])
B5 = np.array([10, 20, 30, 40, 50])
C32 = np.array([[1, 2], [3, 4], [5, 6]])


def kernel(params: str, body: str) -> str:
    lines = body.strip("\n").split("\n")
    return f"def kernel({params}):\n" + "".join(f"    {line}\n" for line in lines)


def assert_numpy(source, *args):
    expected, expected_shapes = numpy_outputs(source, *args)
    satisfied, values, shapes = run_mock_layout(compile_kernel(source), *args)
    assert satisfied, "honest witness rejected"
    assert shapes == expected_shapes
    np.testing.assert_allclose(np.asarray(values, float), np.asarray(expected, float), atol=1e-6)


def accepts(program, *args, backend="mock"):
    if backend == "mock":
        return program.prove(*args, backend="mock").proof_bytes_hex == "mock_satisfied"
    try:
        proof = program.prove(*args, backend=backend)
    except RuntimeError as error:  # halo2-base refuses an unsatisfiable witness
        assert "does not satisfy" in str(error), error
        return False
    return program.verify(proof)


# ── Integer index arrays gather ──────────────────────────────────────────

AB = 'a: Float[Array, "3"], b: Integer[Array, "3"]'
IB = 'i: Integer[Array, "3"], b: Integer[Array, "3"]'
MB = 'm: Integer[Array, "5"], b: Integer[Array, "5"]'
MP = 'm: Bool[Array, "3"], b: Integer[Array, "3"]'
M3 = np.array([True, False, True])
M_X = 'm: Integer[Array, "5"], x: Integer[Array, "5 2"]'
X52 = np.arange(1, 11).reshape(5, 2)

GATHERS = {
    # The reported bug: a runtime integer array of `b`'s shape.
    "argsort": (kernel(AB, "return b[np.argsort(a)]"), (A3, B3)),
    "argsort_float_data": (kernel('a: Float[Array, "3"]', "return a[np.argsort(a)]"), (A3,)),
    "param_permutation": (kernel(IB, "return b[i]"), (I3, B3)),
    "param_zeros_and_ones": (kernel(IB, "return b[i]"), (np.array([1, 0, 1]), B3)),
    "expression": (kernel(IB, "return b[(i + 1) % 3]"), (I3, B3)),
    "where_positions": (kernel(MB, "return b[np.where(m > 0, 0, 4)]"), (M5, B5)),
    # Other shapes than `b`'s, negative positions.
    "longer_with_negatives": (
        kernel('i: Integer[Array, "4"], b: Integer[Array, "3"]', "return b[i]"),
        (np.array([2, 0, -1, -3]), B3)),
    "shorter": (kernel('i: Integer[Array, "2"], b: Integer[Array, "3"]', "return b[i]"),
                (np.array([2, -2]), B3)),
    "index_matrix": (kernel('i: Integer[Array, "2 2"], b: Integer[Array, "3"]', "return b[i]"),
                     (np.array([[0, 2], [1, -1]]), B3)),
    "rows": (kernel('a: Float[Array, "3"], c: Integer[Array, "3 2"]', "return c[np.argsort(a)]"),
             (A3, C32)),
    "rows_same_shape_index": (
        kernel('i: Integer[Array, "2 2"], c: Integer[Array, "2 2"]', "return c[i]"),
        (np.array([[0, 1], [1, 0]]), np.array([[1, 2], [3, 4]]))),
    "rows_then_column": (
        kernel('a: Float[Array, "3"], c: Integer[Array, "3 2"]', "return c[np.argsort(a), 1]"),
        (A3, C32)),
    "constant_negative": (kernel('b: Integer[Array, "3"]', "return b[np.array([2, 0, -1])]"), (B3,)),
    "list_of_runtime_ints": (kernel(IB, "return b[[i[0], i[1]]]"), (I3, B3)),
    # A bool array converted to integers is positions (it used to stay a mask).
    "astype_int_of_mask": (kernel(AB, "return b[(a > 1.5).astype(int)]"), (A3, B3)),
    "array_int_of_mask": (kernel(AB, "return b[np.array(a > 1.5, dtype=int)]"), (A3, B3)),
    "mask_times_one": (kernel(AB, "return b[(a > 1.5) * 1]"), (A3, B3)),
    "list_of_bool_and_int": (kernel(IB, "return b[[i[0] > 5, i[1] + 2]]"), (I3, B3)),
    "list_data": (kernel(AB, "c = b[[0, 1, 2]]\nreturn c[np.argsort(a)]"), (A3, B3)),
    # Conversions of bool arrays whose cells are integers in the circuit
    # (parameters, runtime-shaped arrays, arrays after a write) to integers.
    "astype_int_of_bool_parameter": (kernel(MP, "return b[m.astype(int)]"), (M3, B3)),
    "int_of_bool_parameter_cells": (kernel(MP, "return b[np.array([int(x) for x in m])]"), (M3, B3)),
    "array_int_of_bool_parameter": (kernel(MP, "return b[np.array(m, dtype=int)]"), (M3, B3)),
    "astype_int_after_mask_write": (
        kernel(AB, "m = a > 1.5\nm[a < 2.5] = False\nreturn b[m.astype(int)]"), (A3, B3)),
    "astype_int_of_runtime_shaped_mask": (
        kernel(MB, "c = (b > 25)[m > 0]\nreturn b[c.astype(int)]"), (M5, B5)),
    "astype_int_on_runtime_data": (
        kernel(MB, "c = b[m > 0]\nd = (m[m > 0] > 2).astype(int)\nreturn c[d]"), (M5, B5)),
    # An integer array stays one when bools are assigned into it.
    "int_array_assigned_bools": (
        kernel(AB, "m = np.zeros(3, dtype=int)\nm[:] = a > 1.5\nreturn b[m]"), (A3, B3)),
    "comprehension": (kernel(AB, "return np.array([b[j] for j in np.argsort(a)])"), (A3, B3)),
    # np.take reads its indices as integers.
    "take_runtime": (kernel(IB, "return np.take(b, i)"), (I3, B3)),
    "take_argsort": (kernel(AB, "return np.take(b, np.argsort(a))"), (A3, B3)),
    "take_axis": (kernel('i: Integer[Array, "3"], c: Integer[Array, "3 2"]',
                         "return np.take(c, i % 2, axis=1)"), (I3, C32)),
    # Runtime-shaped index arrays.
    "nonzero_positions": (kernel(MB, "return b[np.nonzero(m)[0]]"), (M5, B5)),
    "flatnonzero_positions": (kernel(MB, "return b[np.flatnonzero(m)]"), (M5, B5)),
    "where_one_argument": (kernel(MB, "return b[np.where(m > 0)[0]]"), (M5, B5)),
    "padding_slots_are_not_checked": (kernel(MB, "return b[np.nonzero(m)[0] - 6]"), (M5, B5)),
    "runtime_data_runtime_positions": (
        kernel(MB, "c = b[m > 0]\nd = m[m > 0]\nreturn c[d % 3]"), (M5, B5)),
    "runtime_data_negative_positions": (
        kernel(MB, "c = b[m > 0]\nd = m[m > 0]\nreturn c[d - 4]"), (M5, B5)),
    "runtime_data_constant_positions": (
        kernel(MB, "c = b[m > 0]\nreturn c[np.array([0, -1])]"), (M5, B5)),
}


@pytest.mark.parametrize("name", GATHERS)
def test_integer_index_array_gathers(name):
    source, args = GATHERS[name]
    assert_numpy(source, *args)


# ── Boolean masks select ─────────────────────────────────────────────────

MASKS = {
    "comparison": (kernel(AB, "return b[a > 1.5]"), (A3, B3)),
    "constant": (kernel('b: Integer[Array, "3"]', "return b[np.array([True, False, True])]"), (B3,)),
    "astype_bool": (kernel(IB, "return b[i.astype(bool)]"), (np.array([1, 0, 2]), B3)),
    "array_dtype_bool": (kernel(IB, "return b[np.array(i, dtype=bool)]"), (np.array([1, 0, 2]), B3)),
    "ones_bool_static_write": (
        kernel('b: Integer[Array, "3"]', "m = np.ones(3, dtype=bool)\nm[1] = False\nreturn b[m]"), (B3,)),
    "zeros_bool_runtime_write": (
        kernel('j: Integer, b: Integer[Array, "3"]', "m = np.zeros(3, dtype=bool)\nm[j] = True\nreturn b[m]"),
        (1, B3)),
    "comparison_runtime_write": (
        kernel(AB + ", j: Integer", "m = a > 1.5\nm[j] = False\nreturn b[m]"), (A3, B3, 0)),
    "full_true": (kernel('j: Integer, b: Integer[Array, "3"]', "m = np.full(3, True)\nm[j] = False\nreturn b[m]"),
                  (2, B3)),
    "zeros_like_bool": (
        kernel(AB, "m = np.zeros_like(a, dtype=bool)\nm[np.argmin(a)] = True\nreturn b[m]"), (A3, B3)),
    "ones_like_of_bool": (kernel(AB, "m = np.ones_like(a > 0)\nm[0] = False\nreturn b[m]"), (A3, B3)),
    "where_true_false": (kernel(AB, "return b[np.where(a > 1.5, True, False)]"), (A3, B3)),
    "where_of_masks": (kernel(AB, "return b[np.where(a > 1.5, a > 2.5, a < 1.5)]"), (A3, B3)),
    "product_of_masks": (kernel(AB, "return b[(a > 1.5) * (a < 2.5)]"), (A3, B3)),
    "sum_of_masks": (kernel(AB, "return b[(a > 2.5) + (a < 1.5)]"), (A3, B3)),
    "sorted_mask": (kernel(AB, "return b[np.sort(a > 1.5)]"), (A3, B3)),
    "minimum_of_masks": (kernel(AB, "return b[np.minimum(a > 1.5, a < 2.5)]"), (A3, B3)),
    "maximum_of_masks": (kernel(AB, "return b[np.maximum(a > 2.5, a < 1.5)]"), (A3, B3)),
    "bool_parameter": (kernel('m: Bool[Array, "3"], b: Integer[Array, "3"]', "return b[m]"),
                       (np.array([True, False, True]), B3)),
    "list_of_runtime_bools": (kernel(AB, "return b[[a[0] > 1.5, a[1] > 1.5, a[2] > 1.5]]"), (A3, B3)),
    "matrix": (kernel('x: Float[Array, "2 2"]', "return x[x > 1.5]"), (np.array([[3.0, 1.0], [2.0, 0.0]]),)),
    "leading_axis": (kernel('a: Float[Array, "3"], c: Integer[Array, "3 2"]', "return c[a > 1.5]"), (A3, C32)),
    "runtime_data": (kernel(MB, "c = b[m > 0]\nreturn c[c > 30]"), (M5, B5)),
    "filtered_mask": (kernel(MB, "c = b[m > 0]\nk = (m > 2)[m > 0]\nreturn c[k]"), (M5, B5)),
    "runtime_astype_bool": (kernel(MB, "c = m[m > 0]\nreturn c[(c % 2).astype(bool)]"), (M5, B5)),
    "nonzero_tuple": (kernel(MB, "return b[np.nonzero(m)]"), (M5, B5)),
    # A choice between cells of bool arrays is a bool, also after a write
    # into the array or a sort of it (their cells are integers in the circuit).
    "where_of_runtime_written_mask": (
        kernel(AB + ", j: Integer", "m = a > 1.5\nm[j] = False\nreturn b[np.where(a > 0, m, False)]"),
        (A3, B3, 0)),
    "where_of_constant_written_mask": (
        kernel(AB, "m = a > 1.5\nm[0] = False\nreturn b[np.where(a > 0, m, False)]"), (A3, B3)),
    "where_of_slice_written_mask": (
        kernel(AB, "m = a > 1.5\nm[1:] = False\nreturn b[np.where(a > 0, m, False)]"), (A3, B3)),
    "where_of_sorted_mask": (kernel(AB, "m = np.sort(a > 1.5)\nreturn b[np.where(a > 0, m, False)]"), (A3, B3)),
    "where_of_bool_parameter": (kernel(MP, "return b[np.where(m, m, False)]"), (M3, B3)),
}


@pytest.mark.parametrize("name", MASKS)
def test_boolean_masks_select(name):
    source, args = MASKS[name]
    assert_numpy(source, *args)


def test_bool_dtype_is_reported():
    assert_numpy(kernel(AB, "m = np.ones(3, dtype=bool)\nreturn m.dtype == bool, (a > 0).dtype == bool, "
                            "np.argsort(a).dtype == bool"), A3, B3)


# ── Assignments ──────────────────────────────────────────────────────────

WRITES = {
    "permutation": (kernel(IB, "c = np.zeros(3, dtype=int)\nc[i] = b\nreturn c"), (I3, B3)),
    "repeated_scalar": (kernel(IB, "c = np.zeros(3, dtype=int)\nc[i] = 7\nreturn c"), (np.array([2, 0, 0]), B3)),
    "argsort": (kernel(AB, "c = np.zeros(3, dtype=int)\nc[np.argsort(a)] = b\nreturn c"), (A3, B3)),
    "augmented_repeated": (kernel(IB, "c = np.zeros(3, dtype=int)\nc[i] += b\nreturn c"),
                           (np.array([1, 0, 1]), B3)),
    "negative": (kernel('i: Integer[Array, "4"], b: Integer[Array, "3"]',
                        "c = np.zeros(3, dtype=int)\nc[i] = 9\nreturn c"), (np.array([2, 0, -1, -1]), B3)),
    "mask": (kernel(AB, "c = np.zeros(3, dtype=int)\nc[a > 1.5] = 5\nreturn c"), (A3, B3)),
    "mask_from_astype": (kernel(IB, "c = np.zeros(3, dtype=int)\nc[i.astype(bool)] = 5\nreturn c"),
                         (np.array([1, 0, 2]), B3)),
    "mask_from_ones_bool": (kernel('b: Integer[Array, "3"]',
                                   "m = np.ones(3, dtype=bool)\nm[1] = False\nc = b.copy()\nc[m] = 0\nreturn c"),
                            (B3,)),
    "mask_list": (kernel(AB, "c = np.zeros(3, dtype=int)\nc[[a[0] > 1.5, a[1] > 1.5, a[2] > 1.5]] = 5\nreturn c"),
                  (A3, B3)),
    "runtime_positions": (kernel(MB, "c = np.zeros(5, dtype=int)\nc[np.nonzero(m)[0]] = 1\nreturn c"), (M5, B5)),
    "runtime_positions_on_runtime_data": (
        kernel(MB, "c = b[m > 0]\nd = m[m > 0]\nc[d % 3] = 0\nreturn c"), (M5, B5)),
    "runtime_mask_on_runtime_data": (kernel(MB, "c = b[m > 0]\nc[c > 30] = 0\nreturn c"), (M5, B5)),
    # A row of a runtime-shaped matrix (it used to write only the row's first cell).
    "runtime_row": (kernel(M_X, "c = x[m > 0]\nc[-1] = 0\nreturn c"), (M5, X52)),
    "runtime_rows_by_positions": (kernel(M_X, "c = x[m > 0]\nc[np.array([0, -1])] = 7\nreturn c"), (M5, X52)),
    "runtime_row_from_array": (kernel(M_X, "c = x[m > 0]\nc[1] = np.array([5, 6])\nreturn c"), (M5, X52)),
}


@pytest.mark.parametrize("name", WRITES)
def test_assignments(name):
    source, args = WRITES[name]
    assert_numpy(source, *args)


# NumPy converts a value assigned into a bool array to its truth value.
BOOL_ARRAY_WRITES = {
    "runtime_element": (kernel('j: Integer, b: Integer[Array, "3"]',
                               "m = np.zeros(3, dtype=bool)\nm[j] = b[0]\nreturn m.sum(), b[m]"), (1, B3)),
    "constant_element": (kernel('b: Integer[Array, "3"]',
                                "m = np.zeros(3, dtype=bool)\nm[1] = 2\nreturn m.sum(), b[m]"), (B3,)),
    "slice": (kernel('b: Integer[Array, "3"]', "m = np.zeros(3, dtype=bool)\nm[1:] = b[1:] - 20\n"
                                               "return m.sum(), b[m]"), (B3,)),
    "mask": (kernel(AB, "m = a > 5\nm[a > 1.5] = b[0]\nreturn m.sum(), b[m]"), (A3, B3)),
    "runtime_shaped": (kernel(MB, "c = (m > 9)[m > 0]\nc[0] = 7\nreturn c.sum(), b[m > 0][c]"), (M5, B5)),
}


@pytest.mark.parametrize("name", BOOL_ARRAY_WRITES)
def test_writes_into_bool_arrays_store_truth_values(name):
    source, args = BOOL_ARRAY_WRITES[name]
    assert_numpy(source, *args)


@pytest.mark.parametrize("obj", ["[True, False, True]", "np.array([True, False, True])", "[True, 2]"])
def test_delete_reads_obj_by_dtype(obj):
    # A bool `obj` is a mask of the entries to delete; `[True, 2]` is positions 1 and 2.
    assert_numpy(kernel('b: Integer[Array, "3"]', f"return np.delete(b, {obj})"), B3)


# ── Wrong outputs and out-of-range indices have no proof ─────────────────

CLAIMED_GATHER = kernel(
    AB + ', expected: Integer[Array, "3"]',
    "assert (b[np.argsort(a)] == expected).all()",
)
CLAIMED_POSITIONS = kernel(
    'i: Integer[Array, "4"], b: Integer[Array, "3"], expected: Integer[Array, "4"]',
    "assert (b[i] == expected).all()",
)
CLAIMED_MASK = kernel(
    'j: Integer, b: Integer[Array, "3"], expected: Integer',
    "m = np.ones(3, dtype=bool)\nm[j] = False\nassert b[m].sum() == expected",
)
CLAIMED_RUNTIME_POSITIONS = kernel(
    MB + ", expected: Integer",
    "assert b[np.nonzero(m)[0]].sum() == expected",
)


@pytest.mark.parametrize("backend", ["mock", "halo2-base"])
def test_gather_output_is_pinned(backend):
    program = compile_kernel(CLAIMED_GATHER)
    assert accepts(program, A3, B3, np.array([20, 30, 10]), backend=backend)
    assert not accepts(program, A3, B3, np.array([10, 20, 30]), backend=backend)  # the old mask read
    assert not accepts(program, A3, B3, np.array([20, 30, 30]), backend=backend)


@pytest.mark.parametrize("backend", ["mock", "halo2-base"])
def test_positions_with_negatives_are_pinned(backend):
    program = compile_kernel(CLAIMED_POSITIONS)
    i = np.array([2, 0, -1, -3])
    assert accepts(program, i, B3, np.array([30, 10, 30, 10]), backend=backend)
    assert not accepts(program, i, B3, np.array([30, 10, 30, 20]), backend=backend)


@pytest.mark.parametrize("backend", ["mock", "halo2-base"])
@pytest.mark.parametrize("i", [[0, 1, 2, 3], [0, 1, 2, -4]])
def test_out_of_range_position_is_rejected(backend, i):
    program = compile_kernel(CLAIMED_POSITIONS)
    # NumPy raises IndexError; no claimed output is provable.
    for claimed in ([10, 20, 30, 10], [10, 20, 30, 0], [10, 20, 30, 30]):
        assert not accepts(program, np.array(i), B3, np.array(claimed), backend=backend)


@pytest.mark.parametrize("backend", ["mock", "halo2-base"])
def test_bool_mask_output_is_pinned(backend):
    program = compile_kernel(CLAIMED_MASK)
    assert accepts(program, 1, B3, 40, backend=backend)
    assert not accepts(program, 1, B3, 30, backend=backend)  # the old gather of b[[1, 0, 1]]
    assert not accepts(program, 1, B3, 60, backend=backend)


@pytest.mark.parametrize("backend", ["mock", "halo2-base"])
def test_runtime_shaped_positions_are_pinned(backend):
    program = compile_kernel(CLAIMED_RUNTIME_POSITIONS)
    assert accepts(program, M5, B5, 110, backend=backend)
    assert not accepts(program, M5, B5, 100, backend=backend)


def test_runtime_shaped_position_out_of_range_is_rejected():
    program = compile_kernel(kernel(MB + ", expected: Integer", "assert b[np.nonzero(m)[0] + 2].sum() == expected"))
    assert accepts(program, np.array([1, 0, 1, 0, 0]), B5, 80)  # positions 2 and 4
    for claimed in (0, 90, 140):  # position 5 is out of range
        assert not accepts(program, M5, B5, claimed)


def test_position_past_runtime_length_is_rejected():
    # `c` has 3 elements (5 slots); NumPy raises for c[3].
    program = compile_kernel(kernel(MB + ", expected: Integer", "c = b[m > 0]\nassert c[np.array([0, 3])].sum() == expected"))
    for claimed in (20, 60, 70):
        assert not accepts(program, M5, B5, claimed)
    assert accepts(program, np.array([1, 1, 1, 1, 0]), B5, 50)


@pytest.mark.parametrize("obj", ["True", "a[0] > 1.5"])
def test_insert_rejects_a_bool_obj(obj):
    # NumPy raises ValueError for a bool scalar `obj` (a rank-0 boolean array).
    with pytest.raises(BaseException, match="boolean array argument obj to insert"):
        compile_kernel(kernel(AB, f"return np.insert(b, {obj}, 0)"))


@pytest.mark.parametrize("backend", ["mock", "halo2-base"])
def test_converted_bool_parameter_positions_are_pinned(backend):
    program = compile_kernel(kernel(MP + ', expected: Integer[Array, "3"]',
                                    "assert (b[m.astype(int)] == expected).all()"))
    assert accepts(program, M3, B3, np.array([20, 10, 20]), backend=backend)
    assert not accepts(program, M3, B3, np.array([10, 30, 30]), backend=backend)  # read as a mask


@pytest.mark.parametrize("assignment", [
    "c[k, 0] = 0",                      # a mask among several indices
    "c[d, 0] = 0",                      # a runtime-shaped index array among several indices
    "c[np.array([True, False, True, False, False]), 0] = 0",
])
def test_index_arrays_among_several_indices_of_runtime_shaped_data_are_rejected(assignment):
    body = f"c = x[m > 0]\nk = (m > 2)[m > 0]\nd = m[m > 0] % 3\n{assignment}\nreturn c"
    with pytest.raises(BaseException, match="combined with other indices is not supported"):
        compile_kernel(kernel(M_X, body))


def test_constant_out_of_range_position_is_a_compile_error():
    with pytest.raises(BaseException, match="out of bounds"):
        compile_kernel(kernel('b: Integer[Array, "3"]', "return b[np.array([0, 5])]"))


def test_constant_out_of_range_position_on_untaken_branch_proves():
    program = compile_kernel(kernel(
        'j: Integer, b: Integer[Array, "3"], expected: Integer',
        "y = 0\nif j > 0:\n    y = b[np.array([0, 5])].sum()\nassert y == expected",
    ))
    assert accepts(program, 0, B3, 0)
    assert not accepts(program, 1, B3, 0)


def test_float_index_array_is_rejected():
    with pytest.raises(BaseException, match="must be of integer"):
        compile_kernel(kernel(AB, "return b[a]"))


# ── Noir ─────────────────────────────────────────────────────────────────

def _nargo_available() -> bool:
    return (Path.home() / ".nargo" / "bin" / "nargo").is_file() or shutil.which("nargo") is not None


def noir_execute(program, *args):
    from zinnia._zinnia_core import execute_noir_circuit
    from zinnia.exec.input_parser import build_circuit_inputs

    witness = build_circuit_inputs(program.program_inputs, args)
    params = {"k": 0, "precision_bits": 32, "lookup_bits": 8}
    return execute_noir_circuit(program._compiled_ir, json.dumps(witness), {}, json.dumps(params))


@pytest.mark.skipif(not _nargo_available(), reason="nargo not installed")
def test_noir_gather_and_mask():
    for source, honest, wrong in (
        (CLAIMED_GATHER, (A3, B3, np.array([20, 30, 10])), (A3, B3, np.array([10, 20, 30]))),
        (CLAIMED_MASK, (1, B3, 40), (1, B3, 30)),
        (CLAIMED_RUNTIME_POSITIONS, (M5, B5, 110), (M5, B5, 100)),
    ):
        program = ZKCircuit.from_source("kernel", source, config=ZinniaConfig(backend="noir")).compile()
        noir_execute(program, *honest)
        with pytest.raises(BaseException):
            noir_execute(program, *wrong)
