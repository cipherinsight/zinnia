"""Outputs of in-place kernels.

An entry function that returns `None` has as outputs the array arguments its
body writes, in parameter order (subscript and slice assignment, `op=`,
`out=`, writes through views, aliases and helpers). They are public, so a
proof binds them: a claimed output that differs from NumPy's is rejected. A
function that returns a value has only its return value as output. Before this
rule such kernels exposed nothing, so their proofs proved nothing.
"""
import numpy as np
import pytest

from testing.lang.mock_outputs import compile_kernel, numpy_in_place_outputs, run_mock_layout
from zinnia import ZKCircuit
from zinnia.debug.exception.base import ZinniaException

BN254 = 21888242871839275222246405745257275088548364400416034343698204186575808495617
HALO2 = {"k": 12, "lookup_bits": 8, "precision_bits": 32}


def written_arguments(program) -> list[str]:
    return [part["output"]["argument"] for part in program.get_output_layout()]


def assert_in_place_matches_numpy(source, names, *args):
    program = compile_kernel(source)
    assert written_arguments(program) == names
    satisfied, values, shapes = run_mock_layout(program, *args)
    expected, expected_shapes = numpy_in_place_outputs(source, names, *args)
    assert satisfied, "honest witness rejected"
    assert shapes == expected_shapes
    np.testing.assert_allclose(np.asarray(values, float), np.asarray(expected, float), atol=1e-6)


STENCIL = '''
def kernel(n: Integer, A: NDArray[float, 6], B: NDArray[float, 6], C: NDArray[float, 6]):
    for i in range(1, 5):
        B[i] = (A[i - 1] + A[i] + A[i + 1]) / 3.0
    A += 1.0
'''


def test_written_arguments_are_the_outputs_in_parameter_order():
    a, b, c = np.arange(6.0), np.zeros(6), np.full(6, 2.0)
    # C is only read (not written), n is a scalar: neither is an output.
    assert_in_place_matches_numpy(STENCIL, ["A", "B"], 3, a, b, c)


def test_writes_through_views_aliases_helpers_and_out():
    source = '''
def kernel(A: NDArray[int, 3, 3], B: NDArray[int, 3], C: NDArray[float, 3], D: NDArray[int, 2]):
    row = A[1]
    row[:] = 7
    alias = B
    alias[0] = -1
    np.multiply(C, 2.0, out=C)
    D[0] = D[1]
'''
    assert_in_place_matches_numpy(
        source, ["A", "B", "C", "D"], np.arange(9).reshape(3, 3), np.array([4, 5, 6]),
        np.array([0.5, 1.5, -2.0]), np.array([1, 9]))


def test_writes_inside_a_helper_reach_the_argument():
    source = '''
def kernel(A: NDArray[int, 3], B: NDArray[float, 2]):
    def bump(x, k):
        x[0] = x[0] + k
    def scale(y):
        y *= 2.0
    bump(A, 10)
    scale(B)
'''
    assert_in_place_matches_numpy(source, ["A", "B"], np.array([1, 2, 3]), np.array([0.5, -1.0]))


def test_conditional_writes_follow_the_branch_taken():
    source = '''
def kernel(A: NDArray[int, 4], t: Integer):
    for i in range(4):
        if A[i] > t:
            A[i] = t
'''
    for t in (0, 2, 10):
        assert_in_place_matches_numpy(source, ["A"], np.array([3, -1, 2, 8]), t)


def test_plain_rebinding_does_not_modify_the_argument():
    source = '''
def kernel(A: NDArray[int, 3], B: NDArray[int, 3]):
    A = A + 1
    B[0] = 5
'''
    # `A = A + 1` makes a new array; the caller's A is unchanged (NumPy).
    assert_in_place_matches_numpy(source, ["B"], np.array([1, 2, 3]), np.array([0, 0, 0]))


def test_writes_before_a_rebinding_reach_the_argument_and_later_ones_do_not():
    source = '''
def kernel(A: NDArray[int, 3]):
    A[0] = 1
    A = A * 2
    A[1] = 3
'''
    # Python: the caller's A gets A[0] = 1; the later write goes to the new array.
    assert_in_place_matches_numpy(source, ["A"], np.array([5, 6, 7]))


def test_swapped_names_write_into_both_arguments():
    source = '''
def kernel(In: NDArray[float, 4], Out: NDArray[float, 4]):
    for it in range(3):
        Out[1:3] = (In[0:2] + In[2:4]) * 0.5
        In, Out = Out, In
'''
    assert_in_place_matches_numpy(source, ["In", "Out"], np.array([1.0, 2.0, 4.0, 8.0]),
                                  np.zeros(4))


def test_a_write_through_a_detached_copy_is_rejected():
    source = '''
def kernel(A: NDArray[int, 3, 2], k: Integer):
    row = A[k]
    row[0] = 9
'''
    with pytest.raises(ZinniaException, match="Reading `A` is not supported"):
        compile_kernel(source)


def test_a_function_returning_a_value_outputs_only_that_value():
    source = '''
def kernel(A: NDArray[int, 3]) -> Integer:
    A[0] = 100
    return A.sum()
'''
    layout = compile_kernel(source).get_output_layout()
    assert [part["output"] for part in layout] == [{"return": []}]


def test_none_annotation_and_bare_returns_are_in_place_kernels():
    source = '''
def kernel(A: NDArray[int, 3], k: Integer) -> None:
    if k > 0:
        A[0] = k
        return
    A[1] = -k
'''
    for k in (4, -2):
        assert_in_place_matches_numpy(source, ["A"], np.array([0, 0, 0]), k)


def test_in_place_outputs_are_bound_by_the_proof():
    program = ZKCircuit.from_source("kernel", STENCIL).compile()
    keys = program.setup(params=HALO2)
    a, b, c = np.arange(6.0), np.zeros(6), np.full(6, 2.0)
    proof = program.prove_with_keys(keys, 3, a, b, c)
    assert proof["public_values"], "an in-place kernel must expose its outputs"
    assert program.verify_with_keys(keys, proof)
    # Claiming B's old contents (or any other value) for an output is rejected.
    for index in range(len(proof["public_values"])):
        tampered = dict(proof, public_values=list(proof["public_values"]))
        value = int.from_bytes(bytes.fromhex(tampered["public_values"][index]), "little")
        tampered["public_values"][index] = ((value + 1) % BN254).to_bytes(32, "little").hex()
        assert not program.verify_with_keys(keys, tampered)
