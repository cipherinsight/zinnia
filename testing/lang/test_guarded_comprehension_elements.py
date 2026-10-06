"""Comprehension elements run only for candidates that pass the filter (audit
finding X9).

The element of `[f(v) for v in xs if g(v)]` is evaluated under the filter's
condition: writes it makes (through a chip) keep the old cells, and chips it
calls see the filter as their entry condition. Bounds checks and divisions
inside the element are still emitted unconditionally by the core operators;
that remaining completeness gap is shared with `if` statements and is handed
off (see the fix report), so it is not tested here.
"""
import numpy as np

from zinnia import ZKCircuit


def accepts(program, *args):
    return program.prove(*args, backend="mock").proof_bytes_hex == "mock_satisfied"


def test_element_side_effect_only_for_kept_candidates():
    source = '''def kernel(x: Integer[Array, "4"], expected: Integer[Array, "4"]):
    def mark(a, i):
        a[i] = a[i] + 100
        return i
    y = x.copy()
    kept = [mark(y, i) for i in range(4) if x[i] > 0]
    assert np.all(y == expected)
'''
    program = ZKCircuit.from_source("kernel", source).compile()
    x = np.array([1, -2, 3, -4])
    assert accepts(program, x, np.array([101, -2, 103, -4]))
    # An unpredicated element would also have marked the filtered-out ones.
    assert not accepts(program, x, np.array([101, 98, 103, 96]))


def test_conditional_expression_arm_side_effect():
    source = '''def kernel(x: Integer[Array, "2"], expected: Integer[Array, "2"]):
    def mark(a):
        a[0] = 7
        return 1
    y = x.copy()
    z = [mark(y) if v > 0 else 0 for v in x]
    assert np.all(y == expected)
'''
    program = ZKCircuit.from_source("kernel", source).compile()
    assert accepts(program, np.array([-1, -2]), np.array([-1, -2]))
    assert not accepts(program, np.array([-1, -2]), np.array([7, -2]))
    assert accepts(program, np.array([-1, 2]), np.array([7, 2]))


def test_chip_call_in_conditional_expression_arm():
    # `checked_source` would also rewrite the helper's `return`, so the check
    # is written out here.
    source = '''def kernel(x: Integer[Array, "3"], expected: Integer[Array, "3"]):
    def bump(a):
        a[0] = a[0] + 100
        return 1
    y = x.copy()
    flag = bump(y) if x[0] > 0 else 0
    assert np.all(y + flag == expected)
'''
    program = ZKCircuit.from_source("kernel", source).compile()
    assert accepts(program, np.array([-1, 2, 3]), np.array([-1, 2, 3]))
    assert not accepts(program, np.array([-1, 2, 3]), np.array([99, 2, 3]))
    assert accepts(program, np.array([1, 2, 3]), np.array([102, 3, 4]))
