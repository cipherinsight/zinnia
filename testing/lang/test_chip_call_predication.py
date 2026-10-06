"""A chip runs only where its call runs (audit finding X1).

A chip, a lifted local ``def`` or a lambda that writes into an array argument,
asserts, or returns must do so only on the paths where the call executes:
inside a private ``if``, after a private ``break``, or after an early
``return``. Every test checks that NumPy's
result is accepted and that the result of an unconditional call is rejected.
"""
import numpy as np
import pytest

from zinnia import ZKChip, ZKCircuit
from studies.user_study.validate import checked_source


def accepts(program, *args):
    return program.prove(*args, backend="mock").proof_bytes_hex == "mock_satisfied"


BUMP_CHIP = 'def bump(a: Integer[Array, "3"]):\n    a[0] = a[0] + 100\n'

# The same mutation written as an explicit chip, a local def lifted into a
# chip, and a lambda lifted into a chip.
HELPERS = {
    "explicit_chip": "",
    "local_def": "    def bump(a):\n        a[0] = a[0] + 100\n",
    "lambda": "    bump = lambda a: np.copyto(a, a + np.array([100, 0, 0]))\n",
}


def compile_kernel(helper, source):
    chips = [ZKChip.from_source("bump", BUMP_CHIP)] if helper == "explicit_chip" else None
    return ZKCircuit.from_source("kernel", source, chips=chips).compile()


@pytest.mark.parametrize("helper", sorted(HELPERS))
def test_mutation_under_private_if(helper):
    source = (
        'def kernel(x: Integer[Array, "3"]):\n'
        + HELPERS[helper]
        + "    y = x.copy()\n"
        "    if x[0] > 0:\n"
        "        bump(y)\n"
        "    return y\n"
    )
    program = compile_kernel(helper, checked_source(source, (3,)))
    x = np.array([-1, 2, 3])
    assert accepts(program, x, np.array([-1, 2, 3]))
    assert not accepts(program, x, np.array([99, 2, 3]))
    x = np.array([1, 2, 3])
    assert accepts(program, x, np.array([101, 2, 3]))
    assert not accepts(program, x, np.array([1, 2, 3]))


@pytest.mark.parametrize("helper", ["explicit_chip", "local_def"])
def test_mutation_after_private_break(helper):
    body = (
        "    y = x.copy()\n"
        "    for i in range(3):\n"
        "        if x[i] < 0:\n"
        "            break\n"
        "        bump2(y, i)\n"
        "    return y\n"
    )
    chip = 'def bump2(a: Integer[Array, "3"], i: Integer):\n    a[i] = a[i] + 100\n'
    if helper == "explicit_chip":
        source = 'def kernel(x: Integer[Array, "3"]):\n' + body
        program = ZKCircuit.from_source(
            "kernel", checked_source(source, (3,)), chips=[ZKChip.from_source("bump2", chip)]
        ).compile()
    else:
        source = ('def kernel(x: Integer[Array, "3"]):\n'
                  "    def bump2(a, i):\n        a[i] = a[i] + 100\n" + body)
        program = ZKCircuit.from_source("kernel", checked_source(source, (3,))).compile()
    x = np.array([1, -2, 3])
    assert accepts(program, x, np.array([101, -2, 3]))
    assert not accepts(program, x, np.array([101, 98, 103]))


def test_mutation_after_early_return_does_not_reach_the_returned_array():
    source = '''def kernel(x: Integer[Array, "3"]):
    def bump(a):
        a[0] = a[0] + 100
    y = x.copy()
    if x[0] < 0:
        return y
    bump(y)
    return y
'''
    program = ZKCircuit.from_source("kernel", source).compile()
    for x in (np.array([-1, 2, 3]), np.array([1, 2, 3])):
        expected = x.copy()
        if x[0] >= 0:
            expected[0] += 100
        public = program.prove(x, backend="mock").public_values
        values = sorted((int(p.split("_")[2].split("=")[0]), int(p.split("=")[1])) for p in public)
        assert [v for _, v in values] == expected.tolist()


def test_assertion_inside_chip_is_predicated_on_the_call():
    source = '''def kernel(x: Integer[Array, "3"]):
    def checked(a):
        assert a[0] > 0
        a[1] = 7
    y = x.copy()
    if x[0] > 0:
        checked(y)
    return y
'''
    program = ZKCircuit.from_source("kernel", checked_source(source, (3,))).compile()
    assert accepts(program, np.array([-1, 2, 3]), np.array([-1, 2, 3]))
    assert accepts(program, np.array([1, 2, 3]), np.array([1, 7, 3]))
    assert not accepts(program, np.array([-1, 2, 3]), np.array([-1, 7, 3]))
