"""Element bounds of `forall` preconditions must not outlive a write (AF-1).

`@requires(forall(a, in_range(lo, hi)))` bounds the elements of the input as
passed. The compiler used to keep that bound for the array after an in-place
write (it was keyed by the array value's identity and by the input's name,
both kept by `a[0] = x`): `np.min(a)` then got `0 <= min(a) <= 4`, so
* the index `c[np.min(a)]` was neither wrapped nor checked: `x = -1` proved
  `c[-1] == 0` where NumPy gives 50, and `x = -6` (NumPy raises) had a proof;
* the divisor `np.min(a) + 1` was taken to be non-zero, so `x = -1` proved
  `7 // 0 == 0`, where the division raises.

Every program here writes the array (directly, through an alias, through a
view, by a mask, by `+=`) before a reduction reads it, and is compared with
NumPy (errors raised) on the mock; one is proved on halo2-base. A bound that
still holds is still used: the unwritten array keeps its discharged check.
"""
import numpy as np
import pytest

from testing.lang.mock_outputs import run_mock
from zinnia import ZKCircuit


def compile_source(source):
    return ZKCircuit.from_source("kernel", source).compile()


def run_numpy(source, *args):
    """The kernel run as NumPy with errors raised; `None` when it raises."""
    import ast
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.arg):
            node.annotation = None
        if isinstance(node, ast.FunctionDef):
            node.decorator_list = []
    env = {"np": np}
    exec(compile(ast.fix_missing_locations(tree), "<kernel>", "exec"), env)
    try:
        with np.errstate(all="raise"):
            return env["kernel"](*[np.copy(a) for a in args])
    except (IndexError, ZeroDivisionError, FloatingPointError):
        return None


HEADER = "@requires(lambda a, x: forall(a, in_range(0, 4)))\ndef kernel(a: Integer[Array, '4'], x: Integer):\n"
USES = {
    "index": "    c = np.array([10, 20, 30, 40, 50])\n    return c[np.min(a)]\n",
    "divisor": "    return 7 // (np.min(a) + 1)\n",
}
WRITES = {
    "in_place": "    a[0] = x\n",
    "second_cell": "    a[2] = x\n",
    "alias": "    b = a\n    b[1] = x\n",
    "view": "    v = a[1:]\n    v[0] = x\n",
    "mask": "    a[a > 2] = x\n",
    "augmented": "    a[3] += x\n",
}


@pytest.mark.parametrize("use", sorted(USES))
@pytest.mark.parametrize("write", sorted(WRITES))
@pytest.mark.parametrize("x", [2, -1, -5, -6, 9])
def test_a_forall_bound_does_not_outlive_a_write(write, use, x):
    source = HEADER + WRITES[write] + USES[use]
    a = np.array([1, 2, 3, 0])
    expected = run_numpy(source, a, x)
    satisfied, outputs = run_mock(compile_source(source), a, x)
    if expected is None:
        assert not satisfied, f"NumPy raises, but {outputs} has a proof"
    else:
        assert satisfied and outputs == [expected]


def test_a_forall_bound_on_an_unwritten_array_is_still_used():
    program = compile_source(HEADER + USES["divisor"])
    classes = [s["ir_instance"]["__class__"] for s in program.get_ir_stmts()]
    # `np.min(a) + 1 >= 1`: no divisor check, only the input domain checks
    # and the precondition's (2 + 2 per element; `x` is unused, so its own
    # domain checks are dropped).
    assert [c for c in classes if c.startswith("Assert")] == ["AssertIR"] * 16
    satisfied, outputs = run_mock(program, np.array([3, 1, 4, 2]), 7)
    assert satisfied and outputs == [3]
    satisfied, _ = run_mock(program, np.array([3, 1, 5, 2]), 7)
    assert not satisfied


@pytest.mark.parametrize("use, expected", [("index", 50), ("divisor", 3)])
def test_halo2_base_proves_the_numpy_value_after_a_write(use, expected):
    program = compile_source(HEADER + WRITES["in_place"] + USES[use])
    args = (np.array([1, 2, 3, 0]), -1) if use == "index" else (np.array([1, 2, 3, 2]), 1)
    circuit = program.prove(*args, backend="halo2-base", params={"k": 12, "precision_bits": 32, "lookup_bits": 8})
    assert [int.from_bytes(bytes.fromhex(h), "little") for h in circuit.public_values] == [expected]


def test_halo2_base_has_no_proof_of_a_division_by_zero_after_a_write():
    program = compile_source(HEADER + WRITES["in_place"] + USES["divisor"])
    with pytest.raises(Exception):
        program.prove(np.array([1, 2, 3, 0]), -1, backend="halo2-base",
                      params={"k": 12, "precision_bits": 32, "lookup_bits": 8})
