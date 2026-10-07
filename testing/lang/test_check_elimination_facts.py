"""`AlwaysSatisfiedElimination` drops a check only when the statements that
define its condition imply it (intervals, then SMT), never on the strength of a
check: not the program's preconditions (whose own checks are among the
candidates), not other assertions.

Before the fact review the pass ran with whatever resolver the graph carried:
the static-only one in the default pipeline (every pass that rebuilds the graph
resets it), so it dropped constant conditions only; with the shortcut and
constant-folding passes turned off it inherited the IR generator's resolver,
which assumes the preconditions.
"""
import re

import pytest

from zinnia import requires, zk_circuit
from zinnia.api.zk_circuit import ZKCircuit
from zinnia.config.optimization_config import OptimizationConfig
from zinnia.config.zinnia_config import ZinniaConfig


def compile_program(fn, **optimization):
    config = ZinniaConfig(optimization_config=OptimizationConfig(**optimization))
    return ZKCircuit.from_method(fn, config=config).compile()


def satisfied(program, *args):
    return program.prove(*args, backend="mock").proof_bytes_hex == "mock_satisfied"


def user_checks(program):
    """The asserted comparisons other than the inputs' domain checks."""
    stmts = {s["stmt_id"]: s for s in program.get_ir_stmts()}
    kinds = []
    for s in stmts.values():
        if s["ir_instance"]["__class__"] != "AssertIR":
            continue
        cond = stmts[s["arguments"][0]]
        operand = stmts[cond["arguments"][0]] if cond["arguments"] else None
        domain = operand is not None and operand["ir_instance"]["__class__"] in ("ReadIntegerIR", "ReadFloatIR") \
            and cond["ir_instance"]["__class__"] in ("GreaterThanOrEqualIIR", "LessThanOrEqualIIR") \
            and abs(stmts[cond["arguments"][1]]["ir_instance"].get("ir_data", {}).get("value", 0)) > 2 ** 62
        if not domain:
            kinds.append(cond["ir_instance"]["__class__"])
    return kinds


@zk_circuit
def implied_and_not(x: int):
    y = abs(x) + 1
    assert y != 0
    assert x != 3
    return y


def test_a_check_its_definitions_imply_is_dropped_and_the_other_kept():
    program = compile_program(implied_and_not)
    assert len(user_checks(program)) == 1, "abs(x) + 1 != 0 needs no check; x != 3 does"
    assert satisfied(program, 5)
    assert satisfied(program, -4)
    assert not satisfied(program, 3)


@zk_circuit
@requires(lambda n: n >= 3)
def at_least_three(n: int):
    return n + 1


@pytest.mark.parametrize("optimization", [{}, {"shortcut_optimization": False, "constant_fold": False}])
def test_a_precondition_does_not_discharge_its_own_check(optimization):
    program = compile_program(at_least_three, **optimization)
    assert satisfied(program, 5)
    assert not satisfied(program, 1)


@zk_circuit
def mutual(x: int, y: int):
    assert x == y
    assert y == x
    return x + y


def test_two_equivalent_checks_do_not_discharge_each_other():
    program = compile_program(mutual)
    assert len(user_checks(program)) >= 1
    assert satisfied(program, 4, 4)
    assert not satisfied(program, 4, 5)
