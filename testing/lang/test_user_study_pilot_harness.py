"""Checks that the pilot evaluator preserves submitted control flow."""

import ast

import numpy as np

from studies.user_study.pilots.validate import instrument, specialize
from studies.user_study.tasks import TASKS
from studies.user_study.validate import python_function
from zinnia import ZKCircuit


def test_pilot_checks_preserve_early_returns_and_avoid_name_collisions():
    text = '''def kernel(x: Integer[Array, "2"]):
    pilot_expected = 1
    pilot_actual = x
    if x[0] < 0:
        return pilot_actual
    return x + pilot_expected
'''
    circuit = ZKCircuit.from_source("kernel", instrument(text, (2,)))
    circuit.compile()
    for values in (np.array([-3, 4]), np.array([3, 4])):
        expected = values if values[0] < 0 else values + 1
        assert circuit.prove(values, expected, backend="mock").proof_bytes_hex == "mock_satisfied"
        assert circuit.prove(values, expected + 1, backend="mock").proof_bytes_hex != "mock_satisfied"


def test_pilot_checks_leave_local_helper_returns_unchanged():
    text = '''def kernel(x):
    def increment(values):
        return values + 1
    return increment(x)
'''
    checked = instrument(text, (2,))
    before = ast.parse(text).body[0].body[0]
    after = ast.parse(checked).body[0].body[0]
    assert ast.dump(before) == ast.dump(after)
    python_function(checked)(np.array([2, 3]), np.array([3, 4]))


def test_pilot_specialization_changes_only_interface_and_public_round_literal():
    text = '''def kernel(x):
    R = 3
    result = x.copy()
    for step in range(R):
        result = result + 1
    return result
'''
    task = next(t for t in TASKS if t.name == "smooth")
    before = ast.parse(text)
    after = ast.parse(specialize(text, task, {"H": 3, "W": 4, "R": 2}))
    after.body[0].args.args[0].annotation = None
    after.body[0].body[0].value = ast.Constant(3)
    assert ast.dump(before) == ast.dump(after)
