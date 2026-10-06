"""Cheaper lowerings of the second performance round (perf-2, 2026-10-06).

Each section covers one change: the outputs equal NumPy position by
position on the mock backend, the halo2-base proof's public values equal the
mock's, inputs outside the program's domain have no proof, and the IR shape
that makes the lowering cheaper is pinned.
"""
import numpy as np
import pytest

from testing.lang.mock_outputs import assert_matches_numpy, compile_kernel
from testing.lang.test_cost_lowerings import HALO2_PARAMS, assert_backends_agree, ir_classes


def proves(source, *args):
    """Whether halo2-base produces a proof that verifies."""
    program = compile_kernel(source)
    try:
        result = program.prove(*args, backend="halo2-base", params=HALO2_PARAMS)
    except Exception:  # noqa: BLE001 - an unsatisfiable witness raises
        return False
    return program.verify(result)


# ── domain preconditions are checked before the operation ────────────────

DOMAIN_OPS = [
    ('def kernel(a: Float[Array, "4"]):\n    return np.sqrt(a)\n', "SqrtFIR",
     np.array([0.0, 2.0, 1e-6, 1.5e6]), np.array([0.0, -1e-9, 4.0, 9.0])),
    ('def kernel(a: Float[Array, "3"]):\n    return np.arcsin(a)\n', "ArcSinFIR",
     np.array([0.5, -1.0, 0.0]), np.array([1.5, 0.0, 0.0])),
]


@pytest.mark.parametrize("source, op, good, bad", DOMAIN_OPS)
def test_precondition_precedes_the_operation(source, op, good, bad):
    classes = ir_classes(compile_kernel(source))
    first_op = classes.index(op)
    # The precondition's assertion (after the input-domain checks) comes
    # before the operation, so the lowering can use the range it proves.
    assert classes[:first_op].count("AssertIR") > 2 * len(good), classes[:first_op + 1]


@pytest.mark.parametrize("source, op, good, bad", DOMAIN_OPS)
def test_domain_ops_match_numpy_and_reject_outside_inputs(source, op, good, bad):
    assert_matches_numpy(source, good)
    assert_backends_agree(source, good)
    assert proves(source, good)
    assert not proves(source, bad)
