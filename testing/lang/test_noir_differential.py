"""Noir backend vs mock backend on a sample of performance-set kernels.

The full sweep is `benchmarking/noir_differential.py` (all cases with an
input fixture); this test runs the same comparison on a sample that covers
the lowering families: Euclidean `//` and `%` (constant and pinned
divisors), signed and interval-driven comparisons, non-negativity asserts,
one- and two-root dynamic gathers, boolean logic, float division, sqrt,
sin, arcsin and integer powers. Each output must equal the mock's
position by position (within the documented tolerance for kernels using
approximated functions).
"""

import shutil
import sys
from pathlib import Path

import pytest

BENCH = Path(__file__).resolve().parents[2] / "benchmarking"
sys.path.insert(0, str(BENCH))

import noir_differential  # noqa: E402

CASES = [
    "0303", "0311", "0318", "0325", "0326", "0329", "0334", "0353",
    "0391", "0414", "0428", "0441", "0452", "0503", "0507",
]


def _nargo_available() -> bool:
    return (Path.home() / ".nargo" / "bin" / "nargo").is_file() or shutil.which("nargo") is not None


@pytest.mark.skipif(not _nargo_available(), reason="nargo not installed")
@pytest.mark.parametrize("case", CASES)
def test_noir_matches_mock(case):
    rec = noir_differential.run_child(case, prove=False, rtol=1e-6, atol=1e-6)
    assert rec["status"] in ("match", "match_approx"), rec
