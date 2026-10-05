"""Run the `noir_fpc` library's own Noir tests (`nargo test`).

`benchmarking/noir_fpc/src/lib.nr` is the fixed-point library linked by the
Noir backend and by the handwritten Noir baselines. Its tests include the
malicious-witness checks: each `#[test(should_fail)]` feeds a constraint
function the hint values a cheating prover would supply (a field-inverse
quotient, an out-of-range remainder, a negative zero, a floor instead of a
rounded square root) and must be rejected.
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

LIB_DIR = Path(__file__).resolve().parents[2] / "benchmarking" / "noir_fpc"


def _nargo() -> str | None:
    explicit = os.environ.get("ZINNIA_NARGO_BIN")
    if explicit:
        return explicit
    home = Path.home() / ".nargo" / "bin" / "nargo"
    if home.is_file():
        return str(home)
    return shutil.which("nargo")


@pytest.mark.skipif(_nargo() is None, reason="nargo not installed")
def test_noir_fpc_nargo_tests_pass():
    proc = subprocess.run(
        [_nargo(), "test", "--silence-warnings"],
        cwd=LIB_DIR,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 0, output[-4000:]
    assert "tests passed" in output
