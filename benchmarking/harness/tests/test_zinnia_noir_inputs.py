"""The harness writes Zinnia's Prover.toml and takes main.nr from `emit_structural`;
both must be exactly what Zinnia's own Noir backend stages for the same input."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from benchmarking.harness import toolchains
from benchmarking.harness.adapters.zinnia_noir import prover_toml
from benchmarking.harness.kernels import compile_kernel

KERNEL = '''from zinnia import *
import numpy as np


@zk_circuit
def kernel(a: Float[Array, "3"], n: Integer, b: Integer[Array, "2"]):
    return a * n + b[1]
'''

#: A parameter the backend renames (`v2`, a name of its own values) and complex
#: inputs (each a flat array of its (real, imaginary) leaves).
COMPLEX_KERNEL = '''from zinnia import *
import numpy as np


@zk_circuit
def kernel(v2: Complex[Array, "2 2"], c: Complex, n: Integer):
    return v2[1, 0] * c + n
'''

pytestmark = pytest.mark.skipif(
    not (toolchains.nargo_path().exists() and toolchains.bb_path().exists()),
    reason="nargo/bb not installed")


def stage_with_backend(tmp_path, monkeypatch, args, source=KERNEL):
    """Compile a toy kernel and let Zinnia's Noir backend stage a package for `args`."""
    kernel = tmp_path / "toy_noir" / "kernel.py"
    kernel.parent.mkdir()
    kernel.write_text(source)
    compiled = compile_kernel(kernel, "kernel")
    staging = tmp_path / "staging"
    staging.mkdir()
    monkeypatch.setenv("TMPDIR", str(staging))
    monkeypatch.setenv("ZINNIA_NOIR_KEEP_TEMP", "1")
    monkeypatch.setenv("ZINNIA_NARGO_BIN", str(toolchains.nargo_path()))
    monkeypatch.setenv("ZINNIA_BB_BIN", str(toolchains.bb_path()))
    try:
        compiled.prove(*args, backend="noir")
    except Exception:  # noqa: BLE001 - the staged package is what is compared
        pass
    staged = [d for d in staging.iterdir() if d.name.startswith("zinnia-noir-")]
    assert len(staged) == 1, os.listdir(staging)
    return compiled, staged[0]


def harness_toml(compiled, args) -> str:
    from zinnia.exec.input_parser import build_circuit_inputs

    return prover_toml(build_circuit_inputs(compiled.program_inputs, args))


@pytest.mark.parametrize("source, args", [
    (KERNEL, [np.array([0.1, -2.75, 1e-3]), -3, np.array([-4, 9])]),
    (COMPLEX_KERNEL, [np.array([[1.5 - 2j, 0.25j], [-3 + 0.5j, 2.0]]), 0.5 - 1.25j, 7]),
])
def test_prover_toml_and_main_nr_match_the_backend(tmp_path, monkeypatch, source, args):
    compiled, staged = stage_with_backend(tmp_path, monkeypatch, args, source)
    assert harness_toml(compiled, args) == (staged / "Prover.toml").read_text()
    assert compiled.emit_structural(*args, backend="noir") == \
        (staged / "src" / "main.nr").read_text()
    assert (staged / "noir_fpc" / "src" / "lib.nr").read_text() == \
        (Path(__file__).resolve().parents[2] / "noir_fpc" / "src" / "lib.nr").read_text()


def test_half_way_values_differ_by_at_most_one_q32_unit(tmp_path, monkeypatch):
    """Exactly half-way values (v * 2^32 = k + 1/2) are not reproduced.

    The harness rounds the exact double half away from zero, as
    `prove::field::quantize` does; the backend quantizes the value its JSON
    parser read, which can be one ULP off, so it sometimes rounds the other
    way. The difference is one unit of 2^-32 in an input.
    """
    args = [np.array([0.5, 1.5, 2.5]) / 2**32, 1, np.array([0, 0])]
    compiled, staged = stage_with_backend(tmp_path, monkeypatch, args)
    ours = harness_toml(compiled, args).splitlines()[0]
    theirs = (staged / "Prover.toml").read_text().splitlines()[0]
    parse = lambda line: [int(v.strip(' "')) for v in line.split("[")[1].rstrip("]").split(",")]
    assert all(abs(a - b) <= 1 for a, b in zip(parse(ours), parse(theirs)))
