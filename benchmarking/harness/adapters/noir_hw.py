"""The handwritten Noir baselines (`<case>/main.nr`), proved like Zinnia -> Noir.

The circuits workstream owns the baselines and the fixture -> `Prover.toml`
writer. The harness looks for that writer as a Python module at
`WRITER_MODULE` exposing

    write_prover_toml(case_dir: Path, fixture_path: Path, input_name: str) -> str
    decode_publics(case_dir: Path, fields: list[int]) -> list   # optional

(`fields` are the proof's public values as signed integers, in order;
`decode_publics` returns the kernel's flattened output). Without that module
the harness writes `Prover.toml` itself from the fixture (parameters in
signature order, floats in Q32, arrays flattened row-major) and decodes the
public values by the type of the reference output at the same position.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path
from types import ModuleType

from benchmarking.harness import paths
from benchmarking.harness.adapters import noir_toolchain
from benchmarking.harness.adapters.base import (Adapter, ProveResult, SetupResult,
                                                StepContext, StepFailure, VerifyResult)
from benchmarking.harness.case_sets import PerfCase
from benchmarking.harness.fixtures import SETUP_INPUT, Fixture
from benchmarking.harness.outputs import (BN254_MODULUS, FIELD_ELEMENTS, LOGICAL,
                                          DecodedPublics, field_to_signed, float_to_q32)

WRITER_MODULE = paths.BASELINES / "noir_hw" / "prover_toml.py"


def _writer_module() -> ModuleType | None:
    if not WRITER_MODULE.exists():
        return None
    spec = importlib.util.spec_from_file_location("noir_hw_prover_toml", WRITER_MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _field_decimal(value: int) -> str:
    return str(value % BN254_MODULUS)


def _encode(dtype: str, value) -> str:
    if dtype == "float":
        return _field_decimal(float_to_q32(float(value)))
    if dtype == "bool":
        return "1" if value else "0"
    return _field_decimal(int(value))


def _flatten(value) -> list:
    if isinstance(value, list):
        return [leaf for item in value for leaf in _flatten(item)]
    return [value]


def generic_prover_toml(fixture: Fixture, input_name: str) -> str:
    """Prover.toml from the fixture alone: one entry per kernel parameter."""
    values = fixture.values(input_name)
    lines = []
    for param in fixture.parameters:
        value = values[param.name]
        if param.kind == "scalar":
            lines.append(f'{param.name} = "{_encode(param.dtype, value)}"')
        else:
            encoded = ", ".join(f'"{_encode(param.dtype, v)}"' for v in _flatten(value))
            lines.append(f"{param.name} = [{encoded}]")
    return "\n".join(lines) + "\n"


def _package_name(case: PerfCase) -> str:
    return "hw_" + re.sub(r"\W", "_", case.case)


class NoirHandwrittenAdapter(Adapter):
    name = "noir_hw"
    description = "handwritten Noir (main.nr) -> nargo 1.0.0-beta.20 + bb UltraHonk (ZK)"
    windows = dict(
        setup="nargo compile + bb write_vk",
        prove_core="wall clock of the `bb prove` process (UltraHonk, ZK), including bb "
                   "start-up and CRS loading",
        end_to_end="`nargo execute` (witness generation; ACIR already compiled) + `bb prove`",
        verify="wall clock of the `bb verify` process with the setup's verification key",
    )

    def missing_program(self, case: PerfCase) -> str | None:
        return None if (case.directory / "main.nr").exists() else "no main.nr"

    def program_files(self, case: PerfCase) -> list[Path]:
        files = [case.directory / "main.nr"]
        return files + ([WRITER_MODULE] if WRITER_MODULE.exists() else [])

    def _inputs(self, ctx: StepContext, input_name: str) -> tuple[str, str]:
        writer = _writer_module()
        if writer is None:
            return generic_prover_toml(ctx.fixture, input_name), "harness generic writer"
        return (writer.write_prover_toml(ctx.case.directory, ctx.case.fixture_path, input_name),
                paths.relative_to_repo(WRITER_MODULE))

    def setup(self, ctx: StepContext) -> SetupResult:
        main_nr = (ctx.case.directory / "main.nr").read_text()
        package = noir_toolchain.NoirPackage.create(ctx.work_dir / "package",
                                                    _package_name(ctx.case), main_nr)
        toml, writer = self._inputs(ctx, SETUP_INPUT)
        package.write_inputs("Prover_A", toml)
        compile_s = package.compile()
        write_vk_s = package.write_vk()
        gates = package.gates()
        ctx.save_state({"bytecode_sha256": noir_toolchain.sha256_file(package.bytecode)})
        return SetupResult(timings_s={"nargo_compile": compile_s, "bb_write_vk": write_vk_s},
                           details={"gates": gates, "prover_toml_writer": writer})

    def prove(self, ctx: StepContext) -> ProveResult:
        state = ctx.load_state()
        package = noir_toolchain.NoirPackage(ctx.work_dir / "package", _package_name(ctx.case))
        toml, _ = self._inputs(ctx, ctx.input_name)
        package.write_inputs("Prover_B", toml)
        witness_s, witness = package.execute("Prover_B", f"witness_{ctx.proof_dir.name}")
        if noir_toolchain.sha256_file(package.bytecode) != state["bytecode_sha256"]:
            raise StepFailure("nondeterministic_build", "nargo execute changed the ACIR")
        prove_s = package.prove(witness, ctx.proof_dir)
        fields = [field_to_signed(f) for f in noir_toolchain.public_fields(ctx.proof_dir)]
        writer = _writer_module()
        if writer is not None and hasattr(writer, "decode_publics"):
            publics = DecodedPublics(LOGICAL, writer.decode_publics(ctx.case.directory, fields))
        else:
            publics = DecodedPublics(FIELD_ELEMENTS, fields)
        proof_path = ctx.proof_dir / "proof"
        return ProveResult(prove_core_s=prove_s, end_to_end_s=witness_s + prove_s,
                           proof_path=str(proof_path), proof_bytes=proof_path.stat().st_size,
                           publics=publics,
                           timings_s={"nargo_execute": witness_s, "bb_prove": prove_s})

    def verify(self, ctx: StepContext) -> VerifyResult:
        package = noir_toolchain.NoirPackage(ctx.work_dir / "package", _package_name(ctx.case))
        return noir_toolchain.verify_proof(package.vk, ctx.proof_path, ctx.tamper,
                                           has_outputs=True)
