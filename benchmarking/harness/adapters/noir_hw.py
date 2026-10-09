"""The handwritten Noir baselines (`<case>/main.nr`; `<program>/main.<function>.nr`
for an NPBench/Pythran entry), proved like Zinnia -> Noir.

The harness drives nargo and bb itself (`noir_toolchain`), so both Noir sides
run the same commands with the same flags. What is specific to the handwritten
programs comes from the circuits workstream's `baselines/hw_check` package:

- `prover_toml.render_prover_toml(abi, fixture, input_name)` writes
  `Prover.toml` from the fixture, laid out by the compiled program's ABI;
- `noir_driver.parse_outputs(main_nr)` reads the `// @outputs:` line every
  `main.nr` carries, and `noir_driver.decode_publics(segments, fields)` turns
  bb's public inputs into the kernel's flattened output.

The package depends on `baselines/noir_hw` (the baselines' gadget library) and
`noir_fpc`, both copied in. Without `hw_check` (a checkout without the circuits
workstream) the harness writes `Prover.toml` from the fixture alone and decodes
outputs by the reference's types; a `main.nr` without an `// @outputs:` line is
decoded the same way.
"""

from __future__ import annotations

import importlib
import json
import re
import sys
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

HW_CHECK = paths.BASELINES / "hw_check"
NOIR_HW_LIBRARY = paths.BASELINES / "noir_hw"
OUTPUTS_LINE = "// @outputs:"


def _hw_check(module: str) -> ModuleType | None:
    """`hw_check.<module>` from the baselines directory, if the package is there."""
    if not (HW_CHECK / "__init__.py").exists():
        return None
    if str(paths.BASELINES) not in sys.path:
        sys.path.insert(0, str(paths.BASELINES))
    return importlib.import_module(f"hw_check.{module}")


def _encode(dtype: str, value) -> str:
    if dtype == "float":
        return str(float_to_q32(float(value)) % BN254_MODULUS)
    if dtype == "bool":
        return "1" if value else "0"
    return str(int(value) % BN254_MODULUS)


def _flatten(value) -> list:
    if isinstance(value, list):
        return [leaf for item in value for leaf in _flatten(item)]
    return [value]


def generic_prover_toml(fixture: Fixture, input_name: str) -> str:
    """Prover.toml from the fixture alone: one entry per kernel parameter, flattened."""
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
    description = ("handwritten Noir (main.nr) -> nargo 1.0.0-beta.20 compile, noir-execute, "
                   "bb UltraHonk (ZK)")
    windows = dict(
        setup="nargo compile + bb write_vk",
        prove_core="wall clock of the `bb prove` process (UltraHonk, ZK), including bb "
                   "start-up and CRS loading",
        end_to_end=noir_toolchain.END_TO_END_WINDOW,
        verify="wall clock of the `bb verify` process with the setup's verification key",
    )

    def missing_program(self, case: PerfCase) -> str | None:
        main_nr = case.baseline_file("main.nr")
        return None if main_nr.exists() else f"no {main_nr.name}"

    def program_files(self, case: PerfCase) -> list[Path]:
        files = [case.baseline_file("main.nr")]
        if NOIR_HW_LIBRARY.exists():
            files += sorted(NOIR_HW_LIBRARY.rglob("*.nr"))
        if HW_CHECK.exists():
            files += [HW_CHECK / "prover_toml.py", HW_CHECK / "noir_driver.py"]
        return files

    def _package(self, ctx: StepContext) -> noir_toolchain.NoirPackage:
        return noir_toolchain.NoirPackage(ctx.work_dir / "package", _package_name(ctx.case))

    def _inputs(self, ctx: StepContext, package: noir_toolchain.NoirPackage,
                input_name: str) -> tuple[str, str]:
        writer = _hw_check("prover_toml")
        if writer is None:
            return generic_prover_toml(ctx.fixture, input_name), "harness generic writer"
        abi = json.loads(package.bytecode.read_text())["abi"]
        fixture = json.loads(ctx.case.fixture_path.read_text())
        try:
            return (writer.render_prover_toml(abi, fixture, input_name),
                    "hw_check.prover_toml")
        except (KeyError, ValueError) as exc:
            raise StepFailure("setup_failed", f"Prover.toml: {exc}") from None

    def setup(self, ctx: StepContext) -> SetupResult:
        main_nr = ctx.case.baseline_file("main.nr").read_text()
        libraries = {"noir_fpc": paths.NOIR_FPC}
        if NOIR_HW_LIBRARY.exists():
            libraries = {"noir_hw": NOIR_HW_LIBRARY, **libraries}
        package = noir_toolchain.NoirPackage.create(ctx.work_dir / "package",
                                                    _package_name(ctx.case), main_nr, libraries)
        compile_s = package.compile()
        toml, writer = self._inputs(ctx, package, SETUP_INPUT)
        package.write_inputs("Prover_A", toml)
        write_vk_s = package.write_vk()
        gates = package.gates()
        ctx.save_state({"bytecode_sha256": noir_toolchain.sha256_file(package.bytecode)})
        return SetupResult(timings_s={"nargo_compile": compile_s, "bb_write_vk": write_vk_s},
                           details={"gates": gates, "prover_toml_writer": writer})

    def prove(self, ctx: StepContext) -> ProveResult:
        state = ctx.load_state()
        package = self._package(ctx)
        toml, _ = self._inputs(ctx, package, ctx.input_name)
        package.write_inputs("Prover_B", toml)
        timings = noir_toolchain.prove_sample(package, "Prover_B",
                                              f"witness_{ctx.proof_dir.name}", ctx.proof_dir,
                                              state["bytecode_sha256"])
        proof_path = ctx.proof_dir / "proof"
        return ProveResult(prove_core_s=timings.bb_prove, end_to_end_s=timings.end_to_end,
                           proof_path=str(proof_path), proof_bytes=proof_path.stat().st_size,
                           publics=self._publics(package, ctx.proof_dir),
                           timings_s=timings.timings_s(),
                           details={"witness_sha256": timings.witness_sha256})

    @staticmethod
    def _publics(package: noir_toolchain.NoirPackage, proof_dir: Path) -> DecodedPublics:
        fields = noir_toolchain.public_fields(proof_dir)
        main_nr = (package.directory / "src" / "main.nr").read_text()
        driver = _hw_check("noir_driver")
        if driver is None or OUTPUTS_LINE not in main_nr:
            return DecodedPublics(FIELD_ELEMENTS, [field_to_signed(f) for f in fields])
        try:
            segments = driver.parse_outputs(main_nr)
            return DecodedPublics(LOGICAL, driver.decode_publics(segments, fields))
        except driver.DriverError as exc:
            raise StepFailure("wrong_output", f"public inputs: {exc}") from None

    def verify(self, ctx: StepContext) -> VerifyResult:
        return noir_toolchain.verify_proof(self._package(ctx).vk, ctx.proof_path, ctx.tamper,
                                           has_outputs=True)
