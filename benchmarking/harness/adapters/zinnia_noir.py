"""Zinnia compiled to Noir source, proved with Barretenberg UltraHonk (ZK).

The harness does not use the backend's own `prove` (which stages a fresh
package per call and proves without zero knowledge). It takes the emitted
`main.nr` from `emit_structural`, checks it is byte-identical for inputs A and
B (the circuit must not depend on the witness), builds one package with it,
compiles it and writes the verification key once in setup, and then proves
input B on that package with the flags `noir_toolchain` uses for both Noir
sides. `Prover.toml` is written from Zinnia's own witness builder, encoded the
way the backend's `emit_prover_toml` encodes it (src/prove/noir/emit.rs); a
test checks both files against what the backend stages. One known difference:
for an input exactly half-way between two Q32 values the backend may round the
other way (it quantizes the value its JSON parser read, which can be one ULP
off), so such an input can differ by one unit of 2^-32.
"""

from __future__ import annotations

import time
from typing import Any

from benchmarking.harness.adapters import noir_toolchain
from benchmarking.harness.adapters.base import (Adapter, ProveResult, SetupResult,
                                                StepContext, StepFailure, VerifyResult)
from benchmarking.harness.adapters.zinnia_common import (compile_case, describe_compiled,
                                                         zinnia_publics)
from benchmarking.harness.fixtures import PROVE_INPUT, SETUP_INPUT
from benchmarking.harness.outputs import BN254_MODULUS, float_to_q32

PACKAGE_NAME = "zinnia_circuit"


# ---------------------------------------------------------------------------
# Prover.toml, mirroring emit_prover_toml (src/prove/noir/emit.rs)
# ---------------------------------------------------------------------------


def _signed_field_decimal(value: int) -> str:
    return str(value) if value >= 0 else str(BN254_MODULUS - (-value))


def _leaf_decimal(node: dict) -> str:
    if "Int" in node:
        # nargo reads field elements: a negative integer is written as p - |v|.
        return _signed_field_decimal(int(node["Int"]))
    if "Float" in node:
        return _signed_field_decimal(float_to_q32(node["Float"]))
    if "Bool" in node:
        return "1" if node["Bool"] else "0"
    if "Hex" in node:
        return "0x" + node["Hex"]
    return "0"


def _flat_leaves(elements: list[dict]) -> list[dict]:
    leaves = []
    for element in elements:
        if "Array" in element:
            leaves.extend(_flat_leaves(element["Array"]["elements"]))
        elif "Sequence" in element:
            leaves.extend(_flat_leaves(element["Sequence"]))
        else:
            leaves.append(element)
    return leaves


def _param_line(name: str, node: dict) -> str:
    if "Array" in node:
        values = ", ".join(f'"{_leaf_decimal(leaf)}"'
                           for leaf in _flat_leaves(node["Array"]["elements"]))
        return f"{name} = [{values}]"
    if "Hex" in node:
        return f'{name} = "{node["Hex"]}"'
    if any(key in node for key in ("Int", "Float", "Bool")):
        return f'{name} = "{_leaf_decimal(node)}"'
    raise StepFailure("unsupported", f"input {name!r}: {next(iter(node))} inputs have no "
                      "Prover.toml encoding in the Noir backend")


def prover_toml(witness: dict) -> str:
    """The Prover.toml the Noir backend would write for this witness."""
    return "".join(_param_line(p["name"], p["value"]) + "\n" for p in witness["params"])


def _witness(program_inputs: list[dict], args: list[Any]) -> dict:
    from zinnia.api.zk_program_input import ZKProgramInput
    from zinnia.exec.input_parser import build_circuit_inputs

    return build_circuit_inputs([ZKProgramInput.import_from(p) for p in program_inputs], args)


def _emit_main_nr(compiled, args: list[Any]) -> str:
    try:
        return compiled.emit_structural(*args, backend="noir")
    except BaseException as exc:  # noqa: BLE001 - PyO3 panics derive from BaseException
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        raise StepFailure("unsupported", f"Noir emission: {type(exc).__name__}: {exc}") from None


class ZinniaNoirAdapter(Adapter):
    name = "zinnia_noir"
    description = "Zinnia -> Noir source -> nargo 1.0.0-beta.20 + bb UltraHonk (ZK)"
    windows = {
        "setup": "compile + emit main.nr for A and B + nargo compile + bb write_vk",
        "prove_core": "wall clock of the `bb prove` process (UltraHonk, ZK), including bb "
                      "start-up and CRS loading",
        "end_to_end": "`nargo execute` (witness generation; ACIR already compiled) + "
                      "`bb prove`",
        "verify": "wall clock of the `bb verify` process with the setup's verification key",
    }

    def setup(self, ctx: StepContext) -> SetupResult:
        started = time.perf_counter()
        compiled = compile_case(ctx.case, backend="noir")
        compile_s = time.perf_counter() - started
        info = describe_compiled(compiled)
        args_a = ctx.fixture.arguments(SETUP_INPUT)
        main_a = _emit_main_nr(compiled, args_a)
        main_b = _emit_main_nr(compiled, ctx.fixture.arguments(PROVE_INPUT))
        if main_a != main_b:
            raise StepFailure("witness_dependent", "emitted main.nr differs between inputs A and B")

        package = noir_toolchain.NoirPackage.create(ctx.work_dir / "package", PACKAGE_NAME, main_a)
        package.write_inputs("Prover_A", prover_toml(_witness(info["program_inputs"], args_a)))
        nargo_compile_s = package.compile()
        write_vk_s = package.write_vk()
        gates = package.gates()
        ctx.save_state({**info, "bytecode_sha256": noir_toolchain.sha256_file(package.bytecode)})
        return SetupResult(
            timings_s={"compile": compile_s, "nargo_compile": nargo_compile_s,
                       "bb_write_vk": write_vk_s},
            details={"gates": gates, "exposed_outputs": len(info["expose_kinds"]),
                     "elided_outputs": len(info["elided"]), "ir_sha256": info["ir_sha256"]})

    def prove(self, ctx: StepContext) -> ProveResult:
        state = ctx.load_state()
        package = noir_toolchain.NoirPackage(ctx.work_dir / "package", PACKAGE_NAME)
        package.write_inputs("Prover_B", prover_toml(_witness(state["program_inputs"],
                                                               ctx.arguments())))
        witness_s, witness = package.execute("Prover_B", f"witness_{ctx.proof_dir.name}")
        if noir_toolchain.sha256_file(package.bytecode) != state["bytecode_sha256"]:
            raise StepFailure("nondeterministic_build", "nargo execute changed the ACIR")
        prove_s = package.prove(witness, ctx.proof_dir)

        # With no exposed output the emitter adds one synthetic public (an input
        # cell) so bb accepts the circuit; it is not an output.
        fields = noir_toolchain.public_fields(ctx.proof_dir) if state["expose_kinds"] else []
        proof_path = ctx.proof_dir / "proof"
        return ProveResult(
            prove_core_s=prove_s, end_to_end_s=witness_s + prove_s,
            proof_path=str(proof_path), proof_bytes=proof_path.stat().st_size,
            publics=zinnia_publics(fields, state),
            timings_s={"nargo_execute": witness_s, "bb_prove": prove_s})

    def verify(self, ctx: StepContext) -> VerifyResult:
        state = ctx.load_state()
        package = noir_toolchain.NoirPackage(ctx.work_dir / "package", PACKAGE_NAME)
        return noir_toolchain.verify_proof(package.vk, ctx.proof_path, ctx.tamper,
                                           has_outputs=bool(state["expose_kinds"]))
