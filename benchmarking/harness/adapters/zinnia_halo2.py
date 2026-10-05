"""Zinnia compiled to its halo2-base backend (IPA over BN254, Blake2b transcript).

Never the legacy `halo2` backend (audit M8): it resolves dynamic memory from
the witness.

The halo2-base API couples key generation with proving: every
`compiled.prove()` call generates IPA parameters, runs `keygen_vk`/`keygen_pk`
with the given witness, synthesizes the circuit twice and calls
`create_proof`. Only `create_proof` (plus transcript finalization) is timed
by the backend itself (`prove_seconds`), so the end-to-end clock of this
adapter includes key generation; see the windows below. Setup therefore runs
one full prove with input A, which is when the parameters are chosen.
"""

from __future__ import annotations

import json
import time

from benchmarking.harness.adapters.base import (Adapter, ProveResult, SetupResult,
                                                StepContext, StepFailure, VerifyResult)
from benchmarking.harness.adapters.zinnia_common import (check_same_build, compile_case,
                                                         describe_compiled, zinnia_publics)
from benchmarking.harness.fixtures import PROVE_INPUT, SETUP_INPUT
from benchmarking.harness.outputs import field_from_le_hex, field_to_le_hex

BACKEND = "halo2-base"
PRECISION_BITS = 32


def _payload(artifact: dict) -> dict:
    """The halo2-base verifying payload (hex of a JSON document)."""
    return json.loads(bytes.fromhex(artifact["vk_bytes"]))


def _prove(compiled, args, params: dict | None):
    try:
        return compiled.prove(*args, backend=BACKEND, params=params)
    except BaseException as exc:  # noqa: BLE001 - PyO3 panics derive from BaseException
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        raise StepFailure("prove_failed", f"{type(exc).__name__}: {exc}") from None


def _fingerprint(compiled, args, params: dict, backend: str) -> str:
    try:
        return compiled.emit_structural(*args, backend=backend, params=params)
    except BaseException as exc:  # noqa: BLE001
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        raise StepFailure("setup_failed", f"structural fingerprint: {exc}") from None


class ZinniaHalo2Adapter(Adapter):
    name = "zinnia_halo2"
    description = "Zinnia -> halo2-base backend (IPA/BN254, Blake2b transcript)"
    windows = {
        "setup": "compile + parameter choice + one full prove with input A (keygen happens "
                 "inside prove)",
        "prove_core": "create_proof + transcript finalize, timed inside the backend "
                      "(ProofArtifact.prove_seconds)",
        "end_to_end": "the whole compiled.prove() call: witness preprocessing, "
                      "ParamsIPA::new(k), keygen_vk + keygen_pk with the input-B witness, "
                      "keygen- and prover-stage synthesis, create_proof. Includes key "
                      "generation because the API does not separate it",
        "verify": "verify_proof_artifact: rebuilds the circuit from the IR in the artifact, "
                  "ParamsIPA::new(k), keygen_vk, verify_proof",
    }

    def setup(self, ctx: StepContext) -> SetupResult:
        started = time.perf_counter()
        compiled = compile_case(ctx.case)
        compile_s = time.perf_counter() - started
        info = describe_compiled(compiled)

        override = ctx.options.get("halo2_params")
        requested = None if override is None else {**override, "precision_bits": PRECISION_BITS}
        started = time.perf_counter()
        proof_a = _prove(compiled, ctx.fixture.arguments(SETUP_INPUT), requested)
        keygen_and_prove_s = time.perf_counter() - started
        artifact = json.loads(proof_a.to_json())
        payload = _payload(artifact)
        params = {"k": artifact["k"], "lookup_bits": payload["lookup_bits"],
                  "precision_bits": payload["precision_bits"]}

        fingerprints = {
            "A": _fingerprint(compiled, ctx.fixture.arguments(SETUP_INPUT), params, BACKEND),
            "B": _fingerprint(compiled, ctx.fixture.arguments(PROVE_INPUT), params, BACKEND),
            "no witness": _fingerprint(compiled, ctx.fixture.arguments(SETUP_INPUT), params,
                                       "halo2-base-witnessless"),
        }
        if len(set(fingerprints.values())) != 1:
            raise StepFailure("witness_dependent", json.dumps(fingerprints))

        ctx.save_state({**info, "params": params, "ir_hash": artifact["ir_hash"],
                        "fingerprint": fingerprints["A"]})
        return SetupResult(
            timings_s={"compile": compile_s, "keygen_and_prove_A": keygen_and_prove_s},
            details={"params": params, "params_source": "override" if override else
                     "Zinnia estimate_params", "cells": fingerprints["A"],
                     "exposed_outputs": len(info["expose_kinds"]),
                     "elided_outputs": len(info["elided"]), "ir_sha256": info["ir_sha256"]})

    def prove(self, ctx: StepContext) -> ProveResult:
        state = ctx.load_state()
        compiled = compile_case(ctx.case)
        check_same_build(compiled, state)
        args = ctx.arguments()
        started = time.perf_counter()
        proof = _prove(compiled, args, state["params"])
        end_to_end_s = time.perf_counter() - started

        ctx.proof_dir.mkdir(parents=True, exist_ok=True)
        proof_path = ctx.proof_dir / "artifact.json"
        proof_path.write_text(proof.to_json())
        fields = [field_from_le_hex(v) for v in proof.public_values]
        return ProveResult(
            prove_core_s=proof.prove_seconds, end_to_end_s=end_to_end_s,
            proof_path=str(proof_path), proof_bytes=len(proof.proof_bytes_hex) // 2,
            publics=zinnia_publics(fields, state),
            timings_s={"create_proof": proof.prove_seconds, "prove_call": end_to_end_s},
            details={"k": proof.k})

    def verify(self, ctx: StepContext) -> VerifyResult:
        from zinnia.compile._bridge import verify_proof_artifact

        state = ctx.load_state()
        artifact = json.loads(ctx.proof_path.read_text())
        mismatch = self._circuit_mismatch(artifact, state)
        if mismatch:
            return VerifyResult(verified=False, verify_s=None,
                                details={"rejected_before_verifier": mismatch})
        if ctx.tamper:
            if not artifact["public_values"]:
                return VerifyResult(verified=False, verify_s=None,
                                    not_applicable="the proof has no public value")
            first = field_from_le_hex(artifact["public_values"][0])
            artifact["public_values"][0] = field_to_le_hex(first + 1)
        started = time.perf_counter()
        try:
            result = json.loads(verify_proof_artifact(json.dumps(artifact)))
            verified, error = bool(result.get("valid")), result.get("error")
        except BaseException as exc:  # noqa: BLE001 - a verifier error rejects the proof
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            verified, error = False, f"{type(exc).__name__}: {exc}"
        verify_s = time.perf_counter() - started
        return VerifyResult(verified=verified, verify_s=verify_s,
                            timings_s={"verify_call": verify_s}, details={"error": error})

    @staticmethod
    def _circuit_mismatch(artifact: dict, state: dict) -> list[str]:
        """Reasons the artifact is not a proof for the circuit setup compiled."""
        problems = []
        if artifact.get("backend") != BACKEND:
            problems.append(f"backend {artifact.get('backend')!r}")
        if artifact.get("ir_hash") != state["ir_hash"]:
            problems.append("ir_hash differs from setup's")
        payload = _payload(artifact)
        for key in ("k", "lookup_bits", "precision_bits"):
            value = artifact["k"] if key == "k" else payload.get(key)
            if value != state["params"][key]:
                problems.append(f"{key}={value}, setup chose {state['params'][key]}")
        return problems
