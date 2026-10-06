"""Zinnia compiled to its halo2-base backend (IPA over BN254, Blake2b transcript).

Never the legacy `halo2` backend (audit M8): it resolves dynamic memory from
the witness.

The steps mirror `halo2_hw`'s (`baselines/halo2_hw/src/scaffold.rs`); each
runs in its own process:

- **setup** (input A): compile; generate the IPA parameters and the proving
  and verifying keys without a witness (`compiled.setup`: the synthesis a
  verifier runs, at the `k` of Zinnia's `estimate_params` unless the run fixes
  it); write them to `<work>/keys` in halo2_hw's encodings (`params.bin`,
  `pk.bin`, `vk.bin` raw bytes, plus `keys.json` with the pinned layout and
  the IR hash). The keygen-stage circuit shape must be the same for input A,
  input B and no witness (`witness_dependent` otherwise). Keys made without a
  witness cannot depend on A; proofs are made with B, so a circuit whose shape
  followed the input would fail to prove or to verify.
- **prove** (input B): recompile (untimed; the IR digest must equal setup's);
  read `params.bin` and `pk.bin` (untimed, reported); witness and proof
  (`compiled.prove_with_keys`); write `proof.bin` and `public_values.json`.
- **verify**: read `params.bin` and `vk.bin` (untimed, reported) and run
  `verify_proof` on `proof.bin` with `public_values.json` (the tamper check
  first adds 1 to the first public value). The verifier never sees the
  circuit description or anything else the prover wrote.

What each clock covers, next to halo2_hw (`contract.PROGRAMS["halo2_hw"]`):

| clock | zinnia_halo2 | halo2_hw |
|---|---|---|
| prove core | `prove_s`: `create_proof` + transcript finalize | `prove_s`: the same two calls |
| end to end | `preprocess_s` + `witness_s` + `prove_s` | `witness_s` + `prove_s` |
| verify | `verify_s`: transcript + `verify_proof` | `verify_s`: the same calls |

`witness_s` on both sides is the synthesis into halo2-base's builder pinned
to keygen's layout (Zinnia walks its IR; halo2_hw runs the handwritten
circuit function). `preprocess_s` is a Zinnia-only pass that evaluates the IR
once to resolve external calls; it runs on every proof, so it counts.

Outside every clock, on both sides: process start-up, compilation (Zinnia;
halo2_hw's circuit is compiled into the binary), reading the inputs (Zinnia:
the Python input builder and `parse_inputs_s`; halo2_hw: `load_input_s`),
reading the parameters and key (`load_params_s`, `load_pk_s` / `load_vk_s`),
starting the proving thread pool (`thread_pool_s`) and writing the proof.
Zinnia-only and outside the clocks: the check that the keys belong to the
compiled circuit (its IR hash, computed once per process; about 16 ms of the
first prove call on 0381's 10k statements). `prove_call_s` and
`verify_call_s` time the whole Python call, everything included.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from benchmarking.harness.adapters.base import (Adapter, ProveResult, SetupResult,
                                                StepContext, StepFailure, VerifyResult)
from benchmarking.harness.adapters.zinnia_common import (check_same_build, compile_case,
                                                         describe_compiled, zinnia_publics)
from benchmarking.harness.fixtures import PROVE_INPUT, SETUP_INPUT
from benchmarking.harness.outputs import field_from_le_hex, field_to_le_hex

PRECISION_BITS = 32
KEYS_DIR = "keys"
PROOF_FILE = "proof.bin"
PUBLICS_FILE = "public_values.json"

#: `load_seconds` entries of a key handle -> the names halo2_hw reports.
_LOAD_TIMING_NAMES = {"params": "load_params_s", "proving_key": "load_pk_s",
                      "verifying_key": "load_vk_s"}


def _is_interrupt(exc: BaseException) -> bool:
    return isinstance(exc, (KeyboardInterrupt, SystemExit))


#: How the halo2-base backend names an IR operation it has no gadget for
#: (`Synthesizer::unsupported`): the program is outside what this backend
#: proves, which is `unsupported`, not a failed setup.
UNSUPPORTED_OPERATION = "halo2-base backend does not yet support"


def _keygen(compiled, params: dict | None):
    try:
        return compiled.setup(params=params)
    except BaseException as exc:  # noqa: BLE001 - PyO3 panics derive from BaseException
        if _is_interrupt(exc):
            raise
        failure_class = "unsupported" if UNSUPPORTED_OPERATION in str(exc) else "setup_failed"
        raise StepFailure(failure_class, f"keygen: {type(exc).__name__}: {exc}") from None


def _load_keys(ctx: StepContext, role: str, state: dict):
    """The setup's keys for `role`, checked to be the keys of setup's circuit."""
    from zinnia import ZKCompiledProgram

    try:
        keys = ZKCompiledProgram.load_keys(ctx.work_dir / KEYS_DIR, role=role)
    except BaseException as exc:  # noqa: BLE001
        if _is_interrupt(exc):
            raise
        raise StepFailure("harness_error", f"reading the setup's keys: {exc}") from None
    if keys.ir_hash != state["ir_hash"] or keys.k != state["params"]["k"]:
        raise StepFailure("harness_error", "the key directory does not hold setup's keys")
    return keys


def _load_timings(keys) -> dict[str, float]:
    return {_LOAD_TIMING_NAMES[name]: seconds for name, seconds in keys.load_seconds.items()}


def _prove(compiled, keys, args) -> dict:
    try:
        return compiled.prove_with_keys(keys, *args)
    except BaseException as exc:  # noqa: BLE001 - PyO3 panics derive from BaseException
        if _is_interrupt(exc):
            raise
        raise StepFailure("prove_failed", f"{type(exc).__name__}: {exc}") from None


def _fingerprint(compiled, args, params: dict, backend: str) -> str:
    try:
        return compiled.emit_structural(*args, backend=backend, params=params)
    except BaseException as exc:  # noqa: BLE001
        if _is_interrupt(exc):
            raise
        raise StepFailure("setup_failed", f"structural fingerprint: {exc}") from None


def _write_proof(proof_dir: Path, out: dict) -> Path:
    proof_dir.mkdir(parents=True, exist_ok=True)
    proof_path = proof_dir / PROOF_FILE
    proof_path.write_bytes(bytes.fromhex(out["proof"]))
    (proof_dir / PUBLICS_FILE).write_text(json.dumps(out["public_values"]))
    return proof_path


def _read_proof(proof_path: Path) -> dict:
    """The proof as `verify_with_keys_report` takes it (hex proof, hex publics)."""
    return {"proof": proof_path.read_bytes().hex(),
            "public_values": json.loads((proof_path.parent / PUBLICS_FILE).read_text())}


class ZinniaHalo2Adapter(Adapter):
    name = "zinnia_halo2"
    description = "Zinnia -> halo2-base backend (IPA/BN254, Blake2b transcript)"
    windows = {
        "setup": "compile + parameter choice + witness-free keygen (ParamsIPA::new, keygen_vk, "
                 "keygen_pk) + writing params/pk/vk to the key directory",
        "prove_core": "prove_s: create_proof + transcript finalize (halo2_hw's prove_s covers "
                      "the same calls)",
        "end_to_end": "preprocess_s (IR evaluation resolving external calls) + witness_s "
                      "(synthesis into the builder pinned to keygen's layout) + prove_s; "
                      "params and proving key read from files beforehand, untimed "
                      "(load_params_s, load_pk_s), as halo2_hw",
        "verify": "verify_s: transcript + verify_proof against the verifying key read from "
                  "the key directory (load_params_s, load_vk_s untimed), as halo2_hw",
    }

    def setup(self, ctx: StepContext) -> SetupResult:
        started = time.perf_counter()
        compiled = compile_case(ctx.case, backend="halo2")
        compile_s = time.perf_counter() - started
        info = describe_compiled(compiled)

        override = ctx.options.get("halo2_params")
        requested = None if override is None else {**override, "precision_bits": PRECISION_BITS}
        keys = _keygen(compiled, requested)
        params = {"k": keys.k, "lookup_bits": keys.lookup_bits,
                  "precision_bits": keys.precision_bits}
        started = time.perf_counter()
        keys.save(ctx.work_dir / KEYS_DIR)
        write_s = time.perf_counter() - started

        fingerprints = {
            "A": _fingerprint(compiled, ctx.fixture.arguments(SETUP_INPUT), params, "halo2-base"),
            "B": _fingerprint(compiled, ctx.fixture.arguments(PROVE_INPUT), params, "halo2-base"),
            "no witness": _fingerprint(compiled, ctx.fixture.arguments(SETUP_INPUT), params,
                                       "halo2-base-witnessless"),
        }
        if len(set(fingerprints.values())) != 1:
            raise StepFailure("witness_dependent", json.dumps(fingerprints))

        ctx.save_state({**info, "params": params, "ir_hash": keys.ir_hash,
                        "fingerprint": fingerprints["A"]})
        key_files = {p.name: p.stat().st_size for p in sorted((ctx.work_dir / KEYS_DIR).iterdir())}
        return SetupResult(
            timings_s={"compile": compile_s, "keygen_s": keys.keygen_seconds, "write_s": write_s},
            details={"params": params, "params_source": "override" if override else
                     "Zinnia estimate_params", "cells": fingerprints["A"],
                     "key_file_bytes": key_files,
                     "exposed_outputs": len(info["expose_kinds"]),
                     "elided_outputs": len(info["elided"]), "ir_sha256": info["ir_sha256"]})

    def prove(self, ctx: StepContext) -> ProveResult:
        state = ctx.load_state()
        compiled = compile_case(ctx.case, backend="halo2")
        check_same_build(compiled, state)
        keys = _load_keys(ctx, "prover", state)
        args = ctx.arguments()
        started = time.perf_counter()
        out = _prove(compiled, keys, args)
        call_s = time.perf_counter() - started

        proof_path = _write_proof(ctx.proof_dir, out)
        fields = [field_from_le_hex(v) for v in out["public_values"]]
        timings = {**_load_timings(keys),
                   "parse_inputs_s": out["parse_inputs_seconds"],
                   "preprocess_s": out["preprocess_seconds"],
                   "witness_s": out["witness_seconds"],
                   "thread_pool_s": out["thread_pool_seconds"],
                   "prove_s": out["prove_seconds"],
                   "prove_call_s": call_s}
        return ProveResult(
            prove_core_s=out["prove_seconds"],
            end_to_end_s=out["preprocess_seconds"] + out["witness_seconds"] + out["prove_seconds"],
            proof_path=str(proof_path), proof_bytes=proof_path.stat().st_size,
            publics=zinnia_publics(fields, state), timings_s=timings, details={"k": out["k"]})

    def verify(self, ctx: StepContext) -> VerifyResult:
        from zinnia import ZKCompiledProgram

        state = ctx.load_state()
        keys = _load_keys(ctx, "verifier", state)
        proof = _read_proof(ctx.proof_path)
        if ctx.tamper:
            if not proof["public_values"]:
                return VerifyResult(verified=False, verify_s=None,
                                    not_applicable="the proof has no public value")
            first = field_from_le_hex(proof["public_values"][0])
            proof["public_values"][0] = field_to_le_hex(first + 1)
        started = time.perf_counter()
        try:
            report = ZKCompiledProgram.verify_with_keys_report(keys, proof)
        except BaseException as exc:  # noqa: BLE001 - a verifier error rejects the proof
            if _is_interrupt(exc):
                raise
            report = {"valid": False, "error": f"{type(exc).__name__}: {exc}",
                      "verify_seconds": None}
        call_s = time.perf_counter() - started
        timings = {**_load_timings(keys), "verify_call_s": call_s}
        if report["verify_seconds"] is not None:
            timings["verify_s"] = report["verify_seconds"]
        return VerifyResult(verified=bool(report["valid"]), verify_s=report["verify_seconds"],
                            timings_s=timings, details={"error": report["error"]})
