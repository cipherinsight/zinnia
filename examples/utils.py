import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from zinnia import ZKCircuit


# High-level purpose:
# This module centralizes the reusable "example pipeline" for Zinnia circuits:
# (1) build/compile a circuit, (2) materialize Halo2 source + input files,
# (3) invoke keygen/prove/verify in the downstream Halo2 project, and
# (4) return strongly-typed result objects for each stage.

DEFAULT_HALO2_FOLDER = "/home/zhantong/halo2-graph"
DEFAULT_K = 16


@dataclass
class Halo2Artifacts:
	# Paths and identifiers shared across keygen/prove/verify actions.
	halo2_folder: Path
	circuit_name: str
	k: int
	source_path: Path
	input_path: Path
	proving_key_path: Path
	verifying_key_path: Path
	snark_path: Path


@dataclass
class KeygenResult:
	# Captures whether key generation succeeded and preserves command output.
	artifacts: Halo2Artifacts
	success: bool
	keygen_output: str


@dataclass
class ProveResult:
	# Captures proving status and emitted logs for diagnostics.
	artifacts: Halo2Artifacts
	success: bool
	prove_output: str


@dataclass
class VerifyResult:
	# Captures both command success and semantic verification outcome.
	artifacts: Halo2Artifacts
	success: bool
	verified: bool
	verify_output: str


def _default_env() -> dict[str, str]:
	# Provide sane defaults for stack/lookup parameters while still allowing
	# callers to override through environment variables.
	env = os.environ.copy()
	env["RUST_MIN_STACK"] = env.get("RUST_MIN_STACK", "536870912")
	env["LOOKUP_BITS"] = env.get("LOOKUP_BITS", "6")
	return env


def _ensure_halo2_paths(circuit_name: str, k: int) -> Halo2Artifacts:
	# Resolve output locations for Rust source, JSON input, keys, and proof.
	folder = Path(DEFAULT_HALO2_FOLDER).expanduser().resolve()
	source_path = folder / "examples" / f"{circuit_name}.rs"
	input_path = folder / "data" / f"{circuit_name}.in"
	proving_key_path = folder / "data" / f"{circuit_name}.pk"
	verifying_key_path = folder / "data" / f"{circuit_name}.vk"
	snark_path = folder / "data" / f"{circuit_name}.snark"

	# Create directories ahead of time so subsequent writes are deterministic.
	source_path.parent.mkdir(parents=True, exist_ok=True)
	input_path.parent.mkdir(parents=True, exist_ok=True)

	return Halo2Artifacts(
		halo2_folder=folder,
		circuit_name=circuit_name,
		k=k,
		source_path=source_path,
		input_path=input_path,
		proving_key_path=proving_key_path,
		verifying_key_path=verifying_key_path,
		snark_path=snark_path,
	)


def _normalize_value(value: Any) -> str:
	# Integer-like lanes are serialized as field-element strings.
	if isinstance(value, bool):
		return "1" if value else "0"
	return str(value)


def _normalize_entry_value(entry: Any) -> Any:
	# Float lanes are deserialized as f64 in generated Rust input structs, so they
	# must remain JSON numbers instead of quoted strings.
	if entry.is_float():
		return float(entry.get_value())
	return _normalize_value(entry.get_value())


def _is_low_level_mapping(payload: Mapping[str, Any]) -> bool:
	# Detect pre-flattened key space expected by lower-level runner flows.
	keys = [str(key) for key in payload.keys()]
	return all(key.startswith("x_") or key.startswith("hash_") for key in keys)


def _to_input_mapping(circuit: ZKCircuit, data: Any) -> dict[str, Any]:
	# Feature highlight:
	# This adapter accepts multiple caller-facing formats and normalizes them
	# into the exact key/value mapping consumed by the Halo2 example binary.
	if isinstance(data, Mapping):
		if _is_low_level_mapping(data):
			return {str(k): _normalize_value(v) for k, v in data.items()}
		# Preferred path: use circuit argparse to encode typed inputs correctly.
		parsed = circuit.argparse(**dict(data))
		return {entry.get_key(): _normalize_entry_value(entry) for entry in parsed.entries}

	if isinstance(data, str):
		# JSON payload support helps scripted pipelines and fixture files.
		payload = json.loads(data)
		if not isinstance(payload, dict):
			raise TypeError("JSON data must decode to a mapping of key-value entries.")
		if _is_low_level_mapping(payload):
			return {str(k): _normalize_value(v) for k, v in payload.items()}
		parsed = circuit.argparse(**payload)
		return {entry.get_key(): _normalize_entry_value(entry) for entry in parsed.entries}

	if isinstance(data, Sequence) and not isinstance(data, (bytes, bytearray, str)):
		# Positional argument path mirrors direct function-call style.
		parsed = circuit.argparse(*data)
		return {entry.get_key(): _normalize_entry_value(entry) for entry in parsed.entries}

	raise TypeError(
		"Unsupported `data` type. Use one of: mapping, JSON string, or a sequence of circuit arguments."
	)


def _write_input_data(circuit: ZKCircuit, data: Any, input_path: Path) -> dict[str, Any]:
	# Convert and persist witness/public values for the Rust runner.
	mapped = _to_input_mapping(circuit, data)
	with input_path.open("w", encoding="utf-8") as fp:
		json.dump(mapped, fp, indent=2)
	return mapped


def _run_halo2_action(artifacts: Halo2Artifacts, action: str, env: Mapping[str, str] | None = None) -> tuple[bool, str]:
	# Build a single command template for keygen/prove/verify actions.
	command = [
		"cargo",
		"run",
		"--example",
		artifacts.circuit_name,
		"--",
		"--name",
		artifacts.circuit_name,
		"-k",
		str(artifacts.k),
		"--input",
		artifacts.input_path.name,
		action,
	]

	# Merge defaults with optional per-call overrides.
	active_env = _default_env()
	if env is not None:
		active_env.update(dict(env))

	# Execute in the Halo2 workspace and return full output for diagnostics.
	print(f"[halo2] running action={action} in {artifacts.halo2_folder}")
	print(f"[halo2] command: {' '.join(command)}")
	process = subprocess.run(
		command,
		cwd=str(artifacts.halo2_folder),
		capture_output=True,
		text=True,
		env=active_env,
	)
	output = process.stdout + process.stderr
	print(f"[halo2] action={action} finished with returncode={process.returncode}")
	return process.returncode == 0, output


def create_circuit(circuit_method, chips: list[Any] | None = None) -> ZKCircuit:
	# Wrap `ZKCircuit.from_method` with lightweight logging for example UX.
	print("[zinnia] creating circuit object from method")
	circuit = ZKCircuit.from_method(circuit_method, chips=chips or [])
	print(
		f"[zinnia] circuit ready: name={circuit.get_name()}, chips={len(circuit.chips)}, externals={len(circuit.externals)}"
	)
	return circuit


def create_prove_verify_keys(
	circuit: ZKCircuit,
	data: Any,
	circuit_name: str | None = None,
	k: int = DEFAULT_K,
) -> KeygenResult:
	# Stage 1: compile the Python circuit into Halo2 Rust source.
	target_name = circuit_name or "target"
	artifacts = _ensure_halo2_paths(target_name, k)

	print("[zinnia] compiling circuit to Halo2 Rust source")
	program = circuit.compile()
	artifacts.source_path.write_text(program.source, encoding="utf-8")

	# Stage 2: serialize input payload and run key generation.
	mapped = _write_input_data(circuit, data, artifacts.input_path)
	print(f"[io] wrote source to {artifacts.source_path}")
	print(f"[io] wrote input to {artifacts.input_path} with {len(mapped)} entries")
	print(f"[io] proving key path: {artifacts.proving_key_path}")
	print(f"[io] verifying key path: {artifacts.verifying_key_path}")

	keygen_success, keygen_output = _run_halo2_action(artifacts, "keygen")
	print(f"[result] keygen success={keygen_success}")
	print(f"[result] proving key exists={artifacts.proving_key_path.exists()}")
	print(f"[result] verifying key exists={artifacts.verifying_key_path.exists()}")
	return KeygenResult(
		artifacts=artifacts,
		success=keygen_success,
		keygen_output=keygen_output,
	)


def prove(circuit: ZKCircuit, keygen_result: KeygenResult, data: Any) -> ProveResult:
	# Stage 3: refresh input and create a SNARK proof using existing keys.
	artifacts = keygen_result.artifacts
	mapped = _write_input_data(circuit, data, artifacts.input_path)
	print(f"[io] refreshed proving input {artifacts.input_path} with {len(mapped)} entries")

	prove_success, output = _run_halo2_action(artifacts, "prove")
	# Fail fast with actionable guidance if proving fails.
	if not prove_success:
		raise RuntimeError(f"[result] prove failed for {artifacts.circuit_name}. Your input data may be invalid or the circuit may not be implemented correctly.")
	# Ensure proof artifact exists before returning success.
	if not artifacts.snark_path.exists():
		raise RuntimeError(f"[result] prove did not generate proof file: {artifacts.snark_path}")
	print(f"[result] prove success={prove_success}")
	print(f"[result] proof file: {artifacts.snark_path}")
	print(f"[result] proof exists={artifacts.snark_path.exists()}")
	return ProveResult(
		artifacts=artifacts,
		success=prove_success,
		prove_output=output,
	)


def verify(keygen_result: KeygenResult) -> VerifyResult:
	# Stage 4: verify generated SNARK against verification key.
	artifacts = keygen_result.artifacts
	verify_success, output = _run_halo2_action(artifacts, "verify")
	# Parse semantic success marker from command output.
	verified = "Snark verified successfully" in output

	print(f"[result] verify command success={verify_success}")
	print(f"[result] proof verified={verified}")
	return VerifyResult(
		artifacts=artifacts,
		success=verify_success,
		verified=verified,
		verify_output=output,
	)
