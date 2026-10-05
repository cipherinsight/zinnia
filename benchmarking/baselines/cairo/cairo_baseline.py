"""Cairo baseline driver (Scarb 2.20.1 / Cairo 2.20.0, Stwo prover via `scarb prove`).

Implements the per-system program contract (eval-redo/BRIEF.md): driven with the
case folder, the fixture, the input name and a mode; prints one JSON object
(also written to --out) with `ok`, `publics`, `proof_path`, `timings_s`; exits
non-zero on failure.

Modes:
  setup    compile the package (`scarb build`, every case's executable). Cairo
           has no per-program key generation.
  execute  `scarb execute` on the bootloader target: public output and the
           execution resources (steps, builtins).
  prove    execute (timed on its own), then `scarb prove --proof-format
           binary` (timed): the proof whose size is reported (Stwo's binary
           format, bzip2-compressed). `scarb verify` reads only the JSON
           format, so the proof is then written again as JSON (`scarb prove`,
           untimed; reported as `timings_s.prove_json_copy`). Stwo proving is
           deterministic: both runs give the same proof (checked on
           empty_program: repeated proofs are byte-identical in each format).
  verify   `scarb verify --proof-file` on the JSON proof. With --tamper, one
           output value in the proof's public claim is changed first, and the
           run succeeds only if verification rejects it.

The program's arguments are private: Scarb writes them into memory through a
hint, outside the public memory. Its return value is written to the output
segment, which is public. The bootloader target is the one `scarb verify`
accepts (`--target standalone` proofs fail verification in 2.20.1).

Timings are wall-clock times of the `scarb` processes, so they include process
start-up and file I/O (Scarb offers no in-process API).

Encoding: floats are Q32 integers (`round(v * 2^32)`), integers as is,
booleans 0/1; arrays are `[len, elements...]` in row-major order (Cairo Serde
of `Array<T>`); negative values are encoded modulo the Stark prime.

The output layout of each case is declared in its `main.cairo` by a line
`// output: <type>` where <type> is `q32`, `i64`, `bool`, `[<type>]` (an
array: length then elements) or `(<type>, ...)` (a tuple). `dim` is an extent
of a data-dependent output shape: committed like an `i64` but reported as
`output_shape`, apart from the values compared with NumPy (e.g. `([dim], [q32])`
for a 2-D crop whose rows and columns depend on the data).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import resource
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
PACKAGE = "ds1000_cairo"
SCARB = os.environ.get(
    "SCARB", str(Path.home() / ".local/share/scarb/scarb-v2.20.1-x86_64-unknown-linux-gnu/bin/scarb"))
STARK_PRIME = 2**251 + 17 * 2**192 + 1
SCALE = 2**32
# The bootloader writes [number of tasks, task output size, program hash]
# before the program's own output.
BOOTLOADER_HEADER = 3


class Failure(Exception):
    pass


# --------------------------------------------------------------------------
# Inputs: fixture -> felts
# --------------------------------------------------------------------------

def quantize(v: float) -> int:
    """round(v * 2^32), half away from zero (Zinnia's quantizer)."""
    scaled = v * SCALE
    return int(scaled + 0.5) if scaled >= 0 else -int(-scaled + 0.5)


def encode_scalar(value, dtype: str) -> int:
    if dtype == "float":
        return quantize(float(value))
    if dtype == "int":
        if not isinstance(value, int) or isinstance(value, bool):
            raise Failure(f"expected an integer, got {value!r}")
        return value
    if not isinstance(value, bool):
        raise Failure(f"expected a boolean, got {value!r}")
    return int(value)


def flatten(value) -> list:
    if isinstance(value, list):
        return [leaf for item in value for leaf in flatten(item)]
    return [value]


def unused_params(case_dir: Path) -> list[str]:
    """Parameters the program declares it never reads (`// unused parameters: a, b`).

    Zinnia removes such inputs from the circuit, so the program does not take
    them either and they are left out of its arguments.
    """
    match = re.search(r"^//\s*unused parameters:\s*(.+?)\s*$", (case_dir / "main.cairo").read_text(), re.M)
    return [name.strip() for name in match.group(1).split(",")] if match else []


def encode_args(fixture: dict, input_name: str, unused: list[str]) -> list[int]:
    values = fixture["inputs"].get(input_name)
    if values is None:
        raise Failure(f"fixture has no input named {input_name!r}")
    names = [param["name"] for param in fixture["parameters"]]
    for name in unused:
        if name not in names:
            raise Failure(f"unused parameter {name!r} is not a kernel parameter")
    felts: list[int] = []
    for param in fixture["parameters"]:
        if param["name"] in unused:
            continue
        value = values[param["name"]]
        if param["kind"] == "array":
            leaves = flatten(value)
            felts.append(len(leaves))
            felts.extend(encode_scalar(v, param["dtype"]) for v in leaves)
        else:
            felts.append(encode_scalar(value, param["dtype"]))
    return felts


# --------------------------------------------------------------------------
# Outputs: felts -> publics
# --------------------------------------------------------------------------

def output_schema(case_dir: Path):
    match = re.search(r"^//\s*output:\s*(.+?)\s*$", (case_dir / "main.cairo").read_text(), re.M)
    if not match:
        raise Failure("main.cairo declares no `// output:` layout")
    schema, rest = parse_type(match.group(1).replace(" ", ""))
    if rest:
        raise Failure(f"cannot parse output layout {match.group(1)!r}")
    return schema


def parse_type(text: str):
    if text.startswith("["):
        inner, rest = parse_type(text[1:])
        if not rest.startswith("]"):
            raise Failure("unbalanced [ in output layout")
        return ("array", inner), rest[1:]
    if text.startswith("("):
        items, rest = [], text[1:]
        while True:
            item, rest = parse_type(rest)
            items.append(item)
            if rest.startswith(","):
                rest = rest[1:]
            elif rest.startswith(")"):
                return ("tuple", items), rest[1:]
            else:
                raise Failure("unbalanced ( in output layout")
    for scalar in ("q32", "i64", "bool", "dim"):
        if text.startswith(scalar):
            return scalar, text[len(scalar):]
    raise Failure(f"unknown type in output layout: {text!r}")


def signed(felt: int) -> int:
    return felt - STARK_PRIME if felt > STARK_PRIME // 2 else felt


def decode_output(report: dict, schema, felts: list[int]) -> None:
    """Sets `publics` (and `output_shape` for `dim` entries) from the output felts."""
    publics: list = []
    shape: list = []
    pos = decode_into(schema, felts, 0, publics, shape)
    if pos != len(felts):
        raise Failure(f"program output has {len(felts) - pos} felts beyond the declared layout")
    report["publics"] = publics
    if shape:
        report["output_shape"] = shape


def decode_into(schema, felts: list[int], pos: int, out: list, shape: list) -> int:
    if schema == "dim":
        shape.append(signed(felts[pos]))
        return pos + 1
    if schema == "q32":
        out.append(signed(felts[pos]) / SCALE)
        return pos + 1
    if schema == "i64":
        out.append(signed(felts[pos]))
        return pos + 1
    if schema == "bool":
        if felts[pos] not in (0, 1):
            raise Failure(f"boolean output is {felts[pos]}")
        out.append(bool(felts[pos]))
        return pos + 1
    kind, inner = schema
    if kind == "array":
        length = felts[pos]
        pos += 1
        for _ in range(length):
            pos = decode_into(inner, felts, pos, out, shape)
        return pos
    for item in inner:
        pos = decode_into(item, felts, pos, out, shape)
    return pos


def program_output(felts: list[int]) -> list[int]:
    """Strips the bootloader's header from the output segment."""
    if len(felts) < BOOTLOADER_HEADER or felts[0] != 1 or felts[1] != len(felts) - 1:
        raise Failure(f"unexpected bootloader output header {felts[:2]}")
    return felts[BOOTLOADER_HEADER:]


def limbs_to_int(limbs: list[int]) -> int:
    return sum(limb << (32 * i) for i, limb in enumerate(limbs))


def proof_output(proof: dict) -> list[int]:
    entries = proof["claim"]["public_data"]["public_memory"]["output"]
    return [limbs_to_int(value) for _, value in entries]


# --------------------------------------------------------------------------
# Scarb
# --------------------------------------------------------------------------

def scarb(report: dict, timing: str, *args: str) -> list[dict]:
    """Runs `scarb --json <args>` in the package, times it, returns its NDJSON."""
    start = time.perf_counter()
    proc = subprocess.run([SCARB, "--json", *args], cwd=PACKAGE_DIR, capture_output=True, text=True)
    report["timings_s"][timing] = time.perf_counter() - start
    messages = []
    for line in proc.stdout.splitlines():
        try:
            messages.append(json.loads(line))
        except json.JSONDecodeError:
            messages.append({"text": line})
    if proc.returncode != 0:
        detail = [m.get("message") or m.get("text") for m in messages if m.get("type") == "error" or "text" in m]
        tail = (proc.stderr.strip().splitlines() or [""])[-1]
        raise Failure(f"scarb {args[0]} failed: {' | '.join(filter(None, detail))[-500:]} {tail}")
    return messages


def execute(report: dict, case: str, case_dir: Path, args_path: Path, schema) -> Path:
    messages = scarb(report, "execute", "execute", "--no-build", "--executable-name", case,
                     "--arguments-file", str(args_path), "--output", "standard", "--target", "bootloader",
                     "--print-program-output", "--print-resource-usage")
    execution_dir = None
    for m in messages:
        if "program_output" in m:
            felts = [int(line) % STARK_PRIME for line in m["program_output"].split()]
            decode_output(report, schema, program_output(felts))
        if m.get("resources"):
            report["resources"] = m["resources"]
        if str(m.get("status", "")).startswith("saving output"):
            execution_dir = PACKAGE_DIR / m["message"]
    if execution_dir is None:
        raise Failure("scarb execute did not report its output directory")
    return execution_dir


def report_proof(report: dict, binary_proof_path: Path, proof_path: Path) -> None:
    report["proof_path"] = str(binary_proof_path)
    report["proof_size_bytes"] = binary_proof_path.stat().st_size
    report["json_proof_path"] = str(proof_path)
    report["json_proof_size_bytes"] = proof_path.stat().st_size


def run(args: argparse.Namespace, report: dict) -> None:
    case_dir = args.case_dir.resolve()
    case = case_dir.name
    report["case"] = case
    work = (args.work_dir or PACKAGE_DIR / "work" / case).resolve()
    work.mkdir(parents=True, exist_ok=True)
    proof_path = work / f"proof-stwo-{args.input}.json"
    binary_proof_path = work / f"proof-stwo-{args.input}.bin"

    if args.mode == "setup":
        scarb(report, "build", "build")
        return

    schema = output_schema(case_dir)
    if args.mode in ("execute", "prove"):
        fixture = json.loads((args.fixture or case_dir / "fixture.json").read_text())
        args_path = work / f"args-{args.input}.json"
        felts = encode_args(fixture, args.input, unused_params(case_dir))
        args_path.write_text(json.dumps([hex(v % STARK_PRIME) for v in felts]))
        execution_dir = execute(report, case, case_dir, args_path, schema)
        if args.mode == "execute":
            shutil.rmtree(execution_dir, ignore_errors=True)
            return
        execution_id = execution_dir.name.removeprefix("execution")
        saved = execution_dir / "proof" / "proof.json"  # Scarb's name, whatever the format
        scarb(report, "prove", "prove", "--execution-id", execution_id, "--proof-format", "binary")
        shutil.move(saved, binary_proof_path)
        scarb(report, "prove_json_copy", "prove", "--execution-id", execution_id, "--proof-format", "json")
        shutil.move(saved, proof_path)
        shutil.rmtree(execution_dir, ignore_errors=True)
        report_proof(report, binary_proof_path, proof_path)
        decode_output(report, schema, program_output(proof_output(json.loads(proof_path.read_text()))))
        return

    # verify
    proof = json.loads(proof_path.read_text())
    report_proof(report, binary_proof_path, proof_path)
    decode_output(report, schema, program_output(proof_output(proof)))
    target = proof_path
    if args.tamper:
        proof["claim"]["public_data"]["public_memory"]["output"][-1][1][0] ^= 1
        tampered = tempfile.NamedTemporaryFile("w", suffix=".json", dir=work, delete=False)
        json.dump(proof, tampered)
        tampered.close()
        target = Path(tampered.name)
    try:
        scarb(report, "verify", "verify", "--proof-file", str(target))
        report["verified"] = True
    except Failure as exc:
        report["verified"] = False
        if not args.tamper:
            raise Failure(f"verification failed: {exc}") from None
    finally:
        if args.tamper:
            target.unlink()
    if args.tamper and report["verified"]:
        raise Failure("a proof with a tampered public output verified")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--input", default="B")
    parser.add_argument("--mode", choices=["setup", "execute", "prove", "verify"], required=True)
    parser.add_argument("--proof-mode", choices=["stwo"], default="stwo")
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--tamper", action="store_true")
    args = parser.parse_args()

    report = {"system": "cairo", "scarb": SCARB, "case": args.case_dir.name, "input": args.input,
              "mode": args.mode, "proof_mode": args.proof_mode, "publics": [], "proof_path": None,
              "timings_s": {}}
    try:
        run(args, report)
        report["ok"] = True
    except Exception as exc:  # report every failure in the JSON, never swallow it
        report["ok"] = False
        report["error"] = f"{type(exc).__name__}: {exc}"
    report["peak_rss_mib"] = round(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024)
    print(json.dumps(report))
    if args.out:
        args.out.write_text(json.dumps(report, indent=1) + "\n")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
