"""Cairo baseline driver (Scarb 2.20.1 / Cairo 2.20.0, Stwo prover via `scarb prove`).

Implements the per-system program contract (eval-redo/BRIEF.md): driven with the
case folder, the fixture, the input name and a mode; prints one JSON object
(also written to --out) with `ok`, `publics`, `proof_path`, `timings_s`; exits
non-zero on failure.

Modes:
  setup    compile this case (`scarb build --target-names <case>`). Cairo
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

Shape-generic entries live in `cairo/<function>.cairo` and take
`Array<Array<i64>>` raw parameter buffers. Each array buffer ends with its
dimensions. They return `Array<i64>` containing exactly the Rust committed
words: the input-shape vector first, then the output. Cairo Serde adds the
outer array length. Reports expose the inner words as `public_words`.

The output layout of each case is declared in its source by a line
`// output: <type>` where <type> is `q32`, `i64`, `bool`, `[<type>]` (an
array: length then elements) or `(<type>, ...)` (a tuple). `dim` is an extent
of a data-dependent output shape: committed like an `i64` but reported as
`output_shape`, apart from the values compared with NumPy (e.g. `([dim], [q32])`
for a 2-D crop whose rows and columns depend on the data). Shape-generic
layouts additionally support `tensor<rank,T>` (dimensions then elements, no
data length), `c32`, `option<T>` (0 for None, 1 then T for Some), and `()`.
The layout describes only the result, not the input-shape prefix.
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


def source_path(case_dir: Path, function: str | None = None) -> Path:
    return case_dir / "cairo" / f"{function}.cairo" if function else case_dir / "main.cairo"


def unused_params(case_dir: Path, function: str | None = None) -> list[str]:
    """Parameters the program declares it never reads (`// unused parameters: a, b`).

    Zinnia removes such inputs from the circuit, so the program does not take
    them either and they are left out of its arguments.
    """
    match = re.search(r"^//\s*unused parameters:\s*(.+?)\s*$", source_path(case_dir, function).read_text(), re.M)
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
        if param["dtype"] == "complex":
            # A complex scalar is a (real, imaginary) Q32 pair. Array
            # lengths count complex elements, not their component felts.
            leaves = flatten(value)
            if len(leaves) % 2:
                raise Failure("complex input must contain real/imaginary pairs")
            if param["kind"] == "array":
                felts.append(len(leaves) // 2)
            elif len(leaves) != 2:
                raise Failure("complex scalar must contain one real/imaginary pair")
            felts.extend(encode_scalar(v, "float") for v in leaves)
            continue
        if param["kind"] == "array":
            leaves = flatten(value)
            felts.append(len(leaves))
            felts.extend(encode_scalar(v, param["dtype"]) for v in leaves)
        else:
            felts.append(encode_scalar(value, param["dtype"]))
    return felts


def encode_shaped_args(fixture: dict, input_name: str, unused: list[str]) -> tuple[list[int], list[int]]:
    """Rust raw word buffers wrapped in Cairo Array<Array<i64>> Serde.

    Dimensions come from actual rectangular input values, checked against the
    fixture metadata just as Rust does. Complex leaves are real/imaginary pairs.
    """
    values = fixture["inputs"][input_name]
    names = [p["name"] for p in fixture["parameters"]]
    if any(n not in names for n in unused):
        raise Failure("unknown unused parameter")
    buffers, shapes = [], []

    def visit(param, value):
        if param["kind"] == "tuple":
            if len(value) != len(param["items"]):
                raise Failure("tuple length mismatch")
            for item, v in zip(param["items"], value):
                visit(item, v)
            return
        dtype = param["dtype"]

        def walk(v):
            if isinstance(v, list):
                children = [walk(item) for item in v]
                dims = children[0][0] if children else []
                if any(d != dims for d, _ in children):
                    raise Failure("ragged array")
                return [len(v), *dims], [w for _, ws in children for w in ws]
            return [], [encode_scalar(v, "float" if dtype == "complex" else dtype)]

        dims, words = walk(value)
        if dtype == "complex":
            if not dims or dims[-1] != 2:
                raise Failure("complex value must be a real/imaginary pair")
            dims.pop()
        if param["kind"] == "scalar":
            if dims:
                raise Failure("scalar given as an array")
        elif param["kind"] == "array":
            if not dims or (param.get("shape") is not None and dims != param["shape"]):
                raise Failure(f"array shape {dims} differs from {param.get('shape')}")
            shapes.extend(dims)
            words += dims
        else:
            raise Failure("unknown parameter kind")
        if any(not -(2**63) <= w < 2**63 for w in words):
            raise Failure("input outside signed 64-bit range")
        buffers.append(words)

    for param in fixture["parameters"]:
        if param["name"] not in unused:
            visit(param, values[param["name"]])
    return [len(buffers), *[w for buf in buffers for w in [len(buf), *buf]]], shapes


# --------------------------------------------------------------------------
# Outputs: felts -> publics
# --------------------------------------------------------------------------

def output_schema(case_dir: Path, function: str | None = None):
    match = re.search(r"^//\s*output:\s*(.+?)\s*$", source_path(case_dir, function).read_text(), re.M)
    if not match:
        raise Failure(f"{source_path(case_dir, function)} declares no `// output:` layout")
    schema, rest = parse_type(match.group(1).replace(" ", ""))
    if rest:
        raise Failure(f"cannot parse output layout {match.group(1)!r}")
    return schema


def parse_type(text: str):
    if text.startswith("tensor<"):
        rank, rest = text[7:].split(",", 1)
        inner, rest = parse_type(rest)
        if not rest.startswith(">") or not rank.isdigit() or not 1 <= int(rank) <= 4:
            raise Failure("invalid tensor layout")
        return ("tensor", (int(rank), inner)), rest[1:]
    if text.startswith("option<"):
        inner, rest = parse_type(text[7:])
        if not rest.startswith(">"):
            raise Failure("invalid option layout")
        return ("option", inner), rest[1:]
    if text.startswith("()"):
        return ("tuple", []), text[2:]
    if text.startswith("c32"):
        return ("tuple", ["q32", "q32"]), text[3:]
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
    if isinstance(schema, tuple) and schema[0] == "shaped":
        # Cairo Serde's outer Array envelope; its contents are Rust's exact words.
        if not felts or felts[0] != len(felts) - 1:
            raise Failure("invalid public word envelope")
        words = [signed(v) for v in felts[1:]]
        if not words or not 0 <= words[0] < len(words):
            raise Failure("invalid input shape commitment")
        report["public_words"] = words
        report["input_shapes"] = words[1:1 + words[0]]
        felts = felts[2 + words[0]:]
        schema = schema[1]
    publics: list = []
    shape: list = []
    pos = decode_into(schema, felts, 0, publics, shape)
    if pos != len(felts):
        raise Failure(f"program output has {len(felts) - pos} felts beyond the declared layout")
    report["publics"] = publics
    if shape:
        report["output_shape"] = shape


def decode_into(schema, felts: list[int], pos: int, out: list, shape: list) -> int:
    if pos >= len(felts) and schema != ("tuple", []):
        raise Failure("truncated public output")
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
    if kind == "option":
        tag = felts[pos]
        if tag not in (0, 1):
            raise Failure("invalid option tag")
        return decode_into(inner, felts, pos + 1, out, shape) if tag else pos + 1
    if kind == "tensor":
        rank, element = inner
        dims = [signed(v) for v in felts[pos:pos + rank]]
        if len(dims) != rank or any(d < 0 for d in dims):
            raise Failure("invalid tensor dimensions")
        length = 1
        for dim in dims:
            length *= dim
        if length > len(felts) - pos - rank:
            raise Failure("truncated tensor")
        shape.extend(dims)
        pos += rank
        for _ in range(length):
            pos = decode_into(element, felts, pos, out, shape)
        return pos
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
    case = case_dir.name + (f"__{args.function}" if args.function else "")
    report["case"] = case
    work = (args.work_dir or PACKAGE_DIR / "work" / case).resolve()
    work.mkdir(parents=True, exist_ok=True)
    proof_path = work / f"proof-stwo-{args.input}.json"
    binary_proof_path = work / f"proof-stwo-{args.input}.bin"

    if args.mode == "setup":
        scarb(report, "build", "build", "--target-names", case)
        return

    schema = output_schema(case_dir, args.function)
    if args.function:
        schema = ("shaped", schema)
    expected_shapes = None
    if args.mode in ("execute", "prove") or args.function:
        candidates = ([case_dir / f"fixture.{args.function}.json", case_dir / f"fixture_{args.function}.json"]
                      if args.function else []) + [case_dir / "fixture.json"]
        fixture_path = args.fixture or next((p for p in candidates if p.exists()), candidates[-1])
        fixture = json.loads(fixture_path.read_text())
        unused = unused_params(case_dir, args.function)
        if args.function:
            felts, expected_shapes = encode_shaped_args(fixture, args.input, unused)
        else:
            felts = encode_args(fixture, args.input, unused)
    if args.mode in ("execute", "prove"):
        args_path = work / f"args-{args.input}.json"
        args_path.write_text(json.dumps([hex(v % STARK_PRIME) for v in felts]))
        execution_dir = execute(report, case, case_dir, args_path, schema)
        if args.function and report.get("input_shapes") != expected_shapes:
            raise Failure("committed input shapes differ from fixture")
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
    if args.function and report.get("input_shapes") != expected_shapes:
        raise Failure("committed input shapes differ from fixture")
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
    parser.add_argument("--function", help="shape-generic entry function (cairo/<function>.cairo)")
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
