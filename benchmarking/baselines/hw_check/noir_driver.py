"""The Noir baseline program: setup / prove / verify a case's `main.nr`.

Contract (shared by every baseline program, see the benchmark brief):

    python -m hw_check.noir_driver <mode> --case <case dir> --workdir <dir>
        [--fixture <path>] [--input A|B] [--out <json>] [--tamper-index <i>]
        [--target noir-recursive]

* `setup`: stages a Nargo package in the work directory (`src/main.nr` plus a
  `Nargo.toml` that depends on `baselines/noir_hw`), compiles it
  (`nargo compile`), writes the verification key (`bb write_vk`) and records
  the gate count (`bb gates`). Nothing here depends on the input.
* `prove`: writes `Prover.toml` for the input (`hw_check.prover_toml`), solves
  the witness (`nargo execute`) and proves (`bb prove`); decodes the public
  outputs from bb's `public_inputs` file.
* `verify`: `bb verify`, optionally after adding 1 to public value `i`.

Proofs are UltraHonk with `--verifier_target noir-recursive` (zero-knowledge,
Poseidon2 transcript) unless `--target` says otherwise.

The decoding of the public values follows the `// @outputs:` line every
`main.nr` carries (one line, segments separated by `;`):

    float[N] | int[N] | bool[N]      N public values
    const int[N] = [...]             values fixed by the circuit, not public
    const float[N] = [...]
    dynamic <kind>[IxU]              one public length L (items), then I*U public
                                     values of which the first L*U are the output
    dynamic2 <kind>[RxC]             public (rows, cols), then R*C public values of
                                     which the first rows*cols (row-major) are the output

Every run prints one JSON object (also written to `--out`):
`{"ok", "case", "mode", "input", "publics", "proof_path", "timings_s", ...}`;
the exit status is non-zero when `ok` is false.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from .prover_toml import BN254_P, SCALE, render_prover_toml

NARGO = os.environ.get("NARGO", str(Path.home() / ".nargo/bin/nargo"))
BB = os.environ.get("BB", str(Path.home() / ".bb/bb"))
NOIR_HW = (Path(__file__).resolve().parents[1] / "noir_hw").resolve()
DEFAULT_TARGET = "noir-recursive"
SCHEME = ["--scheme", "ultra_honk"]


class DriverError(Exception):
    pass


# ----------------------------------------------------------------- outputs

_SEGMENT = re.compile(
    r"^(?:(?P<const>const)\s+(?P<ckind>float|int|bool)\[(?P<cn>\d+)\]\s*=\s*(?P<values>\[.*\])"
    r"|(?P<dyn>dynamic)\s+(?P<dkind>float|int|bool)\[(?P<items>\d+)x(?P<unit>\d+)\]"
    r"|(?P<dyn2>dynamic2)\s+(?P<d2kind>float|int|bool)\[(?P<rows>\d+)x(?P<cols>\d+)\]"
    r"|(?P<kind>float|int|bool)\[(?P<n>\d+)\])$"
)


def parse_outputs(main_nr: str) -> list[dict]:
    """The output layout declared by the `// @outputs:` line."""
    lines = [l for l in main_nr.splitlines() if l.strip().startswith("// @outputs:")]
    if len(lines) != 1:
        raise DriverError("main.nr must carry exactly one `// @outputs:` line")
    spec = lines[0].split(":", 1)[1].strip()
    segments = []
    for part in spec.split(";"):
        m = _SEGMENT.match(part.strip())
        if not m:
            raise DriverError(f"bad output segment {part.strip()!r}")
        if m["const"]:
            values = json.loads(m["values"])
            if len(values) != int(m["cn"]):
                raise DriverError(f"const segment lists {len(values)} values, declares {m['cn']}")
            segments.append({"segment": "constants", "kind": m["ckind"], "values": values})
        elif m["dyn2"]:
            segments.append({"segment": "dynamic2", "kind": m["d2kind"], "rows": int(m["rows"]), "cols": int(m["cols"])})
        elif m["dyn"]:
            segments.append({"segment": "dynamic", "kind": m["dkind"], "items": int(m["items"]), "unit": int(m["unit"])})
        else:
            segments.append({"segment": "cells", "kind": m["kind"], "count": int(m["n"])})
    return segments


def _signed(n: int) -> int:
    return n - BN254_P if n > BN254_P // 2 else n


def _decode(kind: str, n: int):
    v = _signed(n)
    if kind == "float":
        return v / SCALE
    if kind == "bool":
        return v != 0
    return v


def decode_publics(segments: list[dict], fields: list[int]) -> list:
    out, pos = [], 0

    def take(n):
        nonlocal pos
        if pos + n > len(fields):
            raise DriverError("public inputs shorter than the declared outputs")
        chunk = fields[pos:pos + n]
        pos += n
        return chunk

    for seg in segments:
        if seg["segment"] == "cells":
            out += [_decode(seg["kind"], x) for x in take(seg["count"])]
        elif seg["segment"] == "constants":
            out += list(seg["values"])
        elif seg["segment"] == "dynamic2":
            rows, cols = (_signed(x) for x in take(2))
            if not (0 <= rows <= seg["rows"] and 0 <= cols <= seg["cols"]):
                raise DriverError(f"dynamic shape ({rows}, {cols}) outside ({seg['rows']}, {seg['cols']})")
            values = take(seg["rows"] * seg["cols"])
            out += [_decode(seg["kind"], x) for x in values[: rows * cols]]
        else:
            length = _signed(take(1)[0])
            if not 0 <= length <= seg["items"]:
                raise DriverError(f"dynamic length {length} outside [0, {seg['items']}]")
            values = take(seg["items"] * seg["unit"])
            out += [_decode(seg["kind"], x) for x in values[: length * seg["unit"]]]
    if pos != len(fields):
        raise DriverError(f"{len(fields) - pos} public inputs beyond the declared outputs")
    return out


def read_fields(path: Path) -> list[int]:
    data = path.read_bytes()
    if len(data) % 32:
        raise DriverError(f"{path} is not a sequence of 32-byte field elements")
    return [int.from_bytes(data[i:i + 32], "big") for i in range(0, len(data), 32)]


def write_fields(path: Path, fields: list[int]) -> None:
    path.write_bytes(b"".join((x % BN254_P).to_bytes(32, "big") for x in fields))


# ----------------------------------------------------------------- package

def stage_package(case_dir: Path, workdir: Path) -> Path:
    """A Nargo package for the case under `workdir/pkg`."""
    pkg = workdir / "pkg"
    (pkg / "src").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(case_dir / "main.nr", pkg / "src" / "main.nr")
    (pkg / "Nargo.toml").write_text(
        "[package]\n"
        f'name = "{case_dir.name}"\n'
        'type = "bin"\n'
        'authors = [""]\n\n'
        "[dependencies]\n"
        f'noir_hw = {{ path = "{NOIR_HW}" }}\n'
    )
    return pkg


def _run(cmd: list[str], cwd: Path) -> str:
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout).strip().splitlines()[-15:]
        raise DriverError(f"{Path(cmd[0]).name} {cmd[1]} failed:\n" + "\n".join(tail))
    return proc.stdout


def compile_package(pkg: Path) -> dict:
    _run([NARGO, "compile", "--silence-warnings"], pkg)
    return json.loads((pkg / "target" / f"{_package_name(pkg)}.json").read_text())


def _package_name(pkg: Path) -> str:
    m = re.search(r'^name = "([^"]+)"', (pkg / "Nargo.toml").read_text(), re.M)
    return m.group(1)


def _timed(timings: dict, name: str, fn):
    start = time.perf_counter()
    out = fn()
    timings[name] = time.perf_counter() - start
    return out


# ------------------------------------------------------------------- modes

def setup(case_dir: Path, workdir: Path, target: str, timings: dict) -> dict:
    for stale in ("pkg", "keys", "proof"):
        shutil.rmtree(workdir / stale, ignore_errors=True)
    pkg = stage_package(case_dir, workdir)
    name = _package_name(pkg)
    _timed(timings, "compile_s", lambda: compile_package(pkg))
    bytecode = pkg / "target" / f"{name}.json"
    keys = workdir / "keys"
    keys.mkdir()
    _timed(timings, "write_vk_s", lambda: _run([BB, "write_vk", *SCHEME, "-b", str(bytecode), "-o", str(keys), "-t", target], pkg))
    gates_json = _timed(timings, "gates_s", lambda: _run([BB, "gates", *SCHEME, "-b", str(bytecode), "-t", target], pkg))
    gates = json.loads(gates_json[gates_json.index("{"):])["functions"][0]
    (workdir / "setup.json").write_text(json.dumps({"case": case_dir.name, "target": target, "gates": gates}))
    return {"gates": gates["circuit_size"], "acir_opcodes": gates["acir_opcodes"], "vk_path": str(keys / "vk")}


def _load_setup(case_dir: Path, workdir: Path) -> dict:
    path = workdir / "setup.json"
    if not path.exists():
        raise DriverError(f"no setup in {workdir}; run setup first")
    info = json.loads(path.read_text())
    if info["case"] != case_dir.name:
        raise DriverError(f"work directory holds {info['case']}, not {case_dir.name}")
    return info


def prove(case_dir: Path, fixture: dict, input_name: str, workdir: Path, timings: dict) -> dict:
    info = _load_setup(case_dir, workdir)
    target = info["target"]
    pkg = workdir / "pkg"
    name = _package_name(pkg)
    bytecode = pkg / "target" / f"{name}.json"
    abi = json.loads(bytecode.read_text())["abi"]
    _timed(timings, "write_inputs_s",
           lambda: (pkg / "Prover.toml").write_text(render_prover_toml(abi, fixture, input_name)))
    _timed(timings, "execute_s", lambda: _run([NARGO, "execute", "--silence-warnings"], pkg))
    proof_dir = workdir / "proof"
    if proof_dir.exists():
        shutil.rmtree(proof_dir)
    proof_dir.mkdir()
    witness = pkg / "target" / f"{name}.gz"
    _timed(timings, "prove_s", lambda: _run(
        [BB, "prove", *SCHEME, "-b", str(bytecode), "-w", str(witness), "-k", str(workdir / "keys" / "vk"),
         "-o", str(proof_dir), "-t", target], pkg))
    segments = parse_outputs((pkg / "src" / "main.nr").read_text())
    publics = decode_publics(segments, read_fields(proof_dir / "public_inputs"))
    return {
        "gates": info["gates"]["circuit_size"],
        "publics": publics,
        "proof_path": str(proof_dir / "proof"),
        "public_inputs_path": str(proof_dir / "public_inputs"),
        "proof_bytes": (proof_dir / "proof").stat().st_size,
    }


def verify(case_dir: Path, workdir: Path, tamper: int | None, timings: dict) -> tuple[dict, bool]:
    info = _load_setup(case_dir, workdir)
    proof_dir = workdir / "proof"
    publics_path = proof_dir / "public_inputs"
    if tamper is not None:
        fields = read_fields(publics_path)
        if not 0 <= tamper < len(fields):
            raise DriverError(f"tamper index {tamper} outside the {len(fields)} public values")
        fields[tamper] += 1
        publics_path = proof_dir / "public_inputs.tampered"
        write_fields(publics_path, fields)
    cmd = [BB, "verify", *SCHEME, "-p", str(proof_dir / "proof"), "-k", str(workdir / "keys" / "vk"),
           "-i", str(publics_path), "-t", info["target"]]
    proc = _timed(timings, "verify_s", lambda: subprocess.run(cmd, capture_output=True, text=True))
    accepted = proc.returncode == 0
    return {"verified": accepted, "tampered_index": tamper, "proof_path": str(proof_dir / "proof")}, accepted


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["setup", "prove", "verify"])
    ap.add_argument("--case", type=Path, required=True)
    ap.add_argument("--workdir", type=Path, required=True)
    ap.add_argument("--fixture", type=Path)
    ap.add_argument("--input", default="B")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--tamper-index", type=int)
    ap.add_argument("--target", default=DEFAULT_TARGET)
    a = ap.parse_args()

    report: dict = {"case": a.case.name, "mode": a.mode}
    timings: dict = {}
    ok = False
    try:
        if a.mode == "setup":
            report.update(setup(a.case, a.workdir, a.target, timings))
            ok = True
        elif a.mode == "prove":
            report["input"] = a.input
            fixture = json.loads((a.fixture or a.case / "fixture.json").read_text())
            report.update(prove(a.case, fixture, a.input, a.workdir, timings))
            ok = True
        else:
            result, ok = verify(a.case, a.workdir, a.tamper_index, timings)
            report.update(result)
    except (DriverError, KeyError, ValueError, OSError) as e:
        report["error"] = f"{type(e).__name__}: {e}"
    report["timings_s"] = timings
    report["ok"] = ok
    text = json.dumps(report)
    print(text)
    if a.out:
        a.out.write_text(text)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
