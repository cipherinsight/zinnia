#!/usr/bin/env python3
"""A fake baseline program implementing the per-system program contract.

It speaks both command-line styles (`halo2_hw <mode> --case ... --workdir`, and
the hosts' `--case-dir ... --mode ... --work-dir`) and "proves" the toy kernel
`a * 2.0` by storing the outputs in its work directory with a keyed hash (the
key comes from setup), so verification detects a changed public value.
`behavior.txt` next to this script selects a misbehavior for the tests:
`honest`, `wrong_output`, `accept_tamper`, `crash_on_prove`, `sleep_on_prove`,
`allocate_on_prove` (a helper process allocates 1 GB, then the program reports a
plain failure, as a program does that loses a prover to the memory cap).
"""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path


def tag(key: str, publics: list) -> str:
    return hashlib.sha256((key + json.dumps(publics)).encode()).hexdigest()


def parse() -> tuple[argparse.Namespace, bool]:
    halo2_style = len(sys.argv) > 1 and not sys.argv[1].startswith("--")
    parser = argparse.ArgumentParser()
    if halo2_style:
        parser.add_argument("mode")
    for flag in ("--case", "--case-dir", "--function", "--fixture", "--input", "--mode",
                 "--workdir", "--work-dir", "--out", "--proof-mode", "--k", "--lookup-bits",
                 "--tamper-index"):
        parser.add_argument(flag)
    parser.add_argument("--tamper", action="store_true")
    args = parser.parse_args()
    args.work = Path(args.workdir or args.work_dir)
    args.tampering = args.tamper or args.tamper_index is not None
    return args, halo2_style


def main() -> int:
    args, halo2_style = parse()
    names = ({"prove": "prove_s", "witness": "witness_s", "verify": "verify_s"} if halo2_style
             else {"prove": "prove", "witness": "execute", "verify": "verify"})
    behavior = (Path(__file__).parent / "behavior.txt").read_text().strip()
    out = Path(args.out)

    if args.mode == "setup":
        (args.work / "key").write_text(f"key-for-{args.input}")
        out.write_text(json.dumps({"ok": True, "timings_s": {"setup": 0.01}, "k": 7,
                                   "function": args.function}))
        return 0

    key = (args.work / "key").read_text()
    proof_path = args.work / "proof.json"
    if args.mode == "prove":
        if behavior == "crash_on_prove":
            return 3
        if behavior == "sleep_on_prove":
            time.sleep(60)
        if behavior == "allocate_on_prove":
            subprocess.run([sys.executable, "-c", "x = bytearray(1 << 30); x[::4096] = b'1' * "
                            "len(x[::4096])"])
            out.write_text(json.dumps({"ok": False, "error": "prover failed"}))
            return 1
        fixture = json.loads(Path(args.fixture).read_text())
        publics = [2.0 * v for v in fixture["inputs"][args.input]["a"]]
        if behavior == "wrong_output":
            publics = list(reversed(publics))
        proof_path.write_text(json.dumps({"publics": publics, "tag": tag(key, publics)}))
        out.write_text(json.dumps({"ok": True, "publics": publics, "proof_path": str(proof_path),
                                   "timings_s": {names["witness"]: 0.001,
                                                 names["prove"]: 0.002}}))
        return 0

    proof = json.loads(proof_path.read_text())
    publics = proof["publics"]
    if args.tampering:
        publics[0] += 1.0
    verified = behavior == "accept_tamper" or tag(key, publics) == proof["tag"]
    out.write_text(json.dumps({"ok": verified != args.tampering, "verified": verified,
                               "timings_s": {names["verify"]: 0.0005}}))
    return 0 if verified != args.tampering else 1


if __name__ == "__main__":
    sys.exit(main())
