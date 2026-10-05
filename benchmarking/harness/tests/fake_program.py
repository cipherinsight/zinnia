#!/usr/bin/env python3
"""A fake baseline program implementing the per-system program contract.

It "proves" the toy kernel `a * 2.0` by writing the outputs with a keyed hash
(the key comes from setup), so verification detects a changed public value.
`behavior.txt` next to this script selects a misbehavior for the tests:
`honest`, `wrong_output`, `accept_tamper`, `crash_on_prove`, `sleep_on_prove`.
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path


def tag(key: str, publics: list) -> str:
    return hashlib.sha256((key + json.dumps(publics)).encode()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    for flag in ("--case-dir", "--fixture", "--input", "--mode", "--work-dir", "--out",
                 "--proof", "--variant", "--k", "--lookup-bits"):
        parser.add_argument(flag)
    parser.add_argument("--tamper", action="store_true")
    args = parser.parse_args()
    behavior = (Path(__file__).parent / "behavior.txt").read_text().strip()
    work = Path(args.work_dir)
    out = Path(args.out)

    if args.mode == "setup":
        (work / "key").write_text(f"key-for-{args.input}")
        out.write_text(json.dumps({"ok": True, "timings_s": {"setup": 0.01},
                                   "details": {"k": 7, "lookup_bits": 6}}))
        return 0

    key = (work / "key").read_text()
    if args.mode == "prove":
        if behavior == "crash_on_prove":
            return 3
        if behavior == "sleep_on_prove":
            time.sleep(60)
        fixture = json.loads(Path(args.fixture).read_text())
        values = fixture["inputs"][args.input]["a"]
        publics = [2.0 * v for v in values]
        if behavior == "wrong_output":
            publics = list(reversed(publics))
        Path(args.proof).write_text(json.dumps({"publics": publics, "tag": tag(key, publics)}))
        out.write_text(json.dumps({"ok": True, "publics": publics, "proof_path": args.proof,
                                   "timings_s": {"witness": 0.001, "prove": 0.002}}))
        return 0

    proof = json.loads(Path(args.proof).read_text())
    publics = proof["publics"]
    if args.tamper:
        publics[0] += 1.0
    verified = behavior == "accept_tamper" or tag(key, publics) == proof["tag"]
    out.write_text(json.dumps({"ok": True, "verified": verified,
                               "timings_s": {"verify": 0.0005}}))
    return 0 if verified else 1


if __name__ == "__main__":
    sys.exit(main())
