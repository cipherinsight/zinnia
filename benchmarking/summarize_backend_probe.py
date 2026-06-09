from __future__ import annotations

import collections
import json
import sys
from pathlib import Path


def classify(rec: dict) -> str:
    err = rec.get("error") or ""
    rc = rec.get("returncode")
    if rc == -11:
        return "native crash rc=-11"
    if rc == "timeout":
        return "timeout"
    if "does not yet support" in err:
        return err.split("does not yet support ", 1)[1].split("\n", 1)[0]
    if "memory write out of bounds" in err:
        return "memory write out of bounds"
    if err.startswith(" verify:"):
        return "verify-time synthesis/error"
    if not err:
        return "verify false / no error"
    return err[:120]


def main() -> None:
    path = Path(sys.argv[1])
    data = json.loads(path.read_text())
    results = data["results"]
    ok = [r for r in results if r.get("verify_ok")]
    bad = [r for r in results if not r.get("verify_ok")]
    print(f"total {len(results)} ok {len(ok)} bad {len(bad)}")
    print("ok_cases", ",".join(r["case"] for r in ok))

    counts = collections.Counter()
    examples: dict[str, list[str]] = {}
    for rec in bad:
        key = classify(rec)
        counts[key] += 1
        examples.setdefault(key, []).append(rec["case"])

    print("categories")
    for key, count in counts.most_common():
        print(f"{count}\t{key}\t{','.join(examples[key][:16])}")


if __name__ == "__main__":
    main()
