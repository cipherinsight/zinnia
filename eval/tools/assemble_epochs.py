#!/usr/bin/env python3
"""Assemble several harness epochs into one set of shard directories for `harness analyze`.

`harness analyze` refuses a (system, case) record that appears in more than one shard. When a
baseline was fixed and re-timed (e.g. `baselines-r2-fix-*` after `baselines-r1-*`), the newer
record must replace the older one. This script copies each epoch's manifest, case set and
summary-free tree into OUT/<epoch name>, linking raw records, and drops every raw record that a
LATER epoch on the command line also has. Nothing in the source epochs is modified.

Usage: assemble_epochs.py [--case-set FILE] OUT EPOCH [EPOCH ...]   (later epochs win)

--case-set FILE: rewrite each copied case_set.json from the canonical case set (copy-only flag,
tolerance, inclusion), for epochs that were run with `--cases` only (the harness then builds each
case from its folder: default tolerance, no copy-only flag). Only the analysis view changes; the
output checks recorded in the raw files keep the tolerance they were run with.
Prints the assembled directories, ready for `python -m benchmarking.harness analyze ... --out`.
"""
import json
import shutil
import sys
from pathlib import Path


def records(epoch: Path) -> set[tuple[str, str]]:
    raw = epoch / "raw"
    return {(s.name, p.stem) for s in raw.iterdir() if s.is_dir() for p in s.glob("*.json")} if raw.exists() else set()


def canonical_entries(case_set: Path) -> dict[str, dict]:
    raw = json.loads(case_set.read_text())
    return {e["case"]: e for e in (raw["cases"] if isinstance(raw, dict) else raw)}


def rewrite_case_set(path: Path, canonical: dict[str, dict]) -> None:
    doc = json.loads(path.read_text())
    for entry in doc["cases"]:
        c = canonical.get(entry["case"])
        if c is None:
            continue
        entry["data_movement_only"] = bool(c.get("data_movement_only", False))
        tol = c.get("float_tolerance", c.get("tolerance"))
        if tol is not None:
            entry["tolerance"] = {"abs": tol["atol"], "rel": tol["rtol"], "source": "case set (assembled)"}
    doc["source"] = f"{doc.get('source')} (flags from {path.name} rewritten by assemble_epochs.py)"
    path.write_text(json.dumps(doc, indent=1))


def main() -> int:
    args = sys.argv[1:]
    canonical = None
    if args[0] == "--case-set":
        canonical = canonical_entries(Path(args[1]))
        args = args[2:]
    out = Path(args[0])
    epochs = [Path(p).resolve() for p in args[1:]]
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    later: set[tuple[str, str]] = set()
    kept_dirs = []
    dropped = []
    for epoch in reversed(epochs):  # walk newest first so `later` holds what newer epochs have
        mine = records(epoch)
        dest = out / epoch.name
        dest.mkdir()
        for item in epoch.iterdir():
            if item.name in ("raw", "summary"):
                continue
            target = dest / item.name
            (shutil.copytree if item.is_dir() else shutil.copy2)(item, target)
        if canonical is not None and (dest / "case_set.json").exists():
            rewrite_case_set(dest / "case_set.json", canonical)
        for system, case in sorted(mine):
            if (system, case) in later:
                dropped.append((epoch.name, system, case))
                continue
            src = epoch / "raw" / system / f"{case}.json"
            (dest / "raw" / system).mkdir(parents=True, exist_ok=True)
            (dest / "raw" / system / src.name).symlink_to(src)
        later |= mine
        kept_dirs.append(dest)
    for epoch_name, system, case in dropped:
        print(f"superseded: {epoch_name} {system} {case}", file=sys.stderr)
    print(" ".join(str(d) for d in reversed(kept_dirs)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
