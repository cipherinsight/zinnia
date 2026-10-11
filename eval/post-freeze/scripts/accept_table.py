"""Acceptance outcomes per corpus for each run directory, the programs whose outcome differs
from the first run, and (with --list) every program the first run does not accept.
Port of eval-redo/fix/fact-review-data/accept_table.py taking paths.

Usage: python3 accept_table.py [--list] NAME=DIR [NAME=DIR ...]   (DIR holds results.json)
"""
import collections
import json
import sys
from pathlib import Path

args = [a for a in sys.argv[1:] if not a.startswith("--")]
runs = {}
for spec in args:
    name, _, path = spec.partition("=")
    rows = json.load(open(Path(path) / "results.json"))["results"]
    runs[name] = {(r["program"], r["function"]): r for r in rows}
corpora = ("ds1000_numpy", "npbench", "pythran")
outcomes = sorted({r["outcome"] for rs in runs.values() for r in rs.values()})
print("| outcome | " + " | ".join(f"{n} {c}" for n in runs for c in ("ds1000", "npb", "py")) + " |")
print("|---" * (1 + 3 * len(runs)) + "|")
for o in outcomes + ["total"]:
    cells = []
    for rs in runs.values():
        cnt = collections.Counter(r["dataset"] for r in rs.values() if o == "total" or r["outcome"] == o)
        cells += [str(cnt.get(c, 0)) for c in corpora]
    print(f"| {o} | " + " | ".join(cells) + " |")
first = next(iter(runs))
for n, rs in list(runs.items())[1:]:
    only_first = sorted(set(runs[first]) - set(rs))
    only_this = sorted(set(rs) - set(runs[first]))
    diff = [(k, runs[first][k]["outcome"], rs[k]["outcome"]) for k in rs
            if k in runs[first] and rs[k]["outcome"] != runs[first][k]["outcome"]]
    print(f"\n{n} vs {first}: {len(diff)} changed" + (f"; missing {only_first}" if only_first else "")
          + (f"; extra {only_this}" if only_this else ""))
    for k, a, b in sorted(diff):
        print(f"  {rs[k]['dataset']} {k[0]}::{k[1]}: {a} -> {b}  {rs[k].get('detail', '')[:200]!r}")
if "--list" in sys.argv:
    print(f"\nNot accepted in {first}:")
    for k, r in sorted(runs[first].items(), key=lambda kv: (kv[1]["dataset"], kv[1]["outcome"], kv[0])):
        if r["outcome"] != "accepted_correct":
            detail = " ".join(str(r.get("detail", "")).split())[:220]
            print(f"| {r['dataset']} | {k[0]}::{k[1]} | {r['outcome']} | {r.get('construct', '')} | {detail} |")
