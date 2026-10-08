"""The NPBench/Pythran scaling study (see `benchmarking/scaling_npb/__init__.py`).

    python -m benchmarking.scaling_npb generate --data DATA --npbench NPB --pythran PYT [--caps 32,64,128]
    python -m benchmarking.scaling_npb relocate --data DATA
    python -m benchmarking.scaling_npb zkvm-check --data DATA [--out FILE]
    python -m benchmarking.scaling_npb validate --data DATA --cap C --out FILE.jsonl [--systems S,..] [--jobs N]
                                               [--memory 16G] [--timeout S] [--max-elements N] [--cases ...]
    python -m benchmarking.scaling_npb remote-list --data DATA FILE.jsonl...
    python -m benchmarking.scaling_npb delegate-cases --target halo2|noir --case-set FILE MEASURE.json...
    python -m benchmarking.scaling_npb table --data DATA [--validation FILE ...]
    python -m benchmarking.scaling_npb analyze --data DATA [--zinnia EPOCH...] [--zkvm EPOCH...] \\
                                              [--counts FILE...] [--reference-zinnia EPOCH...] \\
                                              [--reference-zkvm EPOCH...] [--out DIR]

NPB and PYT are checkouts of spcl/npbench and serge-sans-paille/pythran at the
commits of `datasets/upstream_sources.json`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _caps(text: str) -> tuple[int, ...]:
    return tuple(int(c) for c in text.split(",") if c)


def cmd_generate(args: argparse.Namespace) -> int:
    from benchmarking.scaling_npb import generate, rule
    if args.data.exists() and any(args.data.iterdir()):
        raise SystemExit(f"{args.data} is not empty; generate into a new directory")
    args.data.mkdir(parents=True, exist_ok=True)
    document = generate.generate(args.data.resolve(), {"npbench": args.npbench.resolve(),
                                                       "pythran": args.pythran.resolve()},
                                 _caps(args.caps) if args.caps else rule.CAPS)
    counts: dict[int, dict[str, int]] = {}
    for point in document["points"]:
        by_status = counts.setdefault(point["cap"], {})
        by_status[point["status"]] = by_status.get(point["status"], 0) + 1
    print(json.dumps(counts, indent=1))
    return 0


def cmd_relocate(args: argparse.Namespace) -> int:
    from benchmarking.scaling_npb import generate
    print(f"{generate.relocate(args.data.resolve())} case folders point into {args.data.resolve()}")
    return 0


def cmd_zkvm_check(args: argparse.Namespace) -> int:
    from benchmarking.scaling_npb import zkvm_check
    return zkvm_check.main(args.data.resolve(), args.out)


def cmd_validate(args: argparse.Namespace) -> int:
    from benchmarking.scaling_npb import validate
    cases = [c for c in (args.cases or "").split(",") if c]
    systems = [s for s in args.systems.split(",") if s]
    unknown = set(systems) - set(validate.SYSTEMS)
    if unknown:
        raise SystemExit(f"unknown systems {sorted(unknown)}; known: {', '.join(validate.SYSTEMS)}")
    return validate.main(args.data.resolve(), args.cap, args.out, args.jobs, args.memory, args.timeout, cases,
                         systems, args.max_elements)


def cmd_remote_list(args: argparse.Namespace) -> int:
    from benchmarking.scaling_npb import validate
    lines = validate.remote_list(args.data.resolve(), args.results)
    (args.data / "validate_on_remote.txt").write_text("".join(f"{line}\n" for line in lines))
    print(f"{len(lines)} (cap, case, system) to validate on the remote -> {args.data / 'validate_on_remote.txt'}")
    return 0


def cmd_table(args: argparse.Namespace) -> int:
    from benchmarking.scaling_npb import table
    print(table.render(args.data.resolve(), args.validation or []))
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    from benchmarking.scaling_npb import analyze
    return analyze.main(args.data.resolve(), args.zinnia, args.zkvm, args.counts, args.out,
                        args.reference_zinnia, args.reference_zkvm)


def cmd_delegate_cases(args: argparse.Namespace) -> int:
    from benchmarking.scaling_npb import analyze
    print(",".join(analyze.differing_cases(args.case_set, args.counts, args.target)))
    return 0


def parser() -> argparse.ArgumentParser:
    top = argparse.ArgumentParser(prog="python -m benchmarking.scaling_npb", description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = top.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="write the scaled entries and the case sets")
    gen.add_argument("--data", type=Path, required=True)
    gen.add_argument("--npbench", type=Path, required=True)
    gen.add_argument("--pythran", type=Path, required=True)
    gen.add_argument("--caps", help="comma-separated caps (default 32,64,128)")
    gen.set_defaults(func=cmd_generate)

    rel = sub.add_parser("relocate", help="point the case sets at DATA's current location")
    rel.add_argument("--data", type=Path, required=True)
    rel.set_defaults(func=cmd_relocate)

    zk = sub.add_parser("zkvm-check", help="diff each scaled zkvm.rs against the original")
    zk.add_argument("--data", type=Path, required=True)
    zk.add_argument("--out", type=Path)
    zk.set_defaults(func=cmd_zkvm_check)

    val = sub.add_parser("validate", help="NumPy reference and Zinnia mock outputs on both targets")
    val.add_argument("--data", type=Path, required=True)
    val.add_argument("--cap", type=int, required=True)
    val.add_argument("--out", type=Path, required=True)
    val.add_argument("--cases", help="comma-separated case ids (default: every scaled case)")
    val.add_argument("--systems", default="zinnia_halo2,zinnia_halo2_delegate,zinnia_noir,zinnia_noir_delegate")
    val.add_argument("--jobs", type=int, default=4)
    val.add_argument("--memory", default="16G")
    val.add_argument("--timeout", type=float, default=1800)
    val.add_argument("--max-elements", type=int, help="leave points with a larger input A to the remote")
    val.set_defaults(func=cmd_validate)

    rem = sub.add_parser("remote-list", help="write DATA/validate_on_remote.txt from local validation results")
    rem.add_argument("--data", type=Path, required=True)
    rem.add_argument("results", type=Path, nargs="+", help="validate outputs (JSON lines)")
    rem.set_defaults(func=cmd_remote_list)

    tab = sub.add_parser("table", help="the classification table (markdown)")
    tab.add_argument("--data", type=Path, required=True)
    tab.add_argument("--validation", type=Path, action="append")
    tab.set_defaults(func=cmd_table)

    dc = sub.add_parser("delegate-cases", help="cases whose @delegate circuit differs (or a count failed)")
    dc.add_argument("--target", choices=["halo2", "noir"], required=True)
    dc.add_argument("--case-set", type=Path, required=True, help="the cap's case set")
    dc.add_argument("counts", type=Path, nargs="+", help="measure.py outputs of final and fix_delegate")
    dc.set_defaults(func=cmd_delegate_cases)

    ana = sub.add_parser("analyze", help="Zinnia vs zkVM prove time and Zinnia vs @delegate gates")
    ana.add_argument("--data", type=Path, required=True)
    ana.add_argument("--zinnia", type=Path, nargs="*", default=[], help="Zinnia epoch directories (stage B)")
    ana.add_argument("--zkvm", type=Path, nargs="*", default=[], help="zkVM epoch directories (stage C)")
    ana.add_argument("--counts", type=Path, nargs="*", default=[],
                     help="measure.py outputs, configurations final and fix_delegate (stage A)")
    ana.add_argument("--reference-zinnia", type=Path, nargs="*", default=[],
                     help="CAP-16 Zinnia epochs of the performance set (optional)")
    ana.add_argument("--reference-zkvm", type=Path, nargs="*", default=[],
                     help="CAP-16 zkVM epochs of the performance set (optional)")
    ana.add_argument("--out", type=Path)
    ana.set_defaults(func=cmd_analyze)
    return top


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
