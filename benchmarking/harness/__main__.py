"""Single entry point of the benchmark harness.

    python -m benchmarking.harness check-env [--systems S,..] [--out FILE]
    python -m benchmarking.harness build [--jobs N] [--baselines halo2_hw,sp1,risc0,cairo]
                                         [--noir-execute] [--skip-zinnia]
    python -m benchmarking.harness run --label L [--systems S,..] [--case-set FILE]
                                       [--cases ID,..] [--fixtures-dir DIR] [--repeats N]
                                       [--timeout S] [--memory-max 16G|none] [--threads N]
                                       [--seed N] [--halo2-k K --halo2-lookup-bits L]
                                       [--shard K/N] [--results-dir DIR]
    python -m benchmarking.harness analyze EPOCH_DIR [EPOCH_DIR ...] [--out DIR]
    python -m benchmarking.harness accept --label L [--case-set FILE] [--programs P,..]
                                          [--timeout S] [--memory-max 8G|none]

See benchmarking/README.md.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from benchmarking.harness import paths
from benchmarking.harness.adapters import ALL_SYSTEMS

DEFAULT_SYSTEMS = ("zinnia_halo2", "halo2_hw", "zinnia_noir", "noir_hw", "sp1", "risc0", "cairo")


def _list(text: str | None) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()] if text else []


def _case_id(text: str) -> str:
    """A bare number names a DS-1000 case (`309` -> `ds1000_case0309`); any
    other text is a case id as the case set writes it."""
    return f"ds1000_case{int(text):04d}" if text.isdigit() else text


def _memory(text: str) -> str | None:
    return None if text.lower() == "none" else text


def _systems(text: str | None) -> list[str]:
    systems = _list(text) or list(DEFAULT_SYSTEMS)
    unknown = [s for s in systems if s not in ALL_SYSTEMS]
    if unknown:
        raise SystemExit(f"unknown systems {unknown}; known: {', '.join(ALL_SYSTEMS)}")
    return systems


def cmd_check_env(args: argparse.Namespace) -> int:
    from benchmarking.harness.toolchains import main_check_env
    return main_check_env(_systems(args.systems), Path(args.out) if args.out else None)


def cmd_build(args: argparse.Namespace) -> int:
    from benchmarking.harness.build import (BASELINE_BUILDS, build_baselines, build_noir_execute,
                                            build_release)
    baselines = _list(args.baselines)
    unknown = [b for b in baselines if b not in BASELINE_BUILDS]
    if unknown:
        raise SystemExit(f"unknown baselines {unknown}; known: {', '.join(BASELINE_BUILDS)}")
    if not args.skip_zinnia:
        build_release(jobs=args.jobs)
    if baselines:
        build_baselines(baselines, jobs=args.jobs)
    if args.noir_execute:
        build_noir_execute(jobs=args.jobs)
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    from benchmarking.harness.case_sets import load_perf_cases, parse_shard, shard_cases
    from benchmarking.harness.protocol import RunConfig
    from benchmarking.harness.run import RunRefused, run_epoch

    case_set = Path(args.case_set) if args.case_set else (
        paths.PERF_CASE_SET if paths.PERF_CASE_SET.exists() and not args.cases else None)
    cases = load_perf_cases(case_set, [_case_id(c) for c in _list(args.cases)],
                            Path(args.fixtures_dir).resolve() if args.fixtures_dir else None)
    shard = parse_shard(args.shard) if args.shard else None
    if shard is not None:
        cases = shard_cases(cases, *shard)
    halo2_params = None
    if args.halo2_k is not None or args.halo2_lookup_bits is not None:
        if args.halo2_k is None or args.halo2_lookup_bits is None:
            raise SystemExit("--halo2-k and --halo2-lookup-bits go together")
        halo2_params = {"k": args.halo2_k, "lookup_bits": args.halo2_lookup_bits}
    config = RunConfig(systems=_systems(args.systems), repeats=args.repeats,
                       step_timeout_s=args.timeout, memory_max=_memory(args.memory_max),
                       threads=args.threads, seed=args.seed, halo2_params=halo2_params,
                       shard=list(shard) if shard else None)
    try:
        root = run_epoch(args.label, cases, config, case_set, Path(args.results_dir).resolve())
    except RunRefused as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    from benchmarking.harness.analysis.aggregate import summarize
    summarize(root)
    print(f"summary: {root / 'summary' / 'summary.md'}")
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    from benchmarking.harness.analysis.aggregate import summarize
    epochs = [Path(e) for e in args.epochs]
    if len(epochs) > 1 and not args.out:
        raise SystemExit("several epochs (the shards of one run) need --out DIR")
    out = Path(args.out) if args.out else epochs[0] / "summary"
    summarize(epochs, out)
    print((out / "summary.md").read_text())
    return 0


def cmd_accept(args: argparse.Namespace) -> int:
    from benchmarking.harness.acceptance.validate import AcceptanceConfig, run_acceptance
    from benchmarking.harness.case_sets import load_acceptance_programs

    programs = load_acceptance_programs(Path(args.case_set))
    wanted = set(_list(args.programs))
    if wanted:
        programs = [p for p in programs if p.program in wanted]
    out_dir = Path(args.epoch) / "acceptance" if args.epoch else (
        paths.RESULTS / f"{time.strftime('%Y-%m-%d')}-{args.label}" / "acceptance")
    if (out_dir / "results.json").exists():
        raise SystemExit(f"{out_dir} already has results; choose another label")
    config = AcceptanceConfig(timeout_s=args.timeout, memory_max=_memory(args.memory_max))
    print(f"results: {run_acceptance(programs, config, out_dir)}")
    return 0


def parser() -> argparse.ArgumentParser:
    top = argparse.ArgumentParser(prog="python -m benchmarking.harness")
    sub = top.add_subparsers(dest="command", required=True)

    check = sub.add_parser("check-env", help="print and record every toolchain version")
    check.add_argument("--systems", help="comma-separated systems (default: all measured)")
    check.add_argument("--out", help="also write the probes as JSON")
    check.set_defaults(func=cmd_check_env)

    build = sub.add_parser("build", help="release build of the extension + build record")
    build.add_argument("--jobs", type=int, default=5)
    build.add_argument("--baselines", help="also build these programs: halo2_hw,sp1,risc0,cairo")
    build.add_argument("--noir-execute", action="store_true",
                       help="also build Noir's artifact executor from the commit nargo reports")
    build.add_argument("--skip-zinnia", action="store_true",
                       help="only build --baselines / --noir-execute")
    build.set_defaults(func=cmd_build)

    run = sub.add_parser("run", help="run one epoch")
    run.add_argument("--label", required=True, help="epoch directory: results/<date>-<label>")
    run.add_argument("--systems", help=f"comma-separated, from {', '.join(ALL_SYSTEMS)}")
    run.add_argument("--case-set", help=f"default: {paths.relative_to_repo(paths.PERF_CASE_SET)}")
    run.add_argument("--cases", help="comma-separated case ids (0309 or ds1000_case0309)")
    run.add_argument("--fixtures-dir", help="read <dir>/<case>/fixture.json instead")
    run.add_argument("--repeats", type=int, default=5, help="timed proves per case (default 5)")
    run.add_argument("--timeout", type=float, default=1800.0, help="per-step timeout, seconds")
    run.add_argument("--memory-max", default="16G", help="per-step memory cap, or none")
    run.add_argument("--threads", type=int, help="set RAYON/OMP/HARDWARE_CONCURRENCY threads")
    run.add_argument("--seed", type=int, default=20261006)
    run.add_argument("--results-dir", default=str(paths.RESULTS),
                     help="parent of the epoch directory (default: benchmarking/results)")
    run.add_argument("--halo2-k", type=int, help="fixed k for both halo2 sides (sensitivity)")
    run.add_argument("--halo2-lookup-bits", type=int)
    run.add_argument("--shard", help="K/N: run only shard K (from 0) of N of the case set's "
                                     "cases (sorted by id, dealt out in turn)")
    run.set_defaults(func=cmd_run)

    analyze = sub.add_parser("analyze", help="summaries of an epoch (or of the shards of one)")
    analyze.add_argument("epochs", nargs="+", metavar="EPOCH_DIR")
    analyze.add_argument("--out", help="summary directory (default: <EPOCH_DIR>/summary; "
                                       "required for several epochs)")
    analyze.set_defaults(func=cmd_analyze)

    accept = sub.add_parser("accept", help="compile + mock-run + reference compare")
    accept.add_argument("--label", default="acceptance")
    accept.add_argument("--epoch", help="write into <epoch>/acceptance instead")
    accept.add_argument("--case-set", default=str(paths.ACCEPTANCE_CASE_SET))
    accept.add_argument("--programs", help="comma-separated program ids")
    accept.add_argument("--timeout", type=float, default=300.0)
    accept.add_argument("--memory-max", default="8G")
    accept.set_defaults(func=cmd_accept)
    return top


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
