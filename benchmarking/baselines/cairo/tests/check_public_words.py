"""Execute Cairo and compare every committed i64 with unchanged native Rust.

Build kernels-native and the selected Cairo targets first. SCARB can name a
memrun wrapper. This script never proves. Names use program/function.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
BASELINES = HERE.parents[1]
BENCH = BASELINES.parent
sys.path.insert(0, str(BASELINES / "zkvm_common"))
from check_outputs import discover_cases, fixture_path  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", required=True, help="comma-separated program/function names")
    parser.add_argument("--input", choices=["A", "B"], help="default: both")
    parser.add_argument("--fixtures-dir", type=Path, help="fixture overrides under DIR/program/")
    args = parser.parse_args()
    deps = BASELINES / "target/release/deps"
    libraries = sorted(deps.glob("libzkvm_kernels-*.rlib"), key=lambda p: p.stat().st_mtime)
    if not libraries:
        parser.error("build kernels-native first (README)")
    cases = discover_cases()
    with tempfile.TemporaryDirectory(prefix="cairo-public-words-") as tmp:
        oracle = Path(tmp) / "oracle"
        subprocess.run(["rustc", "--edition=2021", str(HERE / "public_words.rs"),
                        "--extern", f"zkvm_kernels={libraries[-1]}", "-L", f"dependency={deps}",
                        "-O", "-o", str(oracle)], check=True)
        for name in args.cases.split(","):
            case = cases[name]
            if not case.function or not case.has_program("cairo"):
                parser.error(f"{name} has no shape-generic Cairo entry")
            fixture = fixture_path(case, args.fixtures_dir, None)
            for inp in [args.input] if args.input else ["A", "B"]:
                expected = json.loads(subprocess.check_output([str(oracle), name, str(fixture), inp], text=True))
                result = subprocess.run([sys.executable, str(BASELINES / "cairo/cairo_baseline.py"),
                    "--case-dir", str(case.folder), "--function", case.function, "--fixture", str(fixture),
                    "--input", inp, "--mode", "execute"], capture_output=True, text=True, check=True)
                report = json.loads(result.stdout)
                assert report["ok"], report
                assert report["public_words"] == expected, (name, inp, report["public_words"], expected)
                print(f"ok {name} {inp}: {len(expected)} committed words, bit-identical", flush=True)


if __name__ == "__main__":
    main()
