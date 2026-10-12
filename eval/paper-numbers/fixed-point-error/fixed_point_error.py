"""Fixed-point error of the compiled circuits on the final compiler (2569c9ed), all 214 programs.

Same metric as zinnia-paper/oopsla-2027-r1/fig/data/fixed-point-error.py: for every float output
y of NumPy (float64, the epoch's raw/_reference record) and y' of the circuit (the public outputs
of the first proof sample on fixture B), the error is |y' - y| / max(1, |y|) (absolute below 1,
relative above). Difference from the paper script: the records are taken only from the epoch-fz
proving runs (2026-10-11-fz-*, every manifest checked to be commit 2569c9ed, clean), instead of the
newest record per program across all epochs (which picked the older zk42 runs for 42 programs).

Usage: python fixed_point_error.py <epochs dir> [out.json]
"""
import glob, json, os, statistics, sys
from pathlib import Path

COMMIT = "2569c9edb8a2bf0b88e9351ca5752f164d2fad24"
EPOCHS = Path(sys.argv[1])
ALL = set((Path(__file__).resolve().parent / "rq2-all.txt").read_text().split())
shards = sorted(p for p in EPOCHS.glob("2026-10-11-fz-*") if (p / "manifest.json").exists() and "warm" not in p.name)
for s in shards:
    repo = json.load(open(s / "manifest.json"))["repository"]
    assert repo["commit"] == COMMIT and not repo["dirty"], (s, repo)
out = {"commit": COMMIT, "epochs": [s.name for s in shards], "metric": "|y'-y|/max(1,|y|), float outputs, fixture B", "backends": {}}
for backend in ("zinnia_halo2", "zinnia_noir"):
    records = {}
    for s in shards:
        for path in (s / "raw" / backend).glob("*.json"):
            case = path.stem
            if case in ALL:
                assert case not in records, ("duplicate", case)
                records[case] = path
    assert set(records) == ALL, sorted(ALL - set(records))
    per = {}
    for case, path in sorted(records.items()):
        rec = json.load(open(path))
        samples = [x for x in rec["samples"] if x.get("publics")]
        assert rec["status"] == "ok" and samples, (case, rec["status"])
        leaves = json.load(open(path.parent.parent / "_reference" / f"{case}.json"))["result"]["leaves"]
        values = samples[0]["publics"]["values"]
        assert len(values) == len(leaves), (case, len(values), len(leaves))
        floats = [(y1, y) for y1, (y, kind) in zip(values, leaves) if kind == "float"]
        errs = [abs(y1 - y) / max(1.0, abs(y)) for y1, y in floats]
        absolute = [abs(y1 - y) for y1, y in floats]
        per[case] = {"epoch": path.parent.parent.parent.name, "float_outputs": len(floats),
                     "max_error": max(errs) if errs else None, "max_abs_error": max(absolute) if absolute else None}
    with_f = {c: v for c, v in per.items() if v["max_error"] is not None}
    top = max(with_f, key=lambda c: with_f[c]["max_error"])
    top_abs = max(with_f, key=lambda c: with_f[c]["max_abs_error"])
    vals = sorted(v["max_error"] for v in with_f.values())
    out["backends"][backend] = {"programs": len(per), "with_float_outputs": len(with_f),
        "max_error": with_f[top]["max_error"], "argmax": top, "median_of_program_max": statistics.median(vals),
        "max_plain_abs_error": with_f[top_abs]["max_abs_error"], "argmax_plain_abs": top_abs,
        "programs_ge_1e-6": sorted(c for c, v in with_f.items() if v["max_error"] >= 1e-6),
        "per_program": per}
    print(f"{backend}: {len(per)} programs, {len(with_f)} with float outputs; max {with_f[top]['max_error']:.4g} ({top}), "
          f"median {statistics.median(vals):.3g}; max plain |y'-y| {with_f[top_abs]['max_abs_error']:.4g} ({top_abs}); "
          f">=1e-6: {len(out['backends'][backend]['programs_ge_1e-6'])}")
if len(sys.argv) > 2:
    json.dump(out, open(sys.argv[2], "w"), indent=1)
