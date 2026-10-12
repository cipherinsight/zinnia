"""Write the program lists of the Benchmark Programs appendix.

Source: benchmarking/datasets/case_sets/acceptance.json (entry functions and
their scope status), upstream_sources.json, and FILTERING.md for the Pythran
files removed before the case set (early exclusions, and variants dropped when
keeping the smallest file per family; the variant list is the upstream
pythran/tests/cases listing at commit 4a19d7b9 minus the files used).

usage: python make.py > <paper>/appendix/benchmark-lists.tex
"""
import collections
import json
import pathlib

D = pathlib.Path(__file__).resolve().parents[3] / "benchmarking/datasets"
A = json.loads((D / "case_sets/acceptance.json").read_text())
UP = json.loads((D / "upstream_sources.json").read_text())["programs"]
ENTRIES = [e for e in A["entries"] if e.get("entry_function", True)]

PYTHRAN_EARLY = ["arc_distance_list", "brownian", "count_perm_sig", "empirical",
                 "frequent_itemsets", "grayscott", "julia_pure_python", "kmeans",
                 "loopy_jacob", "monte_carlo_pricer", "pi_buffon"]
PYTHRAN_VARIANTS = ["_histogram", "diffusion_pure_python", "fbcorr",
                    "hyantes_core_numpy", "pairwise"]


def tt(s):
    return r"\texttt{" + s.replace("_", r"\_") + "}"


def name(e):
    ds, prog = e["dataset"], e["program"]
    if ds == "ds1000_numpy":
        return str(int(prog.removeprefix("case")))
    if ds == "npbench":
        return UP[prog]["upstream_path"].rsplit("/", 1)[0]
    return UP[prog]["upstream_path"].rsplit("/", 1)[-1].removesuffix(".py")


def listing(ds, status):
    groups = collections.defaultdict(list)
    for e in ENTRIES:
        if e["dataset"] == ds and e["status"] == status:
            groups[name(e)].append(e["function"])
    out = []
    for n in sorted(groups, key=lambda s: (int(s) if s.isdigit() else 0, s)):
        fs = groups[n]
        out.append(n if ds == "ds1000_numpy" else
                   tt(n) + (" (" + ", ".join(tt(f) for f in sorted(fs)) + ")" if len(fs) > 1 else ""))
    return out, sum(len(v) for v in groups.values())


for ds in ["ds1000_numpy", "npbench", "pythran"]:
    for st in ["in_scope", "out_of_scope"]:
        items, n = listing(ds, st)
        print(f"% {ds} {st}: {n} entry functions")
        print(r"\def\bench" + {"ds1000_numpy": "Ds", "npbench": "Npb", "pythran": "Py"}[ds]
              + {"in_scope": "Kept", "out_of_scope": "Out"}[st] + "{" + ", ".join(items) + "}")
print(r"\def\benchPyEarly{" + ", ".join(tt(s) for s in PYTHRAN_EARLY) + "}")
print(r"\def\benchPyVariants{" + ", ".join(tt(s) for s in PYTHRAN_VARIANTS) + "}")
