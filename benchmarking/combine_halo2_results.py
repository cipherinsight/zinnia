"""Combine hw_halo2_results.json and zinnia_halo2_results_v2.json into a
single side-by-side Markdown table (`HALO2_FULL_COMPARISON_v2.md`).

v2 splits Zinnia's prove time into keygen vs warm prove, mirroring HW's
"keygen separately, time only prove" methodology. The "prove" column on
both sides is now apples-to-apples.

Usage:
    python3 combine_halo2_results.py
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

THIS_DIR = Path(__file__).parent
ZN_FILE_DEFAULT = "zinnia_halo2_results_v2.json"
HW_FILE_DEFAULT = "hw_halo2_results.json"
OUT_DEFAULT = "HALO2_FULL_COMPARISON_v2.md"

# Cases excluded from the headline median because the HW baseline silently
# specialises a runtime kernel parameter as a compile-time constant, or
# otherwise implements a strictly weaker kernel than `kernel.py` declares.
# Three sub-categories below — see the output's "Structurally unfair
# baselines" section for full rationale.
#
# Category A (needs zkRAM): faithful HW baseline is structurally intractable
#   in halo2-graph (1.6B+ cells via naive `select_from_idx`). Zinnia's
#   zkram is the only tractable implementation.
# Category B (tractable but specialised): faithful HW baseline is possible
#   without zkRAM, just O(N) more expensive. HW silently chose to specialise.
# Category C (other silent specialisation): HW silently changes the kernel
#   contract (output shape downgrade, or witness-controlled circuit shape).
UNFAIR_BASELINES = {
    "0415": "[C] HW uses host-provided bin_size as a compile-time loop bound (different inputs -> different circuits); bin_size never constrained in-circuit",
    # Category A — needs zkRAM
    "0388": "[A] HW hardcodes patch_size=4 for 100x200 patch-extract; faithful version needs zkRAM over 20k cells",
    "0390": "[A] HW hardcodes patch_size=10 for 100x200 patch-extract; faithful version needs zkRAM over 20k cells",
    "0490": "[A] HW hardcodes N=5, M=4, K=4 for array bounded N<=64,M<=64,K<=64; faithful version needs zkRAM",
    # Category B — runtime Integer param pinned to a const
    "0383": "[B] HW hardcodes M=10 (top-N from 100-cell argsort); kernel has runtime N: Integer",
    "0389": "[B] HW hardcodes h=6, w=12 (reshape dims); kernel has runtime h, w: Integer",
    "0391": "[B] HW hardcodes low/high for column slice; kernel has runtime low, high: Integer",
    "0392": "[B] HW hardcodes low/high for row slice; kernel has runtime low, high: Integer",
    "0393": "[B] HW hardcodes low/high/clamp for column slice; kernel has runtime low, high: Integer",
    "0398": "[B] HW hardcodes N=8 for recurrence over array bounded at N<=64",
    "0399": "[B] HW hardcodes N=8 for 3-tap recurrence over array bounded at N<=64",
    "0467": "[B] HW hardcodes window size (3,3); kernel has runtime size: Integer[2]",
    "0468": "[B] HW hardcodes window size (3,3) col-major; kernel has runtime size: Integer[2]",
    # Category C — other silent kernel-contract downgrade
    "0298": "[C] HW reads claimed n_unique from input as compile-time loop bound; different proofs would yield different circuits",
    "0322": "[C] HW silently truncates output to first match + length; kernel returns variable-length list",
}


def load(name: str) -> dict[str, dict]:
    path = THIS_DIR / name
    if not path.exists():
        print(f"WARNING: {path} missing")
        return {}
    data = json.loads(path.read_text())
    return {r["case"]: r for r in data["results"]}


def fmt_time(secs):
    if secs is None:
        return "n/a"
    if secs < 1.0:
        return f"{secs*1000:.0f}ms"
    return f"{secs:.2f}s"


def fmt_bytes(b):
    return "n/a" if b is None else f"{b}B"


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--zn", default=ZN_FILE_DEFAULT)
    ap.add_argument("--hw", default=HW_FILE_DEFAULT)
    ap.add_argument("--out", default=OUT_DEFAULT)
    args = ap.parse_args()

    hw = load(args.hw)
    zn = load(args.zn)
    cases = sorted(set(hw) | set(zn))

    out = ["# Zinnia halo2 vs handwritten halo2-graph — full DS-1000 sweep (v2)",
           "",
           "v2 methodology: a single long-lived Python process for the Zinnia sweep,",
           "so the in-memory proving-key cache stays warm across cases. Per case:",
           "one untimed warm-up prove (does keygen + 1 prove), then 3 timed warm",
           "proves. `keygen` = first_prove − median(warm_proves); `prove` =",
           "median(warm_proves). This matches HW's `cargo run prove` flow, which",
           "loads `data/case*.pk` from disk and only times the prove step.",
           "",
           "Both backends: precision_bits=32; lookup_bits=8 (Zinnia) / 6 (HW).",
           "",
           "| Case | Zin k | Zin keygen | Zin prove | Zin proof | Zin v | HW k | HW prove | HW proof | HW adv | HW v |",
           "|------|------:|-----------:|----------:|----------:|:-----:|-----:|---------:|---------:|-------:|:----:|"]

    both_ok = []
    n_zn_only = 0
    n_hw_only = 0
    n_neither = 0

    for c in cases:
        zr = zn.get(c, {})
        hr = hw.get(c, {})
        z_k = zr.get("min_k")
        z_kg = zr.get("keygen_s")
        z_pt = zr.get("prove_s")
        z_pb = zr.get("proof_bytes")
        z_v = zr.get("verify_ok")
        h_k = hr.get("min_k")
        h_pt = statistics.median(hr["prove_times_s"]) if hr.get("prove_times_s") else None
        h_sb = hr.get("snark_size_bytes")
        h_v = hr.get("verify_ok")
        h_adv = hr.get("advice_cells")
        z_vs = "✓" if z_v is True else ("✗" if z_v is False else "—")
        h_vs = "✓" if h_v is True else ("✗" if h_v is False else "—")
        flag = " ⚠️" if c in UNFAIR_BASELINES else ""

        out.append(
            f"| {c}{flag} | {z_k or '—'} | {fmt_time(z_kg)} | {fmt_time(z_pt)} | {fmt_bytes(z_pb)} | {z_vs} "
            f"| {h_k or '—'} | {fmt_time(h_pt)} | {fmt_bytes(h_sb)} | "
            f"{h_adv if h_adv is not None else '—'} | {h_vs} |"
        )

        if c in UNFAIR_BASELINES:
            # Don't fold into the headline stats; surface separately.
            continue

        if z_v and h_v and z_pt and h_pt:
            both_ok.append((c, z_pt, h_pt, z_pt / h_pt))
        elif z_v:
            n_zn_only += 1
        elif h_v:
            n_hw_only += 1
        else:
            n_neither += 1

    # Stats
    out.append("")
    out.append("## Summary")
    out.append("")
    out.append(f"- Cases attempted: **{len(cases)}**")
    out.append(f"- Both sides verified (and apples-to-apples): **{len(both_ok)}**")
    out.append(f"- Zinnia-only verified: **{n_zn_only}**")
    out.append(f"- HW-only verified: **{n_hw_only}**")
    out.append(f"- Neither verified: **{n_neither}**")
    if UNFAIR_BASELINES:
        out.append(f"- Excluded from headline (structurally unfair HW baseline): "
                   f"**{len(UNFAIR_BASELINES)}** ({', '.join(sorted(UNFAIR_BASELINES))})")

    # Document why they were excluded.
    if UNFAIR_BASELINES:
        n_a = sum(1 for v in UNFAIR_BASELINES.values() if v.startswith("[A]"))
        n_b = sum(1 for v in UNFAIR_BASELINES.values() if v.startswith("[B]"))
        n_c = sum(1 for v in UNFAIR_BASELINES.values() if v.startswith("[C]"))
        out.extend([
            "",
            "## Structurally unfair baselines (excluded ⚠️)",
            "",
            "These cases were identified by sweeping every `target.rs` for",
            "patterns where the handwritten halo2-graph baseline silently",
            "implements a strictly weaker kernel than `kernel.py` declares.",
            "Excluding them avoids letting HW's specialisation flatter the",
            "comparison.",
            "",
            f"- **Category A — needs zkRAM ({n_a} cases)**: the kernel uses a",
            "  runtime `Integer` parameter inside indexing of a large static",
            "  array (e.g. 100×200 patch extract with runtime `patch_size`).",
            "  A faithful HW baseline in halo2-graph would need `select_from_idx`",
            "  per access (~4N cells per access for an N-cell source). For a",
            "  20k-cell input × 20k outputs that's 1.6B cells, k≥30, prove time",
            "  in hours. **No tractable handwritten baseline exists; Zinnia's",
            "  zkram is the only viable implementation.** HW dodges this by",
            "  declaring `const PSZ = 4` (or similar) and asserting the runtime",
            "  witness equals that constant.",
            "",
            f"- **Category B — runtime param pinned to a const ({n_b} cases)**:",
            "  the kernel declares an `Integer` parameter (or array of integers)",
            "  but `target.rs` hardcodes a specific value and asserts the runtime",
            "  witness equals it. A faithful HW baseline is *tractable* without",
            "  zkRAM (typically O(N²) total work), but would cost ~5-50× the",
            "  current specialised version. HW silently chose to specialise.",
            "",
            f"- **Category C — other silent kernel-contract downgrade ({n_c} cases)**:",
            "  HW changes the kernel contract in some non-equivalent way — e.g.",
            "  using a host-provided claim as a compile-time loop bound (different",
            "  proofs → different circuits), or truncating a variable-length output",
            "  to a fixed-length witness.",
            "",
            "| Case | Category & rationale |",
            "|------|--------|",
        ])
        for c in sorted(UNFAIR_BASELINES):
            out.append(f"| {c} | {UNFAIR_BASELINES[c]} |")

    if both_ok:
        ratios = [r for _, _, _, r in both_ok]
        slower = [x for x in both_ok if x[3] > 1.0]
        faster = [x for x in both_ok if x[3] < 1.0]
        out.extend([
            "",
            f"- Zinnia / HW prove-time ratio (on {len(both_ok)} both-verified cases): "
            f"median **{statistics.median(ratios):.2f}×**, range "
            f"{min(ratios):.2f}× – {max(ratios):.2f}×",
            f"- Zinnia slower: **{len(slower)} ({100*len(slower)/len(both_ok):.0f}%)**; "
            f"Zinnia faster: **{len(faster)} ({100*len(faster)/len(both_ok):.0f}%)**",
        ])

        # Distribution
        buckets = [(0.0, 0.5, "Zinnia ≤0.5× (much faster)"),
                   (0.5, 1.0, "Zinnia 0.5-1×"),
                   (1.0, 1.5, "Zinnia 1-1.5×"),
                   (1.5, 3.0, "Zinnia 1.5-3×"),
                   (3.0, 10.0, "Zinnia 3-10×"),
                   (10.0, 1e9, "Zinnia >10×")]
        out.extend(["", "### Distribution", "", "| Bucket | Cases | % |", "|---|---:|---:|"])
        for lo, hi, label in buckets:
            cs = [x for x in both_ok if lo <= x[3] < hi]
            out.append(f"| {label} | {len(cs)} | {100*len(cs)/len(both_ok):.0f}% |")

        # Worst hotspots
        worst = sorted(both_ok, key=lambda x: -x[3])[:10]
        out.extend(["", "### Residual hotspots (Zinnia slower by most)",
                    "", "| Case | Zin prove | HW prove | Ratio |",
                    "|------|----------:|---------:|------:|"])
        for c, z, h, r in worst:
            out.append(f"| {c} | {fmt_time(z)} | {fmt_time(h)} | {r:.2f}× |")

        # Biggest Zinnia wins
        best = sorted(both_ok, key=lambda x: x[3])[:10]
        out.extend(["", "### Biggest Zinnia wins (Zinnia faster by most)",
                    "", "| Case | Zin prove | HW prove | Ratio | HW/Zin |",
                    "|------|----------:|---------:|------:|------:|"])
        for c, z, h, r in best:
            out.append(f"| {c} | {fmt_time(z)} | {fmt_time(h)} | {r:.3f}× | {1/r:.1f}× |")

    target = THIS_DIR / args.out
    target.write_text("\n".join(out) + "\n")
    print(f"Wrote {target}")


if __name__ == "__main__":
    main()
