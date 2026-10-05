"""Checks that the halo2_hw crate links the same dependency versions as Zinnia.

Every package that appears in both `Cargo.lock` files (the repository's and
`baselines/halo2_hw/Cargo.lock`) must resolve to the same version and source
(including the git revision), so the handwritten circuits are proved by the
same halo2-base, halo2_proofs and curve code as Zinnia's halo2-base backend.

Usage: python -m hw_check.lock_match
"""
from __future__ import annotations

import sys
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def packages(lock: Path) -> dict[tuple[str, str], str]:
    data = tomllib.loads(lock.read_text())
    return {(p["name"], p["version"]): p.get("source", "local") for p in data["package"]}


def main() -> int:
    zinnia = packages(REPO / "Cargo.lock")
    hw = packages(REPO / "benchmarking" / "baselines" / "halo2_hw" / "Cargo.lock")
    zinnia_names = {name for name, _ in zinnia}
    mismatches = []
    for (name, version), source in sorted(hw.items()):
        if name in ("halo2_hw",) or name not in zinnia_names:
            continue
        if (name, version) not in zinnia or zinnia[(name, version)] != source:
            theirs = sorted(f"{v} {s}" for (n, v), s in zinnia.items() if n == name)
            mismatches.append(f"{name} {version} {source}  vs Zinnia {theirs}")
    for key in ("halo2-base", "halo2_proofs", "halo2curves"):
        print(key, [f"{v} {s}" for (n, v), s in hw.items() if n == key])
    if mismatches:
        print("MISMATCH:\n  " + "\n  ".join(mismatches))
        return 1
    print(f"ok: {sum(1 for n, _ in hw if n in zinnia_names)} shared packages resolve identically")
    return 0


if __name__ == "__main__":
    sys.exit(main())
