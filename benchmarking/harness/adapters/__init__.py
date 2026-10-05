"""The adapter of every proving system, by name.

Zinnia's two backends and the handwritten Noir baselines run in-process in the
step child; the other baselines are external programs behind the program
contract (`adapters.contract`).
"""

from __future__ import annotations

from benchmarking.harness.adapters.base import Adapter

ZINNIA_SYSTEMS = ("zinnia_halo2", "zinnia_noir")
BASELINE_SYSTEMS = ("halo2_hw", "noir_hw", "sp1", "sp1_plonk", "risc0", "cairo")
ALL_SYSTEMS = ZINNIA_SYSTEMS + BASELINE_SYSTEMS

#: The baseline each Zinnia backend is compared with on the same proof system.
SAME_PROOF_SYSTEM = {"zinnia_halo2": "halo2_hw", "zinnia_noir": "noir_hw"}


def get_adapter(name: str) -> Adapter:
    if name == "zinnia_halo2":
        from benchmarking.harness.adapters.zinnia_halo2 import ZinniaHalo2Adapter
        return ZinniaHalo2Adapter()
    if name == "zinnia_noir":
        from benchmarking.harness.adapters.zinnia_noir import ZinniaNoirAdapter
        return ZinniaNoirAdapter()
    if name == "noir_hw":
        from benchmarking.harness.adapters.noir_hw import NoirHandwrittenAdapter
        return NoirHandwrittenAdapter()
    from benchmarking.harness.adapters.contract import CONTRACT_SYSTEMS, ContractProgramAdapter
    if name in CONTRACT_SYSTEMS:
        return ContractProgramAdapter(CONTRACT_SYSTEMS[name])
    raise ValueError(f"unknown system {name!r}; known: {', '.join(ALL_SYSTEMS)}")
