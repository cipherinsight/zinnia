"""The adapter of every proving system, by name.

Zinnia's two backends and the handwritten Noir baselines run in-process in the
step child; the other baselines are external programs behind the program
contract (`adapters.contract`).
"""

from __future__ import annotations

from benchmarking.harness.adapters.base import Adapter

ZINNIA_SYSTEMS = ("zinnia_halo2", "zinnia_noir")
#: Zinnia with every runtime-index array access left to the backend's own memory
#: (`ZINNIA_FORCE_CONSTRUCTION=@delegate`): baselines for the array-lowering layer.
DELEGATE_SYSTEMS = ("zinnia_halo2_delegate", "zinnia_noir_delegate")
BASELINE_SYSTEMS = ("halo2_hw", "noir_hw", "sp1", "sp1_groth16", "risc0", "risc0_groth16",
                    "cairo") + DELEGATE_SYSTEMS
ALL_SYSTEMS = ZINNIA_SYSTEMS + BASELINE_SYSTEMS
#: The systems that compile with the installed Zinnia extension.
EXTENSION_SYSTEMS = ZINNIA_SYSTEMS + DELEGATE_SYSTEMS

#: The baseline each Zinnia backend is compared with on the same proof system.
SAME_PROOF_SYSTEM = {"zinnia_halo2": "halo2_hw", "zinnia_noir": "noir_hw"}


def get_adapter(name: str) -> Adapter:
    if name == "zinnia_halo2":
        from benchmarking.harness.adapters.zinnia_halo2 import ZinniaHalo2Adapter
        return ZinniaHalo2Adapter()
    if name == "zinnia_noir":
        from benchmarking.harness.adapters.zinnia_noir import ZinniaNoirAdapter
        return ZinniaNoirAdapter()
    if name == "zinnia_halo2_delegate":
        from benchmarking.harness.adapters.zinnia_halo2 import ZinniaHalo2DelegateAdapter
        return ZinniaHalo2DelegateAdapter()
    if name == "zinnia_noir_delegate":
        from benchmarking.harness.adapters.zinnia_noir import ZinniaNoirDelegateAdapter
        return ZinniaNoirDelegateAdapter()
    if name == "noir_hw":
        from benchmarking.harness.adapters.noir_hw import NoirHandwrittenAdapter
        return NoirHandwrittenAdapter()
    from benchmarking.harness.adapters.contract import PROGRAMS, ContractProgramAdapter
    if name in PROGRAMS:
        return ContractProgramAdapter(PROGRAMS[name])
    raise ValueError(f"unknown system {name!r}; known: {', '.join(ALL_SYSTEMS)}")
