"""What a `ZKCircuit` hands the Rust core and the reference evaluator.

`revision/phase0/acceptance.md` §2.1: the reference run must see exactly the
AST, configuration, chips and externals phase 0 sees, or the two would not be
running the same program. Both come from `ZinniaCompiler._core_payloads`,
which is what `compile()` itself calls, so this module is the one place the
harnesses reach into the compiler.

Every consumer of a phase 0 record needs both halves — `test_adequacy.py`,
`adequacy_hook.py` and `baseline.py` — so they live here rather than in any
one of them.
"""

from __future__ import annotations

from typing import Any


def core_payloads(zk_circuit: Any) -> tuple[dict, dict, dict, dict]:
    """The four dicts the Rust core reads: AST, config, chips, externals.

    Built exactly as `compile` builds them
    (`ZinniaCompiler._core_payloads`).
    """
    from zinnia.compile.zinnia_compiler import ZinniaCompiler

    compiler = ZinniaCompiler(zk_circuit.config)
    _, ast_json, config, chips, externals = compiler._core_payloads(
        zk_circuit.source,
        zk_circuit.name,
        {name: chip.to_internal_object() for name, chip in zk_circuit.chips.items()},
        {name: ext.to_internal_object() for name, ext in zk_circuit.externals.items()},
    )
    return ast_json, config, chips, externals


def external_impls(zk_circuit: Any) -> dict[str, Any]:
    """The real callables behind `@zk_external` functions, which `E-External`
    has no shape for and the reference run needs (`acceptance.md` §2.5).
    """
    return {name: ext.callable for name, ext in zk_circuit.externals.items()}
