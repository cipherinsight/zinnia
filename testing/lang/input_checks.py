"""Test helper: an IR dump without the per-input domain checks.

Every input element is checked to its dtype's domain (`ir_gen/input_domain.rs`):
`x >= i64::MIN` and `x <= i64::MAX` (each asserted) for integers and float
codes, `AssertBit` for booleans. IR-shape tests that look at other statements
drop these first.
"""

I64_MIN, I64_MAX = -(2**63), 2**63 - 1
_READS = {"ReadIntegerIR", "ReadFloatIR"}
_BOUNDS = {"GreaterThanOrEqualIIR": I64_MIN, "LessThanOrEqualIIR": I64_MAX}


def _class(stmt):
    return stmt["ir_instance"]["__class__"]


def without_input_checks(stmts):
    """`stmts` (an IR dump) minus the domain checks and their constants."""
    by_id = {s["stmt_id"]: s for s in stmts}
    dropped = set()
    for s in stmts:
        args = s.get("arguments", [])
        cls = _class(s)
        if cls in _BOUNDS and len(args) == 2 and _class(by_id[args[0]]) in _READS:
            bound = by_id[args[1]]
            if _class(bound) == "ConstantIntIR" and bound["ir_instance"]["ir_data"]["value"] == _BOUNDS[cls]:
                dropped.add(s["stmt_id"])
        elif cls == "AssertBitIR" and len(args) == 1 and _class(by_id[args[0]]) in _READS:
            dropped.add(s["stmt_id"])
    for s in stmts:
        if _class(s) == "AssertIR" and s.get("arguments", [None])[0] in dropped:
            dropped.add(s["stmt_id"])
    used = {a for s in stmts if s["stmt_id"] not in dropped for a in s.get("arguments", [])}
    return [s for s in stmts if s["stmt_id"] not in dropped
            and not (_class(s) == "ConstantIntIR" and s["stmt_id"] not in used
                     and s["ir_instance"]["ir_data"]["value"] in (I64_MIN, I64_MAX))]
