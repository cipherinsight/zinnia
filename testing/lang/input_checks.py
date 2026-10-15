"""Test helpers: an IR dump without the per-input domain checks, and the range
the asserts of an IR dump confine a value to.

Every input element is checked to its dtype's domain (`ir_gen/input_domain.rs`):
`x >= i64::MIN` and `x <= i64::MAX` (each asserted) for integers and float
codes, `AssertBit` for booleans. IR-shape tests that look at other statements
drop these first. (When other checks bound the same value on both sides, the
lowering merges them all into one range check; see `checked_range`.)
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


def checked_range(stmts, x, end=None):
    """The range `[lo, hi]` the asserts among `stmts[:end]` (an IR dump, all
    of it by default) confine statement `x` to, in the values the circuit
    compares: raw integers, Q32 codes for floats. `None` marks an open end.

    A bound is asserted either as a comparison of `x` with a constant
    (`assert(x >= c)`, ...) or, where the lowering merged the checks bounding
    `x` (`lower::checks::range_merges`), as one range check
    `AssertU128Lt(x - c, h)`, which enforces `c <= x < c + h`."""
    by_id = {s["stmt_id"]: s for s in stmts}

    def constant(i):
        ir = by_id[i]["ir_instance"]
        if ir["__class__"] == "ConstantIntIR":
            return ir["ir_data"]["value"]
        if ir["__class__"] == "ConstantFloatIR":
            return round(ir["ir_data"]["value"] * 2**32)
        return None

    def offset_of(i):
        """`c` when statement `i` is `x - c` (or `x + (-c)`), else `None`."""
        if i == x:
            return 0
        args = by_id[i]["arguments"]
        if len(args) != 2 or args[0] != x or constant(args[1]) is None:
            return None
        return {"SubIIR": constant(args[1]), "AddIIR": -constant(args[1])}.get(_class(by_id[i]))

    lo, hi = None, None
    for s in stmts if end is None else stmts[:end]:
        args = s.get("arguments", [])
        new_lo = new_hi = None
        if _class(s) == "AssertU128LtIR" and constant(args[1]) is not None:
            c = offset_of(args[0])
            if c is not None:
                new_lo, new_hi = c, c + constant(args[1]) - 1
        elif _class(s) == "AssertIR":
            cond = by_id[args[0]]
            operands = cond.get("arguments", [])
            if len(operands) != 2 or operands[0] != x or constant(operands[1]) is None:
                continue
            c, comparison = constant(operands[1]), _class(cond)
            if comparison.startswith("GreaterThanOrEqual"):
                new_lo = c
            elif comparison.startswith("GreaterThan"):
                new_lo = c + 1
            elif comparison.startswith("LessThanOrEqual"):
                new_hi = c
            elif comparison.startswith("LessThan"):
                new_hi = c - 1
        if new_lo is not None:
            lo = new_lo if lo is None else max(lo, new_lo)
        if new_hi is not None:
            hi = new_hi if hi is None else min(hi, new_hi)
    return lo, hi
