//! Checks: the program's assertions read as facts, the cost of a check on
//! each target, and the emission of a check for a fact.

use std::collections::{HashMap, HashSet};

use crate::builder::IRBuilder;
use crate::ir::IRGraph;
use crate::ir_defs::IR;
use crate::types::{StmtId, Value};

use super::construction::Target;
use super::graph::{Bound, Claim, Fact, Scope};

/// An assertion of the program and the facts it establishes. The first
/// form is the fact on the asserted operand itself, on every path; a check
/// the IR generator emitted under an execution predicate `p` (on the
/// guarded operand `select(p, x, safe)`, or as `assert(select(p, c, true))`)
/// also has the form "the fact on `x` where `p` holds", which is
/// equivalent: where `p` is false the guarded operand is the safe value,
/// which passes.
///
/// `consequences` are facts the check implies but that do not imply it
/// back (so they never stand for the check as a requirement); they only
/// let the check make later checks redundant (`dominated`).
#[derive(Clone, Debug)]
pub struct ProgramCheck {
    pub stmt: StmtId,
    pub forms: Vec<Fact>,
    pub consequences: Vec<Fact>,
}

/// The program's assertions the layer understands, in statement order.
pub fn program_checks(graph: &IRGraph) -> Vec<ProgramCheck> {
    let bits: HashSet<StmtId> = graph
        .stmts
        .iter()
        .filter(|s| matches!(s.ir, IR::AssertBit))
        .filter_map(|s| s.arguments.first().copied())
        .collect();
    graph
        .stmts
        .iter()
        .filter_map(|stmt| {
            let forms = check_forms(graph, &stmt.ir, &stmt.arguments);
            let consequences = check_consequences(graph, &bits, &stmt.ir, &stmt.arguments);
            (!forms.is_empty()).then_some(ProgramCheck { stmt: stmt.stmt_id, forms, consequences })
        })
        .collect()
}

/// Facts a check implies that are weaker than it. One kind: NumPy's index
/// wrap `w = select(s, x + k, x)` (`helpers/static_array/base.rs`) checked
/// as `0 <= w < h`, with `s` asserted to be a bit and `k >= 0`, implies
/// `-k <= x < h`: for `s = 1`, `x = w - k` lies in `[-k, h - k)`; for
/// `s = 0`, `x = w` lies in `[0, h)`. (It does not imply the check back:
/// that also pins `s`.) The same index's write window `-n <= x < n`
/// (`AssertU128Lt(x + n, 2n)`) is then redundant after the read's check.
/// The wrap may also be a placeholder (`OpKind::Wrap`, position output).
fn check_consequences(graph: &IRGraph, bits: &HashSet<StmtId>, ir: &IR, args: &[StmtId]) -> Vec<Fact> {
    let stmt = |s: StmtId| &graph.stmts[s as usize];
    let constant = |s: StmtId| match stmt(s).ir {
        IR::ConstantInt { value } => Some(value),
        _ => None,
    };
    let (IR::AssertU128Lt, &[w, hi]) = (ir, args) else { return Vec::new() };
    // The same wrap as an operator placeholder (`lower::memory::wrap`): its
    // position `w` of `x` on an axis of `n`, checked below `h <= n`, is
    // `x` or `x + n`, so `-n <= x < h` whatever construction lowers it.
    if let (IR::ArrayOp(op), Some(h)) = (&stmt(w).ir, constant(hi)) {
        if let (crate::lower::OpKind::Wrap { len }, 0, Some(&x)) = (op.kind, op.output, stmt(w).arguments.first()) {
            if h <= i64::from(len) {
                return vec![Fact::everywhere(Claim::InRange { x, lo: -i64::from(len), hi: Bound::Const(h) })];
            }
        }
        return Vec::new();
    }
    let (Some(h), (IR::SelectI, &[s, shifted, x])) = (constant(hi), (&stmt(w).ir, stmt(w).arguments.as_slice())) else {
        return Vec::new();
    };
    let offset = match (&stmt(shifted).ir, stmt(shifted).arguments.as_slice()) {
        (IR::AddI, &[y, k]) if y == x => constant(k),
        (IR::AddI, &[k, y]) if y == x => constant(k),
        _ => None,
    };
    match offset {
        Some(k) if k >= 0 && bits.contains(&s) => {
            vec![Fact::everywhere(Claim::InRange { x, lo: -k, hi: Bound::Const(h) })]
        }
        _ => Vec::new(),
    }
}

fn check_forms(graph: &IRGraph, ir: &IR, args: &[StmtId]) -> Vec<Fact> {
    let constant = |s: StmtId| match graph.stmts[s as usize].ir {
        IR::ConstantInt { value } => Some(value),
        IR::ConstantBool { value } => Some(i64::from(value)),
        _ => None,
    };
    // `select(p, x, k)` with a constant `k`.
    let guarded = |s: StmtId| match (&graph.stmts[s as usize].ir, graph.stmts[s as usize].arguments.as_slice()) {
        (IR::SelectI | IR::SelectB, &[p, x, k]) => constant(k).map(|k| (p, x, k)),
        _ => None,
    };
    // `x + k` with a constant `k`.
    let shifted = |s: StmtId| match (&graph.stmts[s as usize].ir, graph.stmts[s as usize].arguments.as_slice()) {
        (IR::AddI, &[x, k]) => constant(k).map(|k| (x, k)).or_else(|| constant(x).map(|c| (k, c))),
        _ => None,
    };
    match (ir, args) {
        (IR::AssertU128Lt, &[v, hi]) => {
            let bound = constant(hi).map_or(Bound::Stmt(hi), Bound::Const);
            let mut forms = vec![Fact::everywhere(Claim::InRange { x: v, lo: 0, hi: bound })];
            // `0 <= x + k < h` is `-k <= x < h - k` (NumPy's index window
            // `-n <= i < n` is checked as `AssertU128Lt(i + n, 2n)`).
            if let (Some((x, k)), Bound::Const(h)) = (shifted(v), bound) {
                forms.push(Fact::everywhere(Claim::InRange { x, lo: -k, hi: Bound::Const(h - k) }));
            }
            if let Some((p, x, k)) = guarded(v) {
                if k >= 0 && matches!(bound, Bound::Const(h) if k < h) {
                    forms.push(Fact { claim: Claim::InRange { x, lo: 0, hi: bound }, scope: Scope::When(p) });
                }
            }
            forms
        }
        (IR::AssertNonNegI, &[v]) => {
            let mut forms = vec![Fact::everywhere(Claim::NonNegative(v))];
            if let Some((p, x, k)) = guarded(v) {
                if k >= 0 {
                    forms.push(Fact { claim: Claim::NonNegative(x), scope: Scope::When(p) });
                }
            }
            forms
        }
        (IR::Assert, &[c]) => {
            if let Some((p, c, 1)) = guarded(c) {
                return condition_claim(graph, c, &constant)
                    .map(|claim| vec![Fact { claim, scope: Scope::When(p) }])
                    .unwrap_or_default();
            }
            condition_claim(graph, c, &constant).map(|claim| vec![Fact::everywhere(claim)]).unwrap_or_default()
        }
        _ => Vec::new(),
    }
}

/// The claim a Boolean statement states, when it is one the layer knows.
fn condition_claim(graph: &IRGraph, c: StmtId, constant: &dyn Fn(StmtId) -> Option<i64>) -> Option<Claim> {
    let stmt = &graph.stmts[c as usize];
    let float_zero = |s: StmtId| matches!(graph.stmts[s as usize].ir, IR::ConstantFloat { value } if value == 0.0);
    let zero = |s: StmtId| constant(s) == Some(0) || float_zero(s);
    match (&stmt.ir, stmt.arguments.as_slice()) {
        (IR::GteI | IR::GteF, &[x, z]) | (IR::LteI | IR::LteF, &[z, x]) if zero(z) => Some(Claim::NonNegative(x)),
        (IR::NeI | IR::NeF, &[x, z]) | (IR::NeI | IR::NeF, &[z, x]) if zero(z) => Some(Claim::NonZero(x)),
        (IR::EqI, &[a, b]) => Some(Claim::Equal(a, b)),
        // Integer orders against a constant (an input's int64 window).
        (IR::GteI, &[x, c]) | (IR::LteI, &[c, x]) if constant(c).is_some() => {
            Some(Claim::Within { x, lo: constant(c), hi: None })
        }
        (IR::LteI, &[x, c]) | (IR::GteI, &[c, x]) if constant(c).is_some() => {
            Some(Claim::Within { x, lo: None, hi: constant(c) })
        }
        // `x < c` is `x <= c - 1`, `x > c` is `x >= c + 1` (an element
        // predicate `forall(a, lt_bound(c))`).
        (IR::LtI, &[x, c]) | (IR::GtI, &[c, x]) if super::array_facts::enabled() && constant(c).is_some() => {
            Some(Claim::Within { x, lo: None, hi: constant(c)?.checked_sub(1) })
        }
        (IR::GtI, &[x, c]) | (IR::LtI, &[c, x]) if super::array_facts::enabled() && constant(c).is_some() => {
            Some(Claim::Within { x, lo: constant(c)?.checked_add(1), hi: None })
        }
        // The order of two values (`@requires(is_sorted(a))` asserts each
        // adjacent pair); floats are ordered as their codes.
        (IR::LteI | IR::LteF, &[x, y]) | (IR::GteI | IR::GteF, &[y, x]) if super::array_facts::enabled() => {
            Some(Claim::Ordered { x, y, strict: false })
        }
        (IR::LtI | IR::LtF, &[x, y]) | (IR::GtI | IR::GtF, &[y, x]) if super::array_facts::enabled() => {
            Some(Claim::Ordered { x, y, strict: true })
        }
        _ => None,
    }
}

/// The checks an earlier check implies ("range checks ⇒ bounds": a later
/// `0 <= x < 8` after `0 <= x < 4`, or `x >= 0` after it). The IR has no
/// control flow, so every assertion is a constraint of the circuit; an
/// implied one adds nothing. Only earlier checks count, so a backend that
/// reads the bound a check proves for the statements after it (a cheaper
/// gadget for a value proven in range) still finds a bound as early.
pub fn dominated(checks: &[ProgramCheck]) -> HashSet<StmtId> {
    let mut known: HashMap<StmtId, Vec<Fact>> = HashMap::new();
    let mut dominated = HashSet::new();
    for check in checks {
        let implied = check.forms.iter().any(|form| {
            let Some(&subject) = form.claim.subjects().first() else { return false };
            known.get(&subject).is_some_and(|facts| facts.iter().any(|k| k.implies(form)))
        });
        if implied {
            dominated.insert(check.stmt);
            continue;
        }
        for form in check.forms.iter().chain(&check.consequences) {
            if let Some(&subject) = form.claim.subjects().first() {
                known.entry(subject).or_default().push(*form);
            }
        }
    }
    dominated
}

/// The estimated cost of checking `claim` on `target` (halo2-base cells,
/// Noir gates).
pub fn cost(claim: &Claim, target: Target) -> u64 {
    match (claim, target) {
        (Claim::InRange { .. }, Target::Halo2Base) => 8,
        // The Noir emitter checks `0 <= x < h` for a constant `h` as one
        // range check of `x` when `h` is a power of two, else as range
        // checks of `x` and `h - 1 - x` (measured on 0345 / 0346: 4 gates for
        // h = 6, about 1 for h = 8).
        (&Claim::InRange { hi: Bound::Const(h), .. }, Target::Noir) if h > 0 => {
            if (h as u64).is_power_of_two() {
                1
            } else {
                4
            }
        }
        (Claim::InRange { .. }, Target::Noir) => 5,
        (Claim::NonNegative(_), Target::Halo2Base) => 8,
        (Claim::NonNegative(_), Target::Noir) => 4,
        (Claim::NonZero(_), Target::Halo2Base) => 4,
        (Claim::NonZero(_), Target::Noir) => 3,
        (Claim::Equal(..), _) => 1,
        (Claim::Within { .. }, Target::Halo2Base) => 17,
        (Claim::Within { .. }, Target::Noir) => 3,
        (&Claim::Permutation(a, _), target) => super::permutation::hit_count_cost(a.entries().len() as u64, target, false),
        (Claim::Ordered { .. }, Target::Halo2Base) => 34,
        (Claim::Ordered { .. }, Target::Noir) => 7,
        // The array claims are checked value by value (`emit_claim`).
        (&Claim::ElementsWithin { list, lo, hi }, target) => {
            let sides = u64::from(lo.is_some()) + u64::from(hi.is_some());
            list.stmts().len() as u64 * sides * cost(&Claim::Within { x: 0, lo: Some(0), hi: None }, target)
        }
        (&Claim::NonDecreasing(list) | &Claim::StrictlyIncreasing(list), target) => {
            list.len().saturating_sub(1) as u64 * cost(&Claim::Ordered { x: 0, y: 0, strict: false }, target)
        }
        (&Claim::Distinct(list), target) => {
            let n = list.len() as u64;
            n * n.saturating_sub(1) / 2 * cost(&Claim::NonZero(0), target)
        }
        (&Claim::CountBound { list, .. }, Target::Halo2Base) => 12 * list.len() as u64 + 34,
        (&Claim::CountBound { list, .. }, Target::Noir) => 3 * list.len() as u64 + 7,
    }
}

/// Emit a check of `fact` where its scope holds (through the conditional
/// checks of `builder/guard.rs`); `lookup` maps input statements to their
/// lowered values.
pub fn emit(b: &mut IRBuilder, fact: &Fact, lookup: &HashMap<StmtId, Value>) {
    match fact.scope {
        Scope::Everywhere => emit_claim(b, &fact.claim, lookup),
        Scope::When(p) => {
            let p = lookup[&p].clone();
            b.under_condition(&p, |b| emit_claim(b, &fact.claim, lookup));
        }
    }
}

fn emit_claim(b: &mut IRBuilder, claim: &Claim, lookup: &HashMap<StmtId, Value>) {
    let v = |s: StmtId| lookup[&s].clone();
    match *claim {
        Claim::InRange { x, lo, hi } => {
            let hi = match hi {
                Bound::Const(h) => b.ir_constant_int(h),
                Bound::Stmt(h) => v(h),
            };
            let (x, hi) = if lo == 0 {
                (v(x), hi)
            } else {
                let lo = b.ir_constant_int(lo);
                (b.ir_sub_i(&v(x), &lo), b.ir_sub_i(&hi, &lo))
            };
            b.ir_assert_u128_lt(&x, &hi);
        }
        Claim::NonNegative(x) => {
            b.ir_assert_non_neg_i(&v(x));
        }
        Claim::NonZero(x) => {
            let zero = b.ir_constant_int(0);
            let nonzero = b.ir_not_equal_i(&v(x), &zero);
            b.ir_assert(&nonzero);
        }
        Claim::Equal(x, y) => {
            let equal = b.ir_equal_i(&v(x), &v(y));
            b.ir_assert(&equal);
        }
        Claim::Permutation(values_list, targets_list) => {
            let mut values = |list: super::ListId| -> Vec<Value> {
                list.entries()
                    .iter()
                    .map(|entry| match *entry {
                        super::ListEntry::Stmt(s) => v(s),
                        super::ListEntry::Int(c) => b.ir_constant_int(c),
                    })
                    .collect()
            };
            let (values, targets) = (values(values_list), values(targets_list));
            super::permutation::emit_hit_counts(b, &values, &targets);
        }
        Claim::Within { x, lo, hi } => {
            if let Some(lo) = lo {
                let lo = b.ir_constant_int(lo);
                let above = b.ir_greater_than_or_equal_i(&v(x), &lo);
                b.ir_assert(&above);
            }
            if let Some(hi) = hi {
                let hi = b.ir_constant_int(hi);
                let below = b.ir_less_than_or_equal_i(&v(x), &hi);
                b.ir_assert(&below);
            }
        }
        Claim::Ordered { x, y, strict } => {
            let (x, y) = (v(x), v(y));
            let holds = match (strict, matches!(x, Value::Float(_)) || matches!(y, Value::Float(_))) {
                (false, false) => b.ir_less_than_or_equal_i(&x, &y),
                (true, false) => b.ir_less_than_i(&x, &y),
                (false, true) => b.ir_less_than_or_equal_f(&x, &y),
                (true, true) => b.ir_less_than_f(&x, &y),
            };
            b.ir_assert(&holds);
        }
        Claim::ElementsWithin { list, lo, hi } => {
            for x in list.stmts() {
                emit_claim(b, &Claim::Within { x, lo, hi }, lookup);
            }
        }
        Claim::NonDecreasing(list) | Claim::StrictlyIncreasing(list) => {
            let strict = matches!(claim, Claim::StrictlyIncreasing(_));
            let values = entry_values(b, list, lookup);
            for pair in values.windows(2) {
                let holds = if strict { less(b, &pair[0], &pair[1]) } else { less_or_equal(b, &pair[0], &pair[1]) };
                b.ir_assert(&holds);
            }
        }
        Claim::Distinct(list) => {
            let values = entry_values(b, list, lookup);
            for i in 0..values.len() {
                for j in i + 1..values.len() {
                    let differ = if matches!(values[i], Value::Float(_)) || matches!(values[j], Value::Float(_)) {
                        b.ir_not_equal_f(&values[i], &values[j])
                    } else {
                        b.ir_not_equal_i(&values[i], &values[j])
                    };
                    b.ir_assert(&differ);
                }
            }
        }
        Claim::CountBound { list, max_nonzero } => {
            let mut count = b.ir_constant_int(0);
            for value in entry_values(b, list, lookup) {
                let zero = b.ir_constant_int(0);
                let nonzero = b.ir_not_equal_i(&value, &zero);
                let nonzero = b.ir_int_cast(&nonzero);
                count = b.ir_add_i(&count, &nonzero);
            }
            let bound = b.ir_constant_int(i64::from(max_nonzero));
            let holds = b.ir_less_than_or_equal_i(&count, &bound);
            b.ir_assert(&holds);
        }
    }
}

/// The lowered values of a list's entries (constants for its integers).
fn entry_values(b: &mut IRBuilder, list: super::ListId, lookup: &HashMap<StmtId, Value>) -> Vec<Value> {
    list.entries()
        .iter()
        .map(|entry| match *entry {
            super::ListEntry::Stmt(s) => lookup[&s].clone(),
            super::ListEntry::Int(c) => b.ir_constant_int(c),
        })
        .collect()
}

fn less(b: &mut IRBuilder, x: &Value, y: &Value) -> Value {
    if matches!(x, Value::Float(_)) || matches!(y, Value::Float(_)) {
        b.ir_less_than_f(x, y)
    } else {
        b.ir_less_than_i(x, y)
    }
}

fn less_or_equal(b: &mut IRBuilder, x: &Value, y: &Value) -> Value {
    if matches!(x, Value::Float(_)) || matches!(y, Value::Float(_)) {
        b.ir_less_than_or_equal_f(x, y)
    } else {
        b.ir_less_than_or_equal_i(x, y)
    }
}

// ── Merged range checks ─────────────────────────────────────────────────

/// One check `lo <= x <= hi` (`InRange { x, lo, hi + 1 }`) for every value
/// `x` the program bounds on both sides by checks that hold everywhere, with
/// its cost on `target`: the *range realization* of an element range. The
/// check implies each of the checks it merges (an input's int64 window, the
/// two sides `forall(a, in_range(lo, hi))` asserts, an index bound), so the
/// selection may keep it in their place when it is cheaper; it is emitted
/// where the first of the checks it replaces stood (`emit`).
///
/// It is also how a narrow range becomes visible to the backends, which
/// size a comparison by the range the constraints prove (halo2-base) or by
/// the intervals the asserts deposit (Noir): the one `w`-bit check of
/// `x - lo` proves `x` in `[lo, hi]` (both backends carry a range across a
/// constant offset), where two one-sided checks of a 64-bit value prove
/// nothing narrower than the window. Comparisons, selects, sorts and sums
/// over `x` then work at `w` bits.
///
/// Only ranges narrower than the int64 window are merged (the window alone
/// is the input-domain check, kept as the IR generator emits it).
pub fn range_merges(checks: &[ProgramCheck], target: Target) -> Vec<(Fact, u64)> {
    if !super::array_facts::enabled() {
        return Vec::new();
    }
    // Per value: the tightest bounds the checks state, in statement order.
    let mut bounds: Vec<(StmtId, Option<i64>, Option<i64>)> = Vec::new();
    let mut index: HashMap<StmtId, usize> = HashMap::new();
    for check in checks {
        for form in check.forms.iter().filter(|f| f.scope == Scope::Everywhere) {
            let (x, lo, hi) = match form.claim {
                Claim::Within { x, lo, hi } => (x, lo, hi),
                Claim::InRange { x, lo, hi: Bound::Const(h) } => (x, Some(lo), h.checked_sub(1)),
                Claim::NonNegative(x) => (x, Some(0), None),
                _ => continue,
            };
            let slot = *index.entry(x).or_insert_with(|| {
                bounds.push((x, None, None));
                bounds.len() - 1
            });
            let entry = &mut bounds[slot];
            entry.1 = match (entry.1, lo) {
                (Some(a), Some(b)) => Some(a.max(b)),
                (a, b) => a.or(b),
            };
            entry.2 = match (entry.2, hi) {
                (Some(a), Some(b)) => Some(a.min(b)),
                (a, b) => a.or(b),
            };
        }
    }
    bounds
        .into_iter()
        .filter_map(|(x, lo, hi)| {
            let (lo, hi) = (lo?, hi?);
            let span = (hi as i128) - (lo as i128);
            // Narrower than the window (and not empty: an empty range has no
            // proof, which the checks as written already say).
            if !(0..1i128 << 62).contains(&span) {
                return None;
            }
            let fact = Fact::everywhere(Claim::InRange { x, lo, hi: Bound::Const(hi.checked_add(1)?) });
            Some((fact, range_check_cost(span as u64 + 1, lo != 0, target)))
        })
        .collect()
}

/// The cost of checking `0 <= x - lo < size` (`AssertU128Lt`): halo2-base
/// range-checks the value, and for a size that is no power of two also the
/// gap below it, each `RC(w)` cells at lookup width 8 (limbs, their
/// recombination, a partial top limb), plus the subtraction of `lo`; Noir
/// one range check, or two for a size that is no power of two (as
/// `cost(InRange)` says).
pub fn range_check_cost(size: u64, offset: bool, target: Target) -> u64 {
    let width = u64::from(64 - (size.max(2) - 1).leading_zeros());
    let checks = if size.is_power_of_two() { 1 } else { 2 };
    match target {
        Target::Halo2Base => {
            let limbs = width.div_ceil(8);
            let recombine = if limbs == 1 { 0 } else { 3 * limbs - 2 };
            let partial = if width % 8 == 0 { 0 } else { 5 };
            checks * (limbs + recombine + partial) + u64::from(offset) + 2
        }
        Target::Noir => checks + u64::from(offset),
    }
}

/// Program checks a gadget's postcondition or a definition may imply but
/// that `target`'s backend uses beyond the fact they state, so they are not
/// dropped as established: on Noir, a check `x >= 0` of a value whose root
/// the program takes (`np.sqrt(x)`) also proves `x` below `2^64`, which lets
/// the root take its form without a sign (`qsqrt_of(false, x)`); without
/// the check the root pays for the sign itself, more than the check costs.
/// (halo2-base's root proves its own range at about the check's cost, so
/// there a check `x >= 0` that `x`'s definition implies, such as a sum of
/// squares, is dropped.)
pub fn kept_for_backend(graph: &IRGraph, checks: &[ProgramCheck], target: Target) -> HashSet<StmtId> {
    if target != Target::Noir || !super::array_facts::enabled() {
        return HashSet::new();
    }
    let rooted: HashSet<StmtId> =
        graph.stmts.iter().filter(|s| matches!(s.ir, IR::SqrtF)).filter_map(|s| s.arguments.first().copied()).collect();
    checks
        .iter()
        .filter(|c| c.forms.iter().any(|f| matches!(f.claim, Claim::NonNegative(x) if rooted.contains(&x))))
        .map(|c| c.stmt)
        .collect()
}

/// The bounds the checks state for each value (the tightest of each side),
/// from their forms holding everywhere: `Within`, `InRange` with a constant
/// bound, `NonNegative`.
pub fn stated_ranges<'c>(forms: impl IntoIterator<Item = &'c Fact>) -> HashMap<StmtId, (Option<i64>, Option<i64>)> {
    let mut ranges: HashMap<StmtId, (Option<i64>, Option<i64>)> = HashMap::new();
    for form in forms.into_iter().filter(|f| f.scope == Scope::Everywhere) {
        let (x, lo, hi) = match form.claim {
            Claim::Within { x, lo, hi } => (x, lo, hi),
            Claim::InRange { x, lo, hi: Bound::Const(h) } => (x, Some(lo), h.checked_sub(1)),
            Claim::NonNegative(x) => (x, Some(0), None),
            _ => continue,
        };
        let entry = ranges.entry(x).or_insert((None, None));
        entry.0 = entry.0.max(lo);
        entry.1 = match (entry.1, hi) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        };
    }
    ranges
}
