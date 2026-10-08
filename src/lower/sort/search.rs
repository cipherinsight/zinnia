//! `np.searchsorted` over a sorted table (`OpKind::SearchSorted`).
//!
//! The operator: for a non-decreasing table `Y_0 <= ... <= Y_{n-1}` (the
//! shared operands) and a query `t` (each lane's own operand), the number
//! `u` of entries `<= t` (`right`) or `< t` (left). It is how the frontend
//! lowers `np.searchsorted(Y, t, side)`, and `(Y > t).argmax()` and its
//! mirrored forms when `Y` is a constant sorted array (`ecdf`'s
//! `np.arange(1, n + 1) / n`): `u`, or 0 when `u == n`.
//!
//! The order of the table is NumPy's precondition (its binary search
//! assumes it): every construction lists `NonDecreasing(Y)` among its
//! preconditions. It holds by definition for a sorted constant table and
//! for the outputs of a sort, is established by the program's checks for a
//! table it asserts sorted (`@requires(is_sorted(Y))`), and is otherwise
//! checked (no proof for an unsorted table, as for an index out of range).
//!
//! Terms. A table entry and the query are compared as their *codes*: the
//! integer, or the Q32 fixed-point encoding of a float (`round(v * 2^32)`),
//! the field values the backends compare. A table is *affine* when its
//! codes are `floor((A k + B) / C)` for integers `A >= 0`, `B`, `C > 0`
//! (checked entry by entry at compile time): a quantized arithmetic
//! progression such as `(k + 1) / 50`.
//!
//! Two constructions, the selection takes the cheaper per target:
//! * [`SearchByCount`]: one comparison per entry and a sum (always applies);
//! * [`SearchByPosition`]: the position hinted and pinned by two
//!   comparisons with the entries around it, the entries evaluated in
//!   closed form for an affine table, else read from the table in memory.

use std::ops::Range;

use crate::builder::IRBuilder;
use crate::ir_defs::{HintKind, IR};
use crate::lower::op::{ArrayOp, Constant, OperatorSemantics};
use crate::lower::{Candidate, Construction, Context, EmitCx, Emitter, Group, Lane, OpKind, Requirement, Target};
use crate::provenance::Interval;
use crate::types::{NumberType, ScalarValue, StmtId, Value};

// ── Semantics ───────────────────────────────────────────────────────────

pub struct SearchSortedSemantics;

impl OperatorSemantics for SearchSortedSemantics {
    fn fold(&self, op: &ArrayOp, args: &[Value]) -> Option<Constant> {
        let OpKind::SearchSorted { right } = op.kind else { return None };
        let n = op.shared as usize;
        let query = number(args.get(n)?)?;
        let entries: Vec<f64> = args[..n].iter().map(number).collect::<Option<_>>()?;
        // An unsorted table has no proof (its order is checked).
        if !entries.windows(2).all(|w| w[0] <= w[1]) {
            return None;
        }
        let count = entries.iter().filter(|&&y| if right { y <= query } else { y < query }).count();
        Some(Constant::Int(count as i64))
    }

    fn number_type(&self, _op: &ArrayOp, _args: &[Value]) -> NumberType {
        NumberType::Integer
    }

    fn bounds(&self, op: &ArrayOp, _operands: &[Option<Interval>]) -> Option<Interval> {
        Some(Interval { lo: 0, hi: i64::from(op.shared) })
    }

    fn picks_from(&self, _op: &ArrayOp) -> Option<Range<usize>> {
        None
    }
}

/// A constant's value as a number (`None` when not a known constant).
fn number(v: &Value) -> Option<f64> {
    v.float_val().or_else(|| v.int_val().map(|i| i as f64))
}

// ── Constructions ───────────────────────────────────────────────────────

pub(super) fn constructions() -> Vec<&'static dyn Construction> {
    vec![&SearchByCount, &SearchByPosition]
}

/// `u` as a sum of comparisons: `n - #{t < Y_k}` (right) or `#{Y_k < t}`.
///
/// * Preconditions: the operator's, `NonDecreasing(Y)`.
/// * Postconditions: every lane.
/// * Cost per lane: `n` comparisons and their sum (per entry halo2-base 32
///   cells, Noir 7 gates; measured on `np.searchsorted` of 4 queries, both
///   sides, in sorted tables of 8 to 128 entries,
///   `benchmarking/fact_constructions`, rule `sorted_searchsorted`).
/// * Soundness: no hint; the comparisons are exact on the window, so the
///   count is NumPy's for a sorted table.
pub struct SearchByCount;

/// `u` hinted and pinned by the entries on either side of it.
///
/// * Preconditions: `NonDecreasing(Y)`; for a constant table its codes
///   inside the window with one to spare (for `Y + 1`, `Y - 1`; checked
///   here). A runtime table is read at the hinted position; its codes stay
///   in the window as values of the program do, and `Y + 1`, `Y - 1` are
///   compared as field elements no wider than `2^64 + 1`, inside every
///   comparison's domain on both backends (`emit_order`).
/// * Postconditions: every lane.
/// * Emission per lane (right; left in brackets): `u` = the prover's count
///   (`HintKind::CountAtMost`), `u <= n` asserted; `lo = t` when `u == 0`,
///   else `Y_{u-1}` [`Y_{u-1} + 1`]; `hi = t` when `u == n`, else
///   `Y_u - 1` [`Y_u`]; `lo <= t <= hi` asserted. The entries at the runtime
///   `u` are `floor((A x + B) / C)` for an affine table (one constant
///   division each), else reads of the table, extended by one entry at each
///   end so that both reads stay in bounds for every `u` in `[0, n]`.
/// * Cost per lane: halo2-base about 180 cells in closed form, two reads of
///   `n + 2` slots plus 120 otherwise; Noir about 40 gates, or two ROM reads
///   plus 30 (the tables built once).
/// * Soundness: let `u* = #{k : Y_k <= t}` (right). The table is
///   non-decreasing, so `Y_k <= t` exactly for `k < u*`. For `u = u*` both
///   checks hold (`Y_{u*-1} <= t` as `u* - 1 < u*`; `t < Y_{u*}`, i.e.
///   `t <= Y_{u*} - 1`, as `u*` is not below `u*`; the ends compare `t`
///   with itself). For `u < u*`: `u < n` and `Y_u <= t`, so `t <= Y_u - 1`
///   fails. For `u > u*`: `u >= 1` and `Y_{u-1} > t`, so `lo <= t` fails.
///   So `u*` is the only admissible hint, and `u <= n` keeps it a small
///   integer. Left: the same with `< t` (`Y_{u-1} + 1 <= t`, `t <= Y_u`).
///   Every value compared is a code in the window (the closed form is
///   checked against every entry and bounded at `x = -1` and `x = n`).
pub struct SearchByPosition;

impl Construction for SearchByCount {
    fn name(&self) -> &'static str {
        "searchsorted: count"
    }

    fn candidates(&self, group: &Group, cx: &Context) -> Vec<Candidate> {
        let OpKind::SearchSorted { right } = group.kind else { return Vec::new() };
        let n = group.shared.len() as u64;
        let float = group.shared.iter().any(|&s| matches!(cx.stmt(s).ir, IR::ReadFloat { .. } | IR::ConstantFloat { .. }))
            || group.lanes.iter().any(|l| l.own.iter().any(|&s| is_float_stmt(cx, s)));
        let per_entry = count_cost_per_entry(cx.target, float);
        let emitter = CountEmitter { right, table: group.shared.clone() };
        vec![sorted_table(group, cx, every_lane(group, n * per_entry * group.lanes.len() as u64, Box::new(emitter)))]
    }
}

/// The cost of `lanes` hinted positions in a runtime table of `n` entries
/// (`position_in_sorted`): two reads per lane of the extended table (`n + 2`
/// slots, priced as the sort family prices reads at runtime positions,
/// `cost::UnitCosts::reads`: a selection per read, or table reads on
/// halo2-base for many reads of a long table) and per lane the hint, its
/// range check, two comparisons and the selects at the ends (halo2-base
/// 150 cells, Noir 30 gates). Fitted on `np.searchsorted` of 8 queries in
/// tables of 8 to 128 entries (`benchmarking/fact_constructions`, rule
/// `sorted_searchsorted`): within 15% from 16 entries on.
pub(crate) fn runtime_position_cost(n: usize, lanes: u64, target: Target) -> u64 {
    let fixed = match target {
        Target::Halo2Base => 150,
        Target::Noir => 30,
    };
    super::cost::unit_costs(target).reads(n + 2, 2 * lanes as usize) + fixed * lanes
}

/// A count's cost per entry (`SearchByCount`, `counts::count_query`): a
/// comparison and an addition. Measured on `np.searchsorted` (integers,
/// 32 cells, 7 gates) and on `(np.sort(a) > t).argmax()` (floats, 46 cells,
/// 6 gates), tables and sorts of 8 to 128 entries
/// (`benchmarking/fact_constructions`, rules `sorted_searchsorted`,
/// `sort_counts`).
pub(crate) fn count_cost_per_entry(target: Target, float: bool) -> u64 {
    match (target, float) {
        (Target::Halo2Base, false) => 32,
        (Target::Halo2Base, true) => 46,
        (Target::Noir, _) => 6,
    }
}

/// Whether a statement's value is a float (by the statement that defines
/// it: a float input, constant, arithmetic or cast).
pub(crate) fn is_float_stmt(cx: &Context, s: StmtId) -> bool {
    matches!(
        cx.stmt(s).ir,
        IR::ReadFloat { .. }
            | IR::ConstantFloat { .. }
            | IR::FloatCast
            | IR::AddF
            | IR::SubF
            | IR::MulF
            | IR::DivF
            | IR::SumF { .. }
            | IR::SelectF
            | IR::AbsF
    )
}

/// `candidate` with the operator's precondition, the table's order, unless
/// the table is a sorted constant one (`Table::of` checks it here).
fn sorted_table(group: &Group, cx: &Context, mut candidate: Candidate) -> Candidate {
    if Table::of(group, cx).is_none() {
        let list = crate::lower::ListId::of_stmts(&group.shared);
        let order = crate::lower::Fact::everywhere(crate::lower::Claim::NonDecreasing(list));
        candidate.pre.push(Requirement::Fact(order));
    }
    candidate
}

impl Construction for SearchByPosition {
    fn name(&self) -> &'static str {
        "searchsorted: hinted position"
    }

    fn candidates(&self, group: &Group, cx: &Context) -> Vec<Candidate> {
        let OpKind::SearchSorted { right } = group.kind else { return Vec::new() };
        if group.shared.is_empty() {
            return Vec::new();
        }
        let entries = group.shared.len() as u64 + 2;
        let lanes = group.lanes.len() as u64;
        let Some(table) = Table::of(group, cx) else {
            // A runtime table, read at the hinted position.
            let cost = runtime_position_cost(group.shared.len(), lanes, cx.target);
            let emitter = RuntimePositionEmitter { right, table: group.shared.clone(), extended: None };
            return vec![sorted_table(group, cx, every_lane(group, cost, Box::new(emitter)))];
        };
        let evaluator = match AffineForm::find(&table.codes) {
            Some(form) => Evaluator::Affine(form),
            None => Evaluator::Read { segments: Vec::new() },
        };
        let lanes = group.lanes.len() as u64;
        let cost = match (&evaluator, cx.target) {
            (Evaluator::Affine(_), Target::Halo2Base) => 180 * lanes,
            (Evaluator::Affine(_), Target::Noir) => 40 * lanes,
            (Evaluator::Read { .. }, Target::Halo2Base) => (2 * (10 + 17 * entries) + 120) * lanes,
            (Evaluator::Read { .. }, Target::Noir) => 2 * 3 * entries + (2 * 7 + 30) * lanes,
        };
        let emitter = PositionEmitter { right, table_stmts: group.shared.clone(), table, evaluator };
        vec![sorted_table(group, cx, every_lane(group, cost, Box::new(emitter)))]
    }
}

/// A candidate producing every lane of `group`.
fn every_lane(group: &Group, cost: u64, emitter: Box<dyn Emitter>) -> Candidate {
    Candidate {
        pre: Vec::new(),
        post: group.lanes.iter().map(|lane| Requirement::Lane(lane.stmt)).collect(),
        cost,
        emitter,
    }
}

// ── The table ───────────────────────────────────────────────────────────

/// A constant non-decreasing table: its codes and whether it holds floats.
struct Table {
    codes: Vec<i64>,
    float: bool,
}

impl Table {
    /// The group's table, when every entry is a constant, the codes are
    /// non-decreasing and strictly inside the window (one to spare at each
    /// end for the `± 1` of the pins).
    fn of(group: &Group, cx: &Context) -> Option<Table> {
        let precision = crate::prove::types::ProvingParams::default().precision_bits;
        let mut float = false;
        let codes: Vec<i64> = group
            .shared
            .iter()
            .map(|&s| match cx.stmt(s).ir {
                IR::ConstantInt { value } => Some(value),
                IR::ConstantFloat { value } => {
                    float = true;
                    i64::try_from(crate::prove::field::quantize(value, precision)).ok()
                }
                _ => None,
            })
            .collect::<Option<_>>()?;
        let sorted = codes.windows(2).all(|w| w[0] <= w[1]);
        let inside = codes.iter().all(|&c| c > i64::MIN && c < i64::MAX);
        (sorted && inside && !codes.is_empty()).then_some(Table { codes, float })
    }
}

/// Codes `floor((a k + b) / c)` for `k` in `[0, n)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct AffineForm {
    pub a: i64,
    pub b: i64,
    pub c: i64,
}

impl AffineForm {
    /// Largest denominator tried.
    const MAX_C: i64 = 1 << 12;

    /// A form that reproduces every code, with `a x + b` (adjusted by
    /// `± c`) inside the window for `x` in `[-1, n]`, or `None`. For each
    /// denominator `c` the slope `a` is tried near `c` times the average
    /// slope, and `b` is the smallest offset that keeps every
    /// `c Y_k <= a k + b`; the form holds when also `a k + b < c (Y_k + 1)`.
    pub(super) fn find(codes: &[i64]) -> Option<AffineForm> {
        let n = codes.len();
        let (first, last) = (codes[0] as i128, codes[n - 1] as i128);
        let span = (n.max(2) - 1) as i128;
        for c in 1..=Self::MAX_C as i128 {
            // `a / c` lies within `1 / (n - 1)` of the average slope when a
            // form exists, so `a` within `c / (n - 1) + 1` of `c * slope`.
            let centre = (last - first) * c / span;
            let reach = c / span + 1;
            for a in (centre - reach).max(0)..=centre + reach {
                if let Some(form) = Self::with(codes, a, c) {
                    return Some(form);
                }
            }
        }
        None
    }

    fn with(codes: &[i64], a: i128, c: i128) -> Option<AffineForm> {
        let offsets = codes.iter().enumerate().map(|(k, &y)| c * y as i128 - a * k as i128);
        let b = offsets.clone().max()?;
        let fits = offsets.into_iter().all(|o| b < o + c);
        let n = codes.len() as i128;
        let in_window = |v: i128| v.abs() < 1i128 << 62;
        let bounded = [-1, n].iter().all(|&x| in_window(a * x + b - c) && in_window(a * x + b + c));
        (fits && bounded).then(|| AffineForm { a: a as i64, b: b as i64, c: c as i64 })
    }

    /// `floor((a x + b) / c) + adjust` at the runtime integer `x` (one
    /// multiplication and one constant division; the caller keeps `x` where
    /// the numerator stays in the window, see `with`).
    pub(super) fn emit_at(&self, b: &mut IRBuilder, x: &Value, adjust: i64) -> Value {
        let scale = b.ir_constant_int(self.a);
        let offset = b.ir_constant_int(self.b + adjust * self.c);
        let scaled = b.ir_mul_i(x, &scale);
        let numerator = b.ir_add_i(&scaled, &offset);
        if self.c == 1 {
            return numerator;
        }
        let c = b.ir_constant_int(self.c);
        b.ir_floor_div_i(&numerator, &c)
    }

    /// The code at integer `k` (compile time).
    #[cfg(test)]
    pub(super) fn at(&self, k: i64) -> i64 {
        (self.a as i128 * k as i128 + self.b as i128).div_euclid(self.c as i128) as i64
    }
}

// ── Emission ────────────────────────────────────────────────────────────

struct CountEmitter {
    right: bool,
    table: Vec<StmtId>,
}

impl Emitter for CountEmitter {
    fn lane(&mut self, b: &mut IRBuilder, cx: &mut EmitCx, lane: &Lane) -> Value {
        let t = cx.value(lane.own[0]);
        let entries = cx.values(&self.table);
        let bits: Vec<Value> = entries
            .iter()
            .map(|y| if self.right { less(b, &t, y) } else { less(b, y, &t) })
            .collect();
        let count = sum(b, &bits);
        if self.right {
            let n = b.ir_constant_int(entries.len() as i64);
            b.ir_sub_i(&n, &count)
        } else {
            count
        }
    }
}

/// How the pins evaluate the table at a runtime position.
enum Evaluator {
    Affine(AffineForm),
    /// Reads of the extended table, one segment per adjustment (`-1`, `0`,
    /// `+1`), built on first use.
    Read { segments: Vec<(i64, u32)> },
}

struct PositionEmitter {
    right: bool,
    table_stmts: Vec<StmtId>,
    table: Table,
    evaluator: Evaluator,
}

impl Emitter for PositionEmitter {
    fn lane(&mut self, b: &mut IRBuilder, cx: &mut EmitCx, lane: &Lane) -> Value {
        let n = self.table.codes.len() as i64;
        let t = cx.value(lane.own[0]);
        let entries = cx.values(&self.table_stmts);
        // The hint counts the entries `<= t` (right) or `<= t - 1` (left).
        let query = if self.right {
            code_of(&t)
        } else {
            let one = b.ir_constant_int(1);
            b.ir_sub_i(&code_of(&t), &one)
        };
        let entry_codes: Vec<Value> = entries.iter().map(code_of).collect();
        let u = b.ir_hint(HintKind::CountAtMost, &entry_codes, &query);
        let bound = b.ir_constant_int(n + 1);
        b.ir_assert_u128_lt(&u, &bound);
        let zero = b.ir_constant_int(0);
        let len = b.ir_constant_int(n);
        let first = b.ir_equal_i(&u, &zero);
        let last = b.ir_equal_i(&u, &len);
        let (below_adjust, above_adjust) = if self.right { (0, -1) } else { (1, 0) };
        let below = self.entry(b, &u, -1, below_adjust);
        let above = self.entry(b, &u, 0, above_adjust);
        let lo = select(b, &first, &t, &below);
        let hi = select(b, &last, &t, &above);
        let lo_holds = less_or_equal(b, &lo, &t);
        b.ir_assert(&lo_holds);
        let hi_holds = less_or_equal(b, &t, &hi);
        b.ir_assert(&hi_holds);
        u
    }
}

impl PositionEmitter {
    /// `Y_{u + offset} + adjust` (offset `-1` or `0`), typed like the
    /// table; for `u` in `[0, n]` an entry or, at an end, a value the
    /// caller masks.
    fn entry(&mut self, b: &mut IRBuilder, u: &Value, offset: i64, adjust: i64) -> Value {
        let code = match &mut self.evaluator {
            Evaluator::Affine(form) => {
                // Y_{u + offset} + adjust, with `u` in [0, n]: the form
                // shifted by `offset` entries, at `u`.
                let shifted = AffineForm { b: form.b + form.a * offset, ..*form };
                shifted.emit_at(b, u, adjust)
            }
            Evaluator::Read { segments } => {
                let segment = match segments.iter().find(|(a, _)| *a == adjust) {
                    Some(&(_, segment)) => segment,
                    None => {
                        let segment = extended_table(b, &self.table.codes, adjust);
                        segments.push((adjust, segment));
                        segment
                    }
                };
                // Extended entry `j` is `Y_{j-1}`: read at `u + offset + 1`.
                let address = if offset == -1 {
                    u.clone()
                } else {
                    let one = b.ir_constant_int(1);
                    b.ir_add_i(u, &one)
                };
                crate::helpers::segment::read_typed(b, segment, &address, NumberType::Integer)
            }
        };
        if self.table.float {
            as_float(&code)
        } else {
            code
        }
    }
}

/// `np.searchsorted` of a runtime non-decreasing table by a hinted position
/// (`position_in_sorted`), the extended table built once per group.
struct RuntimePositionEmitter {
    right: bool,
    table: Vec<StmtId>,
    extended: Option<u32>,
}

impl Emitter for RuntimePositionEmitter {
    fn lane(&mut self, b: &mut IRBuilder, cx: &mut EmitCx, lane: &Lane) -> Value {
        let entries = cx.values(&self.table);
        let t = cx.value(lane.own[0]);
        let extended = *self.extended.get_or_insert_with(|| extended_runtime_table(b, &entries));
        position_in_sorted(b, &entries, extended, &t, self.right)
    }
}

/// A segment holding the codes `[Y_0, Y_0, Y_1, ..., Y_{n-1}, Y_{n-1}]` of a
/// runtime table: entry `j` is `Y_{j-1}` clamped to the table, so the reads
/// at `u` and `u + 1` stay in bounds for every `u` in `[0, n]`.
pub(crate) fn extended_runtime_table(b: &mut IRBuilder, entries: &[Value]) -> u32 {
    let n = entries.len();
    let codes: Vec<Value> = (0..n + 2).map(|j| code_of(&entries[j.saturating_sub(1).min(n - 1)])).collect();
    crate::helpers::segment::alloc_and_write_values(b, &codes, NumberType::Integer)
}

/// The number `u` of entries `<= t` (`right`) or `< t` of the non-decreasing
/// `entries` (a precondition the caller establishes), hinted and pinned as
/// [`SearchByPosition`] says: `u` is the prover's count, `u <= n` asserted,
/// `lo = t` at `u == 0` else `Y_{u-1}` (left: `Y_{u-1} + 1`), `hi = t` at
/// `u == n` else `Y_u - 1` (left: `Y_u`), `lo <= t <= hi` asserted, the two
/// entries read from `extended` (`extended_runtime_table`). The comparisons
/// are of codes. For the honest `u` the compared values lie in the window
/// (`Y_{u-1} + 1 <= t` and `t <= Y_u - 1` bound them by `t`); for any other
/// `u` a check fails, or a value one past the window fails the
/// comparison's own window (halo2-base) or is compared exactly (Noir's
/// 64-bit gap holds a difference of `2^64`).
pub(crate) fn position_in_sorted(b: &mut IRBuilder, entries: &[Value], extended: u32, t: &Value, right: bool) -> Value {
    let n = entries.len() as i64;
    let query = if right {
        code_of(t)
    } else {
        let one = b.ir_constant_int(1);
        b.ir_sub_i(&code_of(t), &one)
    };
    let codes: Vec<Value> = entries.iter().map(code_of).collect();
    let u = b.ir_hint(HintKind::CountAtMost, &codes, &query);
    let bound = b.ir_constant_int(n + 1);
    b.ir_assert_u128_lt(&u, &bound);
    let zero = b.ir_constant_int(0);
    let len = b.ir_constant_int(n);
    let first = b.ir_equal_i(&u, &zero);
    let last = b.ir_equal_i(&u, &len);
    let (below_adjust, above_adjust) = if right { (0, -1) } else { (1, 0) };
    let one = b.ir_constant_int(1);
    let next = b.ir_add_i(&u, &one);
    let below = crate::helpers::segment::read_typed(b, extended, &u, NumberType::Integer);
    let below = adjusted(b, below, below_adjust);
    let above = crate::helpers::segment::read_typed(b, extended, &next, NumberType::Integer);
    let above = adjusted(b, above, above_adjust);
    let code = code_of(t);
    let lo = b.ir_select_i(&first, &code, &below);
    let hi = b.ir_select_i(&last, &code, &above);
    let lo_holds = b.ir_less_than_or_equal_i(&lo, &code);
    b.ir_assert(&lo_holds);
    let hi_holds = b.ir_less_than_or_equal_i(&code, &hi);
    b.ir_assert(&hi_holds);
    u
}

/// `code + adjust` (the code itself for 0).
fn adjusted(b: &mut IRBuilder, code: Value, adjust: i64) -> Value {
    if adjust == 0 {
        return code;
    }
    let c = b.ir_constant_int(adjust);
    b.ir_add_i(&code, &c)
}

/// A segment holding `adjust + [Y_0, Y_0, ..., Y_{n-1}, Y_{n-1}]` as codes.
fn extended_table(b: &mut IRBuilder, codes: &[i64], adjust: i64) -> u32 {
    let n = codes.len();
    let values: Vec<Value> = (0..n + 2)
        .map(|j| b.ir_constant_int(codes[j.saturating_sub(1).min(n - 1)] + adjust))
        .collect();
    crate::helpers::segment::alloc_and_write_values(b, &values, NumberType::Integer)
}

// ── Scalar helpers ──────────────────────────────────────────────────────

fn is_float(v: &Value) -> bool {
    matches!(v, Value::Float(_))
}

/// The integer view of a value's code (the same statement).
fn code_of(v: &Value) -> Value {
    match v {
        Value::Float(s) => Value::Integer(ScalarValue::new(None, s.stmt_id)),
        other => other.clone(),
    }
}

/// The float view of a code (the same statement).
fn as_float(code: &Value) -> Value {
    Value::Float(ScalarValue::new(None, code.stmt_id()))
}

fn less(b: &mut IRBuilder, x: &Value, y: &Value) -> Value {
    if is_float(x) || is_float(y) {
        b.ir_less_than_f(x, y)
    } else {
        b.ir_less_than_i(x, y)
    }
}

fn less_or_equal(b: &mut IRBuilder, x: &Value, y: &Value) -> Value {
    if is_float(x) || is_float(y) {
        b.ir_less_than_or_equal_f(x, y)
    } else {
        b.ir_less_than_or_equal_i(x, y)
    }
}

fn select(b: &mut IRBuilder, cond: &Value, t: &Value, f: &Value) -> Value {
    if is_float(t) || is_float(f) {
        b.ir_select_f(cond, t, f)
    } else {
        b.ir_select_i(cond, t, f)
    }
}

fn sum(b: &mut IRBuilder, bits: &[Value]) -> Value {
    let mut total = b.ir_constant_int(0);
    for bit in bits {
        let as_int = b.ir_int_cast(bit);
        total = b.ir_add_i(&total, &as_int);
    }
    total
}

// ── Test fixtures and tests ─────────────────────────────────────────────

/// A search lowered by one construction, its inputs (`t`, the query's
/// code) and NumPy's count.
#[cfg(test)]
pub(crate) struct SearchGraph {
    pub graph: crate::ir::IRGraph,
    pub inputs: Vec<(String, i64)>,
    pub expected: i64,
}

/// The table (floats when `float`, else integers), queried at the code
/// `t` (an input, read as a float's code when `float`), searched on `side`
/// and lowered by `construction` alone; the count exposed.
#[cfg(test)]
pub(crate) fn search_graph(table: &[f64], float: bool, right: bool, t: i64, by_position: bool) -> SearchGraph {
    use crate::circuit_input::InputPath;
    use std::sync::Arc;
    let precision = crate::prove::types::ProvingParams::default().precision_bits;
    let mut b = IRBuilder::new();
    let entries: Arc<[Value]> = table
        .iter()
        .map(|&v| if float { b.ir_constant_float(v) } else { b.ir_constant_int(v as i64) })
        .collect();
    let query = b.ir_read_integer(InputPath::new("t", vec![]), false);
    let query = if float { as_float(&query) } else { query };
    let op = ArrayOp { kind: OpKind::SearchSorted { right }, shared: table.len() as u32, output: 0, position: None };
    let lane = b.ir_array_op(op, &entries, &[query]);
    b.ir_expose_public_i(&lane);
    let construction: &dyn Construction = if by_position { &SearchByPosition } else { &SearchByCount };
    // The search's table reads are lowered in a later round.
    let graph = crate::lower::emit::lower_rounds(b.export_ir_graph(), Target::Halo2Base, &[construction]);
    let code = |v: f64| if float { crate::prove::field::quantize(v, precision) as i64 } else { v as i64 };
    let expected = table.iter().filter(|&&v| if right { code(v) <= t } else { code(v) < t }).count() as i64;
    SearchGraph { graph, inputs: vec![("t".to_string(), t)], expected }
}

/// `graph` with its one `IR::Hint` replaced by the private input `hint`.
#[cfg(test)]
pub(crate) fn hint_as_input(graph: &crate::ir::IRGraph) -> crate::ir::IRGraph {
    use crate::circuit_input::InputPath;
    let mut graph = graph.clone();
    let mut hints = 0;
    for stmt in graph.stmts.iter_mut().filter(|s| matches!(s.ir, IR::Hint { .. })) {
        stmt.ir = IR::ReadInteger { path: InputPath::new("hint", vec![]), is_public: false };
        stmt.arguments.clear();
        stmt.arg_values.clear();
        hints += 1;
    }
    assert_eq!(hints, 1, "one hinted position");
    graph
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lower::sort::test_support::mock_run;

    /// `np.arange(1, n + 1) / n` and its Q32 codes.
    fn ecdf_table(n: usize) -> Vec<f64> {
        (1..=n).map(|k| k as f64 / n as f64).collect()
    }

    fn codes(table: &[f64]) -> Vec<i64> {
        table.iter().map(|&v| crate::prove::field::quantize(v, 32) as i64).collect()
    }

    #[test]
    fn affine_forms_reproduce_quantized_progressions() {
        for n in [1usize, 2, 3, 7, 50, 64, 100] {
            let codes = codes(&ecdf_table(n));
            let form = AffineForm::find(&codes).expect("an ecdf table is affine");
            assert!((0..n).all(|k| form.at(k as i64) == codes[k]), "n = {n}: {form:?}");
        }
        let ints: Vec<i64> = (0..20).map(|k| 3 * k - 7).collect();
        assert_eq!(AffineForm::find(&ints), Some(AffineForm { a: 3, b: -7, c: 1 }));
        // Squares are not an affine progression; neither is a jump.
        let squares: Vec<i64> = (0..10).map(|k| k * k).collect();
        assert_eq!(AffineForm::find(&squares), None);
        assert_eq!(AffineForm::find(&[0, 1, 2, 3, 100]), None);
    }

    /// Queries at, between, below and above the entries, and the window
    /// extremes.
    fn queries(codes: &[i64]) -> Vec<i64> {
        let mut qs: Vec<i64> = codes.iter().flat_map(|&c| [c - 1, c, c + 1]).collect();
        qs.extend([i64::MIN, i64::MAX, 0, -1]);
        qs
    }

    fn assert_counts(table: &[f64], float: bool) {
        let codes: Vec<i64> = if float { codes(table) } else { table.iter().map(|&v| v as i64).collect() };
        for right in [true, false] {
            for by_position in [false, true] {
                for t in queries(&codes) {
                    let search = search_graph(table, float, right, t, by_position);
                    let (ok, got) = mock_run(&search.graph, &search.inputs);
                    assert!(ok, "table {table:?} t {t} right {right} position {by_position}: rejected");
                    assert_eq!(got, vec![search.expected], "table {table:?} t {t} right {right} position {by_position}");
                }
            }
        }
    }

    #[test]
    fn both_constructions_are_numpys_searchsorted() {
        assert_counts(&ecdf_table(50), true); // affine, closed form
        assert_counts(&ecdf_table(1), true);
        assert_counts(&[0.1, 0.25, 0.25, 0.7, 2.0], true); // ties, not affine: reads
        assert_counts(&[-3.0, -3.0, 0.0, 5.0, 9.0, 9.0], false); // integers, reads
        assert_counts(&[2.0, 4.0, 6.0, 8.0], false); // integers, closed form
    }

    /// Every hinted position other than NumPy's count is rejected: every
    /// other position in `[0, n]`, one past each end, far out of range.
    #[test]
    fn the_hinted_position_is_pinned() {
        for (table, float) in [(ecdf_table(20), true), (vec![0.1, 0.25, 0.25, 0.7, 2.0], true)] {
            let codes = codes(&table);
            let n = table.len() as i64;
            for right in [true, false] {
                for t in queries(&codes) {
                    let search = search_graph(&table, float, right, t, true);
                    let graph = hint_as_input(&search.graph);
                    for u in (-1..=n + 1).chain([1 << 40, -(1 << 40)]) {
                        let mut inputs = search.inputs.clone();
                        inputs.push(("hint".to_string(), u));
                        let (ok, _) = mock_run(&graph, &inputs);
                        assert_eq!(ok, u == search.expected, "t {t} right {right}: position {u}");
                    }
                }
            }
        }
    }
}
