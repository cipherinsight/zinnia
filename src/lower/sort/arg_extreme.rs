//! `np.argmax` and `np.argmin` of a numeric line (`OpKind::ArgExtreme`):
//! the position of the first largest (smallest) value, NumPy's rule under
//! ties.
//!
//! The frontend emits one placeholder per reduced line (the whole array, or
//! one fiber of an axis reduction; `lane`), over the line's cells. Two
//! constructions:
//!
//! * [`ArgExtremeBySelects`]: the running extreme, one comparison and two
//!   selects per element after the first (a strict comparison, so the
//!   first extreme wins a tie). Always applies; the lowering the frontend
//!   emitted before the placeholder existed.
//! * [`ArgExtremeOfSorted`]: for a line known to be non-decreasing (the
//!   array claim `NonDecreasing`, `lower::array_facts`), whose argmin is 0
//!   and whose argmax is the first position holding the last value: the
//!   constant `n - 1` when the line is strictly increasing; otherwise that
//!   position counted, or hinted and pinned by its neighbours (below).

use std::ops::Range;

use crate::builder::IRBuilder;
use crate::ir_defs::HintKind;
use crate::lower::op::{ArrayOp, Constant, OperatorSemantics};
use crate::lower::{Candidate, Claim, Construction, Context, EmitCx, Emitter, Fact, Group, Lane, ListId, OpKind, Requirement, Target};
use crate::provenance::Interval;
use crate::types::{NumberType, StmtId, Value};

use super::cost;
use super::emit::{count, Elements, OutputMemory};

/// The placeholder of `np.argmax` (`max`) or `np.argmin` of the cells of
/// one line (at least two), compared as floats when `float`: its lane.
pub fn lane(b: &mut IRBuilder, cells: &[Value], max: bool, float: bool) -> Value {
    let shared: std::sync::Arc<[Value]> = cells.iter().map(|c| b.ensure_ptr(c)).collect();
    let op = ArrayOp { kind: OpKind::ArgExtreme { max, float }, shared: cells.len() as u32, output: 0, position: Some(0) };
    b.ir_array_op(op, &shared, &[])
}

// ── Semantics ───────────────────────────────────────────────────────────

pub struct ArgExtremeSemantics;

impl OperatorSemantics for ArgExtremeSemantics {
    /// NumPy's position of the first extreme of constant cells.
    fn fold(&self, op: &ArrayOp, args: &[Value]) -> Option<Constant> {
        let OpKind::ArgExtreme { max, .. } = op.kind else { return None };
        let values: Vec<f64> = args[..op.shared as usize]
            .iter()
            .map(|v| v.float_val().or_else(|| v.int_val().map(|i| i as f64)).or_else(|| v.bool_val().map(f64::from)))
            .collect::<Option<_>>()?;
        if values.iter().any(|v| v.is_nan()) {
            return None;
        }
        let mut best = 0;
        for (j, v) in values.iter().enumerate().skip(1) {
            if (max && *v > values[best]) || (!max && *v < values[best]) {
                best = j;
            }
        }
        Some(Constant::Int(best as i64))
    }

    fn number_type(&self, _op: &ArrayOp, _args: &[Value]) -> NumberType {
        NumberType::Integer
    }

    fn bounds(&self, op: &ArrayOp, _operands: &[Option<Interval>]) -> Option<Interval> {
        Some(Interval { lo: 0, hi: i64::from(op.shared).saturating_sub(1) })
    }

    fn picks_from(&self, _op: &ArrayOp) -> Option<Range<usize>> {
        None
    }
}

// ── Constructions ───────────────────────────────────────────────────────

pub(super) fn constructions() -> Vec<&'static dyn Construction> {
    vec![&ArgExtremeBySelects, &ArgExtremeOfSorted, &ArgExtremeOfSortedByPosition]
}

/// [`ArgExtremeOfSorted`]'s hinted position, a construction of its own so
/// that measurements can force it (the count, the constants and the argmin
/// are [`ArgExtremeOfSorted`]'s).
pub struct ArgExtremeOfSortedByPosition;

impl Construction for ArgExtremeOfSortedByPosition {
    fn name(&self) -> &'static str {
        "argextreme: sorted line by a hinted position"
    }

    fn candidates(&self, group: &Group, cx: &Context) -> Vec<Candidate> {
        sorted_candidates(group, cx).into_iter().filter(|(how, _)| matches!(how, How::Hinted)).map(|(_, c)| c).collect()
    }
}

/// The running extreme: for `j = 1, ..., n - 1`, `c = a_j > best` (`<` for
/// the minimum), `index = select(c, j, index)`, `best = select(c, a_j,
/// best)`, from `index = 0`, `best = a_0`.
///
/// * Preconditions: none. Postconditions: the lane.
/// * Cost per element after the first: a comparison and two selects
///   (halo2-base 55 cells, Noir 10 gates at the full width; measured on
///   `np.argmax` of 8 and 16 floats).
/// * Soundness: no hint; every value is a function of the cells. A strict
///   comparison keeps the earlier position on a tie, so `index` is the
///   first extreme (NumPy's rule).
pub struct ArgExtremeBySelects;

/// The argmax (argmin) of a non-decreasing line `a_0 <= ... <= a_{n-1}`.
///
/// * Preconditions: `NonDecreasing(a)`, established without a new check
///   (`Context::established`; an assumption about the input, not an
///   obligation of NumPy's semantics).
/// * Postconditions: the lane.
/// * The argmin is 0 (the first value is a smallest). The argmax is `u*`,
///   the first position holding `a_{n-1}`: the values equal to the largest
///   form a suffix `[u*, n)`. With `StrictlyIncreasing(a)` established it is
///   the constant `n - 1`. Otherwise one of:
///   * **count**: `u* = #{j < n - 1 : a_j < a_{n-1}}`, the positions before
///     the suffix (`n - 1` comparisons);
///   * **hinted position**: the prover supplies `u`, and the circuit asserts
///     `0 <= u < n`, `a_u == a_{n-1}`, and `u == 0` or `a_{u-1} < a_{n-1}`
///     (two reads of the line at runtime positions).
/// * Soundness of the hint: `a_u == a_{n-1}` puts `u` in the suffix, so
///   `u >= u*`; for `u > u*`, `u - 1` is in the suffix too and `a_{u-1} ==
///   a_{n-1}` fails the last check. So `u = u*` is the one value the
///   constraints admit, and `u*` satisfies them (NumPy's answer).
/// * Cost: the count, `n - 1` comparisons; the hint, two reads of `n` slots
///   (`UnitCosts::reads`), an equality test, a comparison and a range check.
pub struct ArgExtremeOfSorted;

/// The lane's group as a line: its kind's parameters and the cells.
fn line(group: &Group) -> Option<(bool, bool)> {
    match group.kind {
        OpKind::ArgExtreme { max, float } if group.shared.len() >= 2 => Some((max, float)),
        _ => None,
    }
}

/// Per-element cost of a comparison at the full width, and of a select.
fn comparison_costs(target: Target) -> (u64, u64) {
    match target {
        Target::Halo2Base => (43, 6),
        Target::Noir => (7, 2),
    }
}

impl Construction for ArgExtremeBySelects {
    fn name(&self) -> &'static str {
        "argextreme: running selects"
    }

    fn candidates(&self, group: &Group, cx: &Context) -> Vec<Candidate> {
        let Some((max, float)) = line(group) else { return Vec::new() };
        let (compare, select) = comparison_costs(cx.target);
        let cost = (group.shared.len() as u64 - 1) * (compare + 2 * select);
        vec![every_lane(group, cost, Box::new(SelectsEmitter { cells: group.shared.clone(), max, float }))]
    }
}

impl Construction for ArgExtremeOfSorted {
    fn name(&self) -> &'static str {
        "argextreme: sorted line"
    }

    fn candidates(&self, group: &Group, cx: &Context) -> Vec<Candidate> {
        sorted_candidates(group, cx).into_iter().filter(|(how, _)| !matches!(how, How::Hinted)).map(|(_, c)| c).collect()
    }
}

/// [`ArgExtremeOfSorted`]'s candidates, each with how it finds the position.
fn sorted_candidates(group: &Group, cx: &Context) -> Vec<(How, Candidate)> {
    let Some((max, float)) = line(group) else { return Vec::new() };
    let list = ListId::of_stmts(&group.shared);
    let ordered = Fact::everywhere(Claim::NonDecreasing(list));
    if !cx.established(&ordered) {
        return Vec::new();
    }
    let n = group.shared.len() as u64;
    let candidate = |how: How, cost: u64, fact: Fact| {
        let emitter = Box::new(SortedEmitter { cells: group.shared.clone(), float, how });
        (how, with_pre(every_lane(group, cost, emitter), fact))
    };
    if !max {
        return vec![candidate(How::First, 0, ordered)];
    }
    let increasing = Fact::everywhere(Claim::StrictlyIncreasing(list));
    if cx.established(&increasing) {
        return vec![candidate(How::Last, 0, increasing)];
    }
    let (compare, _) = comparison_costs(cx.target);
    let costs = cost::unit_costs(cx.target);
    let hinted = costs.reads(n as usize, 2) + 2 * compare + crate::lower::checks::range_check_cost(n, false, cx.target);
    vec![candidate(How::Count, (n - 1) * compare, ordered), candidate(How::Hinted, hinted, ordered)]
}

/// [`ArgExtremeOfSorted`] restricted to finding the argmax by a count
/// (`hinted == false`) or by a hinted position (tests).
#[cfg(test)]
pub(crate) struct ForcedSortedArgmax {
    pub hinted: bool,
}

#[cfg(test)]
impl Construction for ForcedSortedArgmax {
    fn name(&self) -> &'static str {
        "argextreme: sorted line (forced)"
    }

    fn candidates(&self, group: &Group, cx: &Context) -> Vec<Candidate> {
        sorted_candidates(group, cx)
            .into_iter()
            .filter(|(how, _)| match how {
                How::Count => !self.hinted,
                How::Hinted => self.hinted,
                How::First | How::Last => true,
            })
            .map(|(_, candidate)| candidate)
            .collect()
    }
}

fn every_lane(group: &Group, cost: u64, emitter: Box<dyn Emitter>) -> Candidate {
    Candidate {
        pre: Vec::new(),
        post: group.lanes.iter().map(|lane| Requirement::Lane(lane.stmt)).collect(),
        cost,
        emitter,
    }
}

fn with_pre(mut candidate: Candidate, fact: Fact) -> Candidate {
    candidate.pre.push(Requirement::Fact(fact));
    candidate
}

// ── Emission ────────────────────────────────────────────────────────────

struct SelectsEmitter {
    cells: Vec<StmtId>,
    max: bool,
    float: bool,
}

impl Emitter for SelectsEmitter {
    fn lane(&mut self, b: &mut IRBuilder, cx: &mut EmitCx, _lane: &Lane) -> Value {
        let cells = cx.values(&self.cells);
        let mut index = b.ir_constant_int(0);
        let mut best = cells[0].clone();
        for (j, element) in cells.iter().enumerate().skip(1) {
            let better = match (self.float, self.max) {
                (true, true) => b.ir_greater_than_f(element, &best),
                (true, false) => b.ir_less_than_f(element, &best),
                (false, true) => b.ir_greater_than_i(element, &best),
                (false, false) => b.ir_less_than_i(element, &best),
            };
            let position = b.ir_constant_int(j as i64);
            index = b.ir_select_i(&better, &position, &index);
            best = if self.float { b.ir_select_f(&better, element, &best) } else { b.ir_select_i(&better, element, &best) };
        }
        index
    }
}

/// How [`ArgExtremeOfSorted`] finds the position.
#[derive(Clone, Copy)]
enum How {
    /// The argmin: 0.
    First,
    /// A strictly increasing line's argmax: `n - 1`.
    Last,
    Count,
    Hinted,
}

struct SortedEmitter {
    cells: Vec<StmtId>,
    float: bool,
    how: How,
}

impl Emitter for SortedEmitter {
    fn lane(&mut self, b: &mut IRBuilder, cx: &mut EmitCx, _lane: &Lane) -> Value {
        let cells = cx.values(&self.cells);
        let n = cells.len();
        let dtype = if self.float { NumberType::Float } else { NumberType::Integer };
        let elements = Elements { values: &cells, dtype, range: None };
        let last = &cells[n - 1];
        match self.how {
            How::First => b.ir_constant_int(0),
            How::Last => b.ir_constant_int(n as i64 - 1),
            How::Count => {
                let below: Vec<Value> = cells[..n - 1].iter().map(|a| elements.less(b, a, last)).collect();
                count(b, &below)
            }
            How::Hinted => {
                // u: the number of values below the last, `#{a_j <= code(a_{n-1}) - 1}`
                // over the codes (the honest value; any other is rejected).
                let codes: Vec<Value> = cells.iter().map(code_of).collect();
                let one = b.ir_constant_int(1);
                let query = b.ir_sub_i(&code_of(last), &one);
                let u = b.ir_hint(HintKind::CountAtMost, &codes, &query);
                let memory = OutputMemory::new(b, &cells, dtype);
                let at_u = memory.read(b, &u);
                let same = elements.equal(b, &at_u, last);
                b.ir_assert(&same);
                // `u - 1`, or 0 where `u` is 0 (a read in range either way;
                // the check below ignores it there).
                let zero = b.ir_constant_int(0);
                let first = b.ir_equal_i(&u, &zero);
                let before = b.ir_sub_i(&u, &one);
                let previous_position = b.ir_select_i(&first, &zero, &before);
                let previous = memory.read(b, &previous_position);
                let smaller = elements.less(b, &previous, last);
                let holds = b.ir_logical_or(&first, &smaller);
                b.ir_assert(&holds);
                u
            }
        }
    }
}

/// The integer view of a value's code (the same statement).
fn code_of(v: &Value) -> Value {
    match v {
        Value::Float(s) => Value::Integer(crate::types::ScalarValue::new(None, s.stmt_id)),
        other => other.clone(),
    }
}
