//! Masked writes `a[mask] = v` (and `a[mask] op= v`), `OpKind::MaskWrite`.
//!
//! Terms. Each cell a masked write may change is one *lane*: its old value,
//! the value written to it, and its mask bit (own operands `[old, written,
//! mask]`). A *chain* is a sequence of masked writes over one cell, each
//! lane's old value the previous lane (`a[m1] = 0; a[m2] = 30; a[m3] += 5`):
//! its *base* is the cell before the first write. The masks are Boolean
//! formulas (`not`, `and`, `or`) over a few *atoms* (comparison bits).
//!
//! # Construction: select per cell ([`select_cell`])
//!
//! `select(mask, written, old)` per lane. Always applies.
//! Cost: halo2-base 8 cells per lane (4 when writing the constant 0); Noir
//! about 2 gates.
//!
//! # Construction: decision list ([`DecisionList`])
//!
//! For a chain over at most three atoms, the cell's final value is a
//! function of the atoms' values: for each of the `2^k` assignments the
//! writes are replayed and the cell ends as one *leaf*: the base, the base
//! plus a constant (a write `old + c`), a constant, or a value independent
//! of the chain. Assignments with the same leaf form a *class*; the final
//! value is `sum over classes with a nonzero leaf: [class] * leaf`, where
//! the indicator `[class]` must be a literal or the conjunction of two
//! literals of the atoms (else the chain keeps the selects). Example
//! (0490): with `a = [x < t1]`, `b = [x < t2]`, `a[a] = 0; a[not b] = 30;
//! a[not a and b] += 5` ends as `30 (1 - b) + (not a and b) (x + 5)`.
//!
//! **Preconditions**: the masks are Boolean (0/1); the intermediate cells
//! of the chain are read only by the chain (its next lane, and its later
//! written values).
//! **Postconditions**: the chain's lanes (the intermediate ones are the
//! selects, then unused).
//! **Soundness**: no hint. The classes partition the assignments, the
//! indicators are 0/1 and exactly one is 1 for the atoms' values, so the
//! sum is the leaf of that assignment, which is the value the selects
//! compute (both evaluate the same writes in the same order).
//! **Cost**: per final cell, one gate per AND indicator, per base shift and
//! per nonzero term (halo2-base about 4 cells each; Noir 1 gate).

use crate::builder::IRBuilder;
use crate::ir::IRStatement;
use crate::ir_defs::IR;
use crate::lower::{Context, Lane, OpKind, Target};
use crate::types::{NumberType, StmtId, Value};

use super::arith::{bit_times, sum};

/// Own operands of a lane.
pub const OLD: usize = 0;
pub const WRITTEN: usize = 1;
pub const MASK: usize = 2;

/// The most atoms a decision list enumerates.
const MAX_ATOMS: usize = 3;

/// The value of one lane as a select.
pub fn select_cell(b: &mut IRBuilder, old: &Value, written: &Value, mask: &Value) -> Value {
    if matches!(old, Value::Float(_)) || matches!(written, Value::Float(_)) {
        b.ir_select_f(mask, written, old)
    } else {
        b.ir_select_i(mask, written, old)
    }
}

/// Cost of one lane as a select.
pub fn select_cost(target: Target, writes_zero: bool) -> u64 {
    match (target, writes_zero) {
        (Target::Halo2Base, true) => 4,
        (Target::Halo2Base, false) => 8,
        (Target::Noir, _) => 2,
    }
}

/// A Boolean formula over atoms (indices into `DecisionList::atoms`).
#[derive(Clone, Debug)]
enum Formula {
    Atom(usize),
    Not(Box<Formula>),
    And(Box<Formula>, Box<Formula>),
    Or(Box<Formula>, Box<Formula>),
}

impl Formula {
    fn eval(&self, row: usize) -> bool {
        match self {
            Formula::Atom(i) => row >> i & 1 == 1,
            Formula::Not(f) => !f.eval(row),
            Formula::And(f, g) => f.eval(row) && g.eval(row),
            Formula::Or(f, g) => f.eval(row) || g.eval(row),
        }
    }
}

/// What a cell holds under one assignment of the atoms.
#[derive(Clone, Debug, PartialEq)]
enum Leaf {
    /// The chain's base.
    Base,
    /// The base plus the constant statement.
    Shifted(StmtId),
    /// A constant statement (compared by value).
    Constant(Constant),
    /// A value the chain does not change.
    Other(StmtId),
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Constant {
    Int(i64, StmtId),
    Float(f64, StmtId),
}

impl Constant {
    fn of(stmt: &IRStatement) -> Option<Constant> {
        match stmt.ir {
            IR::ConstantInt { value } => Some(Constant::Int(value, stmt.stmt_id)),
            IR::ConstantFloat { value } => Some(Constant::Float(value, stmt.stmt_id)),
            _ => None,
        }
    }

    fn is_zero(self) -> bool {
        match self {
            Constant::Int(v, _) => v == 0,
            Constant::Float(v, _) => v == 0.0,
        }
    }

    fn stmt(self) -> StmtId {
        match self {
            Constant::Int(_, s) | Constant::Float(_, s) => s,
        }
    }

    fn same_value(self, other: Constant) -> bool {
        match (self, other) {
            (Constant::Int(a, _), Constant::Int(b, _)) => a == b,
            (Constant::Float(a, _), Constant::Float(b, _)) => a == b,
            _ => false,
        }
    }
}

fn same_leaf(x: &Leaf, y: &Leaf) -> bool {
    match (x, y) {
        (Leaf::Constant(a), Leaf::Constant(b)) => a.same_value(*b),
        _ => x == y,
    }
}

/// An indicator of a class: true everywhere, a literal, or the conjunction
/// of two literals (atom index, polarity).
#[derive(Clone, Copy, Debug)]
enum Indicator {
    Always,
    Literal(usize, bool),
    Both((usize, bool), (usize, bool)),
}

impl Indicator {
    /// The cheapest indicator whose true assignments are exactly `rows`
    /// (bit `r` set: assignment `r`), over `k` atoms.
    fn of(rows: u32, k: usize) -> Option<Indicator> {
        let all = (1u32 << (1 << k)) - 1;
        let literal = |i: usize, v: bool| -> u32 {
            (0..1usize << k).filter(|r| (r >> i & 1 == 1) == v).fold(0, |m, r| m | 1 << r)
        };
        if rows == all {
            return Some(Indicator::Always);
        }
        let literals: Vec<(usize, bool)> = (0..k).flat_map(|i| [(i, true), (i, false)]).collect();
        if let Some(&(i, v)) = literals.iter().find(|&&(i, v)| literal(i, v) == rows) {
            return Some(Indicator::Literal(i, v));
        }
        for (n, &x) in literals.iter().enumerate() {
            for &y in &literals[n + 1..] {
                if x.0 != y.0 && literal(x.0, x.1) & literal(y.0, y.1) == rows {
                    return Some(Indicator::Both(x, y));
                }
            }
        }
        None
    }

    fn gates(self) -> u64 {
        match self {
            Indicator::Always | Indicator::Literal(..) => 0,
            Indicator::Both(..) => 1,
        }
    }
}

/// A chain's final value as a sum over classes (see the module docs).
#[derive(Clone, Debug)]
pub struct DecisionList {
    base: StmtId,
    atoms: Vec<StmtId>,
    /// `(indicator, leaf)` of every class whose leaf is not the constant 0.
    terms: Vec<(Indicator, Leaf)>,
}

impl DecisionList {
    /// The decision list of the chain ending in `lane`, when it has at least
    /// two writes, at most three atoms, and indicators of the cheap forms.
    pub fn of(lane: &Lane, cx: &Context) -> Option<DecisionList> {
        let writes = chain(lane, cx)?;
        let base = cx.stmt(writes[0].stmt).arguments[OLD];
        let mut atoms: Vec<StmtId> = Vec::new();
        let masks: Vec<Formula> =
            writes.iter().map(|w| formula(w.own[MASK], cx, &mut atoms)).collect::<Option<_>>()?;
        if atoms.len() > MAX_ATOMS {
            return None;
        }
        let k = atoms.len();
        // Replay the writes under every assignment.
        let mut leaves: Vec<Leaf> = Vec::with_capacity(1 << k);
        for row in 0..1usize << k {
            let mut cell = Leaf::Base;
            for (w, mask) in writes.iter().zip(&masks) {
                if mask.eval(row) {
                    cell = written_leaf(w, &cell, cx)?;
                }
            }
            leaves.push(cell);
        }
        // Group the assignments by leaf; drop the constant-0 class.
        let mut classes: Vec<(u32, Leaf)> = Vec::new();
        for (row, leaf) in leaves.into_iter().enumerate() {
            match classes.iter_mut().find(|(_, l)| same_leaf(l, &leaf)) {
                Some((rows, _)) => *rows |= 1 << row,
                None => classes.push((1 << row, leaf)),
            }
        }
        let terms = classes
            .into_iter()
            .filter(|(_, leaf)| !matches!(leaf, Leaf::Constant(c) if c.is_zero()))
            .map(|(rows, leaf)| Some((Indicator::of(rows, k)?, leaf)))
            .collect::<Option<Vec<_>>>()?;
        Some(DecisionList { base, atoms, terms })
    }

    /// Gates of the sum (halo2-base about 4 cells each, Noir 1).
    pub fn cost(&self, target: Target) -> u64 {
        let gates: u64 = self
            .terms
            .iter()
            .map(|(ind, leaf)| ind.gates() + u64::from(matches!(leaf, Leaf::Shifted(_))) + 1)
            .sum();
        match target {
            Target::Halo2Base => 4 * gates,
            Target::Noir => gates,
        }
    }

    /// The final value, from the lowered values of the input statements.
    pub fn emit(&self, b: &mut IRBuilder, value: &dyn Fn(StmtId) -> Value) -> Value {
        let base = value(self.base);
        let dtype = if matches!(base, Value::Float(_)) { NumberType::Float } else { NumberType::Integer };
        let atoms: Vec<Value> = self.atoms.iter().map(|&a| value(a)).collect();
        let mut terms = Vec::with_capacity(self.terms.len());
        for (indicator, leaf) in &self.terms {
            let leaf_value = match leaf {
                Leaf::Base => base.clone(),
                Leaf::Shifted(c) => sum(b, &[base.clone(), value(*c)], dtype),
                Leaf::Constant(c) => value(c.stmt()),
                Leaf::Other(s) => value(*s),
            };
            terms.push(match *indicator {
                Indicator::Always => leaf_value,
                Indicator::Literal(i, true) => bit_times(b, &atoms[i], &leaf_value, dtype),
                Indicator::Literal(i, false) => {
                    let zero = super::arith::zero(b, dtype);
                    select(b, &atoms[i], &zero, &leaf_value, dtype)
                }
                Indicator::Both(x, y) => {
                    let both = conjunction(b, &atoms, x, y);
                    bit_times(b, &both, &leaf_value, dtype)
                }
            });
        }
        if terms.is_empty() {
            return super::arith::zero(b, dtype);
        }
        sum(b, &terms, dtype)
    }
}

/// `x and y` for two literals, one gate: `a and b` a product, `not a and
/// b` the select `a ? 0 : b`, `not a and not b` `a ? 0 : not b`.
fn conjunction(b: &mut IRBuilder, atoms: &[Value], x: (usize, bool), y: (usize, bool)) -> Value {
    let zero = b.ir_constant_int(0);
    let (pos, other) = match (x.1, y.1) {
        (true, true) => return b.ir_logical_and(&atoms[x.0], &atoms[y.0]),
        (false, true) => (x.0, y.0),
        (true, false) => (y.0, x.0),
        (false, false) => {
            let not_y = b.ir_logical_not(&atoms[y.0]);
            return b.ir_select_i(&atoms[x.0], &zero, &not_y);
        }
    };
    b.ir_select_i(&atoms[pos], &zero, &atoms[other])
}

fn select(b: &mut IRBuilder, cond: &Value, t: &Value, f: &Value, dtype: NumberType) -> Value {
    match dtype {
        NumberType::Float => b.ir_select_f(cond, t, f),
        _ => b.ir_select_i(cond, t, f),
    }
}

/// One write of a chain: its lane statement and own operands.
struct Write {
    stmt: StmtId,
    own: Vec<StmtId>,
}

/// The writes of the chain ending in `lane`, first to last: at least two,
/// each earlier lane read only by the chain.
fn chain(lane: &Lane, cx: &Context) -> Option<Vec<Write>> {
    let mut writes = vec![Write { stmt: lane.stmt, own: lane.own.clone() }];
    loop {
        let old = writes.last().expect("a write").own[OLD];
        let Some(group) = cx.group_of_lane(old) else { break };
        if group.kind != OpKind::MaskWrite {
            break;
        }
        let earlier = group.lanes.iter().find(|l| l.stmt == old).expect("a lane of its group");
        writes.push(Write { stmt: earlier.stmt, own: earlier.own.clone() });
    }
    writes.reverse();
    if writes.len() < 2 {
        return None;
    }
    // Every intermediate cell is read only by the next lane and by later
    // written values (so it is dead once the chain is a sum).
    let later_written: Vec<StmtId> = writes.iter().map(|w| w.own[WRITTEN]).collect();
    for pair in writes.windows(2) {
        let users = cx.users(pair[0].stmt);
        if !users.iter().all(|u| *u == pair[1].stmt || later_written.contains(u)) {
            return None;
        }
    }
    Some(writes)
}

/// The mask statement as a formula, registering its atoms.
fn formula(stmt: StmtId, cx: &Context, atoms: &mut Vec<StmtId>) -> Option<Formula> {
    let s = cx.stmt(stmt);
    let arg = |i: usize| s.arguments[i];
    Some(match (&s.ir, s.arguments.len()) {
        (IR::LogicalNot, 1) => Formula::Not(Box::new(formula(arg(0), cx, atoms)?)),
        (IR::LogicalAnd, 2) => Formula::And(Box::new(formula(arg(0), cx, atoms)?), Box::new(formula(arg(1), cx, atoms)?)),
        (IR::LogicalOr, 2) => Formula::Or(Box::new(formula(arg(0), cx, atoms)?), Box::new(formula(arg(1), cx, atoms)?)),
        (IR::LtI | IR::LtF | IR::LteI | IR::LteF | IR::GtI | IR::GtF | IR::GteI | IR::GteF | IR::EqI | IR::EqF
            | IR::NeI | IR::NeF, 2) => {
            let index = match atoms.iter().position(|&a| a == stmt) {
                Some(i) => i,
                None => {
                    atoms.push(stmt);
                    atoms.len() - 1
                }
            };
            Formula::Atom(index)
        }
        _ => return None,
    })
}

/// The leaf a write leaves in the cell under an assignment where its mask
/// holds and the cell held `prev`: a constant, a value independent of the
/// chain, or `old + c` of a base.
fn written_leaf(w: &Write, prev: &Leaf, cx: &Context) -> Option<Leaf> {
    let written = cx.stmt(w.own[WRITTEN]);
    if let Some(c) = Constant::of(written) {
        return Some(Leaf::Constant(c));
    }
    let old = w.own[OLD];
    if let (IR::AddF | IR::AddI, [x, y]) = (&written.ir, written.arguments.as_slice()) {
        let shift = match (*x == old, *y == old) {
            (true, false) => Some(*y),
            (false, true) => Some(*x),
            _ => None,
        };
        if let Some(c) = shift.filter(|&c| Constant::of(cx.stmt(c)).is_some()) {
            return match prev {
                Leaf::Base => Some(Leaf::Shifted(c)),
                _ => None,
            };
        }
    }
    // Independent of the chain: it does not read any lane's cell.
    (!written.arguments.contains(&old) && w.own[WRITTEN] != old).then_some(Leaf::Other(w.own[WRITTEN]))
}
