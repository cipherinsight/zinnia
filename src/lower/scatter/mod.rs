//! The scatter family: operators that move cells to positions that depend
//! on runtime values (insertion, deletion, scatters, compaction).
//!
//! * `OpKind::Insert` (`np.insert` at a runtime position): [`insert`], two
//!   forms of one expression per output row over the pinned position
//!   one-hot of [`position`]; the selection takes the cheaper per target.
//!
//! Each construction documents its preconditions, postconditions, cost per
//! backend and its soundness argument next to the code.

pub mod arith;
pub mod insert;
pub mod position;
#[cfg(test)]
pub(crate) mod test_support;
#[cfg(test)]
mod tests;

use std::ops::Range;

use crate::builder::IRBuilder;
use crate::provenance::Interval;
use crate::types::{NumberType, StmtId, Value};

use self::insert::{Form, Insertion};
use super::op::{ArrayOp, Constant, OperatorSemantics};
use super::{Bound, Candidate, Claim, Construction, Context, EmitCx, Emitter, Fact, Group, Lane, OpKind, Requirement, Scope};

/// The scatter family's constructions, in the order the selection breaks
/// ties.
pub fn constructions() -> Vec<&'static dyn Construction> {
    vec![&InsertBySharedProducts, &InsertByComplement]
}

// ── np.insert ───────────────────────────────────────────────────────────

/// The shape of an `OpKind::Insert` group: `n` source rows of `lanes` cells.
fn insert_shape(kind: OpKind) -> Option<(usize, usize)> {
    match kind {
        OpKind::Insert { rows, lanes } => Some((rows as usize, lanes as usize)),
        _ => None,
    }
}

/// What passes may know about an insertion's output cell before it is
/// lowered: it is one of the source cells or one of the inserted cells.
pub struct InsertSemantics;

impl OperatorSemantics for InsertSemantics {
    fn fold(&self, op: &ArrayOp, args: &[Value]) -> Option<Constant> {
        let (n, lanes) = insert_shape(op.kind)?;
        let position = op.position? as usize;
        let (k, j) = (position / lanes, position % lanes);
        let pos = args.get(n * lanes + lanes)?.int_val()?;
        let point = if pos < 0 { pos + n as i64 } else { pos };
        if !(0..=n as i64).contains(&point) {
            return None;
        }
        let point = point as usize;
        let cell = match k.cmp(&point) {
            std::cmp::Ordering::Less => &args[k * lanes + j],
            std::cmp::Ordering::Equal => &args[n * lanes + j],
            std::cmp::Ordering::Greater => &args[(k - 1) * lanes + j],
        };
        match cell {
            Value::Float(_) => cell.float_val().map(Constant::Float),
            _ => cell.int_val().map(Constant::Int),
        }
    }

    fn number_type(&self, op: &ArrayOp, args: &[Value]) -> NumberType {
        let cells = &args[self.picks_from(op).unwrap_or(0..0)];
        if cells.iter().any(|v| matches!(v, Value::Float(_))) {
            NumberType::Float
        } else {
            NumberType::Integer
        }
    }

    fn bounds(&self, op: &ArrayOp, operands: &[Option<Interval>]) -> Option<Interval> {
        operands[self.picks_from(op)?].iter().copied().reduce(|x, y| Some(Interval::union(x?, y?))).flatten()
    }

    fn picks_from(&self, op: &ArrayOp) -> Option<Range<usize>> {
        let (n, lanes) = insert_shape(op.kind)?;
        Some(0..n * lanes + lanes)
    }
}

/// `np.insert` with the products `g[k+1] a[k]` shared by adjacent rows
/// (`insert::Form::SharedProduct`).
pub struct InsertBySharedProducts;

/// `np.insert` with a second running sum `[P >= k]`
/// (`insert::Form::Complement`).
pub struct InsertByComplement;

impl Construction for InsertBySharedProducts {
    fn name(&self) -> &'static str {
        "insert: shared products"
    }

    fn candidates(&self, group: &Group, cx: &Context) -> Vec<Candidate> {
        insert_candidate(group, cx, Form::SharedProduct).into_iter().collect()
    }
}

impl Construction for InsertByComplement {
    fn name(&self) -> &'static str {
        "insert: complement"
    }

    fn candidates(&self, group: &Group, cx: &Context) -> Vec<Candidate> {
        insert_candidate(group, cx, Form::Complement).into_iter().collect()
    }
}

/// Every live lane of an insertion by `form`; it also proves that the
/// position lies in `[-n, n]` (`position::position_one_hot`).
fn insert_candidate(group: &Group, cx: &Context, form: Form) -> Option<Candidate> {
    let (n, lanes) = insert_shape(group.kind)?;
    let pos = *group.shared.last().expect("an insertion has a position operand");
    let in_range = Claim::InRange { x: pos, lo: -(n as i64), hi: Bound::Const(n as i64 + 1) };
    let mut post: Vec<Requirement> = group.lanes.iter().map(|lane| Requirement::Lane(lane.stmt)).collect();
    post.push(Requirement::Fact(Fact { claim: in_range, scope: Scope::Everywhere }));
    Some(Candidate {
        pre: Vec::new(),
        post,
        cost: insert::position_cost(cx.target, n) + form.cost(cx.target, n, lanes),
        emitter: Box::new(InsertEmitter { shared: group.shared.clone(), n, lanes, form, insertion: None }),
    })
}

struct InsertEmitter {
    shared: Vec<StmtId>,
    n: usize,
    lanes: usize,
    form: Form,
    insertion: Option<Insertion>,
}

impl Emitter for InsertEmitter {
    fn begin(&mut self, b: &mut IRBuilder, cx: &mut EmitCx) {
        let values = cx.values(&self.shared);
        let (n, lanes) = (self.n, self.lanes);
        let rows: Vec<Vec<Value>> = values[..n * lanes].chunks(lanes.max(1)).map(<[Value]>::to_vec).collect();
        let inserted = values[n * lanes..n * lanes + lanes].to_vec();
        let pos = &values[n * lanes + lanes];
        let dtype = if values[..n * lanes + lanes].iter().any(|v| matches!(v, Value::Float(_))) {
            NumberType::Float
        } else {
            NumberType::Integer
        };
        self.insertion = Some(Insertion::new(b, rows, inserted, pos, dtype, self.form));
    }

    fn lane(&mut self, b: &mut IRBuilder, _cx: &mut EmitCx, lane: &Lane) -> Value {
        let position = lane.position.expect("insertion lanes have constant positions") as usize;
        let insertion = self.insertion.as_mut().expect("started");
        insertion.cell(b, position / self.lanes, position % self.lanes)
    }
}
