//! The scatter family: operators that move cells to positions that depend
//! on runtime values (insertion, deletion, scatters, compaction).
//!
//! * `OpKind::Insert` (`np.insert` at a runtime position): [`insert`], two
//!   forms of one expression per output row over the pinned position
//!   one-hot of [`position`]; the selection takes the cheaper per target.
//! * `OpKind::IndexWrite` (`a[i, ...] = v` at a runtime index on one axis):
//!   [`write`], a select per cell on the pinned one-hot of the index.
//! * `OpKind::MaskWrite` (`a[mask] = v`): [`mask_write`], a select per cell,
//!   or chains of masked writes as sums over the masks' atoms.
//!
//! Each construction documents its preconditions, postconditions, cost per
//! backend and its soundness argument next to the code.

pub mod arith;
pub mod insert;
pub mod mask_write;
pub mod position;
pub mod write;
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
    vec![&InsertBySharedProducts, &InsertByComplement, &OneHotWrite, &ZeroWriteChain, &MaskWriteBySelects, &MaskWriteByDecisionLists]
}

/// The fact a pinned position one-hot proves about its index: `lo <= x <
/// hi` on every path (the index operand is guarded by the execution
/// predicate).
fn index_window(x: StmtId, lo: i64, hi: i64) -> Requirement {
    Requirement::Fact(Fact { claim: Claim::InRange { x, lo, hi: Bound::Const(hi) }, scope: Scope::Everywhere })
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
    let mut post: Vec<Requirement> = group.lanes.iter().map(|lane| Requirement::Lane(lane.stmt)).collect();
    post.push(index_window(pos, -(n as i64), n as i64 + 1));
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

// ── Writes at a runtime index ───────────────────────────────────────────

/// What passes may know about a cell after a write at a runtime index: it
/// is its old value or the written one.
pub struct IndexWriteSemantics;

impl OperatorSemantics for IndexWriteSemantics {
    fn fold(&self, op: &ArrayOp, args: &[Value]) -> Option<Constant> {
        let OpKind::IndexWrite { axis_len } = op.kind else { return None };
        let own = &args[op.shared as usize..];
        let as_constant = |v: &Value| match v {
            Value::Float(_) => v.float_val().map(Constant::Float),
            _ => v.int_val().map(Constant::Int),
        };
        let hit = args[0].int_val().map(|i| {
            let n = axis_len as i64;
            let p = if i < 0 { i + n } else { i };
            Some(p) == op.position.map(i64::from)
        });
        let runs = own.get(write::PREDICATE).map(|p| p.int_val().or_else(|| p.bool_val().map(i64::from)));
        match (hit, runs) {
            (Some(false), _) | (_, Some(Some(0))) => as_constant(&own[write::OLD]),
            (Some(true), None | Some(Some(_))) => as_constant(&own[write::WRITTEN]),
            _ => {
                let (old, written) = (as_constant(&own[write::OLD])?, as_constant(&own[write::WRITTEN])?);
                (old == written).then_some(old)
            }
        }
    }

    fn number_type(&self, op: &ArrayOp, args: &[Value]) -> NumberType {
        if args[self.picks_from(op).unwrap_or(0..0)].iter().any(|v| matches!(v, Value::Float(_))) {
            NumberType::Float
        } else {
            NumberType::Integer
        }
    }

    fn bounds(&self, op: &ArrayOp, operands: &[Option<Interval>]) -> Option<Interval> {
        operands[self.picks_from(op)?].iter().copied().reduce(|x, y| Some(Interval::union(x?, y?))).flatten()
    }

    fn picks_from(&self, op: &ArrayOp) -> Option<Range<usize>> {
        let first = op.shared as usize;
        Some(first + write::OLD..first + write::WRITTEN + 1)
    }
}

/// Every lane of a group of writes through one runtime index by the pinned
/// one-hot of the index (`write`).
pub struct OneHotWrite;

impl Construction for OneHotWrite {
    fn name(&self) -> &'static str {
        "index write: one-hot"
    }

    fn candidates(&self, group: &Group, cx: &Context) -> Vec<Candidate> {
        let OpKind::IndexWrite { axis_len } = group.kind else { return Vec::new() };
        let n = axis_len as usize;
        let index = group.shared[0];
        let zeros = group
            .lanes
            .iter()
            .filter(|lane| matches!(cx.stmt(lane.own[write::WRITTEN]).ir, crate::ir_defs::IR::ConstantInt { value: 0 }))
            .count();
        let mut post: Vec<Requirement> = group.lanes.iter().map(|lane| Requirement::Lane(lane.stmt)).collect();
        post.push(index_window(index, -(n as i64), n as i64));
        vec![Candidate {
            pre: Vec::new(),
            post,
            cost: write::one_hot_cost(cx.target, n, group.lanes.len(), zeros),
            emitter: Box::new(WriteEmitter { index, n, one_hot: None }),
        }]
    }
}

struct WriteEmitter {
    index: StmtId,
    n: usize,
    one_hot: Option<write::WriteIndex>,
}

impl Emitter for WriteEmitter {
    fn begin(&mut self, b: &mut IRBuilder, cx: &mut EmitCx) {
        let index = cx.value(self.index);
        self.one_hot = Some(write::WriteIndex::new(b, &index, self.n));
    }

    fn lane(&mut self, b: &mut IRBuilder, cx: &mut EmitCx, lane: &Lane) -> Value {
        let coord = lane.position.expect("a written cell has a coordinate") as usize;
        let own = cx.values(&lane.own);
        self.one_hot.as_ref().expect("started").cell(b, coord, &own)
    }
}

/// A chain of zero writes on axes of one length ending in a group, by
/// shared keep products (`write::ZeroWrites`).
pub struct ZeroWriteChain;

impl Construction for ZeroWriteChain {
    fn name(&self) -> &'static str {
        "index write: zero-write chain"
    }

    fn candidates(&self, group: &Group, cx: &Context) -> Vec<Candidate> {
        let OpKind::IndexWrite { axis_len } = group.kind else { return Vec::new() };
        let chain = zero_write_chain(group, cx);
        if chain.len() < 2 {
            return Vec::new();
        }
        let n = axis_len as usize;
        let mut post = Vec::new();
        for g in &chain {
            post.extend(g.lanes.iter().map(|lane| Requirement::Lane(lane.stmt)));
            post.push(index_window(g.shared[0], -(n as i64), n as i64));
        }
        let base: std::collections::HashMap<StmtId, StmtId> = group
            .lanes
            .iter()
            .map(|lane| (lane.stmt, chain_base(lane.stmt, &chain, cx)))
            .collect();
        vec![Candidate {
            pre: Vec::new(),
            post,
            cost: write::zero_chain_cost(cx.target, n, chain.len(), group.lanes.len()),
            emitter: Box::new(ZeroChainEmitter {
                indices: chain.iter().map(|g| g.shared[0]).collect(),
                step_of_group: chain.iter().enumerate().map(|(t, g)| (g.id, t)).collect(),
                lane_group: chain.iter().flat_map(|g| g.lanes.iter().map(move |l| (l.stmt, g.id))).collect(),
                base,
                n,
                writes: None,
            }),
        }]
    }
}

/// Whether `lane` writes the constant 0 unconditionally.
fn writes_zero(lane: &Lane, cx: &Context) -> bool {
    use crate::ir_defs::IR;
    lane.own.len() == write::PREDICATE
        && match cx.stmt(lane.own[write::WRITTEN]).ir {
            IR::ConstantInt { value } => value == 0,
            IR::ConstantFloat { value } => value == 0.0,
            _ => false,
        }
}

/// The groups of zero writes that end in `group`, first to last: each
/// earlier group's lanes are the old cells of the next group's lanes, at
/// the same coordinates, and read by nothing else.
fn zero_write_chain<'a>(group: &'a Group, cx: &'a Context) -> Vec<&'a Group> {
    let mut chain = vec![group];
    loop {
        let last = chain[0];
        if !last.lanes.iter().all(|lane| writes_zero(lane, cx)) {
            return if chain.len() == 1 { Vec::new() } else { chain[1..].to_vec() };
        }
        let previous = |lane: &Lane| -> Option<&'a Group> {
            let old = lane.own[write::OLD];
            let g = cx.group_of_lane(old)?;
            let prior = g.lanes.iter().find(|l| l.stmt == old)?;
            (g.kind == last.kind && prior.position == lane.position && cx.users(old) == [lane.stmt]).then_some(g)
        };
        let Some(earlier) = last.lanes.first().and_then(previous) else { return chain };
        let covers = last.lanes.iter().all(|lane| previous(lane).is_some_and(|g| g.id == earlier.id))
            && earlier.lanes.len() == last.lanes.len();
        if !covers || chain.iter().any(|g| g.id == earlier.id) {
            return chain;
        }
        chain.insert(0, earlier);
    }
}

/// The cell before the first write of `chain` under the last group's lane
/// `lane`.
fn chain_base(lane: StmtId, chain: &[&Group], cx: &Context) -> StmtId {
    let mut cell = lane;
    for g in chain.iter().rev() {
        let l = g.lanes.iter().find(|l| l.stmt == cell).expect("a chain lane");
        cell = l.own[write::OLD];
    }
    let _ = cx;
    cell
}

struct ZeroChainEmitter {
    indices: Vec<StmtId>,
    step_of_group: std::collections::HashMap<super::GroupId, usize>,
    lane_group: std::collections::HashMap<StmtId, super::GroupId>,
    /// The last group's lanes to the cell before the chain.
    base: std::collections::HashMap<StmtId, StmtId>,
    n: usize,
    writes: Option<write::ZeroWrites>,
}

impl Emitter for ZeroChainEmitter {
    fn begin(&mut self, b: &mut IRBuilder, cx: &mut EmitCx) {
        let indices = cx.values(&self.indices);
        self.writes = Some(write::ZeroWrites::new(b, &indices, self.n));
    }

    fn lane(&mut self, b: &mut IRBuilder, cx: &mut EmitCx, lane: &Lane) -> Value {
        let coord = lane.position.expect("a written cell has a coordinate") as usize;
        let step = self.step_of_group[&self.lane_group[&lane.stmt]];
        let own = cx.values(&lane.own);
        let base = match self.base.get(&lane.stmt) {
            Some(&cell) => cx.value(cell),
            None => own[write::OLD].clone(),
        };
        self.writes.as_ref().expect("started").cell(b, step, coord, &own, &base)
    }
}

// ── Masked writes ───────────────────────────────────────────────────────

/// What passes may know about a cell after a masked write: it is its old
/// value or the written one.
pub struct MaskWriteSemantics;

impl OperatorSemantics for MaskWriteSemantics {
    fn fold(&self, _op: &ArrayOp, args: &[Value]) -> Option<Constant> {
        let as_constant = |v: &Value| match v {
            Value::Float(_) => v.float_val().map(Constant::Float),
            _ => v.int_val().map(Constant::Int),
        };
        let mask = &args[mask_write::MASK];
        match mask.bool_val().or_else(|| mask.int_val().map(|v| v != 0)) {
            Some(true) => as_constant(&args[mask_write::WRITTEN]),
            Some(false) => as_constant(&args[mask_write::OLD]),
            None => {
                let (old, written) = (as_constant(&args[mask_write::OLD])?, as_constant(&args[mask_write::WRITTEN])?);
                (old == written).then_some(old)
            }
        }
    }

    fn number_type(&self, op: &ArrayOp, args: &[Value]) -> NumberType {
        if args[self.picks_from(op).unwrap_or(0..0)].iter().any(|v| matches!(v, Value::Float(_))) {
            NumberType::Float
        } else {
            NumberType::Integer
        }
    }

    fn bounds(&self, op: &ArrayOp, operands: &[Option<Interval>]) -> Option<Interval> {
        operands[self.picks_from(op)?].iter().copied().reduce(|x, y| Some(Interval::union(x?, y?))).flatten()
    }

    fn picks_from(&self, _op: &ArrayOp) -> Option<Range<usize>> {
        Some(mask_write::OLD..mask_write::WRITTEN + 1)
    }
}

/// Every masked-write lane as a select (`mask_write::select_cell`).
pub struct MaskWriteBySelects;

/// The chains of masked writes that have a decision list as sums over
/// their atoms, the other lanes as selects (`mask_write::DecisionList`).
pub struct MaskWriteByDecisionLists;

fn writes_constant_zero(lane: &Lane, cx: &Context) -> bool {
    use crate::ir_defs::IR;
    matches!(cx.stmt(lane.own[mask_write::WRITTEN]).ir, IR::ConstantInt { value: 0 })
        || matches!(cx.stmt(lane.own[mask_write::WRITTEN]).ir, IR::ConstantFloat { value } if value == 0.0)
}

impl Construction for MaskWriteBySelects {
    fn name(&self) -> &'static str {
        "mask write: selects"
    }

    fn candidates(&self, group: &Group, cx: &Context) -> Vec<Candidate> {
        if group.kind != OpKind::MaskWrite {
            return Vec::new();
        }
        let cost = group.lanes.iter().map(|l| mask_write::select_cost(cx.target, writes_constant_zero(l, cx))).sum();
        vec![Candidate {
            pre: Vec::new(),
            post: group.lanes.iter().map(|lane| Requirement::Lane(lane.stmt)).collect(),
            cost,
            emitter: Box::new(MaskWriteEmitter { lists: std::collections::HashMap::new() }),
        }]
    }
}

impl Construction for MaskWriteByDecisionLists {
    fn name(&self) -> &'static str {
        "mask write: decision lists"
    }

    fn candidates(&self, group: &Group, cx: &Context) -> Vec<Candidate> {
        if group.kind != OpKind::MaskWrite {
            return Vec::new();
        }
        // Only the last lane of a chain gets a list (the others are dead
        // once it is a sum).
        let continued: std::collections::HashSet<StmtId> =
            group.lanes.iter().map(|lane| lane.own[mask_write::OLD]).collect();
        let lists: std::collections::HashMap<StmtId, mask_write::DecisionList> = group
            .lanes
            .iter()
            .filter(|lane| !continued.contains(&lane.stmt))
            .filter_map(|lane| Some((lane.stmt, mask_write::DecisionList::of(lane, cx)?)))
            .collect();
        if lists.is_empty() {
            return Vec::new();
        }
        // A lane a decision list covers costs nothing as a select (it is
        // dead once the chain's last lane is a sum).
        let covered: std::collections::HashSet<StmtId> = lists
            .keys()
            .flat_map(|&last| chain_lanes(last, cx))
            .collect();
        let cost = group
            .lanes
            .iter()
            .map(|lane| match lists.get(&lane.stmt) {
                Some(list) => list.cost(cx.target),
                None if covered.contains(&lane.stmt) => 0,
                None => mask_write::select_cost(cx.target, writes_constant_zero(lane, cx)),
            })
            .sum();
        vec![Candidate {
            pre: Vec::new(),
            post: group.lanes.iter().map(|lane| Requirement::Lane(lane.stmt)).collect(),
            cost,
            emitter: Box::new(MaskWriteEmitter { lists }),
        }]
    }
}

/// The earlier lanes of the chain that ends in lane `last`.
fn chain_lanes(last: StmtId, cx: &Context) -> Vec<StmtId> {
    let mut lanes = Vec::new();
    let mut cell = cx.stmt(last).arguments[mask_write::OLD];
    while cx.group_of_lane(cell).is_some_and(|g| g.kind == OpKind::MaskWrite) {
        lanes.push(cell);
        cell = cx.stmt(cell).arguments[mask_write::OLD];
    }
    lanes
}

struct MaskWriteEmitter {
    /// Decision lists by the lane they compute.
    lists: std::collections::HashMap<StmtId, mask_write::DecisionList>,
}

impl Emitter for MaskWriteEmitter {
    fn lane(&mut self, b: &mut IRBuilder, cx: &mut EmitCx, lane: &Lane) -> Value {
        if let Some(list) = self.lists.get(&lane.stmt) {
            return list.emit(b, &|stmt| cx.value(stmt));
        }
        let own = cx.values(&lane.own);
        mask_write::select_cell(b, &own[mask_write::OLD], &own[mask_write::WRITTEN], &own[mask_write::MASK])
    }
}
