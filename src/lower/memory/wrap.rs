//! NumPy's position of a runtime index, as an operator the layer lowers per
//! target (`OpKind::Wrap`).
//!
//! NumPy reads an index `i` on an axis of length `n` from the end when it is
//! negative: the position is `i` for `0 <= i < n` and `i + n` for
//! `-n <= i < 0`; any other `i` raises `IndexError`. The frontend
//! (`helpers/static_array/base.rs::normalize_dynamic_index`) emits, for an
//! index whose sign the facts do not decide and that runs on every path, a
//! `Wrap { len: n }` group over the shared operand `i`:
//!
//! * output 0, position 0: the position `w`;
//! * output 1, position `k` (`k < n`): the bit `[w == k]` (no producer
//!   since the frontend's overlays moved into the layer, which reads the
//!   bits from the wrap's `ENCODING`; removed with the old memory code).
//!
//! and the program's check `0 <= w < n` (`AssertU128Lt(w, n)`). Every
//! construction produces NumPy's position for `i` in `[-n, n)`; for any other
//! `i` the circuit has no witness (by the program's check, or by the
//! construction's own constraints where it replaces the check).
//!
//! Constructions:
//!
//! * [`WrapByPinnedSign`]: the sign `s = [i < 0]` a hint asserted to be a
//!   bit, `w = select(s, i + n, i)`; the program's check pins `s`; the bits
//!   `[w == k]` are equality tests (also offered as the one-hot `ENCODING` of
//!   `w`, for element reads by `gather::ElementByOneHot`);
//! * [`WrapByPositionHits`]: the scatter family's pinned position one-hot
//!   (`scatter::position::position_one_hot`): hinted bits `h_p`,
//!   `sum h_p = 1`, `(i - P)(i - P + n) = 0` for `P = sum p h_p`; `w = P`,
//!   `[w == k] = h_k`; it proves `0 <= w < n` itself (the program's check is
//!   dropped) and offers `h` as `ENCODING`.

use std::any::Any;
use std::ops::Range;
use std::rc::Rc;
use std::sync::Arc;

use crate::builder::IRBuilder;
use crate::lower::gather::constructions::{Encodings, ENCODING};
use crate::lower::gather::encoding::{Bit, IndexEncoding};
use crate::lower::op::{ArrayOp, Constant, OperatorSemantics};
use crate::lower::{Bound, Candidate, Claim, Construction, Context, EmitCx, Emitter, Fact, Group, Lane, OpKind, Requirement, Scope};
use crate::provenance::Interval;
use crate::types::{NumberType, StmtId, Value};

/// Output 0: the position.
pub const POSITION: u8 = 0;
/// Output 1: the bit `[w == k]` at position `k`.
pub const HIT: u8 = 1;

/// The position `w` of the runtime index `idx` on an axis of `n` (a
/// placeholder; output 0).
pub fn wrap_lane(b: &mut IRBuilder, idx: &Value, n: usize) -> Value {
    let shared: Arc<[Value]> = Arc::from(vec![b.ensure_ptr(idx)]);
    let op = ArrayOp { kind: OpKind::Wrap { len: n as u32 }, shared: 1, output: POSITION, position: Some(0) };
    b.ir_array_op(op, &shared, &[])
}

/// What passes may know about a wrap lane before it is lowered.
pub struct WrapSemantics;

impl OperatorSemantics for WrapSemantics {
    fn fold(&self, op: &ArrayOp, args: &[Value]) -> Option<Constant> {
        let OpKind::Wrap { len } = op.kind else { return None };
        let (i, n) = (args.first()?.int_val()?, i64::from(len));
        let w = if (-n..0).contains(&i) { i + n } else { i };
        if !(0..n).contains(&w) {
            return None;
        }
        Some(match op.output {
            POSITION => Constant::Int(w),
            _ => Constant::Bool(w == i64::from(op.position?)),
        })
    }

    fn number_type(&self, _op: &ArrayOp, _args: &[Value]) -> NumberType {
        NumberType::Integer
    }

    /// A bit is 0 or 1. The position has no bound before lowering: for an
    /// index outside `[-n, n)` only the program's check (or the chosen
    /// construction's) rejects the witness, and a bound here would let the
    /// range engine drop that check.
    fn bounds(&self, op: &ArrayOp, _operands: &[Option<Interval>]) -> Option<Interval> {
        (op.output == HIT).then_some(Interval { lo: 0, hi: 1 })
    }

    fn picks_from(&self, _op: &ArrayOp) -> Option<Range<usize>> {
        None
    }
}

fn axis_len(group: &Group) -> Option<usize> {
    match group.kind {
        OpKind::Wrap { len } => Some(len as usize),
        _ => None,
    }
}

fn lanes(group: &Group) -> impl Iterator<Item = Requirement> + '_ {
    group.lanes.iter().map(|l| Requirement::Lane(l.stmt))
}

// ── Pinned sign ───────────────────────────────────────────────────────

/// `w = select(s, i + n, i)` with the sign `s` a hint asserted to be a bit;
/// `[w == k]` an equality test each.
///
/// **Preconditions.** None (the program's check `0 <= w < n` stays: it is
/// what pins `s`).
///
/// **Postconditions.** Every lane; with `encoding`, also the one-hot
/// `ENCODING` of `w` (the `n` equality tests).
///
/// **Soundness and completeness.** As `normalize_dynamic_index` documents:
/// for `i` in `[0, n)` only `s = 0` passes the check (`i + n >= n`), for `i`
/// in `[-n, 0)` only `s = 1` (`w = i` would be a field element above
/// `p - n`), for any other `i` neither: `s` is pinned and `w` is NumPy's
/// position. The equality tests are exact gadgets.
///
/// **Cost** (`cost::wrap_by_pinned_sign`): the hint, its bit check and the
/// select; an equality test per bit produced.
pub struct WrapByPinnedSign;

impl Construction for WrapByPinnedSign {
    fn name(&self) -> &'static str {
        "memory: wrap by pinned sign"
    }

    fn candidates(&self, group: &Group, cx: &Context) -> Vec<Candidate> {
        let Some(n) = axis_len(group) else { return Vec::new() };
        let hits = group.lanes.iter().filter(|l| l.output == HIT).count() as u64;
        let plain = Candidate {
            pre: Vec::new(),
            post: lanes(group).collect(),
            cost: super::cost::wrap_by_pinned_sign(hits, cx.target),
            emitter: Box::new(SignEmitter { index: group.shared[0], n, position: None, hits: Vec::new() }),
        };
        let encoded = Candidate {
            pre: Vec::new(),
            post: lanes(group).chain([Requirement::Shared { group: group.id, name: ENCODING }]).collect(),
            cost: super::cost::wrap_by_pinned_sign(n as u64, cx.target),
            emitter: Box::new(SignEmitter { index: group.shared[0], n, position: None, hits: Vec::new() }),
        };
        vec![plain, encoded]
    }
}

/// NumPy's position of an index known to be non-negative: the index
/// itself (`w = i`), `[w == k]` an equality test each.
///
/// **Preconditions.** `NonNegative(i)`, established without a new check
/// (`Context::established`: a merged range check of `i`, a `@requires`
/// bound, a permutation's values): an assumption about the input, not an
/// obligation (NumPy accepts a negative index).
///
/// **Postconditions.** Every lane; with `encoding`, the one-hot `ENCODING`.
///
/// **Soundness and completeness.** For `i >= 0`, NumPy's position is `i`
/// when `i < n`, and NumPy raises otherwise, where the program's check
/// `0 <= w < n` (kept) has no witness. No hint.
///
/// **Cost**: an equality test per bit produced.
pub struct WrapOfNonNegative;

impl Construction for WrapOfNonNegative {
    fn name(&self) -> &'static str {
        "memory: wrap of a non-negative index"
    }

    fn candidates(&self, group: &Group, cx: &Context) -> Vec<Candidate> {
        let Some(n) = axis_len(group) else { return Vec::new() };
        let index = group.shared[0];
        let non_negative = Fact::everywhere(Claim::NonNegative(index));
        if !cx.established(&non_negative) {
            return Vec::new();
        }
        let eq = crate::lower::gather::cost::element_units(cx.target).eq_const;
        let hits = group.lanes.iter().filter(|l| l.output == HIT).count() as u64;
        let emitter = || Box::new(IdentityEmitter { index, n, hits: Vec::new() });
        let plain = Candidate {
            pre: vec![Requirement::Fact(non_negative)],
            post: lanes(group).collect(),
            cost: hits * eq,
            emitter: emitter(),
        };
        let encoded = Candidate {
            pre: vec![Requirement::Fact(non_negative)],
            post: lanes(group).chain([Requirement::Shared { group: group.id, name: ENCODING }]).collect(),
            cost: n as u64 * eq,
            emitter: emitter(),
        };
        vec![plain, encoded]
    }
}

struct IdentityEmitter {
    index: StmtId,
    n: usize,
    /// `[i == k]` once emitted.
    hits: Vec<Option<Value>>,
}

impl IdentityEmitter {
    fn hit(&mut self, b: &mut IRBuilder, cx: &EmitCx, k: usize) -> Value {
        if self.hits.is_empty() {
            self.hits = vec![None; self.n];
        }
        if let Some(hit) = &self.hits[k] {
            return hit.clone();
        }
        let constant = b.ir_constant_int(k as i64);
        let hit = b.ir_equal_i(&cx.value(self.index), &constant);
        self.hits[k] = Some(hit.clone());
        hit
    }
}

impl Emitter for IdentityEmitter {
    fn lane(&mut self, b: &mut IRBuilder, cx: &mut EmitCx, lane: &Lane) -> Value {
        match lane.output {
            POSITION => cx.value(self.index),
            _ => self.hit(b, cx, lane.position.expect("a hit lane has a position") as usize),
        }
    }

    fn shared(&mut self, b: &mut IRBuilder, cx: &mut EmitCx, _group: crate::lower::GroupId, _name: &'static str) -> Rc<dyn Any> {
        let onehot: Vec<Value> = (0..self.n).map(|k| self.hit(b, cx, k)).collect();
        Rc::new(vec![IndexEncoding { lo: 0, onehot, below: None }] as Encodings)
    }
}

struct SignEmitter {
    /// The index operand.
    index: StmtId,
    n: usize,
    position: Option<Value>,
    /// `[w == k]` once emitted.
    hits: Vec<Option<Value>>,
}

impl SignEmitter {
    fn hit(&mut self, b: &mut IRBuilder, k: usize) -> Value {
        if self.hits.is_empty() {
            self.hits = vec![None; self.n];
        }
        if let Some(hit) = &self.hits[k] {
            return hit.clone();
        }
        let constant = b.ir_constant_int(k as i64);
        let hit = b.ir_equal_i(self.position.as_ref().expect("begin ran"), &constant);
        self.hits[k] = Some(hit.clone());
        hit
    }
}

impl Emitter for SignEmitter {
    fn begin(&mut self, b: &mut IRBuilder, cx: &mut EmitCx) {
        let index = cx.value(self.index);
        let zero = b.ir_constant_int(0);
        let negative = b.ir_hint(crate::ir_defs::HintKind::Negative, std::slice::from_ref(&index), &zero);
        b.ir_assert_bit(&negative);
        let len = b.ir_constant_int(self.n as i64);
        let shifted = b.ir_add_i(&index, &len);
        self.position = Some(b.ir_select_i(&negative, &shifted, &index));
    }

    fn lane(&mut self, b: &mut IRBuilder, _cx: &mut EmitCx, lane: &Lane) -> Value {
        match lane.output {
            POSITION => self.position.clone().expect("begin ran"),
            _ => self.hit(b, lane.position.expect("a hit lane has a position") as usize),
        }
    }

    fn shared(&mut self, b: &mut IRBuilder, _cx: &mut EmitCx, _group: crate::lower::GroupId, _name: &'static str) -> Rc<dyn Any> {
        let onehot: Vec<Value> = (0..self.n).map(|k| self.hit(b, k)).collect();
        Rc::new(vec![IndexEncoding { lo: 0, onehot, below: None }] as Encodings)
    }
}

// ── Position one-hot ──────────────────────────────────────────────────

/// The pinned position one-hot of `i` (`scatter::position::position_one_hot`,
/// element positions): `w = sum_p p h_p`, `[w == k] = h_k`.
///
/// **Preconditions.** None.
///
/// **Postconditions.** Every lane; `0 <= w < n`; `-n <= i < n`; the one-hot
/// `ENCODING` of `w` (with its thermometer).
///
/// **Soundness and completeness.** `position_one_hot`'s lemma: the hinted
/// bits are pinned to NumPy's position of `i`, and there is no witness for
/// `i` outside `[-n, n)`. So `w` is the position, in `[0, n)`, and `h_k =
/// [w == k]`. Its checks are unconditional: the frontend emits the
/// placeholder only for code that runs on every path.
///
/// **Cost** (`cost::wrap_by_position_hits`): a hinted bit per position and
/// the relation once; the bits are free for every consumer.
pub struct WrapByPositionHits;

impl Construction for WrapByPositionHits {
    fn name(&self) -> &'static str {
        "memory: wrap by position one-hot"
    }

    fn candidates(&self, group: &Group, cx: &Context) -> Vec<Candidate> {
        let Some(n) = axis_len(group) else { return Vec::new() };
        let bound = group.lanes.iter().filter(|l| l.output == POSITION).map(|l| {
            Requirement::Fact(Fact { claim: Claim::InRange { x: l.stmt, lo: 0, hi: Bound::Const(n as i64) }, scope: Scope::Everywhere })
        });
        // The one-hot also pins the index itself: no witness for `i`
        // outside `[-n, n)` (NumPy's window), as the scatter family's
        // writes through it state.
        let window = Requirement::Fact(Fact {
            claim: Claim::InRange { x: group.shared[0], lo: -(n as i64), hi: Bound::Const(n as i64) },
            scope: Scope::Everywhere,
        });
        let post = lanes(group).chain(bound).chain([window, Requirement::Shared { group: group.id, name: ENCODING }]);
        vec![Candidate {
            pre: Vec::new(),
            post: post.collect(),
            cost: super::cost::wrap_by_position_hits(n as u64, cx.target),
            emitter: Box::new(HitsEmitter { index: group.shared[0], n, hot: Vec::new(), below: Vec::new(), position: None }),
        }]
    }
}

struct HitsEmitter {
    /// The index operand.
    index: StmtId,
    n: usize,
    hot: Vec<Value>,
    below: Vec<Value>,
    position: Option<Value>,
}

impl Emitter for HitsEmitter {
    fn begin(&mut self, b: &mut IRBuilder, cx: &mut EmitCx) {
        let index = cx.value(self.index);
        let one_hot = crate::lower::scatter::position::position_one_hot(
            b,
            &index,
            self.n,
            crate::lower::scatter::position::Positions::Element,
        );
        let mut position = b.ir_constant_int(0);
        for (p, bit) in one_hot.hot.iter().enumerate().skip(1) {
            let weight = b.ir_constant_int(p as i64);
            let term = b.ir_mul_i(bit, &weight);
            position = b.ir_add_i(&position, &term);
        }
        self.position = Some(position);
        self.hot = one_hot.hot;
        self.below = one_hot.below;
    }

    fn lane(&mut self, _b: &mut IRBuilder, _cx: &mut EmitCx, lane: &Lane) -> Value {
        match lane.output {
            POSITION => self.position.clone().expect("begin ran"),
            _ => self.hot[lane.position.expect("a hit lane has a position") as usize].clone(),
        }
    }

    fn shared(&mut self, _b: &mut IRBuilder, _cx: &mut EmitCx, _group: crate::lower::GroupId, _name: &'static str) -> Rc<dyn Any> {
        // `below[k] = [w < k]` for `k` in `1..n` (index 0 unused).
        let below: Vec<Bit> = self.below[..self.n].iter().map(|v| Bit::plain(v.clone())).collect();
        Rc::new(vec![IndexEncoding { lo: 0, onehot: self.hot.clone(), below: Some(below) }] as Encodings)
    }
}
