//! A sort whose input is already in order: [`SortedInput`].
//!
//! `np.sort`, `np.argsort`, `np.max`, `np.min`, `np.median` and
//! `np.percentile` of an array lower to lanes of one sort
//! (`OpKind::Sort`). When the elements `a_0, ..., a_{n-1}` are known to be
//! non-decreasing (the array claim `NonDecreasing(a)`, `lower::array_facts`:
//! from `@requires(is_sorted(a))`, from the outputs of an earlier sort, from
//! a sorted constant table), NumPy's stable sort of `a` is `a` itself and
//! its stable argsort is the identity: equal elements keep their order, and
//! no other element moves. Every lane is then a value of the program:
//!
//! | lane | value |
//! |---|---|
//! | sorted value of rank `r` (`np.max` is rank `n - 1`) | `a_r` |
//! | argsort of rank `r` | the constant `r` |
//! | sorted value at a runtime rank `k` | `a_k`, a read of `a` at `k` |
//! | argsort at a runtime rank `k` | `k` |
//! | payload `p[argsort[r]]` | `p_r` |
//! | count `#{j : a_j <= e}` (`< e`) | one comparison per element and a sum, as every plan |
//!
//! * **Preconditions**: `NonDecreasing(a)`, established without a new check
//!   (`Context::established`): the claim is an assumption about the input,
//!   not something NumPy's sort requires.
//! * **Postconditions**: every lane of the sort.
//! * **Cost**: nothing for lanes at constant ranks; a runtime rank asserts
//!   `0 <= k < n` (a rank outside has no proof, as NumPy raises) and a
//!   sorted value there is a read of `n` slots (`UnitCosts::reads`). Count
//!   lanes cost what they cost under every plan and are left out of every
//!   plan's price.
//! * **Soundness and completeness**: given the claim, the values above are
//!   NumPy's results (stable sort of a non-decreasing sequence), and no
//!   prover hint is involved; the runtime-rank read is the read of a checked
//!   index. The claim holds on every accepted witness by its justification
//!   (a definition, or the program's order checks, which the selection then
//!   keeps).

use crate::builder::IRBuilder;
use crate::lower::{
    Candidate, Claim, Construction, Context, EmitCx, Emitter, Fact, Group, Lane, ListId, OpKind, Requirement,
};
use crate::types::{NumberType, StmtId, Value};

use super::cost;
use super::emit::{Elements, OutputMemory};
use super::semantics::{COUNT_AT_MOST, COUNT_BELOW, INDICES, PAYLOAD, VALUES};

/// The sort of an input already in order (module doc).
pub struct SortedInput;

impl Construction for SortedInput {
    fn name(&self) -> &'static str {
        "sort: input already sorted"
    }

    fn candidates(&self, group: &Group, cx: &Context) -> Vec<Candidate> {
        if group.kind != OpKind::Sort || group.shared.is_empty() {
            return Vec::new();
        }
        let ordered = Fact::everywhere(Claim::NonDecreasing(ListId::of_stmts(&group.shared)));
        if !cx.established(&ordered) {
            return Vec::new();
        }
        let n = group.shared.len();
        let runtime = |output: u8| group.runtime_lanes(output).count();
        let costs = cost::unit_costs(cx.target);
        let rank_check = crate::lower::checks::cost(
            &Claim::InRange { x: 0, lo: 0, hi: crate::lower::Bound::Const(n as i64) },
            cx.target,
        );
        let cost = costs.reads(n, runtime(VALUES)) + rank_check * runtime(INDICES) as u64;
        vec![Candidate {
            pre: vec![Requirement::Fact(ordered)],
            post: group.lanes.iter().map(|lane| Requirement::Lane(lane.stmt)).collect(),
            cost,
            emitter: Box::new(SortedInputEmitter { elements: group.shared.clone(), values_at_runtime: None }),
        }]
    }
}

struct SortedInputEmitter {
    elements: Vec<StmtId>,
    /// The elements in a segment, for reads at runtime ranks (built on
    /// first use).
    values_at_runtime: Option<OutputMemory>,
}

impl Emitter for SortedInputEmitter {
    fn lane(&mut self, b: &mut IRBuilder, cx: &mut EmitCx, lane: &Lane) -> Value {
        let n = self.elements.len();
        match (lane.output, lane.position) {
            (VALUES, Some(r)) => cx.value(self.elements[r as usize]),
            (INDICES, Some(r)) => b.ir_constant_int(i64::from(r)),
            (VALUES, None) => {
                let rank = cx.value(lane.own[0]);
                if self.values_at_runtime.is_none() {
                    let values = cx.values(&self.elements);
                    let dtype = dtype_of(&values);
                    self.values_at_runtime = Some(OutputMemory::new(b, &values, dtype));
                }
                self.values_at_runtime.as_ref().expect("just built").read(b, &rank)
            }
            (INDICES, None) => {
                // The identity permutation: the rank itself, once checked
                // into `[0, n)` (no proof outside, as NumPy raises).
                let rank = cx.value(lane.own[0]);
                let bound = b.ir_constant_int(n as i64);
                b.ir_assert_u128_lt(&rank, &bound);
                rank
            }
            (PAYLOAD, Some(r)) => cx.value(lane.own[r as usize]),
            (COUNT_AT_MOST | COUNT_BELOW, _) => {
                let values = cx.values(&self.elements);
                let elements = Elements { values: &values, dtype: dtype_of(&values), range: None };
                let e = cx.value(lane.own[0]);
                super::counts::count_query(b, &elements, &e, lane.output == COUNT_AT_MOST)
            }
            (output, position) => unreachable!("a sort lane of output {output} at {position:?}"),
        }
    }
}

fn dtype_of(values: &[Value]) -> NumberType {
    if values.iter().any(|v| matches!(v, Value::Float(_))) {
        NumberType::Float
    } else {
        NumberType::Integer
    }
}
