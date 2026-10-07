//! The permutation check: `is_permutation(p)` (a `@requires` precondition),
//! "the `n` integers `p` are a permutation of `0, 1, ..., n - 1`"
//! (`OpKind::IsPermutation`).
//!
//! The frontend (`optim/predicates/witness.rs`) emits one placeholder lane
//! over `p` and asserts it. The lane's value is `1` wherever a proof exists;
//! there is none unless `p` is such a permutation. Each construction proves
//! the fact `Permutation(p, [0, ..., n - 1])` (`Claim::Permutation`), which
//! a permutation scatter `c[p] = v` relies on (`gather::ScatterByHits`):
//!
//! * [`PermutationByHits`]: for every `j`, the number of `k` with
//!   `p[k] == j` is asserted to be 1 (exact; the equality tests are shared
//!   with a scatter by hits over the same `p`, after deduplication);
//! * [`PermutationByGrandProduct`]: `{p_k}` and `{0, ..., n - 1}` are
//!   equal as multisets of field elements, by a grand product at a verifier
//!   challenge (`IR::IdentityCheck`; halo2-base only). Sound except with
//!   probability `n / |F|` (the product's degree); field elements equal to
//!   `0..n-1` are those integers, so `p` is exactly a permutation.

use std::sync::Arc;

use crate::builder::IRBuilder;
use crate::ir_defs::Identity;
use crate::types::{StmtId, Value};

use super::{Candidate, Claim, Construction, Context, EmitCx, Emitter, Fact, Group, Lane, ListId, OpKind, Requirement, Target};

/// The family's constructions, in the order the selection breaks ties.
pub fn constructions() -> Vec<&'static dyn Construction> {
    vec![&PermutationByHits, &PermutationByGrandProduct]
}

/// The frontend's placeholder: the lane of `is_permutation(p)`, which the
/// caller asserts.
pub fn check_lane(b: &mut IRBuilder, p: &[Value]) -> Value {
    let shared: Arc<[Value]> = p.iter().map(|v| b.ensure_ptr(v)).collect();
    let op = super::ArrayOp { kind: OpKind::IsPermutation, shared: shared.len() as u32, output: 0, position: Some(0) };
    b.ir_array_op(op, &shared, &[])
}

/// The fact `p` is a permutation of `0..n` (`n = p.len()`).
pub fn permutation_fact(p: &[StmtId]) -> Fact {
    Fact::everywhere(Claim::Permutation(ListId::of_stmts(p), ListId::range(p.len())))
}

/// Asserts that every value of `targets` (distinct integer constants) is
/// hit exactly once by `values` (as many): then `values` is a permutation
/// of `targets` (each value hits at most one distinct target, and the `n`
/// counts of one sum to `n`). For each target, `n` equality tests summed
/// and the sum asserted to be 1.
pub(crate) fn emit_hit_counts(b: &mut IRBuilder, values: &[Value], targets: &[Value]) {
    let constants: Option<Vec<i64>> = targets.iter().map(Value::int_val).collect();
    let distinct = constants.as_ref().is_some_and(|c| {
        let mut sorted = c.clone();
        sorted.sort_unstable();
        sorted.windows(2).all(|w| w[0] < w[1])
    });
    assert!(
        distinct && values.len() == targets.len() && !values.is_empty(),
        "a hit-count permutation check needs as many values as distinct constant targets"
    );
    let one = b.ir_constant_int(1);
    for target in targets {
        let mut hits = values.iter().map(|v| b.ir_equal_i(v, target)).collect::<Vec<_>>().into_iter();
        let first = hits.next().expect("a non-empty list");
        let count = hits.fold(first, |acc, h| b.ir_add_i(&acc, &h));
        let once = b.ir_equal_i(&count, &one);
        b.ir_assert(&once);
    }
}

/// The cost of the hit-count check of `n` values: `n^2` equality tests
/// against constants (unless `shared_tests`: a scatter by hits over the same
/// values builds them, and they are deduplicated), `n (n - 1)` additions
/// and `n` asserted equalities. halo2-base cells, Noir gates; calibrated on
/// the permutation scatters of 0318 / 0319 (unchanged cells and gates on
/// both backends, `eval-redo/layer/layer-challenge.md`).
pub(crate) fn hit_count_cost(n: u64, target: Target, shared_tests: bool) -> u64 {
    let (test, add, assert) = match target {
        Target::Halo2Base => (7, 1, 2),
        Target::Noir => (2, 0, 1),
    };
    let tests = if shared_tests { 0 } else { n * n * test };
    tests + n * n.saturating_sub(1) * add + n * assert
}

/// The grand product's cells: the constants `0..n` as cells, and two
/// factors of 6 second-phase cells per value.
fn grand_product_cost(n: u64) -> u64 {
    n + 12 * n + 4
}

fn lane_and_fact(group: &Group) -> Vec<Requirement> {
    let mut post: Vec<Requirement> = group.lanes.iter().map(|lane| Requirement::Lane(lane.stmt)).collect();
    post.push(Requirement::Fact(permutation_fact(&group.shared)));
    post
}

/// Whether a permutation scatter over the same values exists (its hits are
/// the check's equality tests) among the groups sharing costs with `group`.
fn scatter_shares_tests(group: &Group, cx: &Context) -> bool {
    let n = group.shared.len();
    cx.sharing_groups(group)
        .any(|g| g.kind == OpKind::PermutationScatter && g.shared.len() == 2 * n && g.shared[..n] == group.shared[..])
}

/// `is_permutation(p)` by hit counts (`emit_hit_counts`). Preconditions:
/// none. Postconditions: the lane and `Permutation(p, 0..n)`. Exact.
pub struct PermutationByHits;

/// `is_permutation(p)` by a grand product at a verifier challenge (the
/// module doc). Preconditions: the challenge. Postconditions: the lane and
/// `Permutation(p, 0..n)`. Cost: `13 n + 4` cells.
pub struct PermutationByGrandProduct;

impl Construction for PermutationByHits {
    fn name(&self) -> &'static str {
        "permutation: hit counts"
    }

    fn candidates(&self, group: &Group, cx: &Context) -> Vec<Candidate> {
        if group.kind != OpKind::IsPermutation || group.shared.is_empty() {
            return Vec::new();
        }
        let cost = hit_count_cost(group.shared.len() as u64, cx.target, scatter_shares_tests(group, cx));
        vec![Candidate {
            pre: Vec::new(),
            post: lane_and_fact(group),
            cost,
            emitter: Box::new(CheckEmitter { p: group.shared.clone(), by_grand_product: false }),
        }]
    }
}

impl Construction for PermutationByGrandProduct {
    fn name(&self) -> &'static str {
        "permutation: grand product"
    }

    fn candidates(&self, group: &Group, cx: &Context) -> Vec<Candidate> {
        if group.kind != OpKind::IsPermutation || group.shared.is_empty() || !super::challenge::available(cx.target) {
            return Vec::new();
        }
        vec![Candidate {
            pre: vec![super::challenge::CHALLENGE],
            post: lane_and_fact(group),
            cost: grand_product_cost(group.shared.len() as u64),
            emitter: Box::new(CheckEmitter { p: group.shared.clone(), by_grand_product: true }),
        }]
    }
}

struct CheckEmitter {
    p: Vec<StmtId>,
    by_grand_product: bool,
}

impl Emitter for CheckEmitter {
    fn begin(&mut self, b: &mut IRBuilder, cx: &mut EmitCx) {
        let p = cx.values(&self.p);
        let targets: Vec<Value> = (0..p.len() as i64).map(|j| b.ir_constant_int(j)).collect();
        if self.by_grand_product {
            let operands: Vec<Value> = p.into_iter().chain(targets).collect();
            b.ir_identity_check(Identity::MultisetEqual { len: self.p.len() as u32, width: 1 }, &operands);
        } else {
            emit_hit_counts(b, &p, &targets);
        }
    }

    fn lane(&mut self, b: &mut IRBuilder, _cx: &mut EmitCx, _lane: &Lane) -> Value {
        b.ir_constant_int(1)
    }
}

/// What passes may know about the lane before it is lowered: `1` when the
/// operands are constants forming a permutation (no fold otherwise: a
/// constant non-permutation has no proof).
pub struct IsPermutationSemantics;

impl super::OperatorSemantics for IsPermutationSemantics {
    fn fold(&self, _op: &super::ArrayOp, args: &[Value]) -> Option<super::Constant> {
        let mut values: Vec<i64> = args.iter().map(Value::int_val).collect::<Option<_>>()?;
        values.sort_unstable();
        values.iter().enumerate().all(|(j, &v)| v == j as i64).then_some(super::Constant::Int(1))
    }

    fn number_type(&self, _op: &super::ArrayOp, _args: &[Value]) -> crate::types::NumberType {
        crate::types::NumberType::Integer
    }

    fn bounds(&self, _op: &super::ArrayOp, _operands: &[Option<crate::provenance::Interval>]) -> Option<crate::provenance::Interval> {
        None
    }

    fn picks_from(&self, _op: &super::ArrayOp) -> Option<std::ops::Range<usize>> {
        None
    }
}

#[cfg(test)]
mod tests {
    use crate::builder::IRBuilder;
    use crate::circuit_input::{CircuitInputs, InputNode, InputParam, InputPath, ResolvedWitness};
    use crate::ir::IRGraph;
    use crate::ir_defs::IR;
    use crate::lower::{Construction, Target};
    use crate::prove::types::ProvingParams;
    use crate::types::Value;

    use super::{PermutationByGrandProduct, PermutationByHits};

    /// `is_permutation(p)` over the inputs `p0..`, asserted, then `p0`
    /// exposed; lowered by `constructions` on `target`.
    fn checked(n: usize, target: Target, constructions: &[&dyn Construction]) -> IRGraph {
        let mut b = IRBuilder::new();
        let p: Vec<Value> = (0..n).map(|k| b.ir_read_integer(InputPath::new(format!("p{k}"), vec![]), false)).collect();
        let lane = super::check_lane(&mut b, &p);
        b.ir_assert(&lane);
        b.ir_expose_public_i(&p[0]);
        crate::lower::emit::lower(&b.export_ir_graph(), target, constructions).expect("a lane")
    }

    fn inputs(p: &[i64]) -> Vec<(String, InputNode)> {
        p.iter().enumerate().map(|(k, v)| (format!("p{k}"), InputNode::Int(*v))).collect()
    }

    fn mock_accepts(graph: &IRGraph, p: &[i64]) -> bool {
        use crate::prove::interpreter::interpret_ir;
        use crate::prove::mock::synthesizer::MockSynthesizer;
        let mut circuit_inputs = CircuitInputs::new();
        for (name, value) in inputs(p) {
            circuit_inputs.params.push(InputParam { name, is_public: false, dtype: serde_json::Value::Null, value });
        }
        let mut synth = MockSynthesizer::new(ResolvedWitness::new(circuit_inputs, 32), ProvingParams::default());
        interpret_ir(graph, &mut synth).expect("mock run");
        synth.satisfied
    }

    fn is_permutation(p: &[i64]) -> bool {
        let mut sorted = p.to_vec();
        sorted.sort_unstable();
        sorted.iter().enumerate().all(|(j, &v)| v == j as i64)
    }

    /// Every list of `n <= 4` values in `[-1, n]`: accepted exactly when it
    /// is a permutation of `0..n`, by both constructions, on the mock.
    #[test]
    fn both_checks_accept_exactly_the_permutations() {
        for n in 1..=4usize {
            let graphs = [
                checked(n, Target::Halo2Base, &[&PermutationByHits]),
                checked(n, Target::Halo2Base, &[&PermutationByGrandProduct]),
            ];
            let span = n as i64 + 2;
            for code in 0..span.pow(n as u32) {
                let p: Vec<i64> = (0..n).map(|k| (code / span.pow(k as u32)) % span - 1).collect();
                for graph in &graphs {
                    assert_eq!(mock_accepts(graph, &p), is_permutation(&p), "{p:?}");
                }
            }
        }
    }

    /// halo2-base: the grand product accepts a permutation and rejects a
    /// duplicate, an out-of-range value and a moved challenge cell.
    #[cfg(feature = "halo2-base-backend")]
    #[test]
    fn grand_product_check_on_halo2_base() {
        use crate::prove::halo2_base::synthesizer::two_phase_tests::{mock_run, Tamper};
        let graph = checked(5, Target::Halo2Base, &[&PermutationByGrandProduct]);
        let run = |p: &[i64], tamper: &Tamper| {
            let named = inputs(p);
            let inputs: Vec<(&str, InputNode)> = named.iter().map(|(n, v)| (n.as_str(), v.clone())).collect();
            mock_run(&graph, &inputs, tamper).accepted
        };
        assert!(run(&[3, 1, 4, 0, 2], &Tamper::default()));
        for p in [[3, 1, 4, 0, 0], [3, 1, 5, 0, 2], [3, 1, 4, -1, 2], [0, 0, 0, 0, 10]] {
            assert!(!run(&p, &Tamper::default()), "{p:?} accepted");
        }
        let moved = Tamper { challenge_offset: crate::prove::field::F::one(), ..Tamper::default() };
        assert!(!run(&[3, 1, 4, 0, 2], &moved));
    }

    /// The selection: hit counts for a short list (n^2 tests are cheap) and
    /// on Noir; the grand product for a long one on halo2-base.
    #[test]
    fn the_cheaper_check_is_selected() {
        let identity_checks = |g: &IRGraph| g.stmts.iter().filter(|s| matches!(s.ir, IR::IdentityCheck(_))).count();
        let all = crate::lower::registry();
        assert_eq!(identity_checks(&checked(8, Target::Halo2Base, &all)), 0);
        assert_eq!(identity_checks(&checked(64, Target::Halo2Base, &all)), 1);
        assert_eq!(identity_checks(&checked(64, Target::Noir, &all)), 0);
    }
}
