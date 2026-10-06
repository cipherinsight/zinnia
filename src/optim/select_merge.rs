//! Collapse a select whose branches agree wherever the other one is taken.
//!
//! `select(p, x, y)` is `x` where `p` holds and `y` elsewhere. When `y`
//! equals `x` at every point where `p` fails, the select is `x` everywhere
//! (and symmetrically `y` when `x` equals `y` where `p` holds). Programs
//! reach this shape when the branches of an `if` / `elif` compute one
//! formula that the source spells out per case: 0374's ECDF lookup returns
//! `0` for `c == 0`, `1` for `c == 50`, and `ys[c - 1] = round(c / 50)`
//! otherwise, and the formula `round(c / 50)` also gives `0` and `1` at the
//! two special points, so the three branches are one expression.
//!
//! **The proof.** The pass picks a *pivot*: an integer statement `s` that
//! the condition `p` compares with a constant, whose interval from the
//! range engine (`optim::range`, the bounds every other pass relies on)
//! holds at most `MAX_POINTS` integers. It then requires `p`, `x` and `y`
//! to be functions of `s` alone: their operands, followed back to `s`, are
//! only constants and statements `prove::interpreter::is_pure_value_op`
//! accepts (total integer arithmetic, integer comparisons, Boolean logic,
//! selects; no input, memory, hint or check). For every integer `v` in the
//! interval it evaluates `p`, `x` and `y` at `s = v` with the mock
//! backend's own operations (`prove::interpreter::pure_value_op`, the
//! dispatch `interpret_ir` uses; float constants at the compiler's Q32
//! precision, as `lower::sort::table` quantizes them) and compares the
//! results as field elements. The select becomes `x` when `x == y` at every
//! `v` where `p` evaluates to 0 (and `y` when `x == y` wherever `p` is not
//! 0).
//!
//! **Soundness.** On every witness the circuit accepts, every statement
//! has the value the IR's semantics give it (the mock is the reference
//! semantics; every backend's constraints enforce it, which the
//! differential tests check), and `s` is an integer in its range-engine
//! interval. `p`, `x` and `y` are then the functions of `s` the evaluation
//! computed, at `s`'s actual value `v`. If `p(v)` is 0 the select's value
//! is `y(v)`, which equals `x(v)`; otherwise it is `x(v)`. So `select(p, x,
//! y) = x` on every accepted witness, and replacing the select's users'
//! operand by `x` changes no value. Nothing is added: `x` is already a
//! statement of the circuit (both branches of a select are computed), and
//! no assertion is added or dropped; statements no longer read are removed
//! by dead-code elimination, which drops only computations.
//!
//! **Cost.** Never more: the select and, once nothing else reads them,
//! `y`'s and `p`'s computations disappear.

use std::collections::{HashMap, HashSet};

use crate::builder::IRBuilder;
use crate::circuit_input::{CircuitInputs, ResolvedWitness};
use crate::ir::IRGraph;
use crate::ir_defs::IR;
use crate::optim::{RangeResolver, Resolver};
use crate::prove::interpreter::{is_pure_value_op, pure_value_op};
use crate::prove::mock::synthesizer::{MockCell, MockSynthesizer};
use crate::prove::traits::Synthesizer;
use crate::prove::types::ProvingParams;
use crate::types::{ScalarValue, StmtId, Value};

use super::IRPass;

pub struct SelectMerge;

/// The most pivot values the proof evaluates, and the most statements a
/// branch or condition may have (bounds the pass's work per select).
const MAX_POINTS: i64 = 128;
const MAX_CONE: usize = 64;

impl IRPass for SelectMerge {
    fn exec(&self, graph: IRGraph) -> IRGraph {
        let mut prover = Prover::new(&graph);
        let collapsed: HashMap<StmtId, StmtId> = graph
            .stmts
            .iter()
            .filter_map(|stmt| match (&stmt.ir, stmt.arguments.as_slice()) {
                (IR::SelectI | IR::SelectF | IR::SelectB, &[p, x, y]) => {
                    prover.collapse(p, x, y).map(|branch| (stmt.stmt_id, branch))
                }
                _ => None,
            })
            .collect();
        if collapsed.is_empty() {
            return graph;
        }
        let mut b = IRBuilder::new();
        let mut lookup: HashMap<StmtId, Value> = HashMap::new();
        for stmt in &graph.stmts {
            let value = match collapsed.get(&stmt.stmt_id) {
                Some(branch) => lookup[branch].clone(),
                None => {
                    let args: Vec<Value> = stmt.arguments.iter().map(|a| lookup[a].clone()).collect();
                    b.create_ir(&stmt.ir, &args)
                }
            };
            lookup.insert(stmt.stmt_id, value);
        }
        let mut out = b.export_ir_graph();
        out.resolver_mut().on_ir_mutated(&[]);
        out
    }
}

/// The proofs of one graph: the range engine and the mock's operations.
struct Prover<'a> {
    graph: &'a IRGraph,
    range: RangeResolver,
    mock: MockSynthesizer,
}

impl<'a> Prover<'a> {
    fn new(graph: &'a IRGraph) -> Self {
        let params = ProvingParams::default();
        let witness = ResolvedWitness::new(CircuitInputs::new(), params.precision_bits);
        Prover { graph, range: RangeResolver::new(), mock: MockSynthesizer::new(witness, params) }
    }

    /// The branch `select(p, x, y)` always equals, when some pivot proves it.
    fn collapse(&mut self, p: StmtId, x: StmtId, y: StmtId) -> Option<StmtId> {
        if self.constant(p).is_some() {
            return None;
        }
        for s in self.pivots(p) {
            let Some((lo, hi)) = self.bounds(s).filter(|(lo, hi)| hi - lo < MAX_POINTS) else { continue };
            let (Some(pc), Some(xc), Some(yc)) = (self.cone(p, s), self.cone(x, s), self.cone(y, s)) else { continue };
            let mut agree_where_false = true;
            let mut agree_where_true = true;
            for v in lo..=hi {
                let (Some(pv), Some(xv), Some(yv)) =
                    (self.eval(&pc, p, s, v), self.eval(&xc, x, s, v), self.eval(&yc, y, s, v))
                else {
                    return None;
                };
                if xv.fp() != yv.fp() {
                    if pv.is_truthy() {
                        agree_where_true = false;
                    } else {
                        agree_where_false = false;
                    }
                }
            }
            if agree_where_false {
                return Some(x);
            }
            if agree_where_true {
                return Some(y);
            }
        }
        None
    }

    fn constant(&self, s: StmtId) -> Option<i64> {
        match self.graph.stmts[s as usize].ir {
            IR::ConstantInt { value } => Some(value),
            IR::ConstantBool { value } => Some(i64::from(value)),
            _ => None,
        }
    }

    /// The integer statements `p` compares with a constant, through `and`,
    /// `or` and `not`.
    fn pivots(&self, p: StmtId) -> Vec<StmtId> {
        let mut out = Vec::new();
        let mut stack = vec![p];
        let mut seen = HashSet::new();
        while let Some(c) = stack.pop() {
            if !seen.insert(c) || seen.len() > MAX_CONE {
                continue;
            }
            let stmt = &self.graph.stmts[c as usize];
            match (&stmt.ir, stmt.arguments.as_slice()) {
                (IR::LogicalAnd | IR::LogicalOr, &[a, b]) => stack.extend([a, b]),
                (IR::LogicalNot, &[a]) => stack.push(a),
                (IR::EqI | IR::NeI | IR::LtI | IR::LteI | IR::GtI | IR::GteI, &[a, b]) => {
                    match (self.constant(a), self.constant(b)) {
                        (None, Some(_)) => out.push(a),
                        (Some(_), None) => out.push(b),
                        _ => {}
                    }
                }
                _ => {}
            }
        }
        out.sort_unstable();
        out.dedup();
        out
    }

    /// The range engine's interval of `s`.
    fn bounds(&mut self, s: StmtId) -> Option<(i64, i64)> {
        let value = Value::Integer(ScalarValue::new(None, Some(s)));
        let lo = self.range.resolve_min_with_stmts(&value, &self.graph.stmts)?;
        let hi = self.range.resolve_max_with_stmts(&value, &self.graph.stmts)?;
        (lo <= hi).then_some((lo, hi))
    }

    /// The statements `root` depends on back to `pivot`, in statement order
    /// (operands first), when all of them are pure value statements and
    /// there are at most `MAX_CONE`; `pivot` itself is not included.
    fn cone(&self, root: StmtId, pivot: StmtId) -> Option<Vec<StmtId>> {
        let mut seen: HashSet<StmtId> = HashSet::new();
        let mut stack = vec![root];
        while let Some(s) = stack.pop() {
            if s == pivot || !seen.insert(s) {
                continue;
            }
            let stmt = &self.graph.stmts[s as usize];
            if seen.len() > MAX_CONE || !is_pure_value_op(&stmt.ir) {
                return None;
            }
            stack.extend(stmt.arguments.iter().copied());
        }
        let mut order: Vec<StmtId> = seen.into_iter().collect();
        order.sort_unstable();
        Some(order)
    }

    /// `root`'s value at `pivot = v`, by the mock's operations over `cone`
    /// (statement ids are in dependency order).
    fn eval(&mut self, cone: &[StmtId], root: StmtId, pivot: StmtId, v: i64) -> Option<MockCell> {
        let mut cells: HashMap<StmtId, MockCell> = HashMap::new();
        cells.insert(pivot, self.mock.constant_int(v).ok()?);
        for &s in cone {
            let stmt = &self.graph.stmts[s as usize];
            let args: Vec<MockCell> = stmt.arguments.iter().map(|a| cells.get(a).cloned()).collect::<Option<_>>()?;
            let cell = pure_value_op(&mut self.mock, &stmt.ir, &args).ok()?;
            cells.insert(s, cell);
        }
        cells.remove(&root)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circuit_input::{InputNode, InputParam, InputPath};

    fn run(graph: &IRGraph, inputs: &[(&str, i64)]) -> (bool, Vec<String>) {
        let mut circuit_inputs = CircuitInputs::new();
        for (name, value) in inputs {
            circuit_inputs.params.push(InputParam {
                name: name.to_string(),
                is_public: false,
                dtype: serde_json::Value::Null,
                value: InputNode::Int(*value),
            });
        }
        let witness = ResolvedWitness::new(circuit_inputs, ProvingParams::default().precision_bits);
        let mut synth = MockSynthesizer::new(witness, ProvingParams::default());
        crate::prove::interpreter::interpret_ir(graph, &mut synth).expect("mock run");
        let mut outputs: Vec<_> = synth.public_outputs.clone().into_iter().collect();
        outputs.sort();
        (synth.satisfied, outputs.into_iter().map(|(_, v)| v).collect())
    }

    fn merged(graph: &IRGraph) -> IRGraph {
        crate::optim::DeadCodeElimination.exec(SelectMerge.exec(graph.clone()))
    }

    fn selects(graph: &IRGraph) -> usize {
        graph.stmts.iter().filter(|s| matches!(s.ir, IR::SelectI)).count()
    }

    /// `c = [a0 < e] + [a1 < e] + [a2 < e]` (in `[0, 3]`), then
    /// `select(c == 0, at_zero, select(c == 3, at_three, 10 c))`.
    fn cases(at_zero: i64, at_three: i64) -> IRGraph {
        let mut b = IRBuilder::new();
        let e = b.ir_read_integer(InputPath::new("e", vec![]), false);
        let bits: Vec<Value> = (0..3)
            .map(|k| {
                let a = b.ir_read_integer(InputPath::new(format!("a{k}"), vec![]), false);
                b.ir_less_than_i(&a, &e)
            })
            .collect();
        let c = bits[1..].iter().fold(bits[0].clone(), |acc, bit| b.ir_add_i(&acc, bit));
        let (zero, three, ten) = (b.ir_constant_int(0), b.ir_constant_int(3), b.ir_constant_int(10));
        let general = b.ir_mul_i(&c, &ten);
        let is_three = b.ir_equal_i(&c, &three);
        let at_three = b.ir_constant_int(at_three);
        let inner = b.ir_select_i(&is_three, &at_three, &general);
        let is_zero = b.ir_equal_i(&c, &zero);
        let at_zero = b.ir_constant_int(at_zero);
        let outer = b.ir_select_i(&is_zero, &at_zero, &inner);
        b.ir_expose_public_i(&outer);
        b.export_ir_graph()
    }

    fn same_on_every_count(original: &IRGraph, merged: &IRGraph) {
        for e in 0..=4 {
            let inputs = [("e", e), ("a0", 1), ("a1", 2), ("a2", 3)];
            assert_eq!(run(merged, &inputs), run(original, &inputs), "e={e}");
        }
    }

    #[test]
    fn branches_that_agree_with_the_general_formula_collapse() {
        // 0 and 30 are what `10 c` gives at c = 0 and c = 3: one expression.
        let graph = cases(0, 30);
        let out = merged(&graph);
        assert_eq!(selects(&out), 0);
        assert!(!out.stmts.iter().any(|s| matches!(s.ir, IR::EqI)));
        same_on_every_count(&graph, &out);
    }

    #[test]
    fn a_branch_that_differs_somewhere_is_kept() {
        for (at_zero, at_three, kept) in [(5, 30, 1), (0, 31, 1), (5, 31, 2)] {
            let graph = cases(at_zero, at_three);
            let out = merged(&graph);
            assert_eq!(selects(&out), kept, "at_zero={at_zero} at_three={at_three}");
            same_on_every_count(&graph, &out);
        }
    }

    #[test]
    fn a_select_whose_branch_reads_an_input_is_not_evaluated() {
        // `select(c == 0, a0 * 0, 10 c)`: equal at c = 0 for every a0, but
        // the branch reads an input, which the evaluation does not cover.
        let mut b = IRBuilder::new();
        let e = b.ir_read_integer(InputPath::new("e", vec![]), false);
        let a = b.ir_read_integer(InputPath::new("a0", vec![]), false);
        let c = b.ir_less_than_i(&a, &e);
        let (zero, ten) = (b.ir_constant_int(0), b.ir_constant_int(10));
        let general = b.ir_mul_i(&c, &ten);
        let opaque = b.ir_mul_i(&a, &zero);
        let is_zero = b.ir_equal_i(&c, &zero);
        let out = b.ir_select_i(&is_zero, &opaque, &general);
        b.ir_expose_public_i(&out);
        let graph = b.export_ir_graph();
        assert_eq!(selects(&merged(&graph)), 1);
    }
}
