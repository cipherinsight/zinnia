//! Count queries against a sort: `#{j : a_j <= e}` and `#{j : a_j < e}` as
//! lanes of the sort (outputs `COUNT_AT_MOST`, `COUNT_BELOW`), and
//! comparisons of a scalar with a sorted value answered from those counts.
//!
//! A count lane is how the frontend lowers the `searchsorted` idiom
//! `(np.sort(a) > e).argmax()` (the number of sorted values `<= e`). Every
//! plan of the sort emits it the same way, one comparison per element and a
//! sum, so it never changes the plan.
//!
//! [`SortCountQueries`] (a pass before the array lowering) rewrites a
//! comparison of a scalar `e` with the sorted value `s_k` of rank `k` into a
//! comparison of a count of the same sort and the same `e` with `k`, when
//! the program already has that count:
//!
//! | comparison | from the count |
//! |---|---|
//! | `e < s_k`  | `#{a_j <= e} <= k` |
//! | `e >= s_k` | `k < #{a_j <= e}` |
//! | `e <= s_k` | `#{a_j < e} <= k` |
//! | `e > s_k`  | `k < #{a_j < e}` |
//!
//! Soundness: `s_k` is the `(k+1)`-th smallest element, so at least `k + 1`
//! elements are `<= e` exactly when `s_k <= e`, and at least `k + 1` are
//! `< e` exactly when `s_k < e` (ties included: equal elements are counted
//! together on both sides). The rewritten comparison computes the same
//! Boolean from a count the circuit constrains anyway; a sorted value no
//! longer read is dead, and its pin (an extreme pin, an order statistic)
//! disappears with it. Costs only fall: the comparison of two small
//! integers replaces a comparison of two window values.

use std::collections::HashMap;

use crate::builder::IRBuilder;
use crate::ir::IRGraph;
use crate::ir_defs::IR;
use crate::lower::OpKind;
use crate::optim::IRPass;
use crate::types::{StmtId, Value};

use super::emit::{count, Elements};
use super::semantics::{COUNT_AT_MOST, COUNT_BELOW, VALUES};

/// `#{j : a_j <= e}` (`at_most`) or `#{j : a_j < e}`: `n - #{e < a_j}` or
/// `#{a_j < e}`, one comparison per element (no negation).
pub(super) fn count_query(b: &mut IRBuilder, elements: &Elements, e: &Value, at_most: bool) -> Value {
    if at_most {
        let above: Vec<Value> = elements.values.iter().map(|a| elements.less(b, e, a)).collect();
        let n_above = count(b, &above);
        let n = b.ir_constant_int(elements.values.len() as i64);
        b.ir_sub_i(&n, &n_above)
    } else {
        let below: Vec<Value> = elements.values.iter().map(|a| elements.less(b, a, e)).collect();
        count(b, &below)
    }
}

/// The pass: comparisons with sorted values answered from the counts the
/// program has (see the module comment). A graph without both is returned
/// unchanged; otherwise the values no longer read are removed.
pub struct SortCountQueries;

impl IRPass for SortCountQueries {
    fn exec(&self, graph: IRGraph) -> IRGraph {
        let answers = answers(&graph);
        if answers.is_empty() {
            return graph;
        }
        let mut b = IRBuilder::new();
        let mut lookup: HashMap<StmtId, Value> = HashMap::new();
        for stmt in graph.get_topological_order(false) {
            if lookup.contains_key(&stmt.stmt_id) {
                continue; // emitted early, as a count's operand
            }
            let answer = answers.get(&stmt.stmt_id);
            // The count may stand later in the program than the comparison
            // (`e < s_0` before `(s > e).argmax()`): it is emitted here when
            // it and its operands are pure.
            let count = answer.and_then(|a| emit_early(&graph, &mut b, &mut lookup, a.count));
            let out = match (answer, count) {
                (Some(answer), Some(count)) => {
                    let rank = b.ir_constant_int(answer.rank);
                    if answer.count_at_most_rank {
                        b.ir_less_than_or_equal_i(&count, &rank)
                    } else {
                        b.ir_less_than_i(&rank, &count)
                    }
                }
                _ => {
                    let args: Vec<Value> = stmt.arguments.iter().map(|a| lookup[a].clone()).collect();
                    b.create_ir(&stmt.ir, &args)
                }
            };
            lookup.insert(stmt.stmt_id, out);
        }
        crate::optim::DeadCodeElimination.exec(b.export_ir_graph())
    }
}

/// The value of `stmt`, emitted now with its operands if it is not yet:
/// only for statements without effects whose order matters (array lanes,
/// casts, constants, arithmetic, comparisons), `None` otherwise.
fn emit_early(graph: &IRGraph, b: &mut IRBuilder, lookup: &mut HashMap<StmtId, Value>, stmt: StmtId) -> Option<Value> {
    if let Some(value) = lookup.get(&stmt) {
        return Some(value.clone());
    }
    let st = &graph.stmts[stmt as usize];
    let pure = matches!(
        st.ir,
        IR::ArrayOp(_)
            | IR::ConstantInt { .. }
            | IR::ConstantFloat { .. }
            | IR::ConstantBool { .. }
            | IR::FloatCast
            | IR::IntCast
            | IR::BoolCast
            | IR::AddI
            | IR::SubI
            | IR::MulI
            | IR::AddF
            | IR::SubF
            | IR::MulF
    );
    if !pure {
        return None;
    }
    let args: Vec<Value> =
        st.arguments.iter().map(|&a| emit_early(graph, b, lookup, a)).collect::<Option<_>>()?;
    let value = b.create_ir(&st.ir, &args);
    lookup.insert(stmt, value.clone());
    Some(value)
}

/// A comparison's replacement: `count <= rank` or `rank < count`.
struct Answer {
    count: StmtId,
    rank: i64,
    count_at_most_rank: bool,
}

/// The comparisons of `graph` that a count lane of the same sort and the
/// same scalar answers.
fn answers(graph: &IRGraph) -> HashMap<StmtId, Answer> {
    // (elements, scalar, at_most) -> the count lane.
    let mut counts: HashMap<(&[StmtId], StmtId, bool), StmtId> = HashMap::new();
    for stmt in &graph.stmts {
        let IR::ArrayOp(op) = &stmt.ir else { continue };
        if op.kind == OpKind::Sort && (op.output == COUNT_AT_MOST || op.output == COUNT_BELOW) {
            let n = op.shared as usize;
            counts.insert((&stmt.arguments[..n], stmt.arguments[n], op.output == COUNT_AT_MOST), stmt.stmt_id);
        }
    }
    if counts.is_empty() {
        return HashMap::new();
    }
    // The sorted value of a constant rank behind a statement.
    let sorted_value = |id: StmtId| match &graph.stmts[id as usize].ir {
        IR::ArrayOp(op) if op.kind == OpKind::Sort && op.output == VALUES => {
            let rank = op.position?;
            Some((&graph.stmts[id as usize].arguments[..op.shared as usize], i64::from(rank)))
        }
        _ => None,
    };
    let mut answers = HashMap::new();
    for stmt in &graph.stmts {
        // `x rel y` as `lower_than`: `x < y` or `x <= y` after swapping.
        let (lower, strict, swapped) = match stmt.ir {
            IR::LtF | IR::LtI => (true, true, false),
            IR::LteF | IR::LteI => (true, false, false),
            IR::GtF | IR::GtI => (true, true, true),
            IR::GteF | IR::GteI => (true, false, true),
            _ => (false, false, false),
        };
        if !lower {
            continue;
        }
        let [x, y] = stmt.arguments[..] else { continue };
        let (small, large) = if swapped { (y, x) } else { (x, y) };
        // `e < s_k`, `e <= s_k` (the value on the right), or `s_k < e`,
        // `s_k <= e`, i.e. `e > s_k`, `e >= s_k`.
        let query = if let Some((elements, rank)) = sorted_value(large) {
            // e < s_k: #{<= e} <= k;  e <= s_k: #{< e} <= k.
            Some((elements, small, rank, strict, true))
        } else if let Some((elements, rank)) = sorted_value(small) {
            // s_k < e: k < #{< e};  s_k <= e: k < #{<= e}.
            Some((elements, large, rank, !strict, false))
        } else {
            None
        };
        let Some((elements, scalar, rank, at_most, count_at_most_rank)) = query else { continue };
        if let Some(&count) = counts.get(&(elements, scalar, at_most)) {
            answers.insert(stmt.stmt_id, Answer { count, rank, count_at_most_rank });
        }
    }
    answers
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::circuit_input::InputPath;
    use crate::lower::sort::test_support::mock_run;
    use crate::lower::{ArrayOp, Target};

    const ELEMENTS: [i64; 7] = [5, -3, 5, 0, -3, 7, 5];

    /// Comparisons of the input `e` with sorted values (both operand
    /// orders), plus the program's own count lanes (before or after the
    /// comparisons); lowered with the count pass first when `answer`.
    /// Returns the graph and how many sorted-value lanes reach the lowering.
    fn comparison_graph(answer: bool, counts_first: bool) -> (IRGraph, usize) {
        let mut b = IRBuilder::new();
        let elements: Arc<[Value]> = (0..ELEMENTS.len())
            .map(|j| b.ir_read_integer(InputPath::new(format!("a{j}"), vec![]), false))
            .collect();
        let e = b.ir_read_integer(InputPath::new("e", vec![]), false);
        let n = ELEMENTS.len() as u32;
        let lane = |b: &mut IRBuilder, output, position, own: &[Value]| {
            b.ir_array_op(ArrayOp { kind: OpKind::Sort, shared: n, output, position }, &elements, own)
        };
        let counts = |b: &mut IRBuilder| {
            for at_most in [true, false] {
                let count = lane(b, if at_most { COUNT_AT_MOST } else { COUNT_BELOW }, None, &[e.clone()]);
                b.ir_expose_public_i(&count);
            }
        };
        if counts_first {
            counts(&mut b);
        }
        for k in [0, 2, 3, 6] {
            let s = lane(&mut b, VALUES, Some(k), &[]);
            let comparisons = [
                b.ir_less_than_i(&e, &s),
                b.ir_less_than_or_equal_i(&e, &s),
                b.ir_greater_than_i(&e, &s),
                b.ir_greater_than_or_equal_i(&e, &s),
                b.ir_less_than_i(&s, &e),
                b.ir_greater_than_or_equal_i(&s, &e),
            ];
            for c in comparisons {
                let as_int = b.ir_int_cast(&c);
                b.ir_expose_public_i(&as_int);
            }
        }
        if !counts_first {
            counts(&mut b);
        }
        let graph = b.export_ir_graph();
        let graph = if answer { SortCountQueries.exec(graph) } else { graph };
        let values_left = graph
            .stmts
            .iter()
            .filter(|s| matches!(&s.ir, IR::ArrayOp(op) if op.output == VALUES))
            .count();
        let lowered = crate::lower::emit::lower(&graph, Target::Halo2Base, &crate::lower::registry())
            .expect("placeholders to lower");
        (crate::optim::DeadCodeElimination.exec(lowered), values_left)
    }

    fn numpy(e: i64, counts_first: bool) -> Vec<i64> {
        let mut sorted = ELEMENTS.to_vec();
        sorted.sort();
        let counts = [
            ELEMENTS.iter().filter(|&&a| a <= e).count() as i64,
            ELEMENTS.iter().filter(|&&a| a < e).count() as i64,
        ];
        let mut out = if counts_first { counts.to_vec() } else { Vec::new() };
        for k in [0, 2, 3, 6] {
            let s = sorted[k];
            out.extend([e < s, e <= s, e > s, e >= s, s < e, s >= e].map(i64::from));
        }
        if !counts_first {
            out.extend(counts);
        }
        out
    }

    #[test]
    fn comparisons_answered_by_counts_match_numpy() {
        for counts_first in [true, false] {
            let (answered, values_left) = comparison_graph(true, counts_first);
            assert_eq!(values_left, 0, "every comparison is answered by a count");
            let (direct, values_left) = comparison_graph(false, counts_first);
            assert_eq!(values_left, 4);
            for e in [-4, -3, -2, 0, 1, 5, 6, 7, 8] {
                let mut inputs: Vec<(String, i64)> =
                    ELEMENTS.iter().enumerate().map(|(j, &a)| (format!("a{j}"), a)).collect();
                inputs.push(("e".to_string(), e));
                for graph in [&answered, &direct] {
                    let (ok, got) = mock_run(graph, &inputs);
                    assert!(ok, "e = {e}");
                    assert_eq!(got, numpy(e, counts_first), "e = {e} counts first {counts_first}");
                }
            }
        }
    }
}
