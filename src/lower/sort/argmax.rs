//! Tests of a gathered element at the first-True index, answered without
//! the gather: `h[argmax(h)]` is `any(h)`.
//!
//! The program `r = ((x == a) & (y == b)).argmax(); x[r] != a or y[r] != b`
//! reads `x` and `y` at `r` and compares them again, although the program
//! already holds every comparison `x_k == a`, `y_k == b` and their
//! conjunction `h_k`. Two identities answer such a test from `h`:
//!
//! 1. A gather commutes with an elementwise operation: for `r` in
//!    `[0, n)`, `x[r] == c` is `(x == c)[r]`, `p[r] and q[r]` is
//!    `(p and q)[r]`, `not p[r]` is `(not p)[r]`.
//! 2. `r = argmax(h)` (NumPy: the first True, 0 when there is none) picks a
//!    True when there is one, so `h[r]` is `any(h)`: with a True `h[r] = 1`;
//!    without one `r = 0` and `h[0] = 0`.
//!
//! The pass [`ArgmaxQueries`] finds `r` as the gather family's argmax lane
//! (`OpKind::ArgmaxOfFlags` over `h`) or as a first-True select chain
//! (`select(h_0, 0, select(h_1, 1, .. select(h_{n-1}, n-1, 0)))`), views a
//! statement built from reads at `r` (`ArrayRead`s of known cells) by
//! equality tests, `and`, `or` and `not` as `B[r]` (or `not B[r]`) for a
//! vector `B` of statements the program has, and replaces it by `any(h)`
//! (or its negation) when `B` is `h`. Everything it relies on is a
//! statement of the program, matched by identity; no hint. `any(h)` is the
//! sum of the bits tested non-zero.
//!
//! A selection between the first-True index and a default under that test
//! (`r if h[r] else d`, the program's `result = -1`) is the first-True chain
//! itself with `d` as its last default: without a True the chain ends at
//! its default, with one it stops at the first True. No `any` is needed.

use std::collections::HashMap;

use crate::builder::IRBuilder;
use crate::ir::IRGraph;
use crate::ir_defs::IR;
use crate::lower::OpKind;
use crate::optim::IRPass;
use crate::types::{StmtId, Value};

pub struct ArgmaxQueries;

impl IRPass for ArgmaxQueries {
    fn exec(&self, graph: IRGraph) -> IRGraph {
        let answers = answers(&graph);
        if answers.is_empty() {
            return graph;
        }
        let mut b = IRBuilder::new();
        let mut lookup: HashMap<StmtId, Value> = HashMap::new();
        let mut any_of: HashMap<Vec<StmtId>, Value> = HashMap::new();
        for stmt in graph.get_topological_order(false) {
            let out = match answers.get(&stmt.stmt_id) {
                Some(Answer { chain_default: Some(default), bits, .. }) => {
                    let mut r = lookup[default].clone();
                    for (k, bit) in bits.iter().enumerate().rev() {
                        let at = b.ir_constant_int(k as i64);
                        r = b.ir_select_i(&lookup[bit], &at, &r);
                    }
                    r
                }
                Some(answer) => {
                    let any = any_of
                        .entry(answer.bits.clone())
                        .or_insert_with(|| {
                            let bits: Vec<Value> = answer.bits.iter().map(|s| lookup[s].clone()).collect();
                            any(&mut b, &bits)
                        })
                        .clone();
                    if answer.negated {
                        b.ir_logical_not(&any)
                    } else {
                        any
                    }
                }
                None => {
                    let args: Vec<Value> = stmt.arguments.iter().map(|a| lookup[a].clone()).collect();
                    b.create_ir(&stmt.ir, &args)
                }
            };
            lookup.insert(stmt.stmt_id, out);
        }
        // The reads the rewrite left unused go with the dead statements.
        crate::optim::DeadCodeElimination.exec(b.export_ir_graph())
    }
}

/// `[bits contain a 1]`: their sum tested non-zero.
fn any(b: &mut IRBuilder, bits: &[Value]) -> Value {
    let mut total = b.ir_constant_int(0);
    for bit in bits {
        let as_int = b.ir_int_cast(bit);
        total = b.ir_add_i(&total, &as_int);
    }
    let zero = b.ir_constant_int(0);
    b.ir_not_equal_i(&total, &zero)
}

/// A statement's replacement: `any(bits)`, negated or not; or, for a
/// selection between the first-True index over `bits` and a default, that
/// chain ending at the default.
struct Answer {
    bits: Vec<StmtId>,
    negated: bool,
    chain_default: Option<StmtId>,
}

/// `B[r]` (or `not B[r]` when `negated`) for the statements `B`.
#[derive(Clone, PartialEq, Eq)]
struct Gathered {
    bits: Vec<StmtId>,
    negated: bool,
}

struct Facts<'a> {
    graph: &'a IRGraph,
    /// Statements by (operation, operands), for finding the elementwise
    /// comparisons and conjunctions the program has.
    by_operands: HashMap<(std::mem::Discriminant<IR>, Vec<StmtId>), StmtId>,
}

fn answers(graph: &IRGraph) -> HashMap<StmtId, Answer> {
    let facts = Facts::new(graph);
    let mut answers = HashMap::new();
    for stmt in &graph.stmts {
        if !matches!(stmt.ir, IR::EqI | IR::NeI | IR::LogicalAnd | IR::LogicalOr | IR::LogicalNot) {
            continue;
        }
        let mut index = None;
        let Some(view) = facts.gathered(stmt.stmt_id, &mut index) else { continue };
        let Some(r) = index else { continue };
        if facts.first_true_bits(r).as_ref() == Some(&view.bits) {
            answers.insert(stmt.stmt_id, Answer { bits: view.bits, negated: view.negated, chain_default: None });
        }
    }
    // `select(not h[r], d, r)` and `select(h[r], r, d)` with `r` the chain.
    for stmt in &graph.stmts {
        let (IR::SelectI, &[cond, t, f]) = (&stmt.ir, &stmt.arguments[..]) else { continue };
        let Some(test) = answers.get(&cond) else { continue };
        let (index, default) = if test.negated { (f, t) } else { (t, f) };
        if facts.first_true_bits(index).as_ref() == Some(&test.bits) {
            let bits = test.bits.clone();
            answers.insert(stmt.stmt_id, Answer { bits, negated: false, chain_default: Some(default) });
        }
    }
    answers
}

impl<'a> Facts<'a> {
    fn new(graph: &'a IRGraph) -> Facts<'a> {
        let mut by_operands = HashMap::new();
        for stmt in &graph.stmts {
            if matches!(stmt.ir, IR::EqI | IR::LogicalAnd) {
                by_operands.insert((std::mem::discriminant(&stmt.ir), stmt.arguments.clone()), stmt.stmt_id);
            }
        }
        Facts { graph, by_operands }
    }

    fn stmt(&self, id: StmtId) -> &crate::ir::IRStatement {
        &self.graph.stmts[id as usize]
    }

    /// The program's statement `op(x, y)` or `op(y, x)` (both ops here are
    /// symmetric).
    fn symmetric(&self, op: &IR, x: StmtId, y: StmtId) -> Option<StmtId> {
        let key = |a, b| (std::mem::discriminant(op), vec![a, b]);
        self.by_operands.get(&key(x, y)).or_else(|| self.by_operands.get(&key(y, x))).copied()
    }

    /// `stmt` as `B[r]` or `not B[r]`, recording `r` (one index for all).
    fn gathered(&self, stmt: StmtId, index: &mut Option<StmtId>) -> Option<Gathered> {
        let st = self.stmt(stmt);
        match (&st.ir, &st.arguments[..]) {
            (IR::EqI | IR::NeI, &[x, y]) => {
                let (slots, c) = match self.read_at(x, index) {
                    Some(slots) => (slots, y),
                    None => (self.read_at(y, index)?, x),
                };
                let bits: Vec<StmtId> =
                    slots.iter().map(|&slot| self.symmetric(&IR::EqI, slot, c)).collect::<Option<_>>()?;
                Some(Gathered { bits, negated: matches!(st.ir, IR::NeI) })
            }
            (IR::LogicalNot, &[x]) => {
                let inner = self.gathered(x, index)?;
                Some(Gathered { negated: !inner.negated, ..inner })
            }
            (IR::LogicalAnd | IR::LogicalOr, &[x, y]) => {
                let (p, q) = (self.gathered(x, index)?, self.gathered(y, index)?);
                // `and` of two plain views, or (De Morgan) `or` of two
                // negated ones: the conjunction's bits.
                let conjunction = match st.ir {
                    IR::LogicalAnd => !p.negated && !q.negated,
                    _ => p.negated && q.negated,
                };
                if !conjunction || p.bits.len() != q.bits.len() {
                    return None;
                }
                let bits: Vec<StmtId> = p
                    .bits
                    .iter()
                    .zip(&q.bits)
                    .map(|(&a, &b)| self.symmetric(&IR::LogicalAnd, a, b))
                    .collect::<Option<_>>()?;
                Some(Gathered { bits, negated: p.negated })
            }
            _ => None,
        }
    }

    /// The cells `stmt` reads (a read of known cells: `ArrayRead` of an
    /// `ArrayInit`), when it reads at the same runtime index as the rest of
    /// the view.
    fn read_at(&self, stmt: StmtId, index: &mut Option<StmtId>) -> Option<Vec<StmtId>> {
        let st = self.stmt(stmt);
        let (cells, at) = match &st.ir {
            IR::ArrayRead(_) => {
                let read = crate::lower::arrays::element_read(&self.graph.stmts, stmt)?;
                (read.line, read.index?)
            }
            _ => return None,
        };
        if index.is_some_and(|r| r != at) {
            return None;
        }
        *index = Some(at);
        Some(cells)
    }

    /// The bits `h` when `r` is NumPy's argmax of them: an argmax lane of
    /// the gather family, or the first-True chain
    /// `select(h_0, 0, select(h_1, 1, ... select(h_{n-1}, n-1, 0)))`.
    fn first_true_bits(&self, r: StmtId) -> Option<Vec<StmtId>> {
        if let IR::ArrayOp(op) = &self.stmt(r).ir {
            let n = op.shared as usize;
            return (op.kind == OpKind::ArgmaxOfFlags).then(|| self.stmt(r).arguments[..n].to_vec());
        }
        let constant = |id: StmtId| match self.stmt(id).ir {
            IR::ConstantInt { value } => Some(value),
            _ => None,
        };
        let mut bits = Vec::new();
        let mut at = r;
        loop {
            let st = self.stmt(at);
            match (&st.ir, &st.arguments[..]) {
                (IR::SelectI, &[cond, then, rest]) if constant(then) == Some(bits.len() as i64) => {
                    bits.push(cond);
                    at = rest;
                }
                (IR::ConstantInt { value: 0 }, _) if bits.len() > 1 => return Some(bits),
                _ => return None,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circuit_input::InputPath;
    use crate::lower::sort::test_support::mock_run;

    /// The same with the gather family's placeholders (`placeholders`): the
    /// argmax lane and element lanes at it, lowered by the registry.
    fn first_match_with(rewrite: bool, placeholders: bool) -> IRGraph {
        let mut b = IRBuilder::new();
        let n = 6;
        let read = |b: &mut IRBuilder, name: &str| b.ir_read_integer(InputPath::new(name, vec![]), false);
        let x: Vec<Value> = (0..n).map(|k| read(&mut b, &format!("x{k}"))).collect();
        let y: Vec<Value> = (0..n).map(|k| read(&mut b, &format!("y{k}"))).collect();
        let (a, c) = (read(&mut b, "a"), read(&mut b, "c"));
        let h: Vec<Value> = (0..n)
            .map(|k| {
                let ex = b.ir_equal_i(&x[k], &a);
                let ey = b.ir_equal_i(&y[k], &c);
                b.ir_logical_and(&ex, &ey)
            })
            .collect();
        let r = if placeholders {
            crate::lower::gather::scan_lane(&mut b, crate::lower::gather::ScanKind::Argmax, &h)
        } else {
            let mut r = b.ir_constant_int(0);
            for k in (0..n).rev() {
                let at = b.ir_constant_int(k as i64);
                r = b.ir_select_i(&h[k], &at, &r);
            }
            r
        };
        let gathered = |b: &mut IRBuilder, cells: &[Value]| {
            if placeholders {
                return crate::lower::gather::element_lane(b, cells, &r).expect("an encoded index");
            }
            let segment = crate::helpers::segment::alloc_and_write_values(b, cells, crate::types::NumberType::Integer);
            b.storage_read_at(segment, &r)
        };
        let xr = gathered(&mut b, &x);
        let yr = gathered(&mut b, &y);
        let nx = b.ir_not_equal_i(&xr, &a);
        let ny = b.ir_not_equal_i(&yr, &c);
        let miss = b.ir_logical_or(&nx, &ny);
        let minus_one = b.ir_constant_int(-1);
        let result = b.ir_select_i(&miss, &minus_one, &r);
        let miss = b.ir_int_cast(&miss);
        b.ir_expose_public_i(&r);
        b.ir_expose_public_i(&miss);
        b.ir_expose_public_i(&result);
        let graph = b.export_ir_graph();
        let graph = if rewrite { ArgmaxQueries.exec(graph) } else { graph };
        let reads = graph.stmts.iter().filter(|s| matches!(&s.ir, IR::ArrayRead(_))).count();
        assert_eq!(reads == 0, rewrite, "the reads die exactly when rewritten");
        crate::lower::emit::lower_rounds(graph, crate::lower::Target::Halo2Base, &crate::lower::registry())
    }

    #[test]
    fn a_test_at_the_first_true_index_is_any() {
        for placeholders in [false, true] {
            let rewritten = first_match_with(true, placeholders);
            let original = first_match_with(false, placeholders);
            assert_first_matches(&rewritten, &original);
        }
    }

    fn assert_first_matches(rewritten: &IRGraph, original: &IRGraph) {
        let cases: [([i64; 6], [i64; 6]); 5] = [
            ([1, 2, 3, 2, 5, 2], [9, 9, 4, 4, 4, 7]),  // first match at 3
            ([2, 2, 3, 2, 5, 2], [4, 9, 4, 4, 4, 7]),  // at 0
            ([1, 1, 1, 1, 1, 2], [0, 0, 0, 0, 0, 4]),  // at 5
            ([1, 1, 1, 1, 1, 1], [4, 4, 4, 4, 4, 4]),  // none: r = 0, x[0] != a
            ([2, 1, 1, 1, 1, 1], [5, 4, 4, 4, 4, 4]),  // none, x[0] == a but y[0] != c
        ];
        for (x, y) in cases {
            let mut inputs: Vec<(String, i64)> = Vec::new();
            inputs.extend(x.iter().enumerate().map(|(k, &v)| (format!("x{k}"), v)));
            inputs.extend(y.iter().enumerate().map(|(k, &v)| (format!("y{k}"), v)));
            inputs.extend([("a".to_string(), 2), ("c".to_string(), 4)]);
            let (ok, got) = mock_run(rewritten, &inputs);
            assert!(ok);
            assert_eq!(got, mock_run(original, &inputs).1, "x {x:?} y {y:?}");
        }
    }
}
