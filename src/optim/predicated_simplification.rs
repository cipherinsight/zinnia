//! Simplify each branch of a select under the select's own condition.
//!
//! `select(p, x, y)` is `x` exactly where `p` holds and `y` where it does
//! not, so `x` may be replaced by any `x'` that equals `x` wherever `p`
//! holds (and `y` by any `y'` equal to it wherever `p` fails). The frontend
//! guards code that runs under a runtime condition this way
//! (`builder/guard.rs`: `select(p, v, safe)`), and code under a condition
//! often computes things the condition decides: in
//! `if 1 <= c <= 49: ys[argmax(xs > e) - 1]` the argmax is `select(c < 50,
//! c, 0)` and NumPy's negative-index wrap of `argmax - 1` is
//! `select(argmax - 1 < 0, argmax - 1 + n, argmax - 1)`; under `1 <= c <=
//! 49` both tests are decided and the index is `c - 1`.
//!
//! **The rewrite.** From `p` being true (or false) the pass reads
//! *assumptions*: the truth of `p` and of the conjuncts it implies through
//! `and`, `or` (when false) and `not`, and for each conjunct that compares
//! an integer statement with a constant, an interval for that statement.
//! It then walks the branch's operands (up to a depth and a node budget):
//! a comparison or logical statement whose value the assumptions and the
//! operands' intervals decide becomes that constant; a select whose
//! condition is decided becomes its chosen branch; an integer addition,
//! subtraction or multiplication whose operands changed is rebuilt on the
//! new operands. Everything else is kept as it is. Intervals combine the
//! range engine's bounds (`optim::range`, read off the original statements)
//! with the assumptions, through `+`, `-`, `*` and decided selects.
//!
//! **Soundness.** Every rewrite step preserves the statement's value on
//! every witness the circuit accepts on which the assumptions hold:
//! the assumptions are consequences of `p` having that truth value (a
//! select's condition is a Boolean, 0 or 1, on every accepted witness, as
//! every comparison and logical statement is exact in the IR's semantics),
//! a decided comparison has that value under them, and a decided select
//! takes its chosen branch. So `x'` equals `x` wherever `p` holds, and the
//! select's value is unchanged on every accepted witness. Apart from the
//! checks below, no assertion is added or dropped (only operands of selects
//! change); statements no longer read are removed by dead-code elimination
//! afterwards, which drops only computations, never checks.
//!
//! **Checks the guard decides.** An `AssertU128Lt(v, n)` (`0 <= v < n`)
//! with a constant `n`, whose operand is a select `select(p, x, y)` with
//! `x` in `[0, n)` under `p` and `y` in `[0, n)` under `not p` (intervals as
//! above), holds on every witness: it is dropped. This is the bound check of
//! a guarded index `select(p, i, 0)` whose guard bounds `i`. Sound for the
//! same reason: on every accepted witness `p` is 0 or 1, and either way the
//! select's value lies in `[0, n)`.
//!
//! **Cost.** Only decided selects and comparisons disappear; the rebuilt
//! statements are integer `+`, `-`, `*` on the simplified operands, which
//! replace the originals when nothing else reads them.

use std::collections::{HashMap, HashSet};

use crate::builder::IRBuilder;
use crate::ir::IRGraph;
use crate::ir_defs::IR;
use crate::optim::{RangeResolver, Resolver};
use crate::types::{ScalarValue, StmtId, Value};

use super::IRPass;

pub struct PredicatedSimplification;

/// How deep below a branch the walk goes, and how many statements it
/// visits per branch (keeps the pass linear in the graph's size).
const MAX_DEPTH: u32 = 12;
const NODE_BUDGET: usize = 256;

impl IRPass for PredicatedSimplification {
    fn exec(&self, graph: IRGraph) -> IRGraph {
        let mut base = BaseBounds { graph: &graph, range: RangeResolver::new(), cache: HashMap::new() };
        let mut b = IRBuilder::new();
        let mut lookup: HashMap<StmtId, Value> = HashMap::new();
        let mut changed = false;
        for stmt in &graph.stmts {
            if guard_decides_check(&graph, &mut base, stmt) {
                lookup.insert(stmt.stmt_id, Value::None);
                changed = true;
                continue;
            }
            let mut args: Vec<Value> = stmt.arguments.iter().map(|a| lookup[a].clone()).collect();
            if let (IR::SelectI | IR::SelectF | IR::SelectB, &[p, x, y]) = (&stmt.ir, stmt.arguments.as_slice()) {
                if !is_constant(&graph, p) {
                    for (slot, branch, truth) in [(1, x, true), (2, y, false)] {
                        let assumptions = Assumptions::of(&graph, p, truth);
                        if assumptions.bounds.is_empty() && assumptions.truths.len() == 1 && !reads_stmt(&graph, branch, p) {
                            continue;
                        }
                        let mut walk = Walk::new(&graph, &mut base, &assumptions, &lookup);
                        if let Some(simplified) = walk.rewrite(&mut b, branch, 0) {
                            args[slot] = simplified;
                            changed = true;
                        }
                    }
                }
            }
            let value = b.create_ir(&stmt.ir, &args);
            lookup.insert(stmt.stmt_id, value);
        }
        if !changed {
            return graph;
        }
        let mut out = b.export_ir_graph();
        out.resolver_mut().on_ir_mutated(&[]);
        out
    }
}

/// Whether `stmt` is `AssertU128Lt(select(p, x, y), n)` with a constant `n`
/// and both branches in `[0, n)` under the condition's truth that picks
/// them.
fn guard_decides_check<'a>(graph: &'a IRGraph, base: &mut BaseBounds<'a>, stmt: &crate::ir::IRStatement) -> bool {
    let (IR::AssertU128Lt, &[v, hi]) = (&stmt.ir, stmt.arguments.as_slice()) else { return false };
    let Some(n) = constant_of(graph, hi) else { return false };
    let select = &graph.stmts[v as usize];
    let (IR::SelectI, &[p, x, y]) = (&select.ir, select.arguments.as_slice()) else { return false };
    let no_values = HashMap::new();
    [(x, true), (y, false)].into_iter().all(|(branch, truth)| {
        let assumptions = Assumptions::of(graph, p, truth);
        let mut walk = Walk::new(graph, base, &assumptions, &no_values);
        walk.interval(branch, 0).is_some_and(|(lo, hi)| 0 <= lo && hi < n)
    })
}

fn is_constant(graph: &IRGraph, s: StmtId) -> bool {
    matches!(graph.stmts[s as usize].ir, IR::ConstantInt { .. } | IR::ConstantBool { .. })
}

/// Whether `branch` is `p` or reads it directly (a nested guard on the same
/// condition, which `p`'s truth alone decides).
fn reads_stmt(graph: &IRGraph, branch: StmtId, p: StmtId) -> bool {
    branch == p || graph.stmts[branch as usize].arguments.contains(&p)
}

fn constant_of(graph: &IRGraph, s: StmtId) -> Option<i64> {
    match graph.stmts[s as usize].ir {
        IR::ConstantInt { value } => Some(value),
        IR::ConstantBool { value } => Some(i64::from(value)),
        _ => None,
    }
}

/// An integer interval `[lo, hi]`.
type Interval = (i64, i64);

fn intersect(a: Option<Interval>, b: Option<Interval>) -> Option<Interval> {
    match (a, b) {
        (Some((l1, h1)), Some((l2, h2))) => Some((l1.max(l2), h1.min(h2))),
        (a, None) => a,
        (None, b) => b,
    }
}

/// The range engine's bounds of the original statements, cached.
struct BaseBounds<'a> {
    graph: &'a IRGraph,
    range: RangeResolver,
    cache: HashMap<StmtId, Option<Interval>>,
}

impl BaseBounds<'_> {
    fn of(&mut self, s: StmtId) -> Option<Interval> {
        if let Some(cached) = self.cache.get(&s) {
            return *cached;
        }
        let value = Value::Integer(ScalarValue::new(None, Some(s)));
        let min = self.range.resolve_min_with_stmts(&value, &self.graph.stmts);
        let max = self.range.resolve_max_with_stmts(&value, &self.graph.stmts);
        let bounds = min.zip(max).filter(|(l, h)| l <= h);
        self.cache.insert(s, bounds);
        bounds
    }
}

/// What a select's condition having one truth value implies.
#[derive(Default)]
struct Assumptions {
    /// Boolean statements with a known value.
    truths: HashMap<StmtId, bool>,
    /// Intervals of integer statements compared with constants.
    bounds: HashMap<StmtId, Interval>,
}

impl Assumptions {
    fn of(graph: &IRGraph, p: StmtId, truth: bool) -> Assumptions {
        let mut out = Assumptions::default();
        out.assume(graph, p, truth, 0);
        out
    }

    fn assume(&mut self, graph: &IRGraph, c: StmtId, truth: bool, depth: u32) {
        if depth > MAX_DEPTH || self.truths.insert(c, truth).is_some() {
            return;
        }
        let stmt = &graph.stmts[c as usize];
        match (&stmt.ir, stmt.arguments.as_slice(), truth) {
            (IR::LogicalAnd, &[a, b], true) | (IR::LogicalOr, &[a, b], false) => {
                self.assume(graph, a, truth, depth + 1);
                self.assume(graph, b, truth, depth + 1);
            }
            (IR::LogicalNot, &[a], _) => self.assume(graph, a, !truth, depth + 1),
            _ => {
                if let Some((x, interval)) = comparison_bound(graph, c, truth) {
                    let narrowed = intersect(self.bounds.get(&x).copied(), Some(interval)).expect("an interval");
                    self.bounds.insert(x, narrowed);
                }
            }
        }
    }
}

/// An integer comparison of a statement with a constant.
#[derive(Clone, Copy)]
enum Order {
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Ne,
}

impl Order {
    fn of(ir: &IR) -> Option<Order> {
        Some(match ir {
            IR::LtI | IR::LtIBounded => Order::Lt,
            IR::LteI => Order::Le,
            IR::GtI | IR::GtIBounded => Order::Gt,
            IR::GteI => Order::Ge,
            IR::EqI => Order::Eq,
            IR::NeI => Order::Ne,
            _ => return None,
        })
    }

    /// The order with its operands swapped (`c < x` is `x > c`).
    fn swapped(self) -> Order {
        match self {
            Order::Lt => Order::Gt,
            Order::Le => Order::Ge,
            Order::Gt => Order::Lt,
            Order::Ge => Order::Le,
            other => other,
        }
    }

    /// The order of the negation.
    fn negated(self) -> Order {
        match self {
            Order::Lt => Order::Ge,
            Order::Le => Order::Gt,
            Order::Gt => Order::Le,
            Order::Ge => Order::Lt,
            Order::Eq => Order::Ne,
            Order::Ne => Order::Eq,
        }
    }

    /// `x op c` as an interval of `x` (`None` for `!=`).
    fn interval(self, c: i64) -> Option<Interval> {
        Some(match self {
            Order::Lt => (i64::MIN, c.checked_sub(1)?),
            Order::Le => (i64::MIN, c),
            Order::Gt => (c.checked_add(1)?, i64::MAX),
            Order::Ge => (c, i64::MAX),
            Order::Eq => (c, c),
            Order::Ne => return None,
        })
    }

    /// `x op y` when the intervals decide it.
    fn decide(self, (xl, xh): Interval, (yl, yh): Interval) -> Option<bool> {
        let (always, never) = match self {
            Order::Lt => (xh < yl, xl >= yh),
            Order::Le => (xh <= yl, xl > yh),
            Order::Gt => (xl > yh, xh <= yl),
            Order::Ge => (xl >= yh, xh < yl),
            Order::Eq => (xl == xh && yl == yh && xl == yl, xh < yl || xl > yh),
            Order::Ne => (xh < yl || xl > yh, xl == xh && yl == yh && xl == yl),
        };
        if always {
            Some(true)
        } else if never {
            Some(false)
        } else {
            None
        }
    }
}

/// `(x, interval)` when `c` compares the integer statement `x` with a
/// constant and `c == truth` confines `x` to `interval`.
fn comparison_bound(graph: &IRGraph, c: StmtId, truth: bool) -> Option<(StmtId, Interval)> {
    let stmt = &graph.stmts[c as usize];
    let order = Order::of(&stmt.ir)?;
    let &[a, b] = stmt.arguments.as_slice() else { return None };
    let (x, order, k) = match (constant_of(graph, a), constant_of(graph, b)) {
        (None, Some(k)) => (a, order, k),
        (Some(k), None) => (b, order.swapped(), k),
        _ => return None,
    };
    let order = if truth { order } else { order.negated() };
    Some((x, order.interval(k)?))
}

/// One branch's walk under its assumptions.
struct Walk<'a, 'b> {
    graph: &'a IRGraph,
    base: &'b mut BaseBounds<'a>,
    assumptions: &'b Assumptions,
    lookup: &'b HashMap<StmtId, Value>,
    intervals: HashMap<StmtId, Option<Interval>>,
    decided: HashMap<StmtId, Option<bool>>,
    rewritten: HashMap<StmtId, Option<Value>>,
    visited: HashSet<StmtId>,
}

impl<'a, 'b> Walk<'a, 'b> {
    fn new(
        graph: &'a IRGraph,
        base: &'b mut BaseBounds<'a>,
        assumptions: &'b Assumptions,
        lookup: &'b HashMap<StmtId, Value>,
    ) -> Self {
        Walk {
            graph,
            base,
            assumptions,
            lookup,
            intervals: HashMap::new(),
            decided: HashMap::new(),
            rewritten: HashMap::new(),
            visited: HashSet::new(),
        }
    }

    fn within_budget(&mut self, s: StmtId, depth: u32) -> bool {
        self.visited.insert(s);
        depth <= MAX_DEPTH && self.visited.len() <= NODE_BUDGET
    }

    /// `s`'s interval under the assumptions.
    fn interval(&mut self, s: StmtId, depth: u32) -> Option<Interval> {
        if let Some(cached) = self.intervals.get(&s) {
            return *cached;
        }
        let mut out = intersect(self.base.of(s), self.assumptions.bounds.get(&s).copied());
        if self.within_budget(s, depth) {
            let stmt = &self.graph.stmts[s as usize];
            let derived = match (&stmt.ir, stmt.arguments.as_slice()) {
                (IR::ConstantInt { value }, _) => Some((*value, *value)),
                (IR::AddI, &[a, b]) => self.combine(a, b, depth, |x, y| x.checked_add(y)),
                (IR::SubI, &[a, b]) => {
                    let (x, y) = (self.interval(a, depth + 1), self.interval(b, depth + 1));
                    x.zip(y).and_then(|((xl, xh), (yl, yh))| Some((xl.checked_sub(yh)?, xh.checked_sub(yl)?)))
                }
                (IR::MulI, &[a, b]) => self.combine(a, b, depth, |x, y| x.checked_mul(y)),
                (IR::SelectI, &[c, t, e]) => match self.decide(c, depth + 1) {
                    Some(true) => self.interval(t, depth + 1),
                    Some(false) => self.interval(e, depth + 1),
                    None => None,
                },
                _ => None,
            };
            out = intersect(out, derived);
        }
        let out = out.filter(|(l, h)| l <= h);
        self.intervals.insert(s, out);
        out
    }

    /// `op` over the corners of the operands' intervals (exact for `+` and
    /// for `*`, whose extremes over a box are at its corners).
    fn combine(&mut self, a: StmtId, b: StmtId, depth: u32, op: fn(i64, i64) -> Option<i64>) -> Option<Interval> {
        let ((xl, xh), (yl, yh)) = (self.interval(a, depth + 1)?, self.interval(b, depth + 1)?);
        let corners = [op(xl, yl)?, op(xl, yh)?, op(xh, yl)?, op(xh, yh)?];
        Some((*corners.iter().min()?, *corners.iter().max()?))
    }

    /// The value of the Boolean statement `c` under the assumptions, when
    /// they decide it.
    fn decide(&mut self, c: StmtId, depth: u32) -> Option<bool> {
        if let Some(cached) = self.decided.get(&c) {
            return *cached;
        }
        let mut out = self.assumptions.truths.get(&c).copied();
        if out.is_none() && self.within_budget(c, depth) {
            let stmt = &self.graph.stmts[c as usize];
            out = match (&stmt.ir, stmt.arguments.as_slice()) {
                (IR::ConstantBool { value }, _) => Some(*value),
                (IR::LogicalNot, &[a]) => self.decide(a, depth + 1).map(|v| !v),
                (IR::LogicalAnd, &[a, b]) => match (self.decide(a, depth + 1), self.decide(b, depth + 1)) {
                    (Some(false), _) | (_, Some(false)) => Some(false),
                    (Some(true), Some(true)) => Some(true),
                    _ => None,
                },
                (IR::LogicalOr, &[a, b]) => match (self.decide(a, depth + 1), self.decide(b, depth + 1)) {
                    (Some(true), _) | (_, Some(true)) => Some(true),
                    (Some(false), Some(false)) => Some(false),
                    _ => None,
                },
                (ir, &[a, b]) => match Order::of(ir) {
                    Some(order) => {
                        let (x, y) = (self.interval(a, depth + 1), self.interval(b, depth + 1));
                        x.zip(y).and_then(|(x, y)| order.decide(x, y))
                    }
                    None => None,
                },
                _ => None,
            };
        }
        self.decided.insert(c, out);
        out
    }

    /// `s` simplified under the assumptions, or `None` when nothing in it
    /// changes. New statements are emitted into `b`.
    fn rewrite(&mut self, b: &mut IRBuilder, s: StmtId, depth: u32) -> Option<Value> {
        if let Some(cached) = self.rewritten.get(&s) {
            return cached.clone();
        }
        let out = if depth > MAX_DEPTH {
            None
        } else {
            self.rewrite_uncached(b, s, depth)
        };
        self.rewritten.insert(s, out.clone());
        out
    }

    fn rewrite_uncached(&mut self, b: &mut IRBuilder, s: StmtId, depth: u32) -> Option<Value> {
        let stmt = &self.graph.stmts[s as usize];
        match (&stmt.ir, stmt.arguments.as_slice()) {
            (IR::SelectI | IR::SelectF | IR::SelectB, &[c, t, e]) => {
                let chosen = match self.decide(c, depth + 1)? {
                    true => t,
                    false => e,
                };
                Some(self.rewrite(b, chosen, depth + 1).unwrap_or_else(|| self.lookup[&chosen].clone()))
            }
            (IR::AddI | IR::SubI | IR::MulI, &[x, y]) => {
                let (nx, ny) = (self.rewrite(b, x, depth + 1), self.rewrite(b, y, depth + 1));
                if nx.is_none() && ny.is_none() {
                    return None;
                }
                let x = nx.unwrap_or_else(|| self.lookup[&x].clone());
                let y = ny.unwrap_or_else(|| self.lookup[&y].clone());
                Some(b.create_ir(&stmt.ir, &[x, y]))
            }
            (ir, _) if is_boolean_test(ir) => {
                let value = self.decide(s, depth)?;
                Some(b.ir_constant_bool(value))
            }
            _ => None,
        }
    }
}

/// Integer comparisons and logical connectives: statements `decide` reads.
fn is_boolean_test(ir: &IR) -> bool {
    Order::of(ir).is_some() || matches!(ir, IR::LogicalAnd | IR::LogicalOr | IR::LogicalNot)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circuit_input::{CircuitInputs, InputNode, InputParam, InputPath, ResolvedWitness};
    use crate::prove::types::ProvingParams;

    /// The mock's verdict and public outputs on `inputs`.
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
        let mut synth = crate::prove::mock::synthesizer::MockSynthesizer::new(witness, ProvingParams::default());
        crate::prove::interpreter::interpret_ir(graph, &mut synth).expect("mock run");
        let mut outputs: Vec<_> = synth.public_outputs.clone().into_iter().collect();
        outputs.sort();
        (synth.satisfied, outputs.into_iter().map(|(_, v)| v).collect())
    }

    fn simplified(graph: &IRGraph) -> IRGraph {
        crate::optim::DeadCodeElimination.exec(PredicatedSimplification.exec(graph.clone()))
    }

    fn count(graph: &IRGraph, f: fn(&IR) -> bool) -> usize {
        graph.stmts.iter().filter(|s| f(&s.ir)).count()
    }

    /// 0374's lookup: under `1 <= c <= n - 1`, `ys[argmax - 1]` with
    /// `argmax = select(c < n, c, 0)` and NumPy's wrap of `argmax - 1`,
    /// guarded by the branch's condition and checked into `[0, check)`.
    fn guarded_wrap_checked_against(n: i64, check: i64) -> IRGraph {
        let mut b = IRBuilder::new();
        let c = b.ir_read_integer(InputPath::new("c", vec![]), false);
        let (zero, one, len, last) =
            (b.ir_constant_int(0), b.ir_constant_int(1), b.ir_constant_int(n), b.ir_constant_int(n - 1));
        let at_most_zero = b.ir_less_than_or_equal_i(&c, &zero);
        let positive = b.ir_logical_not(&at_most_zero);
        let below_last = b.ir_less_than_or_equal_i(&c, &last);
        let p = b.ir_logical_and(&below_last, &positive);
        let some = b.ir_less_than_i(&c, &len);
        let argmax = b.ir_select_i(&some, &c, &zero);
        let index = b.ir_sub_i(&argmax, &one);
        let negative = b.ir_less_than_i(&index, &zero);
        let shifted = b.ir_add_i(&index, &len);
        let wrapped = b.ir_select_i(&negative, &shifted, &index);
        let guarded = b.ir_select_i(&p, &wrapped, &zero);
        let bound = b.ir_constant_int(check);
        b.ir_assert_u128_lt(&guarded, &bound);
        b.ir_expose_public_i(&guarded);
        b.export_ir_graph()
    }

    fn guarded_wrap(n: i64) -> IRGraph {
        guarded_wrap_checked_against(n, n)
    }

    /// `guarded_wrap(50)` with the guarded index checked against `check`.
    fn guarded_wrap_checked(check: i64) -> IRGraph {
        guarded_wrap_checked_against(50, check)
    }

    #[test]
    fn a_guarded_index_wrap_that_its_guard_decides_is_removed() {
        let graph = guarded_wrap(50);
        let out = simplified(&graph);
        // Both tests inside the branch are decided by `1 <= c <= 49`.
        assert_eq!(count(&out, |ir| matches!(ir, IR::LtI)), 0);
        assert_eq!(count(&out, |ir| matches!(ir, IR::SelectI)), 1);
        // The guarded index's bound `0 <= select(p, c - 1, 0) < 50` holds
        // on both sides of the guard: no check left.
        assert_eq!(count(&out, |ir| matches!(ir, IR::AssertU128Lt)), 0);
        for c in -3..=53 {
            assert_eq!(run(&out, &[("c", c)]), run(&graph, &[("c", c)]), "c={c}");
        }
        // A bound the guard does not imply stays: `c - 1 < 48` fails at 49.
        let narrower = simplified(&guarded_wrap_checked(48));
        assert_eq!(count(&narrower, |ir| matches!(ir, IR::AssertU128Lt)), 1);
        assert!(!run(&narrower, &[("c", 49)]).0);
        assert!(run(&narrower, &[("c", 48)]).0);
    }

    /// `select(c > k, a, select(c > k, b, d))` and the same with an
    /// undecided inner test `c > k + 1`.
    fn nested(inner_offset: i64) -> IRGraph {
        let mut b = IRBuilder::new();
        let c = b.ir_read_integer(InputPath::new("c", vec![]), false);
        let [x, y, z] = ["a", "b", "d"].map(|n| b.ir_read_integer(InputPath::new(n, vec![]), false));
        let k = b.ir_constant_int(3);
        let k2 = b.ir_constant_int(3 + inner_offset);
        let outer = b.ir_greater_than_i(&c, &k);
        let inner = b.ir_greater_than_i(&c, &k2);
        let else_branch = b.ir_select_i(&inner, &y, &z);
        let out = b.ir_select_i(&outer, &x, &else_branch);
        b.ir_expose_public_i(&out);
        b.export_ir_graph()
    }

    #[test]
    fn the_else_branch_is_simplified_under_the_negated_condition() {
        let graph = nested(0);
        let out = simplified(&graph);
        assert_eq!(count(&out, |ir| matches!(ir, IR::SelectI)), 1);
        // `c > 4` is not decided by `c <= 3`... but it is: `c <= 3` makes it
        // false. `c > 2` is not decided by `c <= 3`.
        let decided = simplified(&nested(1));
        assert_eq!(count(&decided, |ir| matches!(ir, IR::SelectI)), 1);
        let undecided = simplified(&nested(-1));
        assert_eq!(count(&undecided, |ir| matches!(ir, IR::SelectI)), 2);
        for g in [nested(0), nested(1), nested(-1)] {
            let s = simplified(&g);
            for c in -1..=8 {
                let inputs = [("c", c), ("a", 10), ("b", 20), ("d", 30)];
                assert_eq!(run(&s, &inputs), run(&g, &inputs), "c={c}");
            }
        }
    }
}
