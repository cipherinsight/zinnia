//! The framework: selection (exact against greedy, sharing, acyclic
//! justification) and the driver's discharge of checks.

use std::sync::Arc;

use crate::builder::IRBuilder;
use crate::circuit_input::InputPath;
use crate::ir::IRGraph;
use crate::ir_defs::IR;
use crate::types::{StmtId, Value};

use super::select::{select, select_exact, select_greedy, Node, Problem};
use super::{
    Bound, Candidate, Claim, Construction, Context, EmitCx, Emitter, Fact, Group, Lane, Requirement, Target,
};

// ── Selection ───────────────────────────────────────────────────────────

fn lane(s: StmtId) -> Requirement {
    Requirement::Lane(s)
}

fn shared(name: &'static str) -> Requirement {
    Requirement::Shared { group: 0, name }
}

fn node(pre: Vec<Requirement>, post: Vec<Requirement>, cost: u64) -> Node {
    Node { pre, post, cost }
}

/// Two lanes, each either produced alone (35) or by a cheap node (10) that
/// needs one shared value (40): sharing it (60) beats producing each lane
/// alone (70), which the greedy choice, pricing the shared value for each
/// lane, takes.
fn shared_graph() -> Problem {
    Problem {
        nodes: vec![
            node(vec![shared("segment")], vec![lane(1)], 10),
            node(vec![shared("segment")], vec![lane(2)], 10),
            node(vec![], vec![shared("segment")], 40),
            node(vec![], vec![lane(1)], 35),
            node(vec![], vec![lane(2)], 35),
        ],
        roots: vec![lane(1), lane(2)],
    }
}

/// `true` when `nodes` (in order) justify every root and every
/// precondition by an earlier node.
fn justified_in_order(problem: &Problem, nodes: &[usize]) -> bool {
    let produced_before = |i: usize, r: &Requirement| {
        nodes[..i].iter().any(|&m| problem.nodes[m].post.iter().any(|p| super::select::justifies(p, r)))
    };
    let roots = problem.roots.iter().all(|r| produced_before(nodes.len(), r));
    let pres = nodes.iter().enumerate().all(|(i, &n)| problem.nodes[n].pre.iter().all(|r| produced_before(i, r)));
    roots && pres
}

#[test]
fn exact_selection_pays_a_shared_node_once() {
    let problem = shared_graph();
    let exact = select_exact(&problem).expect("a selection");
    assert_eq!(exact.cost, 60);
    assert_eq!(exact.nodes.len(), 3);
    assert_eq!(exact.nodes[0], 2, "the shared value comes first");
    assert!(justified_in_order(&problem, &exact.nodes));
}

#[test]
fn greedy_selection_is_valid_but_not_optimal_on_a_shared_graph() {
    let problem = shared_graph();
    let greedy = select_greedy(&problem).expect("a selection");
    assert_eq!(greedy.cost, 70);
    assert!(justified_in_order(&problem, &greedy.nodes));
}

/// `X` by a node needing `Y`, `Y` by a node needing `X` (a cycle, cost 2)
/// or by an expensive node: only the acyclic justification is valid.
#[test]
fn justification_is_acyclic() {
    let problem = Problem {
        nodes: vec![
            node(vec![shared("y")], vec![shared("x")], 1),
            node(vec![shared("x")], vec![shared("y")], 1),
            node(vec![], vec![shared("y")], 100),
        ],
        roots: vec![shared("x")],
    };
    for selection in [select_exact(&problem), select_greedy(&problem)] {
        let selection = selection.expect("a selection");
        assert_eq!(selection.nodes, vec![2, 0]);
        assert_eq!(selection.cost, 101);
    }
}

#[test]
fn ties_go_to_the_earliest_node_and_unjustifiable_roots_have_no_selection() {
    let tie = Problem { nodes: vec![node(vec![], vec![lane(1)], 5), node(vec![], vec![lane(1)], 5)], roots: vec![lane(1)] };
    assert_eq!(select(&tie).expect("a selection").nodes, vec![0]);
    let missing = Problem { nodes: vec![node(vec![shared("never")], vec![lane(1)], 1)], roots: vec![lane(1)] };
    assert_eq!(select_exact(&missing), None);
    assert_eq!(select_greedy(&missing), None);
}

#[test]
fn a_fact_justifies_any_weaker_fact() {
    let narrow = Fact::everywhere(Claim::InRange { x: 7, lo: 0, hi: Bound::Const(4) });
    let wide = Fact::everywhere(Claim::InRange { x: 7, lo: 0, hi: Bound::Const(8) });
    assert!(narrow.implies(&wide));
    assert!(!wide.implies(&narrow));
    assert!(narrow.implies(&Fact::everywhere(Claim::NonNegative(7))));
    let guarded = Fact { claim: wide.claim, scope: super::Scope::When(3) };
    assert!(wide.implies(&guarded));
    assert!(!guarded.implies(&wide));
}

// ── Discharge through the driver ────────────────────────────────────────

/// A program with an index check `0 <= x < 4` and one sort lane.
struct Program {
    graph: IRGraph,
    x: StmtId,
}

fn program(with_check: bool) -> Program {
    let mut b = IRBuilder::new();
    let x = b.ir_read_integer(InputPath::new("x", vec![]), false);
    if with_check {
        let four = b.ir_constant_int(4);
        b.ir_assert_u128_lt(&x, &four);
    }
    let elements: Arc<[Value]> =
        (0..2).map(|j| b.ir_read_integer(InputPath::new(format!("a{j}"), vec![]), false)).collect();
    let out = b.ir_sorted(&elements, 0, false);
    b.ir_expose_public_i(&out);
    Program { graph: b.export_ir_graph(), x: x.stmt_id().expect("a statement") }
}

/// A test construction: the lane is the first element; optionally its
/// candidate proves `0 <= x < 4` or relies on it.
struct FirstElement {
    cost: u64,
    proves: Option<StmtId>,
    needs: Option<StmtId>,
}

struct FirstElementEmitter(StmtId);

impl Emitter for FirstElementEmitter {
    fn lane(&mut self, _b: &mut IRBuilder, cx: &mut EmitCx, _lane: &Lane) -> Value {
        cx.value(self.0)
    }
}

fn index_fact(x: StmtId) -> Requirement {
    Requirement::Fact(Fact::everywhere(Claim::InRange { x, lo: 0, hi: Bound::Const(4) }))
}

impl Construction for FirstElement {
    fn name(&self) -> &'static str {
        "test: first element"
    }

    fn candidates(&self, group: &Group, _cx: &Context) -> Vec<Candidate> {
        let mut post: Vec<Requirement> = group.lanes.iter().map(|l| Requirement::Lane(l.stmt)).collect();
        post.extend(self.proves.map(index_fact));
        let pre = self.needs.map(index_fact).into_iter().collect();
        vec![Candidate { pre, post, cost: self.cost, emitter: Box::new(FirstElementEmitter(group.shared[0])) }]
    }
}

fn index_checks(graph: &IRGraph) -> usize {
    graph.stmts.iter().filter(|s| matches!(s.ir, IR::AssertU128Lt)).count()
}

fn lower_with(program: &Program, constructions: &[&dyn Construction]) -> IRGraph {
    super::emit::lower(&program.graph, Target::Halo2Base, constructions).expect("a graph with a lane")
}

#[test]
fn a_check_a_selected_postcondition_proves_is_not_emitted() {
    let p = program(true);
    let proving = FirstElement { cost: 5, proves: Some(p.x), needs: None };
    let plain = FirstElement { cost: 1, proves: None, needs: None };
    // 5 (proves the check) < 1 + the check's cost (8): the check is dropped.
    let lowered = lower_with(&p, &[&plain, &proving]);
    assert_eq!(index_checks(&lowered), 0);
    assert!(!lowered.stmts.iter().any(|s| matches!(s.ir, IR::ArrayOp(_))));
}

#[test]
fn a_check_is_kept_when_proving_it_costs_more() {
    let p = program(true);
    let proving = FirstElement { cost: 20, proves: Some(p.x), needs: None };
    let plain = FirstElement { cost: 1, proves: None, needs: None };
    assert_eq!(index_checks(&lower_with(&p, &[&plain, &proving])), 1);
}

#[test]
fn an_unproven_precondition_is_emitted_as_a_check_once() {
    let p = program(false);
    let needing = FirstElement { cost: 1, proves: None, needs: Some(p.x) };
    assert_eq!(index_checks(&lower_with(&p, &[&needing])), 1);
    // The program's own check justifies the precondition: no second check.
    let p = program(true);
    let needing = FirstElement { cost: 1, proves: None, needs: Some(p.x) };
    assert_eq!(index_checks(&lower_with(&p, &[&needing])), 1);
}

/// Two bound checks of one index, `bounds` in statement order, then a sort.
fn checked_twice(bounds: [i64; 2]) -> IRGraph {
    let mut b = IRBuilder::new();
    let x = b.ir_read_integer(InputPath::new("x", vec![]), false);
    for hi in bounds {
        let hi = b.ir_constant_int(hi);
        b.ir_assert_u128_lt(&x, &hi);
    }
    let elements: Arc<[Value]> =
        (0..2).map(|j| b.ir_read_integer(InputPath::new(format!("a{j}"), vec![]), false)).collect();
    let out = b.ir_sorted(&elements, 0, false);
    b.ir_expose_public_i(&out);
    b.export_ir_graph()
}

#[test]
fn a_check_an_earlier_check_implies_is_not_emitted() {
    let lowered = |bounds| super::emit::lower(&checked_twice(bounds), Target::Halo2Base, &super::registry()).unwrap();
    assert_eq!(index_checks(&lowered([4, 8])), 1);
    // A weaker check first keeps both: dropping it would leave the
    // statements between the two without the bound it proves.
    assert_eq!(index_checks(&lowered([8, 4])), 2);
}

/// One side of a window is not the other: `x >= c` does not imply `x <= d`
/// (a field element above the window passes the first), while a range
/// `lo <= x < hi` implies both sides it covers.
#[test]
fn one_sided_bounds_imply_only_their_side() {
    let above = Fact::everywhere(Claim::Within { x: 7, lo: Some(i64::MIN), hi: None });
    let below = Fact::everywhere(Claim::Within { x: 7, lo: None, hi: Some(i64::MAX) });
    assert!(!above.implies(&below) && !below.implies(&above));
    let index = Fact::everywhere(Claim::InRange { x: 7, lo: -4, hi: Bound::Const(4) });
    assert!(index.implies(&above) && index.implies(&below));
    assert!(!index.implies(&Fact::everywhere(Claim::Within { x: 7, lo: Some(-3), hi: None })));
    assert!(!index.implies(&Fact::everywhere(Claim::Within { x: 7, lo: None, hi: Some(2) })));
}

/// NumPy's index wrap `w = select(s, x + n, x)` checked as `0 <= w < n`,
/// then the write window `-n <= x < n` checked as `AssertU128Lt(x + n, 2n)`,
/// then a sort lane. The sign `s` is a free input (a tampered hint),
/// asserted to be a bit when `bit` is set.
fn wrap_then_window(n: i64, bit: bool) -> IRGraph {
    let mut b = IRBuilder::new();
    let x = b.ir_read_integer(InputPath::new("x", vec![]), false);
    let s = b.ir_read_integer(InputPath::new("s", vec![]), false);
    if bit {
        b.ir_assert_bit(&s);
    }
    let len = b.ir_constant_int(n);
    let shifted = b.ir_add_i(&x, &len);
    let wrapped = b.ir_select_i(&s, &shifted, &x);
    b.ir_assert_u128_lt(&wrapped, &len);
    let twice = b.ir_constant_int(2 * n);
    let shifted_again = b.ir_add_i(&x, &len);
    b.ir_assert_u128_lt(&shifted_again, &twice);
    let elements: Arc<[Value]> =
        (0..2).map(|j| b.ir_read_integer(InputPath::new(format!("a{j}"), vec![]), false)).collect();
    let out = b.ir_sorted(&elements, 0, false);
    b.ir_expose_public_i(&out);
    b.export_ir_graph()
}

/// The wrap's check implies the write window (for `s` a bit: `x = w - n`
/// or `x = w`), so the window is not emitted; the circuit still accepts
/// exactly NumPy's indices `-n <= x < n` with the honest sign `[x < 0]`,
/// and no other `(x, s)`.
#[test]
fn the_index_wrap_check_implies_the_write_window() {
    use crate::circuit_input::{CircuitInputs, InputNode, InputParam, ResolvedWitness};
    use crate::prove::types::ProvingParams;
    let n = 3;
    let lower = |bit| super::emit::lower(&wrap_then_window(n, bit), Target::Halo2Base, &super::registry()).unwrap();
    assert_eq!(index_checks(&lower(true)), 1);
    // Without the bit assertion `s` may be any field element, and the
    // window is a separate fact.
    assert_eq!(index_checks(&lower(false)), 2);
    let lowered = lower(true);
    for x in -2 * n - 1..=2 * n {
        for s in [-1, 0, 1, 2] {
            let mut inputs = CircuitInputs::new();
            for (name, value) in [("x", x), ("s", s), ("a0", 5), ("a1", 2)] {
                inputs.params.push(InputParam {
                    name: name.to_string(),
                    is_public: false,
                    dtype: serde_json::Value::Null,
                    value: InputNode::Int(value),
                });
            }
            let witness = ResolvedWitness::new(inputs, ProvingParams::default().precision_bits);
            let mut synth = crate::prove::mock::synthesizer::MockSynthesizer::new(witness, ProvingParams::default());
            crate::prove::interpreter::interpret_ir(&lowered, &mut synth).expect("mock run");
            let honest = -n <= x && x < n && s == i64::from(x < 0);
            assert_eq!(synth.satisfied, honest, "x={x} s={s}");
        }
    }
}

// ── The program-wide challenge requirement ──────────────────────────────

/// `groups` independent sorts of two elements, each lane exposed.
fn independent_sorts(groups: usize) -> IRGraph {
    let mut b = IRBuilder::new();
    for g in 0..groups {
        let elements: Arc<[Value]> =
            (0..2).map(|j| b.ir_read_integer(InputPath::new(format!("a{g}_{j}"), vec![]), false)).collect();
        let out = b.ir_sorted(&elements, 0, false);
        b.ir_expose_public_i(&out);
    }
    b.export_ir_graph()
}

/// A test construction: the lane is element `element` of its group, at
/// `cost`, needing the verifier challenge when `challenged`.
struct ElementAt {
    element: usize,
    cost: u64,
    challenged: bool,
}

impl Construction for ElementAt {
    fn name(&self) -> &'static str {
        if self.challenged {
            "test: challenged"
        } else {
            "test: plain"
        }
    }

    fn candidates(&self, group: &Group, _cx: &Context) -> Vec<Candidate> {
        let pre = if self.challenged { vec![super::challenge::CHALLENGE] } else { Vec::new() };
        vec![Candidate {
            pre,
            post: group.lanes.iter().map(|l| Requirement::Lane(l.stmt)).collect(),
            cost: self.cost,
            emitter: Box::new(FirstElementEmitter(group.shared[self.element])),
        }]
    }
}

/// How many exposed values are the second element of their sort (the
/// challenged construction's choice).
fn challenged_groups(graph: &IRGraph) -> usize {
    graph
        .stmts
        .iter()
        .filter(|s| matches!(s.ir, IR::ExposePublicI))
        .filter(|s| match &graph.stmts[s.arguments[0] as usize].ir {
            IR::ReadInteger { path, .. } => path.param.ends_with("_1"),
            _ => false,
        })
        .count()
}

/// The challenge costs `C` once. A construction needing it saves `0.6 C`
/// per group: one group alone does not pay for it, two together do, and
/// then both use it (it is paid once). On Noir it has no producer.
#[test]
fn the_challenge_is_paid_once_when_the_groups_together_justify_it() {
    let c = super::challenge::HALO2_BASE_CHALLENGE_COST;
    let plain = ElementAt { element: 0, cost: c * 7 / 10, challenged: false };
    let challenged = ElementAt { element: 1, cost: c / 10, challenged: true };
    let lowered = |groups, target| super::emit::lower(&independent_sorts(groups), target, &[&plain, &challenged]).unwrap();
    assert_eq!(challenged_groups(&lowered(1, Target::Halo2Base)), 0, "one group alone pays the challenge");
    assert_eq!(challenged_groups(&lowered(2, Target::Halo2Base)), 2, "two groups do not share the challenge");
    assert_eq!(challenged_groups(&lowered(3, Target::Halo2Base)), 3);
    assert_eq!(challenged_groups(&lowered(3, Target::Noir)), 0, "a challenge on Noir");
}

/// A challenged construction that is the only producer of a lane cannot be
/// selected on Noir: the lowering has no selection there.
#[test]
fn a_challenged_construction_alone_has_no_selection_on_noir() {
    let challenged = ElementAt { element: 1, cost: 1, challenged: true };
    let graph = independent_sorts(1);
    assert_eq!(challenged_groups(&super::emit::lower(&graph, Target::Halo2Base, &[&challenged]).unwrap()), 1);
    let noir = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| super::emit::lower(&graph, Target::Noir, &[&challenged])));
    assert!(noir.is_err(), "selected without a challenge");
}
