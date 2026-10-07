//! The framework: selection (exact against greedy, sharing, acyclic
//! justification) and the driver's discharge of checks.

use std::any::Any;
use std::rc::Rc;
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

/// Four lanes, each by a plain node (8) or by one that also provides a
/// shared value the program needs once (10): any one lane can provide it, so
/// cost and sum of positions tie four ways (the shape of DS-1000 case 0345,
/// whose circuit changed between runs with Z3's arbitrary pick). The rule
/// picks the set that leaves out the highest-numbered nodes: the first lane
/// provides the value.
#[test]
fn equal_cost_and_position_ties_have_one_answer() {
    let mut nodes = Vec::new();
    for l in 1..=4 {
        nodes.push(node(vec![], vec![lane(l)], 8));
        nodes.push(node(vec![], vec![lane(l), shared("range")], 10));
    }
    let problem = Problem { nodes, roots: vec![lane(1), lane(2), lane(3), lane(4), shared("range")] };
    let mut chosen = select_exact(&problem).expect("a selection").nodes;
    chosen.sort_unstable();
    assert_eq!(chosen, vec![1, 2, 4, 6]);
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

/// Claims' value lists are numbered per lowering: an id does not depend on
/// what was interned before the lowering started.
#[test]
fn list_ids_are_numbered_per_lowering() {
    use super::graph::{ListEntry, ListId};
    let first = |earlier: usize| {
        for k in 0..earlier {
            ListId::intern(&[ListEntry::Int(k as i64 + 100)]);
        }
        let graph = independent_sorts(1);
        super::emit::lower(&graph, Target::Halo2Base, &super::registry()).expect("a lane");
        ListId::range(3)
    };
    assert_eq!(first(0), first(5));
}

// ── Local selection (measurement only, `ZINNIA_ABLATE_JOINT`) ───────────

fn joint() -> super::emit::Options {
    super::emit::Options::default()
}

fn local() -> super::emit::Options {
    super::emit::Options { local: true, ..Default::default() }
}

fn lower_opts(
    graph: &IRGraph,
    target: Target,
    constructions: &[&dyn Construction],
    options: &super::emit::Options,
) -> IRGraph {
    super::emit::lower_with(graph, target, constructions, options).expect("lanes to lower")
}

/// The challenge costs `C` once. Alone, a group does not pay it for a saving
/// of `0.6 C`; together, two or three groups would, and the compiler takes
/// it. Deciding each group on its own, no group takes it. A saving larger
/// than `C` makes every group take it alone (and the challenge is still one
/// requirement of the program).
#[test]
fn locally_each_group_pays_the_challenge_alone() {
    let c = super::challenge::HALO2_BASE_CHALLENGE_COST;
    let plain = ElementAt { element: 0, cost: c * 7 / 10, challenged: false };
    let challenged = ElementAt { element: 1, cost: c / 10, challenged: true };
    let lowered = |groups, options: &super::emit::Options| {
        lower_opts(&independent_sorts(groups), Target::Halo2Base, &[&plain, &challenged], options)
    };
    assert_eq!(challenged_groups(&lowered(3, &joint())), 3);
    assert_eq!(challenged_groups(&lowered(3, &local())), 0, "a group alone does not pay the challenge");
    let costly = ElementAt { element: 0, cost: c * 2, challenged: false };
    let graph = independent_sorts(3);
    assert_eq!(challenged_groups(&lower_opts(&graph, Target::Halo2Base, &[&costly, &challenged], &local())), 3);
}

/// A test construction for one group (`on`): its lane is element `element`,
/// at `cost`; it may need or produce the shared value `INDEX`'s statement
/// value `SHARED` (a `SharedAt` of `at`), produce it without a lane
/// (`producer`), or prove `0 <= x < 4` for `x = at`. Equal `name`s make
/// candidates variants of one construction.
struct OnGroup {
    name: &'static str,
    on: usize,
    element: usize,
    cost: u64,
    needs_shared: bool,
    producer: bool,
    proves: bool,
    needs_fact: bool,
    at: StmtId,
}

const SHARED: &str = "test: shared";

impl OnGroup {
    fn new(name: &'static str, on: usize, element: usize, cost: u64, at: StmtId) -> OnGroup {
        OnGroup { name, on, element, cost, needs_shared: false, producer: false, proves: false, needs_fact: false, at }
    }
}

struct SharedProducerEmitter;

impl Emitter for SharedProducerEmitter {
    fn lane(&mut self, _b: &mut IRBuilder, _cx: &mut EmitCx, _lane: &Lane) -> Value {
        unreachable!("no lane")
    }

    fn shared_at(&mut self, _b: &mut IRBuilder, _cx: &mut EmitCx, _at: StmtId, _name: &'static str) -> Rc<dyn Any> {
        Rc::new(())
    }
}

/// The lane is element `.0`; the shared value is read first when `.1`.
struct ElementReadingShared(StmtId, Option<StmtId>);

impl Emitter for ElementReadingShared {
    fn lane(&mut self, b: &mut IRBuilder, cx: &mut EmitCx, _lane: &Lane) -> Value {
        if let Some(at) = self.1 {
            cx.shared_at::<()>(b, at, SHARED);
        }
        cx.value(self.0)
    }
}

impl Construction for OnGroup {
    fn name(&self) -> &'static str {
        self.name
    }

    fn candidates(&self, group: &Group, _cx: &Context) -> Vec<Candidate> {
        if group.id != self.on {
            return Vec::new();
        }
        let shared = Requirement::SharedAt { at: self.at, name: SHARED };
        if self.producer {
            return vec![Candidate { pre: Vec::new(), post: vec![shared], cost: self.cost, emitter: Box::new(SharedProducerEmitter) }];
        }
        let mut pre = Vec::new();
        let mut post: Vec<Requirement> = group.lanes.iter().map(|l| Requirement::Lane(l.stmt)).collect();
        if self.needs_shared {
            pre.push(shared);
        }
        if self.needs_fact {
            pre.push(index_fact(self.at));
        }
        if self.proves {
            post.push(index_fact(self.at));
        }
        let reads = self.needs_shared.then_some(self.at);
        vec![Candidate { pre, post, cost: self.cost, emitter: Box::new(ElementReadingShared(group.shared[self.element], reads)) }]
    }
}

/// `x` read first (checked `0 <= x < 4` when `checked`), then `groups`
/// independent sorts of two elements.
fn sorts_after_x(groups: usize, checked: bool) -> (IRGraph, StmtId) {
    let mut b = IRBuilder::new();
    let x = b.ir_read_integer(InputPath::new("x", vec![]), false);
    if checked {
        let four = b.ir_constant_int(4);
        b.ir_assert_u128_lt(&x, &four);
    }
    for g in 0..groups {
        let elements: Arc<[Value]> =
            (0..2).map(|j| b.ir_read_integer(InputPath::new(format!("a{g}_{j}"), vec![]), false)).collect();
        let out = b.ir_sorted(&elements, 0, false);
        b.ir_expose_public_i(&out);
    }
    (b.export_ir_graph(), x.stmt_id().expect("a statement"))
}

/// Two groups, each reading a shared value of `x` (built by either group's
/// lane-less producer, 40) for 10, or not at all for 35. Together the value
/// pays (60 < 70); alone it does not (50 > 35).
#[test]
fn locally_a_shared_value_is_paid_by_each_group() {
    let (graph, x) = sorts_after_x(2, false);
    let mut constructions: Vec<OnGroup> = Vec::new();
    for g in 0..2 {
        constructions.push(OnGroup { needs_shared: true, ..OnGroup::new("test: reads shared", g, 1, 10, x) });
        constructions.push(OnGroup { producer: true, ..OnGroup::new("test: shared producer", g, 0, 40, x) });
        constructions.push(OnGroup::new("test: alone", g, 0, 35, x));
    }
    let all: Vec<&dyn Construction> = constructions.iter().map(|c| c as &dyn Construction).collect();
    assert_eq!(challenged_groups(&lower_opts(&graph, Target::Halo2Base, &all, &joint())), 2, "the shared value pays together");
    assert_eq!(challenged_groups(&lower_opts(&graph, Target::Halo2Base, &all, &local())), 0, "alone it does not");
}

/// The program checks `0 <= x < 4` (8). Each of two groups lowers plainly
/// (4, element 0) or by a node that also proves the check (10, element 1).
/// Together one proving node pays (10 + 4 < 4 + 4 + 8) and drops the check
/// for both. Alone, each group counts the check against its own plain node
/// (4 + 8 > 10), so both take the proving node; the check is dropped once.
#[test]
fn locally_each_group_counts_a_program_check_it_could_discharge() {
    let (graph, x) = sorts_after_x(2, true);
    let mut constructions: Vec<OnGroup> = Vec::new();
    for g in 0..2 {
        constructions.push(OnGroup::new("test: plain", g, 0, 4, x));
        constructions.push(OnGroup { proves: true, ..OnGroup::new("test: proving", g, 1, 10, x) });
    }
    let all: Vec<&dyn Construction> = constructions.iter().map(|c| c as &dyn Construction).collect();
    let joint_lowered = lower_opts(&graph, Target::Halo2Base, &all, &joint());
    assert_eq!((challenged_groups(&joint_lowered), index_checks(&joint_lowered)), (1, 0));
    let local_lowered = lower_opts(&graph, Target::Halo2Base, &all, &local());
    assert_eq!((challenged_groups(&local_lowered), index_checks(&local_lowered)), (2, 0));
}

/// Group 0 offers a plain node (1) and one that also exposes a shared value
/// (3); group 1 reads that value (1) or lowers alone (10). When the exposing
/// node is a variant of the plain one (the same construction), group 1 takes
/// it at its full price (4 < 10) and it replaces the plain node; when it is
/// another construction, group 0's decision stands and group 1 lowers alone.
#[test]
fn locally_a_later_group_may_only_extend_an_earlier_decision() {
    struct Exposing(&'static str, StmtId);
    impl Construction for Exposing {
        fn name(&self) -> &'static str {
            self.0
        }
        fn candidates(&self, group: &Group, _cx: &Context) -> Vec<Candidate> {
            if group.id != 0 {
                return Vec::new();
            }
            let mut post: Vec<Requirement> = group.lanes.iter().map(|l| Requirement::Lane(l.stmt)).collect();
            post.push(Requirement::SharedAt { at: self.1, name: SHARED });
            vec![Candidate { pre: Vec::new(), post, cost: 3, emitter: Box::new(ExposingEmitter(group.shared[1])) }]
        }
    }
    struct ExposingEmitter(StmtId);
    impl Emitter for ExposingEmitter {
        fn lane(&mut self, _b: &mut IRBuilder, cx: &mut EmitCx, _lane: &Lane) -> Value {
            cx.value(self.0)
        }
        fn shared_at(&mut self, _b: &mut IRBuilder, _cx: &mut EmitCx, _at: StmtId, _name: &'static str) -> Rc<dyn Any> {
            Rc::new(())
        }
    }
    let (graph, x) = sorts_after_x(2, false);
    let plain = OnGroup::new("test: group 0", 0, 0, 1, x);
    let reading = OnGroup { needs_shared: true, ..OnGroup::new("test: group 1", 1, 1, 1, x) };
    let alone = OnGroup::new("test: group 1 alone", 1, 0, 10, x);
    let variant = Exposing("test: group 0", x);
    let other = Exposing("test: group 0 exposing", x);
    let with_variant: [&dyn Construction; 4] = [&plain, &variant, &reading, &alone];
    assert_eq!(challenged_groups(&lower_opts(&graph, Target::Halo2Base, &with_variant, &local())), 2);
    let with_other: [&dyn Construction; 4] = [&plain, &other, &reading, &alone];
    assert_eq!(challenged_groups(&lower_opts(&graph, Target::Halo2Base, &with_other, &joint())), 2);
    assert_eq!(challenged_groups(&lower_opts(&graph, Target::Halo2Base, &with_other, &local())), 0);
}

/// A node offered for group 0 that also produces group 1's lane (as the RAM
/// trace of a written array produces its reads), for `cost`, against each
/// group's own node (4). Together it pays from 8 down; group 0 alone takes it
/// only below 4, and then it decides group 1 too.
#[test]
fn locally_a_node_producing_two_groups_lanes_decides_both() {
    struct Both(u64);
    impl Construction for Both {
        fn name(&self) -> &'static str {
            "test: both groups"
        }
        fn candidates(&self, group: &Group, cx: &Context) -> Vec<Candidate> {
            if group.id != 0 {
                return Vec::new();
            }
            let second: std::collections::HashMap<StmtId, StmtId> =
                cx.groups.iter().flat_map(|g| g.lanes.iter().map(move |l| (l.stmt, g.shared[1]))).collect();
            let post = second.keys().map(|&s| Requirement::Lane(s)).collect();
            vec![Candidate { pre: Vec::new(), post, cost: self.0, emitter: Box::new(SecondEmitter(second)) }]
        }
    }
    struct SecondEmitter(std::collections::HashMap<StmtId, StmtId>);
    impl Emitter for SecondEmitter {
        fn lane(&mut self, _b: &mut IRBuilder, cx: &mut EmitCx, lane: &Lane) -> Value {
            cx.value(self.0[&lane.stmt])
        }
    }
    let (graph, x) = sorts_after_x(2, false);
    let own: Vec<OnGroup> = (0..2).map(|g| OnGroup::new("test: own", g, 0, 4, x)).collect();
    for (cost, joint_groups, local_groups) in [(5, 2, 0), (3, 2, 2)] {
        let both = Both(cost);
        let all: [&dyn Construction; 3] = [&own[0], &own[1], &both];
        assert_eq!(challenged_groups(&lower_opts(&graph, Target::Halo2Base, &all, &joint())), joint_groups);
        assert_eq!(challenged_groups(&lower_opts(&graph, Target::Halo2Base, &all, &local())), local_groups, "cost {cost}");
    }
}

// ── Fixed constructions (measurement only, `ZINNIA_FORCE_CONSTRUCTION`) ─

fn forced(names: &[&str]) -> super::emit::Options {
    super::emit::Options { forcing: super::forcing::Forcing::of(names), ..Default::default() }
}

/// A forced construction is used even where it costs more; one that cannot
/// be selected (a challenge on Noir) is released for its group, which is
/// then lowered as usual.
#[test]
fn a_forced_construction_is_used_where_it_can_be_selected() {
    let c = super::challenge::HALO2_BASE_CHALLENGE_COST;
    let plain = ElementAt { element: 0, cost: c / 10, challenged: false };
    let challenged = ElementAt { element: 1, cost: c / 10, challenged: true };
    let graph = independent_sorts(2);
    let lowered = |target, options: &super::emit::Options| lower_opts(&graph, target, &[&plain, &challenged], options);
    assert_eq!(challenged_groups(&lowered(Target::Halo2Base, &joint())), 0);
    assert_eq!(challenged_groups(&lowered(Target::Halo2Base, &forced(&["test: challenged"]))), 2);
    assert_eq!(challenged_groups(&lowered(Target::Noir, &forced(&["test: challenged"]))), 0, "released on Noir");
}

/// A construction applies to a group only when its candidates produce every
/// lane; the configurations expand to registered constructions; an unknown
/// name is refused.
#[test]
fn forcing_names_and_configurations() {
    use super::forcing::{Forcing, CONFIGURATIONS};
    let names: Vec<&str> = super::registry().iter().map(|c| c.name()).collect();
    for (configuration, families) in CONFIGURATIONS {
        for (family, listed) in *families {
            assert!(listed.iter().all(|name| names.contains(name)), "{listed:?}");
            assert!(Forcing::parse(&format!("@{configuration}")).fixes_family(family));
        }
    }
    let mixed = Forcing::parse("@delegate, sort: network");
    assert_eq!(mixed.names().last().map(String::as_str), Some("sort: network"));
    assert!(!mixed.fixes_family("sort"));
    assert!(std::panic::catch_unwind(|| Forcing::parse("@nothing")).is_err());
    assert!(std::panic::catch_unwind(|| Forcing::parse("sort: no such")).is_err());
    assert!(Forcing::keeps("memory: element by index bits", "memory: index bits"));
    assert!(!Forcing::keeps("memory: element by table read", "memory: index bits"));
}
