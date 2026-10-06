//! The pass: build the graph of each operator group, select, emit.
//!
//! 1. **Groups.** Placeholders with the same kind and shared operands form
//!    one group; their lanes are the operator's live outputs.
//! 2. **Nodes.** Every registered construction offers candidates for every
//!    group. Each precondition fact that no free fact proves and no
//!    candidate establishes gets a *check node*: the framework emits the
//!    check through `builder/guard.rs`, under the fact's scope.
//! 3. **Program checks.** An assertion of the program that an earlier one
//!    implies, or whose fact a scalar gadget's postcondition establishes
//!    (`facts`), is dropped. One whose fact some candidate establishes, or
//!    that implies a candidate's precondition, becomes a root requirement
//!    whose own producer is the assertion itself (kept when selected, at its
//!    cost): a postcondition can discharge it, and it can justify a
//!    precondition.
//! 4. **Selection** per connected part of the graph (`select`).
//! 5. **Emission** in a topological order of statements, node starts and
//!    lanes: a node starts right before its first lane, with its inputs
//!    lowered; every other statement keeps its place.

use std::any::Any;
use std::cmp::Reverse;
use std::collections::{BTreeMap, BinaryHeap, HashMap, HashSet};
use std::rc::Rc;

use crate::builder::IRBuilder;
use crate::ir::IRGraph;
use crate::ir_defs::IR;
use crate::optim::IRPass;
use crate::types::{StmtId, Value};

use super::checks::{self, ProgramCheck};
use super::construction::{Candidate, Construction, Context, Emitter, Target};
use super::facts::FreeFacts;
use super::graph::{Fact, Group, GroupId, Lane, Requirement};
use super::select::{self, justifies, Node, Problem};

/// The array lowering pass for one compile target.
pub struct ArrayLowering {
    pub target: Target,
}

impl IRPass for ArrayLowering {
    /// Lowers every placeholder and discharges the checks that facts
    /// justify, then simplifies what it emitted as the earlier passes did
    /// (`select(c, 1, 0)` is `c`, ...), deduplicates it, and drops what no
    /// output needs. A graph with nothing to do is returned unchanged.
    fn exec(&self, graph: IRGraph) -> IRGraph {
        let Some(lowered) = lower(&graph, self.target, &super::registry()) else {
            return graph;
        };
        let simplified = crate::optim::PatternMatchOptim.exec(lowered);
        let deduplicated = crate::optim::DuplicateCodeElimination.exec(simplified);
        crate::optim::DeadCodeElimination.exec(deduplicated)
    }
}

/// Lower `graph` with `constructions`; `None` when it has no placeholder
/// and no check to drop.
pub fn lower(graph: &IRGraph, target: Target, constructions: &[&dyn Construction]) -> Option<IRGraph> {
    let groups = collect_groups(graph);
    let facts = FreeFacts::new(graph);
    let program_checks = checks::program_checks(graph);
    let mut freely_proven = checks::dominated(&program_checks);
    freely_proven.extend(
        program_checks.iter().filter(|c| c.forms.iter().any(|f| facts.established_by_gadget(f))).map(|c| c.stmt),
    );
    if groups.is_empty() && freely_proven.is_empty() {
        return None;
    }

    let users = users_of(graph);
    let lane_group: HashMap<StmtId, GroupId> =
        groups.iter().flat_map(|g| g.lanes.iter().map(move |l| (l.stmt, g.id))).collect();
    let cx = Context { target, graph, groups: &groups, facts: &facts, users: &users, lane_group: &lane_group };
    let mut nodes: Vec<NodeEntry> = Vec::new();
    let forced = forced_construction();
    for group in &groups {
        let offered: Vec<(&dyn Construction, Vec<super::Candidate>)> =
            constructions.iter().map(|c| (*c, c.candidates(group, &cx))).collect();
        let only_forced = forced
            .as_deref()
            .is_some_and(|name| offered.iter().any(|(c, cands)| c.name() == name && !cands.is_empty()));
        for (construction, candidates) in offered {
            if only_forced && Some(construction.name()) != forced.as_deref() {
                continue;
            }
            for candidate in candidates {
                nodes.push(NodeEntry::offered(construction.name(), candidate, &facts));
            }
        }
    }
    let established: Vec<Fact> = nodes.iter().flat_map(|n| n.post_facts()).collect();
    let wanted: Vec<Fact> = nodes.iter().flat_map(|n| n.pre_facts()).collect();
    let mut roots: Vec<Requirement> = groups.iter().flat_map(|g| g.lanes.iter().map(|l| Requirement::Lane(l.stmt))).collect();
    for check in program_checks.iter().filter(|c| !freely_proven.contains(&c.stmt)) {
        let provable = check.forms.iter().any(|f| established.iter().any(|e| e.implies(f)));
        let useful = check.forms.iter().any(|f| wanted.iter().any(|w| f.implies(w)));
        if provable || useful {
            // The forms are equivalent; the root is one a node establishes
            // when there is one (so that node can justify the check).
            let root = check.forms.iter().find(|f| established.iter().any(|e| e.implies(f))).unwrap_or(&check.forms[0]);
            roots.push(Requirement::Fact(*root));
            nodes.push(NodeEntry::program_check(check, target));
        }
    }
    add_check_nodes(&mut nodes, target);

    let selected = select_all(&nodes, &roots);
    let mut dropped = freely_proven;
    dropped.extend(unselected_checks(&nodes, &selected));
    Some(emit(graph, &groups, nodes, &selected, &dropped))
}

/// Measurement only: the environment variable `ZINNIA_FORCE_CONSTRUCTION`
/// set to a construction's [`Construction::name`] makes it the only
/// construction offered for every group it applies to (other groups are
/// lowered as usual), so cost curves and scaling demonstrations can compile
/// one program with each alternative. The benchmark harness never sets it;
/// a forced construction is sound like any other, it may only cost more.
fn forced_construction() -> Option<String> {
    std::env::var("ZINNIA_FORCE_CONSTRUCTION").ok().filter(|s| !s.is_empty())
}

// ── Groups ──────────────────────────────────────────────────────────────

fn collect_groups(graph: &IRGraph) -> Vec<Group> {
    let mut by_key: BTreeMap<(super::OpKind, Vec<StmtId>), GroupId> = BTreeMap::new();
    let mut groups: Vec<Group> = Vec::new();
    for stmt in &graph.stmts {
        let IR::ArrayOp(op) = &stmt.ir else { continue };
        let shared = op.shared as usize;
        let key = (op.kind, stmt.arguments[..shared].to_vec());
        let id = *by_key.entry(key).or_insert_with(|| {
            groups.push(Group { id: groups.len(), kind: op.kind, shared: stmt.arguments[..shared].to_vec(), lanes: Vec::new() });
            groups.len() - 1
        });
        groups[id].lanes.push(Lane {
            stmt: stmt.stmt_id,
            output: op.output,
            position: op.position,
            own: stmt.arguments[shared..].to_vec(),
        });
    }
    groups
}

fn users_of(graph: &IRGraph) -> HashMap<StmtId, Vec<StmtId>> {
    let mut users: HashMap<StmtId, Vec<StmtId>> = HashMap::new();
    for stmt in &graph.stmts {
        for &arg in &stmt.arguments {
            users.entry(arg).or_default().push(stmt.stmt_id);
        }
    }
    users
}

// ── Nodes ───────────────────────────────────────────────────────────────

/// An implementation node of the pass.
struct NodeEntry {
    /// The construction's name, or the framework's for checks.
    name: &'static str,
    node: Node,
    kind: NodeKind,
}

enum NodeKind {
    /// A construction's candidate.
    Offered(Box<dyn Emitter>),
    /// A program's assertion, kept when selected.
    ProgramCheck(StmtId),
    /// A check the framework emits for an unproven precondition.
    Check(Fact),
}

impl NodeEntry {
    /// A candidate, with the preconditions free facts prove left out.
    fn offered(name: &'static str, candidate: Candidate, facts: &FreeFacts) -> NodeEntry {
        let Candidate { pre, post, cost, emitter } = candidate;
        let pre = pre
            .into_iter()
            .filter(|r| !matches!(r, Requirement::Fact(f) if facts.proves(f)))
            .collect();
        NodeEntry { name, node: Node { pre, post, cost }, kind: NodeKind::Offered(emitter) }
    }

    fn program_check(check: &ProgramCheck, target: Target) -> NodeEntry {
        let post = check.forms.iter().map(|f| Requirement::Fact(*f)).collect();
        let cost = checks::cost(&check.forms[0].claim, target);
        NodeEntry { name: "program check", node: Node { pre: Vec::new(), post, cost }, kind: NodeKind::ProgramCheck(check.stmt) }
    }

    fn post_facts(&self) -> impl Iterator<Item = Fact> + '_ {
        facts_of(&self.node.post)
    }

    fn pre_facts(&self) -> impl Iterator<Item = Fact> + '_ {
        facts_of(&self.node.pre)
    }
}

fn facts_of(requirements: &[Requirement]) -> impl Iterator<Item = Fact> + '_ {
    requirements.iter().filter_map(|r| match r {
        Requirement::Fact(f) => Some(*f),
        _ => None,
    })
}

/// A check node for every precondition fact no node establishes.
fn add_check_nodes(nodes: &mut Vec<NodeEntry>, target: Target) {
    let wanted: Vec<Fact> = nodes.iter().flat_map(|n| n.pre_facts()).collect();
    let mut added: HashSet<Fact> = HashSet::new();
    for fact in wanted {
        let required = Requirement::Fact(fact);
        let produced = nodes.iter().any(|n| n.node.post.iter().any(|p| justifies(p, &required)));
        if !produced && added.insert(fact) {
            let node = Node { pre: Vec::new(), post: vec![required], cost: checks::cost(&fact.claim, target) };
            nodes.push(NodeEntry { name: "check", node, kind: NodeKind::Check(fact) });
        }
    }
}

// ── Selection ───────────────────────────────────────────────────────────

/// The selected nodes, producers first, over every connected part of the
/// graph (nodes are connected through the requirements they share).
fn select_all(nodes: &[NodeEntry], roots: &[Requirement]) -> Vec<usize> {
    let parts = connected_parts(nodes, roots);
    let mut selected = Vec::new();
    for (members, part_roots) in parts {
        let problem = Problem {
            nodes: members.iter().map(|&i| nodes[i].node.clone()).collect(),
            roots: part_roots,
        };
        let Some(selection) = select::select(&problem) else {
            let names: Vec<&str> = members.iter().map(|&i| nodes[i].name).collect();
            panic!("array lowering: no selection justifies every requirement (nodes: {names:?})");
        };
        selected.extend(selection.nodes.into_iter().map(|i| members[i]));
    }
    selected
}

/// Union of the nodes that share a requirement (or a root), with the
/// roots of each part. Lanes and shared values are matched by key; facts
/// (few) by implication.
fn connected_parts(nodes: &[NodeEntry], roots: &[Requirement]) -> Vec<(Vec<usize>, Vec<Requirement>)> {
    let mut parts = UnionFind::new(nodes.len());
    let mut first_with: HashMap<&Requirement, usize> = HashMap::new();
    let (mut produced, mut consumed): (Vec<(usize, Fact)>, Vec<(usize, Fact)>) = (Vec::new(), Vec::new());
    for (i, entry) in nodes.iter().enumerate() {
        for (r, is_post) in entry.node.post.iter().map(|r| (r, true)).chain(entry.node.pre.iter().map(|r| (r, false))) {
            match r {
                Requirement::Fact(f) if is_post => produced.push((i, *f)),
                Requirement::Fact(f) => consumed.push((i, *f)),
                _ => match first_with.get(r) {
                    Some(&j) => parts.union(i, j),
                    None => {
                        first_with.insert(r, i);
                    }
                },
            }
        }
    }
    let producers_of = |f: &Fact| -> Vec<usize> { produced.iter().filter(|(_, p)| p.implies(f)).map(|(j, _)| *j).collect() };
    for (i, f) in &consumed {
        for j in producers_of(f) {
            parts.union(*i, j);
        }
    }
    let mut root_part: Vec<(usize, Requirement)> = Vec::new();
    for r in roots {
        let producers: Vec<usize> = match r {
            Requirement::Fact(f) => producers_of(f),
            other => first_with.get(other).into_iter().copied().collect(),
        };
        let Some(&first) = producers.first() else {
            panic!("array lowering: no construction produces {r:?}");
        };
        for &j in &producers[1..] {
            parts.union(first, j);
        }
        root_part.push((first, r.clone()));
    }
    let mut by_part: BTreeMap<usize, (Vec<usize>, Vec<Requirement>)> = BTreeMap::new();
    for i in 0..nodes.len() {
        by_part.entry(parts.find(i)).or_default().0.push(i);
    }
    for (node, r) in root_part {
        by_part.entry(parts.find(node)).or_default().1.push(r);
    }
    by_part.into_values().filter(|(_, roots)| !roots.is_empty()).collect()
}

struct UnionFind(Vec<usize>);

impl UnionFind {
    fn new(n: usize) -> UnionFind {
        UnionFind((0..n).collect())
    }

    fn find(&mut self, mut i: usize) -> usize {
        while self.0[i] != i {
            self.0[i] = self.0[self.0[i]];
            i = self.0[i];
        }
        i
    }

    fn union(&mut self, a: usize, b: usize) {
        let (a, b) = (self.find(a), self.find(b));
        self.0[a] = b;
    }
}

fn unselected_checks(nodes: &[NodeEntry], selected: &[usize]) -> HashSet<StmtId> {
    let check_of = |i: usize| match nodes[i].kind {
        NodeKind::ProgramCheck(s) => Some(s),
        _ => None,
    };
    let kept: HashSet<StmtId> = selected.iter().filter_map(|&i| check_of(i)).collect();
    (0..nodes.len()).filter_map(check_of).filter(|s| !kept.contains(s)).collect()
}

// ── Emission ────────────────────────────────────────────────────────────

/// What the emitters see: the lowered values of the input graph's
/// statements, and the selected nodes (for shared values and checks).
pub struct EmitCx<'a> {
    lookup: &'a HashMap<StmtId, Value>,
    state: &'a mut EmitState,
}

struct EmitState {
    emitters: HashMap<usize, Option<Box<dyn Emitter>>>,
    started: HashSet<usize>,
    /// The selected producer of each shared value, and the values emitted.
    shared_producer: HashMap<SharedKey, usize>,
    shared_values: HashMap<SharedKey, Rc<dyn Any>>,
    /// The selected check nodes each node's preconditions need.
    checks_for: HashMap<usize, Vec<Fact>>,
    emitted_checks: HashSet<Fact>,
}

impl EmitCx<'_> {
    /// The lowered value of an input-graph statement (an operand).
    pub fn value(&self, stmt: StmtId) -> Value {
        self.lookup[&stmt].clone()
    }

    pub fn values(&self, stmts: &[StmtId]) -> Vec<Value> {
        stmts.iter().map(|s| self.value(*s)).collect()
    }

    /// The shared value `name` of `group`, emitted once by its selected
    /// producer.
    pub fn shared<T: 'static>(&mut self, b: &mut IRBuilder, group: GroupId, name: &'static str) -> Rc<T> {
        self.shared_by_key(b, SharedKey::Group(group, name), |e, b, cx| e.shared(b, cx, group, name))
    }

    /// The shared value `name` of statement `at` (`Requirement::SharedAt`),
    /// emitted once by its selected producer.
    pub fn shared_at<T: 'static>(&mut self, b: &mut IRBuilder, at: StmtId, name: &'static str) -> Rc<T> {
        self.shared_by_key(b, SharedKey::At(at, name), |e, b, cx| e.shared_at(b, cx, at, name))
    }

    fn shared_by_key<T: 'static>(
        &mut self,
        b: &mut IRBuilder,
        key: SharedKey,
        produce: impl FnOnce(&mut dyn Emitter, &mut IRBuilder, &mut EmitCx) -> Rc<dyn Any>,
    ) -> Rc<T> {
        if !self.state.shared_values.contains_key(&key) {
            let producer = *self.state.shared_producer.get(&key).unwrap_or_else(|| panic!("no producer of {key:?}"));
            let value = self.with_emitter(b, producer, produce);
            self.state.shared_values.insert(key, value);
        }
        self.state.shared_values[&key].clone().downcast::<T>().unwrap_or_else(|_| panic!("{key:?}: wrong type"))
    }

    /// Run `f` on node `node`'s emitter, started first.
    fn with_emitter<R>(&mut self, b: &mut IRBuilder, node: usize, f: impl FnOnce(&mut dyn Emitter, &mut IRBuilder, &mut EmitCx) -> R) -> R {
        let mut emitter = self.state.emitters.get_mut(&node).and_then(Option::take).expect("an emitter in use (cyclic justification)");
        if self.state.started.insert(node) {
            for fact in self.state.checks_for.get(&node).cloned().unwrap_or_default() {
                if self.state.emitted_checks.insert(fact) {
                    checks::emit(b, &fact, self.lookup);
                }
            }
            emitter.begin(b, self);
        }
        let result = f(emitter.as_mut(), b, self);
        self.state.emitters.insert(node, Some(emitter));
        result
    }
}

/// The key of a shared value: per group (`Requirement::Shared`) or per
/// statement (`Requirement::SharedAt`).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum SharedKey {
    Group(GroupId, &'static str),
    At(StmtId, &'static str),
}

/// A unit of emission: an input statement, the start of a node, or a lane.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
enum Unit {
    Start(usize),
    Stmt(StmtId),
}

fn emit(graph: &IRGraph, groups: &[Group], nodes: Vec<NodeEntry>, selected: &[usize], dropped: &HashSet<StmtId>) -> IRGraph {
    let selected_set: HashSet<usize> = selected.iter().copied().collect();
    let mut lane_producer: HashMap<StmtId, usize> = HashMap::new();
    let mut state = EmitState {
        emitters: HashMap::new(),
        started: HashSet::new(),
        shared_producer: HashMap::new(),
        shared_values: HashMap::new(),
        checks_for: HashMap::new(),
        emitted_checks: HashSet::new(),
    };
    let check_facts: Vec<Fact> = selected
        .iter()
        .filter_map(|&i| match nodes[i].kind {
            NodeKind::Check(f) => Some(f),
            _ => None,
        })
        .collect();
    let mut starts: Vec<usize> = Vec::new();
    for (i, entry) in nodes.into_iter().enumerate() {
        if !selected_set.contains(&i) {
            continue;
        }
        for r in &entry.node.post {
            match r {
                Requirement::Lane(s) => {
                    lane_producer.entry(*s).or_insert(i);
                }
                Requirement::Shared { group, name } => {
                    state.shared_producer.entry(SharedKey::Group(*group, name)).or_insert(i);
                }
                Requirement::SharedAt { at, name } => {
                    state.shared_producer.entry(SharedKey::At(*at, name)).or_insert(i);
                }
                Requirement::Fact(_) => {}
            }
        }
        let needed: Vec<Fact> = entry
            .node
            .pre
            .iter()
            .filter_map(|r| match r {
                Requirement::Fact(f) => check_facts.iter().find(|c| c.implies(f)).copied(),
                _ => None,
            })
            .collect();
        state.checks_for.insert(i, needed);
        if let NodeKind::Offered(emitter) = entry.kind {
            if entry.node.post.iter().any(|r| matches!(r, Requirement::Lane(_))) {
                starts.push(i);
            }
            state.emitters.insert(i, Some(emitter));
        }
    }

    let lanes: HashMap<StmtId, &Lane> = groups.iter().flat_map(|g| g.lanes.iter().map(|l| (l.stmt, l))).collect();
    let check_inputs: HashMap<usize, Vec<StmtId>> = state
        .checks_for
        .iter()
        .map(|(&node, facts)| (node, facts.iter().flat_map(Fact::subjects).collect()))
        .collect();
    let order = emission_order(graph, groups, &lane_producer, &starts, &check_inputs);

    let mut b = IRBuilder::new();
    b.reserve_segment_ids_below(next_free_segment_id(graph));
    let mut lookup: HashMap<StmtId, Value> = HashMap::new();
    for unit in order {
        match unit {
            Unit::Start(node) => {
                let mut cx = EmitCx { lookup: &lookup, state: &mut state };
                cx.with_emitter(&mut b, node, |_, _, _| ());
            }
            Unit::Stmt(s) => {
                let value = if let Some(lane) = lanes.get(&s) {
                    let producer = lane_producer[&s];
                    let mut cx = EmitCx { lookup: &lookup, state: &mut state };
                    cx.with_emitter(&mut b, producer, |e, b, cx| e.lane(b, cx, lane))
                } else if dropped.contains(&s) {
                    Value::None
                } else {
                    let stmt = &graph.stmts[s as usize];
                    let args: Vec<Value> = stmt.arguments.iter().map(|a| lookup[a].clone()).collect();
                    b.create_ir(&stmt.ir, &args)
                };
                lookup.insert(s, value);
            }
        }
    }
    b.export_ir_graph()
}

/// Statements in their order, each node's start placed right before the
/// first lane it produces once its inputs are emitted (Kahn's algorithm,
/// ties by position).
fn emission_order(
    graph: &IRGraph,
    groups: &[Group],
    lane_producer: &HashMap<StmtId, usize>,
    starts: &[usize],
    check_inputs: &HashMap<usize, Vec<StmtId>>,
) -> Vec<Unit> {
    let position: HashMap<StmtId, usize> = graph.stmts.iter().enumerate().map(|(i, s)| (s.stmt_id, i)).collect();
    let mut deps: HashMap<Unit, Vec<Unit>> = HashMap::new();
    let mut priority: HashMap<Unit, usize> = HashMap::new();
    for stmt in &graph.stmts {
        let unit = Unit::Stmt(stmt.stmt_id);
        let mut d: Vec<Unit> = stmt.arguments.iter().map(|a| Unit::Stmt(*a)).collect();
        if let Some(&node) = lane_producer.get(&stmt.stmt_id) {
            d.push(Unit::Start(node));
            let p = priority.entry(Unit::Start(node)).or_insert(usize::MAX);
            *p = (*p).min(2 * position[&stmt.stmt_id]);
        }
        deps.insert(unit, d);
        priority.insert(unit, 2 * position[&stmt.stmt_id] + 1);
    }
    let group_of: HashMap<StmtId, &Group> = groups.iter().flat_map(|g| g.lanes.iter().map(move |l| (l.stmt, g))).collect();
    for &node in starts {
        let mut inputs: Vec<Unit> = Vec::new();
        for (lane, _) in lane_producer.iter().filter(|(_, &p)| p == node) {
            inputs.extend(group_of[lane].shared.iter().map(|s| Unit::Stmt(*s)));
        }
        inputs.extend(check_inputs.get(&node).into_iter().flatten().map(|s| Unit::Stmt(*s)));
        inputs.sort_unstable();
        inputs.dedup();
        deps.insert(Unit::Start(node), inputs);
    }

    let mut waiting: HashMap<Unit, usize> = HashMap::new();
    let mut dependents: HashMap<Unit, Vec<Unit>> = HashMap::new();
    let mut ready: BinaryHeap<Reverse<(usize, Unit)>> = BinaryHeap::new();
    for (unit, d) in &deps {
        waiting.insert(*unit, d.len());
        for dep in d {
            dependents.entry(*dep).or_default().push(*unit);
        }
        if d.is_empty() {
            ready.push(Reverse((priority[unit], *unit)));
        }
    }
    let mut order = Vec::with_capacity(deps.len());
    while let Some(Reverse((_, unit))) = ready.pop() {
        order.push(unit);
        for next in dependents.get(&unit).cloned().unwrap_or_default() {
            let w = waiting.get_mut(&next).expect("a known unit");
            *w -= 1;
            if *w == 0 {
                ready.push(Reverse((priority[&next], next)));
            }
        }
    }
    assert_eq!(order.len(), deps.len(), "array lowering: the emission order has a cycle");
    order
}

/// One above the largest memory segment id `graph` uses.
fn next_free_segment_id(graph: &IRGraph) -> u32 {
    graph
        .stmts
        .iter()
        .filter_map(|s| match s.ir {
            IR::AllocateMemory { segment_id, .. }
            | IR::WriteMemory { segment_id }
            | IR::ReadMemory { segment_id }
            | IR::MemoryTraceEmit { segment_id, .. }
            | IR::DynamicNDArrayGetItem { segment_id, .. }
            | IR::DynamicNDArraySetItem { segment_id, .. } => Some(segment_id + 1),
            _ => None,
        })
        .max()
        .unwrap_or(0)
}
