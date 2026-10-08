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
//!
//! Measurement switches ([`Options`], `docs/ablations.md`) change steps 2-4:
//! no free facts, a fixed construction per group (`super::forcing`), or each
//! group deciding on its own (`super::local`); with every switch off the pass
//! is the compiler's.

use std::any::Any;
use std::cmp::Reverse;
use std::collections::{BTreeMap, BTreeSet, BinaryHeap, HashMap, HashSet};
use std::rc::Rc;

use crate::builder::IRBuilder;
use crate::ir::IRGraph;
use crate::ir_defs::IR;
use crate::optim::IRPass;
use crate::types::{StmtId, Value};

use super::checks::{self, ProgramCheck};
use super::construction::{Candidate, Construction, Context, Emitter, Target};
use super::facts::FreeFacts;
use super::forcing::Forcing;
use super::graph::{CellSource, Fact, Group, GroupId, Lane, Requirement, VersionLine};
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
    ///
    /// Lowering runs in rounds: an emitter may leave a read it does not
    /// build itself as an `ArrayRead` of an `ArrayInit` (the reads of an
    /// overlay's versions, a sort's reads at hinted ranks), which the next
    /// round lowers with every other element read.
    fn exec(&self, graph: IRGraph) -> IRGraph {
        lower_rounds(graph, self.target, &super::registry())
    }
}

/// Lower `graph` in rounds until no placeholder or array read is left: the
/// first round with `first` (every construction in production; one under
/// test in the layer's tests), the later ones, which lower the reads the
/// emitters leave for them, with the whole registry. After each round the
/// emitted code is simplified, deduplicated and pruned.
///
/// # Panics
/// When array operations are left after `MAX_ROUNDS` rounds.
pub fn lower_rounds(graph: IRGraph, target: Target, first: &[&dyn Construction]) -> IRGraph {
    lower_rounds_with(graph, target, first, &Options::from_env())
}

/// [`lower_rounds`] with explicit switches.
pub fn lower_rounds_with(graph: IRGraph, target: Target, first: &[&dyn Construction], options: &Options) -> IRGraph {
    let registry = super::registry();
    let mut graph = graph;
    for round in 0..MAX_ROUNDS {
        let constructions = if round == 0 { first } else { &registry };
        let Some(lowered) = lower_with(&graph, target, constructions, options) else {
            return graph;
        };
        let simplified = crate::optim::PatternMatchOptim.exec(lowered);
        let deduplicated = crate::optim::DuplicateCodeElimination.exec(simplified);
        graph = crate::optim::DeadCodeElimination.exec(deduplicated);
        if !has_array_operations(&graph) {
            return graph;
        }
    }
    panic!("array lowering: array operations left after {MAX_ROUNDS} rounds");
}

/// The most lowering rounds a program may need (an overlay's reads, then a
/// table's; a sort's reads at hinted ranks).
const MAX_ROUNDS: usize = 4;

/// Whether `graph` still has something for the layer: a placeholder or an
/// array read or write (an `ArrayInit` alone is dead once its reads are
/// lowered).
fn has_array_operations(graph: &IRGraph) -> bool {
    graph.stmts.iter().any(|s| matches!(s.ir, IR::ArrayOp(_) | IR::ArrayRead(_) | IR::ArrayWrite(_)))
}

/// Lower `graph` with `constructions`; `None` when it has no placeholder
/// and no check to drop. The measurement switches are read from the
/// environment ([`Options::from_env`]).
pub fn lower(graph: &IRGraph, target: Target, constructions: &[&dyn Construction]) -> Option<IRGraph> {
    lower_with(graph, target, constructions, &Options::from_env())
}

/// The measurement switches of the pass (`docs/ablations.md`). The default
/// (every switch off) is the compiler's behaviour.
#[derive(Clone, Debug, Default)]
pub struct Options {
    /// `ZINNIA_ABLATE_LAYER_FACTS=1`: no free facts, and the program's checks
    /// take no part (none is dropped, none justifies a precondition), so every
    /// precondition fact gets a check node and every program check is kept as
    /// the IR generator emitted it.
    pub no_facts: bool,
    /// `ZINNIA_ABLATE_ARRAY_FACTS=1`: no array claim (`super::array_facts`):
    /// none is established, implied or composed, so the constructions keyed
    /// on them do not apply.
    pub no_array_facts: bool,
    /// `ZINNIA_ABLATE_JOINT=1`: each group chooses its construction on its
    /// own instead of all groups together (`super::local`).
    pub local: bool,
    /// `ZINNIA_FORCE_CONSTRUCTION`: one fixed construction per group
    /// (`super::forcing`).
    pub forcing: Forcing,
    /// `ZINNIA_LAYER_REPORT=<path>`: append what each lowering chose to this
    /// file (`super::report`).
    pub report: Option<std::path::PathBuf>,
}

impl Options {
    pub fn from_env() -> Options {
        Options {
            no_facts: crate::ablated("ZINNIA_ABLATE_LAYER_FACTS"),
            no_array_facts: crate::ablated("ZINNIA_ABLATE_ARRAY_FACTS"),
            local: crate::ablated("ZINNIA_ABLATE_JOINT"),
            forcing: Forcing::from_env(),
            report: std::env::var_os("ZINNIA_LAYER_REPORT").filter(|p| !p.is_empty()).map(Into::into),
        }
    }
}

/// [`lower`] with explicit switches.
pub fn lower_with(
    graph: &IRGraph,
    target: Target,
    constructions: &[&dyn Construction],
    options: &Options,
) -> Option<IRGraph> {
    // Claims' value lists are numbered per lowering (`ListId`).
    super::graph::ListId::reset();
    super::array_facts::set_enabled(!options.no_array_facts);
    let (groups, direct) = collect_groups(graph);
    let facts = if options.no_facts { FreeFacts::none(graph) } else { FreeFacts::new(graph) };
    let program_checks = if options.no_facts { Vec::new() } else { checks::program_checks(graph) };
    let mut freely_proven = checks::dominated(&program_checks);
    let kept_for_backend = checks::kept_for_backend(graph, &program_checks, target);
    freely_proven.extend(
        program_checks
            .iter()
            .filter(|c| !kept_for_backend.contains(&c.stmt))
            .filter(|c| c.forms.iter().any(|f| facts.established_by_gadget(f)))
            .map(|c| c.stmt),
    );
    // Checks that each realize a value's two-sided range (`checks::
    // range_merges`), offered in place of the program checks they imply.
    let merges = checks::range_merges(&program_checks, target);
    if groups.is_empty() && direct.is_empty() && freely_proven.is_empty() && merges.is_empty() {
        return None;
    }

    let users = users_of(graph);
    let lane_group: HashMap<StmtId, GroupId> =
        groups.iter().flat_map(|g| g.lanes.iter().map(move |l| (l.stmt, g.id))).collect();
    let cx = Context {
        target,
        graph,
        groups: &groups,
        facts: &facts,
        checks: &program_checks,
        users: &users,
        lane_group: &lane_group,
        alone: options.local,
    };
    // A forced construction that cannot lower its group together with the
    // rest of the program is released for that group (the group is lowered
    // as usual), and the graph is built again.
    let mut released: BTreeSet<GroupId> = BTreeSet::new();
    let (nodes, selected, offers) = loop {
        let (mut nodes, offers) = offer(&groups, &cx, constructions, &options.forcing, &released);
        add_composition_nodes(&mut nodes, &facts);
        for &(fact, cost) in &merges {
            let node = Node { pre: Vec::new(), post: vec![Requirement::Fact(fact)], cost };
            nodes.push(NodeEntry { name: "merged range check", group: None, node, kind: NodeKind::Check(fact) });
        }
        let (roots, check_roots) = root_requirements(&groups, &program_checks, &freely_proven, &mut nodes, target);
        add_check_nodes(&mut nodes, target, options.no_facts);
        add_program_nodes(&mut nodes, target);
        let selection = if options.local {
            super::local::select(&nodes, &groups, &roots, &check_roots)
        } else {
            select_members(&nodes, &(0..nodes.len()).collect::<Vec<_>>(), &roots)
        };
        match selection {
            Ok(selected) => break (nodes, selected, offers),
            Err(stuck) => {
                let release: Vec<GroupId> = stuck
                    .iter()
                    .filter_map(|r| match r {
                        Requirement::Lane(s) => lane_group.get(s).copied(),
                        _ => None,
                    })
                    .filter(|g| offers[*g].forced.is_some() && !released.contains(g))
                    .collect();
                if release.is_empty() {
                    let names: Vec<&str> = nodes.iter().map(|n| n.name).collect();
                    panic!(
                        "array lowering: no selection justifies every requirement \
                         (unjustified: {stuck:?}; nodes: {names:?})"
                    );
                }
                released.extend(release);
            }
        }
    };
    if let Some(path) = &options.report {
        super::report::append(path, target, options, &groups, &nodes, &selected, &offers, &released);
    }
    let mut dropped = freely_proven;
    dropped.extend(unselected_checks(&nodes, &selected));
    let in_place = checks_in_place_of(graph, &nodes, &selected, &program_checks, &dropped);
    Some(emit(graph, &groups, &direct, nodes, &selected, &dropped, &in_place))
}

/// Where each selected check stands that replaces dropped program checks
/// (a merged range check, `checks::range_merges`, or a precondition's check
/// that also implies a program check): at the first of the program checks
/// it implies that are dropped, by statement position. Its subjects are
/// lowered there, as that check's condition reads them.
fn checks_in_place_of(
    graph: &IRGraph,
    nodes: &[NodeEntry],
    selected: &[usize],
    program_checks: &[ProgramCheck],
    dropped: &HashSet<StmtId>,
) -> HashMap<StmtId, Vec<Fact>> {
    let position: HashMap<StmtId, usize> = graph.stmts.iter().enumerate().map(|(i, s)| (s.stmt_id, i)).collect();
    let mut in_place: HashMap<StmtId, Vec<Fact>> = HashMap::new();
    for &i in selected {
        let NodeKind::Check(fact) = nodes[i].kind else { continue };
        let first = program_checks
            .iter()
            .filter(|c| dropped.contains(&c.stmt) && c.forms.iter().any(|f| fact.implies(f)))
            .min_by_key(|c| position[&c.stmt]);
        if let Some(check) = first {
            in_place.entry(check.stmt).or_default().push(fact);
        }
    }
    in_place
}

/// What the constructions offered one group: the constructions with
/// candidates for it, and the one forced on it (`None`: lowered as usual).
pub(super) struct Offer {
    pub(super) applicable: Vec<&'static str>,
    pub(super) forced: Option<&'static str>,
}

/// The implementation nodes the constructions offer for every group, and
/// what each group was offered. Under a forcing, a group whose forced
/// construction applies keeps only that construction's candidates (and its
/// helpers', `Forcing::keeps`); the groups in `released` are lowered as
/// usual.
fn offer(
    groups: &[Group],
    cx: &Context,
    constructions: &[&dyn Construction],
    forcing: &Forcing,
    released: &BTreeSet<GroupId>,
) -> (Vec<NodeEntry>, Vec<Offer>) {
    let mut nodes: Vec<NodeEntry> = Vec::new();
    let mut offers: Vec<Offer> = Vec::with_capacity(groups.len());
    for group in groups {
        let offered: Vec<(&dyn Construction, Vec<Candidate>)> =
            constructions.iter().map(|c| (*c, c.candidates(group, cx))).collect();
        let only = if released.contains(&group.id) { None } else { forcing.choice(group, &offered) };
        let applicable = offered.iter().filter(|(_, candidates)| !candidates.is_empty()).map(|(c, _)| c.name()).collect();
        offers.push(Offer { applicable, forced: only });
        for (construction, candidates) in offered {
            if only.is_some_and(|name| !Forcing::keeps(name, construction.name())) {
                continue;
            }
            for mut candidate in candidates {
                // Reading a written version's cells needs them (its memory
                // group's overlay produces them).
                if let Some(line) = &group.version {
                    let produces_own_lane =
                        candidate.post.iter().any(|r| matches!(r, Requirement::Lane(s) if group.lanes.iter().any(|l| l.stmt == *s)));
                    if produces_own_lane {
                        candidate.pre.push(super::memory::versions::cells_requirement(line.version));
                    }
                }
                nodes.push(NodeEntry::offered(construction.name(), group.id, candidate, cx.facts));
            }
        }
    }
    (nodes, offers)
}

/// The root requirements: every live lane, then each program check that a
/// node can establish or that implies a node's precondition (its own
/// producer, the assertion kept, is added to `nodes`). Also returns each
/// such check's root with the index of its node.
fn root_requirements(
    groups: &[Group],
    program_checks: &[ProgramCheck],
    freely_proven: &HashSet<StmtId>,
    nodes: &mut Vec<NodeEntry>,
    target: Target,
) -> (Vec<Requirement>, Vec<(Requirement, usize)>) {
    let established: Vec<Fact> = nodes.iter().flat_map(|n| n.post_facts()).collect();
    let wanted: Vec<Fact> = nodes.iter().flat_map(|n| n.pre_facts()).collect();
    let mut roots: Vec<Requirement> = groups.iter().flat_map(|g| g.lanes.iter().map(|l| Requirement::Lane(l.stmt))).collect();
    let mut check_roots = Vec::new();
    for check in program_checks.iter().filter(|c| !freely_proven.contains(&c.stmt)) {
        let provable = check.forms.iter().any(|f| established.iter().any(|e| e.implies(f)));
        let useful = check.forms.iter().any(|f| wanted.iter().any(|w| f.implies(w)));
        if provable || useful {
            // The forms are equivalent; the root is one a node establishes
            // when there is one (so that node can justify the check).
            let root = check.forms.iter().find(|f| established.iter().any(|e| e.implies(f))).unwrap_or(&check.forms[0]);
            roots.push(Requirement::Fact(*root));
            check_roots.push((Requirement::Fact(*root), nodes.len()));
            nodes.push(NodeEntry::program_check(check, target));
        }
    }
    (roots, check_roots)
}

// ── Groups ──────────────────────────────────────────────────────────────

/// The groups of `graph`'s placeholders and array reads, and the reads of a
/// known cell at a constant address (read → cell), which need no
/// construction:
///
/// * a placeholder's group: its kind and shared operands;
/// * an element group: the reads of one line of known cells (`ArrayRead`s
///   of an `ArrayInit`, `super::arrays`), or of one line of a written
///   version's cells (`Group::version`);
/// * a memory group (`OpKind::Versions`): a version tree with writes
///   (`super::memory::versions`), whose lanes are its reads at constant
///   addresses.
fn collect_groups(graph: &IRGraph) -> (Vec<Group>, HashMap<StmtId, StmtId>) {
    type Key = (super::OpKind, Vec<StmtId>, Option<VersionLine>);
    let mut by_key: BTreeMap<Key, GroupId> = BTreeMap::new();
    let mut groups: Vec<Group> = Vec::new();
    let mut direct: HashMap<StmtId, StmtId> = HashMap::new();
    let mut add_lane = |key: Key, lane: Lane| {
        let id = *by_key.entry(key.clone()).or_insert_with(|| {
            let (kind, shared, version) = key;
            groups.push(Group { id: groups.len(), kind, shared, lanes: Vec::new(), version, tree: None });
            groups.len() - 1
        });
        groups[id].lanes.push(lane);
    };
    let trees = super::memory::versions::trees(&graph.stmts);
    let root_of_read: HashMap<StmtId, StmtId> =
        trees.iter().flat_map(|(&root, t)| t.reads.iter().map(move |r| (r.stmt, root))).collect();
    for stmt in &graph.stmts {
        match &stmt.ir {
            IR::ArrayOp(op) => {
                let shared = op.shared as usize;
                let lane = Lane {
                    stmt: stmt.stmt_id,
                    output: op.output,
                    position: op.position,
                    own: stmt.arguments[shared..].to_vec(),
                };
                add_lane((op.kind, stmt.arguments[..shared].to_vec(), None), lane);
            }
            IR::ArrayRead(access) => match super::arrays::element_read(&graph.stmts, stmt.stmt_id) {
                // A line of one cell is that cell: its position is 0
                // wherever the read runs (the program checks it), and
                // nothing else reads it.
                Some(super::arrays::ElementRead { line, index: Some(index) }) if line.len() > 1 => {
                    let lane = Lane { stmt: stmt.stmt_id, output: 0, position: None, own: vec![index] };
                    add_lane((super::OpKind::Element, line, None), lane);
                }
                Some(super::arrays::ElementRead { line, .. }) => {
                    direct.insert(stmt.stmt_id, line[0]);
                }
                // A read of a written version: its tree's (`trees`).
                None => match stmt.arguments.get(1) {
                    Some(&index) => {
                        let line = VersionLine { version: stmt.arguments[0], addresses: access.line() };
                        let lane = Lane { stmt: stmt.stmt_id, output: 0, position: None, own: vec![index] };
                        add_lane((super::OpKind::Element, Vec::new(), Some(line)), lane);
                    }
                    None => {
                        let root = root_of_read[&stmt.stmt_id];
                        let lane = Lane { stmt: stmt.stmt_id, output: 0, position: Some(access.offset), own: Vec::new() };
                        add_lane((super::OpKind::Versions, vec![root], None), lane);
                    }
                },
            },
            _ => {}
        }
    }
    // Every tree is a memory group (with no lane of its own when all its
    // reads are at runtime positions).
    for (root, tree) in trees {
        let key = (super::OpKind::Versions, vec![root], None);
        let id = *by_key.entry(key).or_insert_with(|| {
            groups.push(Group { id: groups.len(), kind: super::OpKind::Versions, shared: vec![root], lanes: Vec::new(), version: None, tree: None });
            groups.len() - 1
        });
        groups[id].tree = Some(std::rc::Rc::new(tree));
    }
    (groups, direct)
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
pub(super) struct NodeEntry {
    /// The construction's name, or the framework's for checks.
    pub(super) name: &'static str,
    /// The group a construction offered the node for (`None` for the
    /// framework's nodes).
    pub(super) group: Option<GroupId>,
    pub(super) node: Node,
    pub(super) kind: NodeKind,
}

pub(super) enum NodeKind {
    /// A construction's candidate.
    Offered(Box<dyn Emitter>),
    /// A program's assertion, kept when selected.
    ProgramCheck(StmtId),
    /// A check the framework emits for an unproven precondition.
    Check(Fact),
    /// The producer of a program-wide requirement (`challenge`): nothing to
    /// emit here (the backend lays out what the consumers' IR needs).
    Program,
    /// An array claim composed of the scalar facts that are its basis
    /// (`super::array_facts::basis`): nothing to emit; the checks its basis
    /// needs are emitted before the nodes that rely on the claim.
    Composition,
}

impl NodeEntry {
    /// A candidate, with the preconditions free facts prove left out.
    fn offered(name: &'static str, group: GroupId, candidate: Candidate, facts: &FreeFacts) -> NodeEntry {
        let Candidate { pre, post, cost, emitter } = candidate;
        let pre = pre
            .into_iter()
            .filter(|r| !matches!(r, Requirement::Fact(f) if facts.proves(f)))
            .collect();
        NodeEntry { name, group: Some(group), node: Node { pre, post, cost }, kind: NodeKind::Offered(emitter) }
    }

    fn program_check(check: &ProgramCheck, target: Target) -> NodeEntry {
        let post = check.forms.iter().map(|f| Requirement::Fact(*f)).collect();
        let cost = checks::cost(&check.forms[0].claim, target);
        let node = Node { pre: Vec::new(), post, cost };
        NodeEntry { name: "program check", group: None, node, kind: NodeKind::ProgramCheck(check.stmt) }
    }

    /// The lanes the node produces.
    pub(super) fn lanes(&self) -> impl Iterator<Item = StmtId> + '_ {
        self.node.post.iter().filter_map(|r| match r {
            Requirement::Lane(s) => Some(*s),
            _ => None,
        })
    }

    pub(super) fn post_facts(&self) -> impl Iterator<Item = Fact> + '_ {
        facts_of(&self.node.post)
    }

    pub(super) fn pre_facts(&self) -> impl Iterator<Item = Fact> + '_ {
        facts_of(&self.node.pre)
    }
}

fn facts_of(requirements: &[Requirement]) -> impl Iterator<Item = Fact> + '_ {
    requirements.iter().filter_map(|r| match r {
        Requirement::Fact(f) => Some(*f),
        _ => None,
    })
}

/// A composition node for every array claim some node relies on that no
/// free fact proves: it establishes the claim from the scalar facts of its
/// basis (`super::array_facts::basis`), at no cost. Those facts are then
/// justified like any precondition (by the program's checks, typically).
fn add_composition_nodes(nodes: &mut Vec<NodeEntry>, facts: &FreeFacts) {
    let wanted: BTreeSet<Fact> =
        nodes.iter().flat_map(|n| n.pre_facts()).filter(|f| f.claim.is_array_claim()).collect();
    for fact in wanted {
        let Some(basis) = super::array_facts::basis_facts(&fact) else { continue };
        let pre = basis.into_iter().filter(|f| !facts.proves(f)).map(Requirement::Fact).collect();
        let node = Node { pre, post: vec![Requirement::Fact(fact)], cost: 0 };
        nodes.push(NodeEntry { name: "array claim", group: None, node, kind: NodeKind::Composition });
    }
}

/// A check node for every precondition fact no node establishes (with
/// `every`, for every precondition fact: the selection may still pick a
/// construction's postcondition when that is cheaper).
fn add_check_nodes(nodes: &mut Vec<NodeEntry>, target: Target, every: bool) {
    let wanted: Vec<Fact> = nodes.iter().flat_map(|n| n.pre_facts()).collect();
    let mut added: HashSet<Fact> = HashSet::new();
    for fact in wanted {
        let required = Requirement::Fact(fact);
        let produced = nodes.iter().any(|n| n.node.post.iter().any(|p| justifies(p, &required)));
        if (every || !produced) && added.insert(fact) {
            let node = Node { pre: Vec::new(), post: vec![required], cost: checks::cost(&fact.claim, target) };
            nodes.push(NodeEntry { name: "check", group: None, node, kind: NodeKind::Check(fact) });
        }
    }
}

/// A producer node for every program-wide requirement some node needs and
/// the target can produce (`challenge::producer_cost`). Nodes needing one
/// the target cannot produce are never selected.
fn add_program_nodes(nodes: &mut Vec<NodeEntry>, target: Target) {
    let names: BTreeSet<&'static str> = nodes.iter().flat_map(|n| program_requirements(&n.node.pre)).collect();
    for name in names {
        if let Some(cost) = super::challenge::producer_cost(name, target) {
            let node = Node { pre: Vec::new(), post: vec![Requirement::Program { name }], cost };
            nodes.push(NodeEntry { name: "program-wide requirement", group: None, node, kind: NodeKind::Program });
        }
    }
}

fn program_requirements(requirements: &[Requirement]) -> impl Iterator<Item = &'static str> + '_ {
    requirements.iter().filter_map(|r| match r {
        Requirement::Program { name } => Some(*name),
        _ => None,
    })
}

// ── Selection ───────────────────────────────────────────────────────────

/// The selected nodes among `members` (indices into `nodes`) that justify
/// `roots`, producers first, over every connected part of the graph (nodes
/// are connected through the requirements they share, the program-wide ones
/// excepted). The compiler selects among all nodes; the local selection
/// (`super::local`) among those one group may use.
///
/// A program-wide requirement (the verifier challenge) is paid once for all
/// parts together, so it is decided globally: for each set `S` of the
/// program-wide requirements the target can produce, every part is solved
/// with those in `S` free and the nodes needing any other left out; the
/// total is the cost of `S` plus the parts' costs. The cheapest total wins,
/// ties to the smaller set (a requirement is used only when it pays). Exact
/// whenever each part's selection is.
///
/// `Err` holds the roots no selection justifies: those no member produces,
/// or the roots of the parts that have no selection even with every
/// program-wide requirement available.
pub(super) fn select_members(
    nodes: &[NodeEntry],
    members: &[usize],
    roots: &[Requirement],
) -> Result<Vec<usize>, Vec<Requirement>> {
    let parts = connected_parts(nodes, members, roots)?;
    let producers: Vec<(usize, &'static str)> = members
        .iter()
        .filter_map(|&i| match (&nodes[i].kind, nodes[i].node.post.as_slice()) {
            (NodeKind::Program, [Requirement::Program { name }]) => Some((i, *name)),
            _ => None,
        })
        .collect();
    assert!(producers.len() <= 4, "array lowering: too many program-wide requirements");
    // A part's selection depends only on the available requirements its
    // nodes need: solved once per such set.
    let needs: Vec<BTreeSet<&'static str>> = parts
        .iter()
        .map(|(members, _)| members.iter().flat_map(|&i| program_requirements(&nodes[i].node.pre)).collect())
        .collect();
    let mut solved: HashMap<(usize, Vec<&'static str>), Option<(u64, Vec<usize>)>> = HashMap::new();
    let mut solve = |p: usize, names: Vec<&'static str>| {
        let (members, part_roots) = &parts[p];
        solved.entry((p, names.clone())).or_insert_with(|| select_part(nodes, members, part_roots, &names)).clone()
    };
    let mut best: Option<(u64, Vec<usize>)> = None;
    for subset in 0..1usize << producers.len() {
        let available: Vec<(usize, &'static str)> =
            producers.iter().enumerate().filter(|(j, _)| subset >> j & 1 == 1).map(|(_, p)| *p).collect();
        let mut cost: u64 = available.iter().map(|&(i, _)| nodes[i].node.cost).sum();
        let mut selected: Vec<usize> = available.iter().map(|&(i, _)| i).collect();
        let feasible = (0..parts.len()).all(|p| {
            let names: Vec<&'static str> =
                available.iter().map(|&(_, name)| name).filter(|name| needs[p].contains(name)).collect();
            let Some((part_cost, part_nodes)) = solve(p, names) else { return false };
            cost += part_cost;
            selected.extend(part_nodes);
            true
        });
        if feasible && best.as_ref().is_none_or(|(c, _)| cost < *c) {
            best = Some((cost, selected));
        }
    }
    match best {
        Some((_, selected)) => Ok(selected),
        None => Err((0..parts.len())
            .filter(|&p| {
                let names: Vec<&'static str> =
                    producers.iter().map(|&(_, name)| name).filter(|name| needs[p].contains(name)).collect();
                solve(p, names).is_none()
            })
            .flat_map(|p| parts[p].1.clone())
            .collect()),
    }
}

/// The cheapest selection of one part, with the program-wide requirements
/// `available` free and nodes needing another left out: its cost and nodes
/// (indices into `nodes`). `None` when it has none.
fn select_part(
    nodes: &[NodeEntry],
    members: &[usize],
    roots: &[Requirement],
    available: &[&'static str],
) -> Option<(u64, Vec<usize>)> {
    let usable: Vec<usize> = members
        .iter()
        .copied()
        .filter(|&i| program_requirements(&nodes[i].node.pre).all(|name| available.contains(&name)))
        .collect();
    let problem = Problem {
        nodes: usable
            .iter()
            .map(|&i| {
                let mut node = nodes[i].node.clone();
                node.pre.retain(|r| !matches!(r, Requirement::Program { .. }));
                node
            })
            .collect(),
        roots: roots.to_vec(),
    };
    let selection = select::select(&problem)?;
    Some((selection.cost, selection.nodes.into_iter().map(|i| usable[i]).collect()))
}

/// Union of the `members` that share a requirement (or a root), with the
/// roots of each part. Lanes and shared values are matched by key; facts
/// (few) by implication. `Err` holds the roots no member mentions.
fn connected_parts(
    nodes: &[NodeEntry],
    members: &[usize],
    roots: &[Requirement],
) -> Result<Vec<(Vec<usize>, Vec<Requirement>)>, Vec<Requirement>> {
    let mut parts = UnionFind::new(nodes.len());
    let mut first_with: HashMap<&Requirement, usize> = HashMap::new();
    let (mut produced, mut consumed): (Vec<(usize, Fact)>, Vec<(usize, Fact)>) = (Vec::new(), Vec::new());
    for &i in members {
        let entry = &nodes[i];
        for (r, is_post) in entry.node.post.iter().map(|r| (r, true)).chain(entry.node.pre.iter().map(|r| (r, false))) {
            match r {
                Requirement::Fact(f) if is_post => produced.push((i, *f)),
                Requirement::Fact(f) => consumed.push((i, *f)),
                // Decided for all parts together (`select_members`).
                Requirement::Program { .. } => {}
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
    let mut unproduced: Vec<Requirement> = Vec::new();
    for r in roots {
        let producers: Vec<usize> = match r {
            Requirement::Fact(f) => producers_of(f),
            other => first_with.get(other).into_iter().copied().collect(),
        };
        let Some(&first) = producers.first() else {
            unproduced.push(r.clone());
            continue;
        };
        for &j in &producers[1..] {
            parts.union(first, j);
        }
        root_part.push((first, r.clone()));
    }
    if !unproduced.is_empty() {
        return Err(unproduced);
    }
    let mut by_part: BTreeMap<usize, (Vec<usize>, Vec<Requirement>)> = BTreeMap::new();
    for &i in members {
        by_part.entry(parts.find(i)).or_default().0.push(i);
    }
    for (node, r) in root_part {
        by_part.entry(parts.find(node)).or_default().1.push(r);
    }
    Ok(by_part.into_values().filter(|(_, roots)| !roots.is_empty()).collect())
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

    /// The values of an element group's cells (`Group::cells`): statements'
    /// values, or a written version's cells from its memory group.
    pub fn cells(&mut self, b: &mut IRBuilder, cells: &CellSource) -> Vec<Value> {
        match cells {
            CellSource::Stmts(stmts) => self.values(stmts),
            CellSource::Version(line) => {
                let all = self.shared_at::<Vec<Value>>(b, line.version, super::memory::versions::CELLS);
                line.addresses.iter().map(|&a| all[a].clone()).collect()
            }
        }
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

fn emit(
    graph: &IRGraph,
    groups: &[Group],
    direct: &HashMap<StmtId, StmtId>,
    nodes: Vec<NodeEntry>,
    selected: &[usize],
    dropped: &HashSet<StmtId>,
    in_place: &HashMap<StmtId, Vec<Fact>>,
) -> IRGraph {
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
    // The selected compositions: each array claim with the facts it is
    // composed of.
    let compositions: Vec<(Fact, Vec<Fact>)> = selected
        .iter()
        .filter(|&&i| matches!(nodes[i].kind, NodeKind::Composition))
        .filter_map(|&i| {
            let post = nodes[i].post_facts().next()?;
            Some((post, nodes[i].pre_facts().collect()))
        })
        .collect();
    let mut starts: Vec<usize> = Vec::new();
    let mut version_inputs: HashMap<usize, Vec<StmtId>> = HashMap::new();
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
                Requirement::Fact(_) | Requirement::Program { .. } => {}
            }
        }
        let needed: Vec<Fact> = entry
            .node
            .pre
            .iter()
            .filter_map(|r| match r {
                Requirement::Fact(f) => Some(f),
                _ => None,
            })
            .flat_map(|f| checks_justifying(f, &check_facts, &compositions))
            .collect();
        state.checks_for.insert(i, needed);
        // A version's cells (`SharedAt { at: version }`) are built from its
        // write's operands: the node that reads them starts after those.
        let versions: Vec<StmtId> = entry
            .node
            .pre
            .iter()
            .filter_map(|r| match r {
                Requirement::SharedAt { at, .. } => Some(*at),
                _ => None,
            })
            .collect();
        version_inputs.insert(i, versions);
        if let NodeKind::Offered(emitter) = entry.kind {
            if entry.node.post.iter().any(|r| matches!(r, Requirement::Lane(_))) {
                starts.push(i);
            }
            state.emitters.insert(i, Some(emitter));
        }
    }

    let lanes: HashMap<StmtId, &Lane> = groups.iter().flat_map(|g| g.lanes.iter().map(|l| (l.stmt, l))).collect();
    // What each node's start waits for besides the shared operands of the
    // groups it produces lanes of: the subjects of the checks it needs and
    // the versions whose cells it reads.
    let start_inputs: HashMap<usize, Vec<StmtId>> = state
        .checks_for
        .iter()
        .map(|(&node, facts)| {
            let subjects = facts.iter().flat_map(Fact::subjects);
            (node, subjects.chain(version_inputs.get(&node).into_iter().flatten().copied()).collect())
        })
        .collect();
    let order = emission_order(graph, groups, &lane_producer, &starts, &start_inputs);

    let mut b = IRBuilder::new();
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
                } else if let Some(cell) = direct.get(&s) {
                    lookup[cell].clone()
                } else if dropped.contains(&s) {
                    for fact in in_place.get(&s).into_iter().flatten() {
                        if state.emitted_checks.insert(*fact) {
                            checks::emit(&mut b, fact, &lookup);
                        }
                    }
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

/// The selected check facts that justify `fact`: the check that implies it,
/// or, for an array claim a selected composition establishes, the checks of
/// the facts it is composed of (each emitted before the node relying on the
/// claim starts).
fn checks_justifying(fact: &Fact, checks: &[Fact], compositions: &[(Fact, Vec<Fact>)]) -> Vec<Fact> {
    if let Some(check) = checks.iter().find(|c| c.implies(fact)) {
        return vec![*check];
    }
    compositions
        .iter()
        .find(|(post, _)| post.implies(fact))
        .map(|(_, basis)| basis.iter().flat_map(|f| checks.iter().find(|c| c.implies(f)).copied()).collect())
        .unwrap_or_default()
}

/// Statements in their order, each node's start placed right before the
/// first lane it produces once its inputs are emitted (Kahn's algorithm,
/// ties by position).
fn emission_order(
    graph: &IRGraph,
    groups: &[Group],
    lane_producer: &HashMap<StmtId, usize>,
    starts: &[usize],
    start_inputs: &HashMap<usize, Vec<StmtId>>,
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
            let group = group_of[lane];
            inputs.extend(group.shared.iter().map(|s| Unit::Stmt(*s)));
        }
        inputs.extend(start_inputs.get(&node).into_iter().flatten().map(|s| Unit::Stmt(*s)));
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
