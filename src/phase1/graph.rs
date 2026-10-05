//! The candidate graph G of `revision/phase1/README.md` §2.
//!
//! G is an AND/OR graph with two kinds of nodes. A *requirement* is a
//! question — `Represent`, `Fetch` or `Enforce` — and points at every
//! implementation node that answers it, of which phase 2 selects exactly
//! one. An *implementation node* is one instance of one rule of the library
//! and points at every requirement it requires, all of which are selected
//! with it. G contains each requirement once; the identity that makes it
//! once is [`RequirementId`].
//!
//! The invariant checker at the end of the file is §2's "Invariants" list,
//! as far as a structural check can see it: acyclicity, no dangling edge,
//! every root present, the epoch discipline below an `Enforce`, and the
//! answer record holding exactly the questions the nodes asked.

use std::collections::{HashMap, HashSet};

use crate::phase0::desc::ValueKey;
use crate::phase0::term::{BoolId, TermId};

use crate::phase1::interface::{Batch, Family, Interface};
use crate::phase1::oracle::answers::AnswerId;
use crate::phase1::rules::RuleId;

/// Index of a requirement node in [`Graph::requirements`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ReqId(pub u32);

/// Index of an implementation node in [`Graph::implementations`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ImplId(pub u32);

/// What a `Represent` asks about: a value of Δ, or a term of Φ
/// (`rules/README.md` §3.2).
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Target {
    Value(ValueKey),
    /// An integer term, by its term id.
    Term(TermId),
    /// A Boolean term, by its term id.
    BoolTerm(BoolId),
}

impl Target {
    pub fn as_value(&self) -> Option<&ValueKey> {
        match self {
            Target::Value(key) => Some(key),
            _ => None,
        }
    }

    pub fn render(&self) -> String {
        match self {
            Target::Value(key) => key.encode(),
            Target::Term(t) => format!("t{}", t.index()),
            Target::BoolTerm(b) => format!("b{}", b.index()),
        }
    }
}

/// Which check an `Enforce` root is about.
///
/// `../README.md` §2 makes one root per check of Φ whose CANFAIL answer is
/// not "no", and one per `precondition` constraint over several variables
/// with φ = true. Both are constraints of Φ, so both are named by the index
/// of the constraint; only a check has a position in the check order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum EnforceTarget {
    /// The constraint at this index of Φ, which is a check.
    Check { constraint: usize },
    /// The constraint at this index of Φ, which is a `precondition`.
    Precondition { constraint: usize },
}

impl EnforceTarget {
    pub fn constraint(self) -> usize {
        match self {
            EnforceTarget::Check { constraint } | EnforceTarget::Precondition { constraint } => {
                constraint
            }
        }
    }
}

/// The form of a requirement (`../README.md` §2).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Form {
    Represent { target: Target, interface: Interface },
    Fetch { value: ValueKey, batch: Batch },
    Enforce { target: EnforceTarget },
}

impl Form {
    pub fn family(&self) -> Option<Family> {
        match self {
            Form::Represent { interface, .. } => Some(interface.family()),
            Form::Fetch { .. } => Some(Family::Values),
            Form::Enforce { .. } => None,
        }
    }
}

/// One parameter of an implementation node (`../README.md` §2,
/// "Invariants").
///
/// Every capacity, query count and bit width is an integer. The one
/// exception `../../README.md` §5.7 allows is "a rule-internal parameter
/// with a real trade-off", which becomes "an integer variable that phase 2
/// receives with its range": the total query count `Q_o` of a ROM context
/// (`rules/README.md` §4.4), whose value is the sum over the readers phase 2
/// selects. It is phase 2's variable and not a variable of Φ, so the
/// invariant "no node of G mentions a variable of Φ" still holds by typing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Parameter {
    Fixed(i128),
    Ranged { lo: i128, hi: i128 },
}

impl Parameter {
    /// The value, when the parameter is an integer.
    pub fn fixed(self) -> Option<i128> {
        match self {
            Parameter::Fixed(value) => Some(value),
            Parameter::Ranged { .. } => None,
        }
    }

    /// The range phase 2 chooses within: a fixed parameter's own value,
    /// twice.
    pub fn range(self) -> (i128, i128) {
        match self {
            Parameter::Fixed(value) => (value, value),
            Parameter::Ranged { lo, hi } => (lo, hi),
        }
    }
}

/// The sharing identity of `../README.md` §5.1: two requirements that differ
/// only in it are two nodes, and Ω caps how many a value may have.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SharingId(pub u32);

impl SharingId {
    /// The identity every consumer shares unless a rule asks for a
    /// recomputation instance.
    pub const PRIMARY: SharingId = SharingId(0);
}

/// The epoch filter of `rules/README.md` §6.2, the one restriction a row
/// carries in its identity (`../README.md` §2, §4).
///
/// Below `Enforce(k)` no node may have assumed check k or a later one, so
/// the filter has to be part of the requirement's identity: phase 2 must not
/// be free to select an excluded provider for the wire that verifies k.
///
/// Acyclicity is not a filter of this kind. §4 keeps a stack of the rows in
/// progress ([`crate::phase1::table::Table::is_filling`]): a port that asks
/// for a row already on the stack drops that candidate, and a row that
/// dropped one is provisional — its requirement nodes are its own
/// ([`RequirementId::provisional`]) and it is not memoised, so a later
/// consumer outside the cycle refills it unrestricted. Putting the
/// ancestors in the identity instead would give every consumer of a value
/// its own row, which is the opposite of what §4 asks for: "the consumers of
/// one value share its providers and phase 2 counts them once".
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct EpochFilter {
    /// `Some(p)`: only nodes with epoch < p may appear.
    pub epoch_below: Option<u32>,
}

impl EpochFilter {
    pub fn is_empty(&self) -> bool {
        self.epoch_below.is_none()
    }

    /// The filter below `Enforce(k)`: this one tightened to position(k).
    pub fn below_check(&self, position: u32) -> EpochFilter {
        EpochFilter {
            epoch_below: Some(match self.epoch_below {
                Some(previous) => previous.min(position),
                None => position,
            }),
        }
    }

    /// Whether a node of this epoch may appear in a row carrying this set.
    pub fn admits_epoch(&self, epoch: u32) -> bool {
        match self.epoch_below {
            Some(position) => epoch < position,
            None => true,
        }
    }
}

/// The identity of a requirement node (`../README.md` §2).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RequirementId {
    pub form: Form,
    /// `g`, the execution condition the requirement is asked under.
    pub condition: BoolId,
    pub sharing: SharingId,
    pub filter: EpochFilter,
    /// The epoch of the nodes that provide it; `None` for an `Enforce`,
    /// which is provided at no epoch of its own.
    pub epoch: Option<u32>,
    /// `Some(n)` when the row that created it was provisional (§4,
    /// "Acyclicity by construction"): the number is that fill's, so two
    /// fills of one row never share a requirement node. A provisional fill
    /// is missing the candidates the in-progress cut dropped, and attaching
    /// a later, unrestricted fill's providers to the same node would put the
    /// dropped candidate back and close the cycle in G.
    pub provisional: Option<u32>,
}

/// A requirement node with the implementation nodes that answer it.
#[derive(Debug, Clone)]
pub struct RequirementNode {
    pub id: RequirementId,
    /// The OR edges: exactly one is selected in phase 2.
    pub providers: Vec<ImplId>,
}

/// What makes two implementation nodes the same node.
///
/// §3.4: "One implementation node may be attached to several requirement
/// nodes when its output bundles are projections of one set of wires". The
/// rows of the two requirements are filled separately, so the same instance
/// is offered twice; this is what recognises it as one node, with one
/// selection variable in phase 2 and its cost counted once.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct NodeIdentity {
    rule: RuleId,
    variant: Option<&'static str>,
    matched: Vec<ValueKey>,
    inputs: Vec<ReqId>,
    outputs: Vec<ReqId>,
    parameters: Vec<(String, Parameter)>,
    epoch: u32,
    cost: i128,
    readers: Vec<(ReqId, i128)>,
}

impl NodeIdentity {
    fn of(node: &ImplNode) -> NodeIdentity {
        NodeIdentity {
            rule: node.rule,
            variant: node.variant,
            matched: node.matched.clone(),
            inputs: node.inputs.clone(),
            outputs: node.outputs.clone(),
            parameters: node.parameters.clone(),
            epoch: node.epoch,
            cost: node.cost,
            readers: node.readers.clone(),
        }
    }
}

/// An implementation node: one instance of one rule (`../README.md` §2).
#[derive(Debug, Clone)]
pub struct ImplNode {
    pub rule: RuleId,
    /// The keys of Δ the rule's `source_pattern` matched.
    pub matched: Vec<ValueKey>,
    /// The AND edges, one per required input port.
    pub inputs: Vec<ReqId>,
    /// The requirements this node's wires provide (§3.4: one node may be
    /// attached to several).
    pub outputs: Vec<ReqId>,
    /// Every capacity, query count and bit width, as an integer — or, for
    /// the one parameter `../../README.md` §5.7 admits, as a range.
    pub parameters: Vec<(String, Parameter)>,
    /// The position in Φ's check order whose checks the node assumed.
    pub epoch: u32,
    pub cost: i128,
    /// The primitives of `rules/README.md` §7 the emitter counts, which the
    /// cost function turned into `cost`. Phase 3 reads them to know which
    /// gadget the node denotes — an `fop⟨name⟩` names the backend
    /// instruction outright — and a test asserts a row against them.
    pub primitives: Vec<(crate::phase1::cost::Primitive, i128)>,
    /// The ids in A of the questions this instance asked.
    pub answers: Vec<AnswerId>,
    /// The readers of a ROM context, with the queries each registers
    /// (`rules/README.md` §4.4): the list phase 2 sums `Q_o` over. Empty on
    /// every other node.
    pub readers: Vec<(ReqId, i128)>,
    /// The rule's variant name, when it has one.
    pub variant: Option<&'static str>,
    /// Whether the dominance pruning of §4 dropped the node. A dropped node
    /// is in no requirement's provider list and so in no subgraph phase 2
    /// can select; it stays in the record because the questions it asked are
    /// in A.
    pub dropped: bool,
}

impl ImplNode {
    /// The float instructions this node emits, in emitter order and with
    /// their counts (`rules/README.md` §7, `fop⟨name⟩`).
    pub fn fops(&self) -> Vec<(&'static str, i128)> {
        self.primitives
            .iter()
            .filter_map(|&(primitive, times)| match primitive {
                crate::phase1::cost::Primitive::Float(op) => Some((op.as_str(), times)),
                _ => None,
            })
            .collect()
    }

    /// The value of an integer parameter; `None` for a ranged one.
    pub fn parameter(&self, name: &str) -> Option<i128> {
        self.parameter_of(name).and_then(Parameter::fixed)
    }

    pub fn parameter_of(&self, name: &str) -> Option<Parameter> {
        self.parameters
            .iter()
            .find(|(key, _)| key == name)
            .map(|&(_, value)| value)
    }
}

/// The candidate graph.
#[derive(Debug, Default)]
pub struct Graph {
    pub requirements: Vec<RequirementNode>,
    pub implementations: Vec<ImplNode>,
    /// The roots of §2: one `Represent(v, Export)` per delivered value and
    /// the `Enforce` roots.
    pub roots: Vec<ReqId>,
    index: HashMap<RequirementId, ReqId>,
    nodes: HashMap<NodeIdentity, ImplId>,
}

impl Graph {
    /// The requirement node for `id`, created if absent. G contains each
    /// requirement once (§2).
    pub fn requirement(&mut self, id: RequirementId) -> ReqId {
        if let Some(&existing) = self.index.get(&id) {
            return existing;
        }
        let at = ReqId(self.requirements.len() as u32);
        self.index.insert(id.clone(), at);
        self.requirements.push(RequirementNode { id, providers: Vec::new() });
        at
    }

    pub fn find(&self, id: &RequirementId) -> Option<ReqId> {
        self.index.get(id).copied()
    }

    pub fn node(&self, at: ReqId) -> &RequirementNode {
        &self.requirements[at.0 as usize]
    }

    pub fn implementation(&self, at: ImplId) -> &ImplNode {
        &self.implementations[at.0 as usize]
    }

    /// Add an implementation node and attach it to every requirement it
    /// outputs, or return the node that is already there.
    ///
    /// A node attached to several requirements (§3.4) is offered once per
    /// row those requirements were filled in, and is one node.
    pub fn add_implementation(&mut self, node: ImplNode) -> ImplId {
        let identity = NodeIdentity::of(&node);
        if let Some(&existing) = self.nodes.get(&identity) {
            return existing;
        }
        let at = ImplId(self.implementations.len() as u32);
        for &output in &node.outputs {
            self.requirements[output.0 as usize].providers.push(at);
        }
        self.nodes.insert(identity, at);
        self.implementations.push(node);
        at
    }

    /// Drop an implementation node from every requirement that names it.
    ///
    /// Used by the dominance pruning of §4: the node stays in the vector so
    /// that no index moves, no requirement points at it any more — which is
    /// what [`Graph::live_implementations`] counts — and its identity leaves
    /// the index, so that a later row offering the same instance builds its
    /// own node and prunes it on its own merits.
    pub fn detach(&mut self, at: ImplId) {
        for requirement in &mut self.requirements {
            requirement.providers.retain(|&provider| provider != at);
        }
        let node = &mut self.implementations[at.0 as usize];
        node.dropped = true;
        let identity = NodeIdentity::of(node);
        self.nodes.remove(&identity);
    }

    /// The implementation nodes some requirement still points at.
    pub fn live_implementations(&self) -> Vec<ImplId> {
        let mut live: Vec<ImplId> = self
            .requirements
            .iter()
            .flat_map(|requirement| requirement.providers.iter().copied())
            .collect();
        live.sort();
        live.dedup();
        live
    }

    pub fn requirement_count(&self) -> usize {
        self.requirements.len()
    }
}

/// What the invariant checker of §2 found.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Violation {
    pub invariant: &'static str,
    pub detail: String,
}

impl std::fmt::Display for Violation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.invariant, self.detail)
    }
}

/// Check the invariants of `../README.md` §2 on a finished graph.
///
/// The three that a structural check can see are checked here: G is
/// acyclic, every edge lands on a node that exists and every root is a
/// requirement of G; the wires that verify a check come from nodes that
/// assumed no check at or after it; and A holds exactly the questions the
/// nodes asked. "No node mentions a variable of Φ" holds by typing —
/// [`ImplNode::parameters`] is a list of integers — and is checked as such.
pub fn violations(graph: &Graph, answers: &crate::phase1::oracle::answers::Answers) -> Vec<Violation> {
    let mut found = Vec::new();
    check_edges(graph, &mut found);
    check_acyclic(graph, &mut found);
    check_epoch_filter(graph, &mut found);
    check_answers(graph, answers, &mut found);
    found
}

fn check_edges(graph: &Graph, found: &mut Vec<Violation>) {
    let requirements = graph.requirements.len() as u32;
    let implementations = graph.implementations.len() as u32;
    for (at, node) in graph.implementations.iter().enumerate() {
        for edge in node.inputs.iter().chain(node.outputs.iter()) {
            if edge.0 >= requirements {
                found.push(Violation {
                    invariant: "edges land on nodes of G",
                    detail: format!("implementation {} names requirement {}", at, edge.0),
                });
            }
        }
    }
    for (at, node) in graph.requirements.iter().enumerate() {
        for provider in &node.providers {
            if provider.0 >= implementations {
                found.push(Violation {
                    invariant: "edges land on nodes of G",
                    detail: format!("requirement {} names implementation {}", at, provider.0),
                });
                continue;
            }
            if !graph.implementations[provider.0 as usize]
                .outputs
                .contains(&ReqId(at as u32))
            {
                found.push(Violation {
                    invariant: "a provider names the requirement it answers",
                    detail: format!(
                        "requirement {} names implementation {}, which does not output it",
                        at, provider.0
                    ),
                });
            }
        }
    }
    for root in &graph.roots {
        if root.0 >= requirements {
            found.push(Violation {
                invariant: "every root is a requirement of G",
                detail: format!("root {}", root.0),
            });
        }
    }
}

/// G is acyclic by construction (§2), so a depth-first search over the
/// requirement → implementation → requirement edges finds no back edge.
fn check_acyclic(graph: &Graph, found: &mut Vec<Violation>) {
    #[derive(Clone, Copy, PartialEq)]
    enum Mark {
        Open,
        Done,
    }
    let mut mark: HashMap<ReqId, Mark> = HashMap::new();
    let mut stack: Vec<(ReqId, usize)> = Vec::new();
    for start in (0..graph.requirements.len()).map(|at| ReqId(at as u32)) {
        if mark.contains_key(&start) {
            continue;
        }
        stack.push((start, 0));
        mark.insert(start, Mark::Open);
        while let Some(&mut (at, ref mut cursor)) = stack.last_mut() {
            let children = successors(graph, at);
            if *cursor >= children.len() {
                mark.insert(at, Mark::Done);
                stack.pop();
                continue;
            }
            let next = children[*cursor];
            *cursor += 1;
            match mark.get(&next) {
                Some(Mark::Open) => found.push(Violation {
                    invariant: "G is acyclic",
                    detail: format!("requirement {} reaches itself through {}", at.0, next.0),
                }),
                Some(Mark::Done) => {}
                None => {
                    mark.insert(next, Mark::Open);
                    stack.push((next, 0));
                }
            }
        }
    }
}

fn successors(graph: &Graph, at: ReqId) -> Vec<ReqId> {
    let mut out = Vec::new();
    for provider in &graph.node(at).providers {
        out.extend(graph.implementation(*provider).inputs.iter().copied());
    }
    out.sort();
    out.dedup();
    out
}

/// The wires that verify a check are computed by nodes that assumed no check
/// at or after it (§2, fourth invariant): every node reachable below an
/// `Enforce(k)` has epoch < position(k).
fn check_epoch_filter(graph: &Graph, found: &mut Vec<Violation>) {
    for root in &graph.roots {
        let node = graph.node(*root);
        let Some(position) = node.id.filter.epoch_below else {
            continue;
        };
        let mut seen: HashSet<ReqId> = HashSet::new();
        let mut stack = vec![*root];
        while let Some(at) = stack.pop() {
            if !seen.insert(at) {
                continue;
            }
            for provider in &graph.node(at).providers {
                let implementation = graph.implementation(*provider);
                if implementation.epoch >= position {
                    found.push(Violation {
                        invariant: "no check is verified on a capacity it justified",
                        detail: format!(
                            "{} at epoch {} provides a wire below Enforce at position {}",
                            implementation.rule.as_str(),
                            implementation.epoch,
                            position
                        ),
                    });
                }
                stack.extend(implementation.inputs.iter().copied());
            }
        }
    }
}

/// A contains exactly the questions the nodes asked (§2, fifth invariant).
fn check_answers(
    graph: &Graph,
    answers: &crate::phase1::oracle::answers::Answers,
    found: &mut Vec<Violation>,
) {
    let mut asked: HashSet<AnswerId> = HashSet::new();
    for node in &graph.implementations {
        asked.extend(node.answers.iter().copied());
    }
    for id in answers.roots() {
        asked.insert(id);
    }
    for id in answers.ids() {
        if !asked.contains(&id) {
            found.push(Violation {
                invariant: "A contains exactly the questions nodes asked",
                detail: format!("answer {} is in A and in no node", id.0),
            });
        }
    }
    for id in asked {
        if answers.get(id).is_none() {
            found.push(Violation {
                invariant: "every question a node asked is in A",
                detail: format!("answer {} is named by a node and not in A", id.0),
            });
        }
    }
}
