//! Local selection (measurement only, `ZINNIA_ABLATE_JOINT=1`,
//! `docs/ablations.md`): each group chooses its construction on its own.
//!
//! The compiler chooses the constructions of all groups in one optimization
//! (`emit::select_members` over every node): a value two groups share, a
//! fact one construction establishes for another, a program check one
//! construction discharges for another, and a program-wide facility (the
//! verifier challenge, the RAM trace) are each paid once, so a construction
//! that is costly alone can be the cheapest choice in the program.
//!
//! Here only the **decision** changes: which node produces each lane.
//!
//! 1. **Decisions.** The groups decide one at a time, in group order (the
//!    order of their first lanes in the program). A group's decision is the
//!    cheapest selection that justifies the group's own requirements with
//!    **every node at its full cost**: what an earlier group selected is not
//!    free, so the group pays for every value, fact, check and facility it
//!    uses. Its requirements are its live lanes and the program checks its
//!    constructions can establish or rely on (exactly the checks the
//!    compiler makes requirements of, attached to every group whose
//!    candidates mention them). Free facts stay free, and the program's
//!    checks stay what they are in the compiler: a group may rely on one or
//!    discharge it by its own postcondition, paying the check where it keeps
//!    it.
//! 2. **Commitment.** The soundness of some constructions depends on which
//!    construction produces another group's lanes (a read through the
//!    one-hot of an index wrap is sound only if that wrap's lanes are the
//!    ones the program checks), so decisions are committed: every lane a
//!    selected node produces is produced by that node. A later group may use
//!    a node that produces a committed lane only if it is that lane's
//!    producer, or a variant of it (the same construction producing a strict
//!    superset of its requirements, such as the same wrap also exposing its
//!    one-hot), which then replaces it. A node that produces lanes of several
//!    groups (the RAM trace of a written array produces the reads of its
//!    versions) commits all of them; a group whose lanes are all committed
//!    has nothing left to decide.
//! 3. **Completion.** With every lane's producer fixed, the rest is the
//!    compiler's selection: the program checks, check nodes, shared values
//!    and program-wide facilities the decided nodes need are selected
//!    together, at minimum cost, among the nodes that produce no lane. So a
//!    check a decided node proves is dropped, a shared value two groups both
//!    chose is built once, and a facility is paid once: only the decision is
//!    ablated, not the emission's deduplication.

use std::collections::{BTreeSet, HashMap};

use crate::types::StmtId;

use super::emit::{select_members, NodeEntry};
use super::graph::{Fact, Group, Requirement};

/// The selected nodes (indices into `nodes`) when every group decides on its
/// own; `Err` with the requirements of the first group that has no
/// selection consistent with the earlier decisions, or of the completion.
///
/// `roots` are the compiler's root requirements; `check_roots` the program
/// checks among them, each with the index of its own node (the assertion
/// kept).
pub(super) fn select(
    nodes: &[NodeEntry],
    groups: &[Group],
    roots: &[Requirement],
    check_roots: &[(Requirement, usize)],
) -> Result<Vec<usize>, Vec<Requirement>> {
    let producers = Producers::new(nodes);
    let attached = attach_checks(nodes, groups.len(), check_roots);
    let mut decisions = Decisions { committed: HashMap::new() };
    for group in groups {
        let mut group_roots: Vec<Requirement> = group
            .lanes
            .iter()
            .filter(|lane| !decisions.committed.contains_key(&lane.stmt))
            .map(|lane| Requirement::Lane(lane.stmt))
            .collect();
        group_roots.extend(attached[group.id].iter().cloned());
        if group_roots.is_empty() {
            continue;
        }
        let allowed: Vec<bool> = (0..nodes.len()).map(|i| decisions.consistent(nodes, i)).collect();
        let members = producers.reachable(nodes, &allowed, &group_roots);
        for node in select_members(nodes, &members, &group_roots)? {
            decisions.commit(nodes, node);
        }
    }
    // Completion: every lane by its committed producer, the rest selected
    // among the nodes that produce no lane.
    let members: Vec<usize> = (0..nodes.len())
        .filter(|&i| {
            let mut lanes = nodes[i].lanes().peekable();
            lanes.peek().is_none() || lanes.all(|lane| decisions.committed.get(&lane) == Some(&i))
        })
        .collect();
    let selected = select_members(nodes, &members, roots)?;
    for lane in groups.iter().flat_map(|g| &g.lanes) {
        let count = selected.iter().filter(|&&i| nodes[i].lanes().any(|s| s == lane.stmt)).count();
        assert_eq!(count, 1, "local selection: lane {} has {count} selected producers", lane.stmt);
    }
    Ok(selected)
}

/// The decisions taken so far: the producer of every committed lane.
struct Decisions {
    committed: HashMap<StmtId, usize>,
}

impl Decisions {
    /// Whether node `i` fits the decisions: each lane it produces is
    /// uncommitted, or committed to `i` or to a node `i` is a variant of.
    fn consistent(&self, nodes: &[NodeEntry], i: usize) -> bool {
        nodes[i].lanes().all(|lane| match self.committed.get(&lane) {
            None => true,
            Some(&p) => p == i || is_variant(nodes, p, i),
        })
    }

    /// Commit the lanes of the selected node `i` to it (a variant replaces
    /// the node it extends).
    fn commit(&mut self, nodes: &[NodeEntry], i: usize) {
        for lane in nodes[i].lanes() {
            if let Some(p) = self.committed.insert(lane, i).filter(|&p| p != i) {
                assert!(
                    is_variant(nodes, p, i),
                    "local selection: lane {lane} produced by two nodes ({} and {})",
                    nodes[p].name,
                    nodes[i].name
                );
            }
        }
    }
}

/// Whether node `wider` is a variant of node `p`: offered by the same
/// construction, producing every requirement `p` produces and more.
fn is_variant(nodes: &[NodeEntry], p: usize, wider: usize) -> bool {
    let (narrow, wide) = (&nodes[p], &nodes[wider]);
    narrow.name == wide.name
        && narrow.node.post.len() < wide.node.post.len()
        && narrow.node.post.iter().all(|r| wide.node.post.contains(r))
}

/// For every group, the program checks among `check_roots` that a node
/// offered for it can establish or relies on (the compiler's reasons for
/// making a check a requirement, restricted to the group's candidates).
fn attach_checks(nodes: &[NodeEntry], groups: usize, check_roots: &[(Requirement, usize)]) -> Vec<Vec<Requirement>> {
    let mut attached = vec![Vec::new(); groups];
    for (root, check) in check_roots {
        let forms: Vec<Fact> = nodes[*check].post_facts().collect();
        for (g, roots) in attached.iter_mut().enumerate() {
            let related = nodes.iter().filter(|n| n.group == Some(g)).any(|n| {
                let provable = n.post_facts().any(|e| forms.iter().any(|f| e.implies(f)));
                let useful = n.pre_facts().any(|w| forms.iter().any(|f| f.implies(&w)));
                provable || useful
            });
            if related {
                roots.push(root.clone());
            }
        }
    }
    attached
}

/// The producers of every requirement, for the closure of one group's
/// problem: lanes, shared values and program-wide requirements by key,
/// facts by implication.
struct Producers {
    by_key: HashMap<Requirement, Vec<usize>>,
    facts: Vec<(usize, Fact)>,
}

impl Producers {
    fn new(nodes: &[NodeEntry]) -> Producers {
        let mut by_key: HashMap<Requirement, Vec<usize>> = HashMap::new();
        let mut facts = Vec::new();
        for (i, entry) in nodes.iter().enumerate() {
            for r in &entry.node.post {
                match r {
                    Requirement::Fact(f) => facts.push((i, *f)),
                    other => by_key.entry(other.clone()).or_default().push(i),
                }
            }
        }
        Producers { by_key, facts }
    }

    fn of(&self, required: &Requirement) -> Vec<usize> {
        match required {
            Requirement::Fact(f) => self.facts.iter().filter(|(_, p)| p.implies(f)).map(|(i, _)| *i).collect(),
            other => self.by_key.get(other).cloned().unwrap_or_default(),
        }
    }

    /// The allowed nodes that can take part in justifying `roots`: their
    /// producers, the producers of those nodes' preconditions, and so on
    /// (index order).
    fn reachable(&self, nodes: &[NodeEntry], allowed: &[bool], roots: &[Requirement]) -> Vec<usize> {
        let mut seen = vec![false; nodes.len()];
        let mut asked: BTreeSet<Requirement> = BTreeSet::new();
        let mut queue: Vec<Requirement> = roots.to_vec();
        while let Some(required) = queue.pop() {
            if !asked.insert(required.clone()) {
                continue;
            }
            for i in self.of(&required) {
                if allowed[i] && !seen[i] {
                    seen[i] = true;
                    queue.extend(nodes[i].node.pre.iter().cloned());
                }
            }
        }
        (0..nodes.len()).filter(|&i| seen[i]).collect()
    }
}
