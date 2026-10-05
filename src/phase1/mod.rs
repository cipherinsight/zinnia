//! Phase 1: the candidate graph.
//!
//! Phase 1 is the second stage of the revised compiler
//! (`revision/phase1/README.md`): it reads the relation system phase 0
//! produced and builds, for every delivered value, the ways it can be
//! implemented, with every capacity and every cost already an integer.
//! Phase 2 selects one way per requirement; phase 3 emits wires. Phase 1 is
//! the only stage that calls a solver.
//!
//! ```text
//! (Φ, Δ, R, P, Lib, Ω, cost) ↦ (G, A)   or   reject
//! ```
//!
//! # Module layout
//!
//! | Module | Content |
//! |---|---|
//! | [`interface`] | the interface catalogue of `rules/README.md` §4: the families, the layout λ, the μ grammar |
//! | [`graph`] | requirement and implementation nodes, their identity, the edges, and the invariant checker of §2 |
//! | [`table`] | the provider table of §4: the memo key, the rows, the weaker-condition lookup |
//! | [`oracle`] | the five kinds of question, answered cheapest layer first, over a sliced Φ_e |
//! | [`build`] | the construction order of §4: the CANFAIL sweep, the roots, the depth-first expansion, the pruning |
//! | [`ast_index`] | key → AST node: the inverse of phase 0's key assignment |
//! | [`rules`] | the `Rule` trait and the library, one module per rule file |
//! | [`cost`] | the cost function and the unit cost over the primitives of `rules/README.md` §7 |
//! | [`omega`] | Ω, the candidate policy of §6.4 |
//! | [`export`] | the JSON of G, A and the statistics |
//!
//! # State of the implementation
//!
//! Step 1 built the construction algorithm of §4 with nine scalar rules.
//! Step 2 adds the 22 rules that cover programs whose arrays have fixed
//! shapes — the representation, mapping, reading, elementwise and reduction
//! groups, with `INDEX_NORMALIZE`, `SHAPE_CHECK` and `CONSTANT_SIZED` — so
//! that a `Fetch` row is filled by the reading rules of §4 step 4 and an
//! array value has providers. Step 2b gives phase 1 a term arena of its own
//! ([`build::Builder::terms`], §1.1), so a condition, a comparison to prove
//! and a composed coordinate are terms it builds rather than shapes it works
//! around; and it adds the Float rows, one `fop⟨name⟩` per backend float
//! instruction ([`cost::FloatOp`]). The rules that need a run-time length
//! (`BUILD_MASKED`, `PACK_BY_KEY`, `PACK_SEGMENTS`, `CONCAT`, `SETITEM`, the
//! masks and the ordering group) are step 3's.
//!
//! # Building and testing
//!
//! ```text
//! Z3LIB=$(.venv/bin/python -c "import z3,pathlib;\
//!   p=pathlib.Path(z3.__file__).parent;print(next(p.glob('**/libz3.so')).parent)")
//! PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 LIBRARY_PATH=$Z3LIB LD_LIBRARY_PATH=$Z3LIB \
//!   cargo test --lib --no-default-features --features halo2-backend phase1
//! ```

pub mod ast_index;
pub(crate) mod build;
pub mod cost;
pub mod export;
pub mod graph;
pub mod interface;
pub mod omega;
pub mod oracle;
pub(crate) mod rules;
pub mod table;

#[cfg(test)]
mod tests;

use std::collections::HashMap;

use crate::ast::ASTCircuit;
use crate::phase0::Phase0Result;

use cost::Cost;
use graph::Graph;
use omega::Omega;
use oracle::answers::Answers;
use oracle::stats::Stats;
use oracle::Limits;

/// Why phase 1 refused a program.
///
/// A rejection is a property of the program together with the rule library:
/// a value with no provider names the rule that would provide it, which is
/// what tells a missing rule from a defect.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Reject {
    /// A root requirement no rule of the library answers.
    NoProvider { requirement: String, detail: String },
    /// A key of R with no entry in Δ, which phase 0 rules out.
    NoDescriptor { value: String },
    /// G grew past `Ω.node_budget` (`rules/README.md` §6.4).
    NodeBudget { budget: usize },
    /// A port reads a value under a condition that does not imply the
    /// value's ψ, so §4's "Conditions of child requirements" does not hold
    /// for it: the consumer cannot name the value it asks for.
    ChildCondition { value: String, under: String, psi: String },
}

impl Reject {
    pub fn reason(&self) -> &'static str {
        match self {
            Reject::NoProvider { .. } => "no_provider",
            Reject::NoDescriptor { .. } => "no_descriptor",
            Reject::NodeBudget { .. } => "node_budget",
            Reject::ChildCondition { .. } => "child_condition",
        }
    }

    pub fn detail(&self) -> String {
        match self {
            Reject::NoProvider { requirement, detail } => {
                format!("{}: {}", requirement, detail)
            }
            Reject::NoDescriptor { value } => value.clone(),
            Reject::NodeBudget { budget } => format!("the node budget is {}", budget),
            Reject::ChildCondition { value, under, psi } => format!(
                "{} is read under `{}`, which does not imply its ψ `{}`",
                value, under, psi
            ),
        }
    }

    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({"reason": self.reason(), "detail": self.detail()})
    }
}

impl std::fmt::Display for Reject {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.reason(), self.detail())
    }
}

/// The result of phase 1: the candidate graph G, the answer record A, the
/// term arena the two are written over, and the statistics that say what the
/// oracle cost.
pub struct Phase1Result {
    pub graph: Graph,
    pub answers: Answers,
    /// Phase 0's arena, cloned and grown with the terms phase 1 built
    /// (`revision/phase1/README.md` §1.1). Every id below [`Phase1Result::base_nodes`]
    /// is phase 0's term; every id at or above it is phase 1's.
    pub terms: crate::phase0::term::TermArena,
    /// `terms.node_count()` of phase 0's arena, where phase 1's terms start.
    pub base_nodes: usize,
    pub stats: Stats,
    /// Rows that had no provider, with the rule that would have filled them.
    /// Not a rejection: a row a consumer asked and did not need is a note.
    pub missing: Vec<String>,
}

impl Phase1Result {
    /// The invariants of `revision/phase1/README.md` §2, as far as a
    /// structural check sees them.
    pub fn violations(&self) -> Vec<graph::Violation> {
        graph::violations(&self.graph, &self.answers)
    }
}

/// Run phase 1 on one phase 0 record.
pub fn build(
    result: &Phase0Result,
    circuit: &ASTCircuit,
    omega: &Omega,
    cost: &dyn Cost,
) -> Result<Phase1Result, Reject> {
    build_with(result, circuit, &HashMap::new(), omega, cost, Limits::default())
}

/// [`build`] with the chips the circuit inlines and the oracle's time
/// limits, which the pyo3 binding and the tests need.
/// The stack the construction runs on.
///
/// §4's expansion is depth first and recurses once per row, and a row's
/// frame carries the request, the rule's candidates and the child rows. A
/// program whose scalar values form a chain thousands long — an unrolled
/// loop that folds one accumulator — recurses that deep, which is more than
/// the eight megabytes a thread is given by default. The construction
/// therefore runs on a thread of its own with room for it, rather than
/// crashing on a program phase 0 accepted.
const STACK_BYTES: usize = 512 * 1024 * 1024;

pub fn build_with(
    result: &Phase0Result,
    circuit: &ASTCircuit,
    chips: &HashMap<String, serde_json::Value>,
    omega: &Omega,
    cost: &dyn Cost,
    limits: Limits,
) -> Result<Phase1Result, Reject> {
    std::thread::scope(|scope| {
        let worker = std::thread::Builder::new()
            .stack_size(STACK_BYTES)
            .name("zinnia-phase1".to_string())
            .spawn_scoped(scope, || {
                construct(result, circuit, chips, omega, cost, limits)
            })
            .expect("a thread for the construction");
        match worker.join() {
            Ok(outcome) => outcome,
            // The construction panicked; the panic belongs to the caller,
            // not to this thread.
            Err(panic) => std::panic::resume_unwind(panic),
        }
    })
}

fn construct(
    result: &Phase0Result,
    circuit: &ASTCircuit,
    chips: &HashMap<String, serde_json::Value>,
    omega: &Omega,
    cost: &dyn Cost,
    limits: Limits,
) -> Result<Phase1Result, Reject> {
    let mut builder = build::Builder::new(result, circuit, chips, omega, cost, limits);
    // §4, in order: the CANFAIL sweep, the roots, the depth-first expansion.
    let enforce = builder.canfail_sweep();
    builder.expand(&enforce)?;
    let build::Builder { graph, oracle, table, missing, terms, base_nodes, .. } = builder;
    let mut stats = oracle.stats;
    stats.requirements = graph.requirement_count() as u64;
    stats.implementations = graph.live_implementations().len() as u64;
    stats.roots = graph.roots.len() as u64;
    stats.rows = table.len() as u64;
    stats.terms_built = (terms.node_count() - base_nodes) as u64;
    Ok(Phase1Result { graph, answers: oracle.answers, terms, base_nodes, stats, missing })
}
