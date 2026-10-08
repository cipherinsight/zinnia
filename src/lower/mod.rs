//! The array-operator lowering layer.
//!
//! The IR generator emits one placeholder per live output lane of an array
//! operator (`IR::ArrayOp`, see [`op`]) instead of expanding the operator
//! element by element. After dead-code elimination, this layer lowers every
//! placeholder by choosing, per operator group, the cheapest set of
//! *constructions* on the compile target that produces every live lane and
//! justifies every obligation:
//!
//! * [`graph`]: requirement nodes (lanes, shared values, facts) and groups;
//! * [`construction`]: the [`Construction`] trait (applicability →
//!   candidates with preconditions, postconditions, cost, emitter) and the
//!   compile [`Target`];
//! * [`select`]: minimum-cost acyclic justification (Z3, greedy fallback);
//! * [`emit`]: the pass [`ArrayLowering`] (graph building, selection,
//!   emission in topological order, discharge of checks);
//! * [`facts`] and [`checks`]: free facts, program checks as facts, and the
//!   emission of checks through `builder/guard.rs`;
//! * measurement only (`docs/ablations.md`): [`forcing`] (one fixed
//!   construction per group), `local` (each group decides on its own) and
//!   `report` (what each lowering chose).
//!
//! Families: [`sort`], [`gather`], [`stats`], [`memory`], [`linear`], [`permutation`]; scatter registers its own in [`registry`].
//! [`challenge`]: the program-wide verifier challenge of the constructions that check
//! identities at a random point (halo2-base only), and their probabilistic soundness.
//! See `eval-redo/layer/core-api.md` for how to add a family.

pub mod array_facts;
pub mod arrays;
pub mod challenge;
pub mod checks;
pub mod construction;
pub mod emit;
pub mod facts;
pub mod forcing;
pub mod gather;
pub mod graph;
pub mod linear;
mod local;
pub mod memory;
pub mod op;
pub mod permutation;
mod report;
pub mod scatter;
pub mod select;
mod simplify;
pub mod sort;
pub mod stats;
#[cfg(test)]
pub(crate) mod array_facts_tests;
#[cfg(test)]
mod tests;

pub use construction::{Candidate, Construction, Context, Emitter, Target};
pub use emit::{ArrayLowering, EmitCx};
pub use graph::{Bound, Claim, Fact, Group, GroupId, Lane, ListEntry, ListId, Requirement, Scope};
pub use op::{ArrayOp, Constant, OpKind, OperatorSemantics};

/// Every registered construction, family by family. The order is the
/// selection's tie-break among equally cheap candidates.
pub fn registry() -> Vec<&'static dyn Construction> {
    let mut all = Vec::new();
    all.extend(sort::constructions());
    all.extend(gather::constructions());
    all.extend(scatter::constructions());
    all.extend(stats::constructions());
    all.extend(memory::constructions());
    all.extend(linear::constructions());
    all.extend(permutation::constructions());
    all
}
