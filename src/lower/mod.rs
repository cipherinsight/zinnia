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
//!   emission of checks through `builder/guard.rs`.
//!
//! Families: [`sort`], [`gather`]; scatter registers its own in [`registry`].
//! See `eval-redo/layer/core-api.md` for how to add a family.

pub mod checks;
pub mod construction;
pub mod emit;
pub mod facts;
pub mod gather;
pub mod graph;
pub mod op;
pub mod scatter;
pub mod select;
pub mod sort;
#[cfg(test)]
mod tests;

pub use construction::{Candidate, Construction, Context, Emitter, Target};
pub use emit::{ArrayLowering, EmitCx};
pub use graph::{Bound, Claim, Fact, Group, GroupId, Lane, Requirement, Scope};
pub use op::{ArrayOp, Constant, OpKind, OperatorSemantics};

/// Every registered construction, family by family. The order is the
/// selection's tie-break among equally cheap candidates.
pub fn registry() -> Vec<&'static dyn Construction> {
    let mut all = Vec::new();
    all.extend(sort::constructions());
    all.extend(gather::constructions());
    all.extend(scatter::constructions());
    all
}
