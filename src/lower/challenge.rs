//! The verifier challenge: a program-wide requirement of the constructions
//! that check an identity at a random point.
//!
//! Some identities are far cheaper to check than to compute: a multiset
//! equality as a grand product (`sort`'s grand-product plan), a matrix
//! product by Freivalds' check, a convolution as a product of polynomials
//! (`linear`). Each needs a field element the prover cannot predict when it
//! fixes the values being checked. On halo2-base this is a verifier
//! challenge of the proof system: every first-phase advice column (every
//! value the program computes or the prover supplies) is committed, the
//! challenge is squeezed from the transcript, and the checks are laid out in
//! a second phase (`prove/halo2_base/circuit.rs`). Noir has no such
//! challenge (an in-circuit hash would cost more than the checks save), so
//! the requirement has no producer there and no challenge construction is
//! ever selected or offered on Noir.
//!
//! **Probabilistic soundness.** A construction that reads the challenge is
//! sound except with probability at most `d / |F|` per check, where `d` is
//! the degree of the check's polynomial in the challenges (Schwartz-Zippel:
//! a nonzero polynomial of total degree `d` vanishes on at most a `d / |F|`
//! fraction of points) and `|F| ≈ 2^254` is the BN254 scalar field. Each
//! construction's doc comment states its `d` (at most a few thousand for
//! the shapes compiled here, so the bound is below `2^-240`). Every other
//! construction of the layer is exact. The challenge-based constructions
//! only establish facts (an identity) about values the prover committed in
//! the first phase, or order such values; no first-phase value depends on a
//! challenge (the backend asserts this).
//!
//! **Selection.** The challenge is [`CHALLENGE`], a `Requirement::Program`:
//! every challenge construction lists it as a precondition, and its producer
//! has a one-time cost ([`producer_cost`]). The selection pays it at most
//! once and uses challenge constructions only when the whole program is
//! cheaper with them (`emit::select_all`).

use super::construction::Target;
use super::graph::Requirement;

/// The name of the challenge requirement.
pub const CHALLENGE_NAME: &str = "verifier challenge";

/// The program-wide requirement "verifier challenges are available".
pub const CHALLENGE: Requirement = Requirement::Program { name: CHALLENGE_NAME };

/// The one-time cost of enabling the second phase on halo2-base, in cells.
///
/// Its own cells are few (one challenge cell each; the powers and shifts a
/// check precomputes are priced by the check), but the second phase adds
/// committed polynomials: the challenge column, at least one second-phase
/// gate column, their share of the permutation argument, and a selector per
/// challenge, about four columns. At `k = 9` (every perf case that can use
/// a challenge construction) that is about 2,000 cells. Measured
/// (`eval-redo/layer/layer-challenge.md`, argsorts of 64 to 160 elements,
/// interleaved proofs at `k = 9`): the grand-product sort of 64 saves 4,300
/// cells against the routed network and proves in about the same time, of
/// 100 saves 7,900 cells and proves about 1.5 times faster. Above `k = 9`
/// a column holds more cells, but the constructions' savings grow faster
/// (the scaling tables there).
pub const HALO2_BASE_CHALLENGE_COST: u64 = 2_000;

/// The one-time cost of producing the program-wide requirement `name` on
/// `target`; `None` when the target cannot produce it.
pub fn producer_cost(name: &str, target: Target) -> Option<u64> {
    match (name, target) {
        (CHALLENGE_NAME, Target::Halo2Base) => Some(HALO2_BASE_CHALLENGE_COST),
        _ => None,
    }
}

/// Whether `target` offers verifier challenges (constructions that need
/// one offer no candidate elsewhere).
pub fn available(target: Target) -> bool {
    producer_cost(CHALLENGE_NAME, target).is_some()
}
