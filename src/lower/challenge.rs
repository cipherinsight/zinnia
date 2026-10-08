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
//! cheaper with them (`emit::select_members`).

use super::construction::Target;
use super::graph::Requirement;

/// The name of the challenge requirement.
pub const CHALLENGE_NAME: &str = "verifier challenge";

/// The program-wide requirement "verifier challenges are available".
pub const CHALLENGE: Requirement = Requirement::Program { name: CHALLENGE_NAME };

/// The cells one halo2-base column holds at `k = 9`, the size the layer's
/// halo2-base prices assume: `2^9` rows less the 20 rows halo2-base keeps
/// unusable (`prove/halo2_base/mod.rs`, `UNUSABLE_ROWS`).
pub const HALO2_BASE_COLUMN_CELLS: u64 = 512 - 20;

/// The one-time cost of enabling the second phase on halo2-base, in cells.
///
/// The second phase's own cells (the challenge powers, the checks' factors)
/// are priced by the constructions that lay them out. What the phase adds
/// beyond them is columns its cells do not fill: the challenge column (one
/// cell per challenge), and the unused rows of the second phase's last gate
/// column, which is rounded up separately from the first phase's columns
/// (half a column on average). Both are advice columns with equality
/// enabled, so each carries its share of the permutation argument like any
/// column of cells. One and a half columns of cells: 738 cells at `k = 9`.
///
/// Measured on the circuits as configured
/// (`prove::halo2_base::configured_polys`, `eval-redo/layer/
/// cost-model-misses.md`): for the eleven perf programs compiled with a
/// grand-product sort or permutation, laying the second-phase cells out in
/// the first phase and dropping the challenge column would commit 1 to 3
/// fewer polynomials (mean 1.8), where one and a half columns are 2.25 (a
/// column of cells is 1.5 committed polynomials with its permutation
/// share). The earlier price, 2,000 cells (about four columns), came from
/// proving times on a loaded machine (`eval-redo/layer/layer-challenge.md`)
/// that could not resolve a difference of this size.
pub const HALO2_BASE_CHALLENGE_COST: u64 = HALO2_BASE_COLUMN_CELLS * 3 / 2;

/// The name of the program-wide RAM trace (halo2-base,
/// `lower::memory::versions::RamTrace`).
pub const RAM_NAME: &str = "RAM trace";

/// The one-time cost of the RAM trace on halo2-base, in cells: its 8 advice
/// and 11 fixed columns, a shuffle and a lookup argument (about 11
/// committed polynomials of `2^k` rows; `prove/halo2_base/circuit/ram.rs`),
/// at `k = 9`.
pub const HALO2_BASE_RAM_COST: u64 = 11 * 512;

/// The one-time cost of producing the program-wide requirement `name` on
/// `target`; `None` when the target cannot produce it.
pub fn producer_cost(name: &str, target: Target) -> Option<u64> {
    match (name, target) {
        (CHALLENGE_NAME, Target::Halo2Base) => Some(HALO2_BASE_CHALLENGE_COST),
        (RAM_NAME, Target::Halo2Base) => Some(HALO2_BASE_RAM_COST),
        _ => None,
    }
}

/// Whether `target` offers verifier challenges (constructions that need
/// one offer no candidate elsewhere).
pub fn available(target: Target) -> bool {
    producer_cost(CHALLENGE_NAME, target).is_some()
}
