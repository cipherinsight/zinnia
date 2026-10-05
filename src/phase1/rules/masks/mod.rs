//! Rule group `masks` (`revision/phase1/rules/masks/README.md`).
//!
//! The counting quantities of a Boolean value m: per physical position of a
//! bundle that holds m, whether the position holds a true element and how
//! many true elements precede it (`PREFIX_RANK`); the number of true
//! elements, the count variable `c = Count(m)` of Φ, on one wire (`COUNT`,
//! an output of the same node); and the packed array of the flat indices of
//! the true elements (`NONZERO`).
//!
//! **One node per (mask, bundle).** The row of `Ranks` is keyed by the mask
//! identity — the defining key of the mask value, which is what phase 0 keys
//! `Count` by — together with the bundle I it ranks
//! ([`crate::phase1::interface::RanksOf`], `../README.md` §4.5). So `x[m]`
//! and `y[m]` over one bundle of m name one row, and `PREFIX_RANK` and
//! `COUNT` build the same candidate in their two rows, which is what makes
//! the node one node (§3.4).

pub(crate) mod count;
pub(crate) mod nonzero;
pub(crate) mod prefix_rank;

use crate::phase0::desc::ValueKey;

use crate::phase1::build::Builder;
use crate::phase1::interface::{Interface, LayoutKind, OwnerRef, RanksOf};

/// Whether a bundle's positions are in the logical order of the value
/// (`README.md` §2, item 4): the valid positions in physical order are the
/// elements in logical order, which is what makes a rank the logical rank.
/// A keyed layout fails it and gets no `PREFIX_RANK` node.
pub fn ordered(owner: &OwnerRef) -> bool {
    owner.layout.is_ordered()
}

/// The bundle a concrete interface of the Store family is, as the subject of
/// a `Ranks` row (`../README.md` §4.5).
pub fn subject_of(interface: &Interface, value: &ValueKey, valid: bool) -> Option<RanksOf> {
    let owner = super::reading::owner_of(interface, value)?;
    ordered(&owner).then_some(RanksOf { owner, valid })
}

/// How many positions of a bundle have both a validity wire and a value
/// wire, which is what one `and` per position is charged for
/// (`prefix_rank.md`, `resource_relations`, `B_v`).
///
/// Decided by construction and never by a question: a box `Store` whose grid
/// is the value's own literal extents has the constant 1 on every slot, and
/// every other bundle carries a wire per slot. A `Packed` with canon = yes
/// has `data[i] = 0` beyond ℓ, so its value wire is already the active bit.
pub fn wired_positions(
    builder: &Builder<'_>,
    value: &ValueKey,
    owner: &OwnerRef,
    valid: bool,
) -> i128 {
    // Variant `valid` reads the validity wires and computes nothing.
    if valid {
        return 0;
    }
    match &owner.layout {
        LayoutKind::Box(grid) => {
            let dims = builder
                .array(value)
                .map(|array| array.dims.clone())
                .unwrap_or_default();
            let constant = dims.len() == grid.len()
                && dims
                    .iter()
                    .zip(grid)
                    .all(|(&dimension, &extent)| {
                        builder.terms.constant_value(dimension) == Some(extent)
                    });
            if constant {
                0
            } else {
                owner.capacity
            }
        }
        LayoutKind::Compact(_) if owner.canon => 0,
        _ => owner.capacity,
    }
}

/// Whether the bundle carries no validity wire of its own, so that the
/// validity of a position is the thermometer of ℓ (`../README.md` §4.1).
///
/// True of every `Packed`, canonical or not: `canon = yes` says the *values*
/// beyond ℓ are zero, not that a consumer can tell which positions those
/// are. The thermometer is the `Ranks(I, valid)` node of the bundle, emitted
/// there once and shared.
pub fn validity_is_a_thermometer(owner: &OwnerRef) -> bool {
    owner.family == crate::phase1::interface::Family::Packed
}
