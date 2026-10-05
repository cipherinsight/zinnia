//! Rule group `ordering` (`revision/phase1/rules/ordering/README.md`).
//!
//! The rules whose result depends on the order of an operand's element
//! values: `SORT`, `ARGSORT`, `UNIQUE`. All three share one construction —
//! the prover supplies the result as witness, one `perm(B, k)` checks that
//! it is a rearrangement of the operand's valid elements, and `B − 1` gated
//! adjacent range checks check that it is in order — so this module holds
//! what the three files share: how a bundle is read (§2.1), the encoding
//! `ord_τ` (§2.2) and the order check (§2.3).

pub(crate) mod argsort;
pub(crate) mod sort;
pub(crate) mod unique;

use crate::phase0::desc::Dtype;

use crate::phase1::build::Builder;
use crate::phase1::cost::{FloatOp, Primitive};
use crate::phase1::interface::{Capacity, Interface, LayoutKind};
use crate::phase1::rules::Candidate;

/// How one Store-family bundle reads as the five things the emitters need
/// (`README.md` §2.1).
#[derive(Debug, Clone, Copy)]
pub struct Operand {
    /// `B`, the slots the operand occupies.
    pub capacity: Capacity,
    /// Whether the prefix validity `m′` of the result costs a thermometer:
    /// a box bundle's own validity is already a prefix, every other layout's
    /// is not.
    pub thermometer: bool,
    /// *dense* of §4.1: the logical index of a slot is the slot, so the
    /// constant fold can drop the validity column altogether.
    pub dense: bool,
}

/// The bundle a concrete interface is, or `None` for one no emitter reads.
pub fn operand_of(interface: &Interface) -> Option<Operand> {
    let layout = interface.lambda()?;
    Some(match &layout.layout {
        LayoutKind::Box(grid) => Operand {
            capacity: grid.iter().product(),
            thermometer: false,
            dense: true,
        },
        LayoutKind::Compact(capacity) => Operand {
            capacity: *capacity,
            thermometer: true,
            dense: true,
        },
        LayoutKind::Ranked { capacity, .. } => Operand {
            capacity: *capacity,
            thermometer: true,
            dense: false,
        },
        LayoutKind::Keyed { capacity } => Operand {
            capacity: *capacity,
            thermometer: true,
            dense: false,
        },
    })
}

/// What two elements are ordered on (`README.md` §2.2, §2.3).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderKey {
    /// An integer-family dtype: `ord_τ` maps every element into `[0, 2^β)`,
    /// so a pair is ordered by one range check on the difference, and
    /// `ARGSORT` folds the logical index into the same number at `b` further
    /// bits.
    Encoded { beta: u32 },
    /// Float. The backend's encoding admits no order-preserving map into a
    /// bounded interval, so `ord_Float` is the identity on the wire and the
    /// pair is ordered by the backend's own comparison instead: one
    /// `fop⟨lte_f⟩` (strict: `fop⟨lt_f⟩`), and, where an index breaks ties,
    /// `fop⟨lt_f⟩` and `fop⟨eq_f⟩` beside the index comparison, which is the
    /// two-comparison form `argsort.md` step 4 already states for a key
    /// wider than `Ω.max_width`. `perm` is unchanged: a Float element is one
    /// field element and its encoding is injective, so a multiset equality
    /// over the wires is one over the values.
    Float,
}

/// The order key of a dtype, or `None` for one no order is stated for
/// (`README.md` §2.2). Complex is one: it has no order, and no rule of this
/// group matches it.
pub fn order_key(builder: &Builder<'_>, dtype: Dtype) -> Option<OrderKey> {
    if crate::phase1::rules::sources::is_float(dtype) {
        return Some(OrderKey::Float);
    }
    let width = builder.dtype_width(dtype);
    (width > 0).then(|| OrderKey::Encoded { beta: builder.omega.round(width) })
}

/// `β`, the width `ord_τ` encodes into, for the rules that report it as a
/// parameter; 0 for a Float, whose wire carries no width (`../README.md` §5).
pub fn ord_width(key: OrderKey) -> u32 {
    match key {
        OrderKey::Encoded { beta } => beta,
        OrderKey::Float => 0,
    }
}

/// The order check of `README.md` §2.3 over `pairs` adjacent pairs.
///
/// `index_width` is `b`, the width of the logical index the key carries, 0
/// where the check orders on the value alone. `strict` asks for `<` rather
/// than `≤`. `gated` says the prefix validity is a wire, which costs one
/// `mul` per pair to force the check only where the pair is valid; where it
/// is a constant the forcing is a `lin`.
pub fn order_check(
    candidate: Candidate,
    key: OrderKey,
    pairs: Capacity,
    index_width: u32,
    strict: bool,
    gated: bool,
) -> Candidate {
    let pairs = pairs.max(0);
    let candidate = candidate.emit(Primitive::Mul, if gated { pairs } else { 0 });
    match key {
        OrderKey::Encoded { beta } => {
            candidate.emit(Primitive::Range { width: beta + index_width }, pairs)
        }
        // The value comparison, and the tie clause when an index follows it.
        OrderKey::Float if index_width == 0 => candidate.emit(
            Primitive::Float(if strict { FloatOp::Lt } else { FloatOp::Lte }),
            pairs,
        ),
        OrderKey::Float => candidate
            .emit(Primitive::Float(FloatOp::Lt), pairs)
            .emit(Primitive::Float(FloatOp::Eq), pairs)
            .emit(Primitive::Cmp { width: index_width }, pairs)
            .emit(Primitive::And, pairs)
            .emit(Primitive::Or, pairs),
    }
}
