//! Rule group `scalar_control`
//! (`revision/phase1/rules/scalar_control/README.md`).
//!
//! Scalars, terms, merges, checks and delivery. Every rule of the group
//! produces one wire per scalar, or no wire at all. This module holds the
//! three conventions §2 of the group README states and every file of the
//! group uses.
//!
//! **Comparison widths.** `[x < y]` is `cmp_w(x, y)` with `w = W(x − y)`;
//! `[x ≤ y]` is `cmp_w(x, y + 1)` with `w = W(x − y − 1)`; `[x = y]` is
//! `eq(x, y)`. The arguments `y + 1` and `x − k` are `lin` and free.
//!
//! **Interval-decided gates.** A comparison whose interval decides it is the
//! constant 1 or 0 at no cost; a `select`, `min`, `max` or `ite` whose
//! condition is a constant is the selected arm's wire at no cost. These are
//! interval facts of `../README.md` §6.1, never solver answers. Every count
//! below is stated before this folding; the emitter applies it.
//!
//! **Where a term reads its variables.** A term's children nest as phase 0's
//! path conditions do. Every sub-term is raised under the node's own `g`
//! rather than under the nested `g ∧ b` that the sub-term sits inside: `g`
//! is the weaker condition, so a provider under it serves the nested one
//! too, and one row per sub-term is shared by every occurrence
//! (`rules/README.md` §3.2 states the same rule for sub-terms).

pub(crate) mod enforce_check;
pub(crate) mod index_normalize;
pub(crate) mod shape_check;
pub(crate) mod export;
pub(crate) mod len_shape;
pub(crate) mod scalar_op;
pub(crate) mod select;
pub(crate) mod term_to_wires;

use crate::phase0::interval::Interval;
use crate::phase0::term::NodeView;

use crate::phase1::build::{bits_for, width_of, Builder};
use crate::phase1::cost::Primitive;
use crate::phase1::oracle::layers::{evaluate_bool, Trivalent};

/// `I_{a − b}`, for the width of a comparison whose difference term phase 0
/// never built.
pub fn difference(a: Interval, b: Interval) -> Interval {
    Interval::new(
        a.lo.zip(b.hi).map(|(lo, hi)| lo - hi),
        a.hi.zip(b.lo).map(|(hi, lo)| hi - lo),
    )
}

/// The interval of the arena node at `index`, when it is integer-sorted.
pub fn node_interval(builder: &Builder<'_>, index: u32) -> Interval {
    match builder.terms.int_at(index) {
        Some(term) => builder.interval(term),
        None => Interval::UNBOUNDED,
    }
}

/// The primitives one term node emits (`term_to_wires.md`, §emitter), with
/// the interval-decided folding of the group README applied.
pub fn node_gadget(builder: &Builder<'_>, index: u32) -> Vec<(Primitive, i128)> {
    let node = builder.terms.node_at(index as usize);
    // A Boolean node the ranges decide is a constant at no cost.
    if node.is_bool() {
        if let Some(formula) = builder.terms.bool_at(index) {
            if evaluate_bool(&builder.terms, &builder.result.phi, formula)
                != Trivalent::Unknown
            {
                return Vec::new();
            }
        }
    }
    match node {
        // A constant is substituted and a variable is the port's wire.
        NodeView::Const(_) | NodeView::Var(_) | NodeView::BVar(_) => Vec::new(),
        NodeView::True | NodeView::False => Vec::new(),
        // `lin`, free.
        NodeView::Add(..) | NodeView::Sub(..) | NodeView::Mul(..) => Vec::new(),
        NodeView::Div(t, k) | NodeView::Mod(t, k) => vec![(divmod(builder, t, k), 1)],
        NodeView::Min(a, b) | NodeView::Max(a, b) => {
            let width = builder.omega.round(width_of(difference(
                node_interval(builder, a),
                node_interval(builder, b),
            )));
            vec![(Primitive::Cmp { width }, 1), (Primitive::Select, 1)]
        }
        NodeView::Ite(condition, _, _) => {
            // An `ite` on a condition the ranges decide is the selected arm.
            if let Some(formula) = builder.terms.bool_at(condition) {
                if evaluate_bool(&builder.terms, &builder.result.phi, formula)
                    != Trivalent::Unknown
                {
                    return Vec::new();
                }
            }
            vec![(Primitive::Select, 1)]
        }
        NodeView::Lt(a, b) => {
            let width = builder.omega.round(width_of(difference(
                node_interval(builder, a),
                node_interval(builder, b),
            )));
            vec![(Primitive::Cmp { width }, 1)]
        }
        NodeView::Le(a, b) => {
            let mut range = difference(node_interval(builder, a), node_interval(builder, b));
            range.lo = range.lo.map(|lo| lo - 1);
            range.hi = range.hi.map(|hi| hi - 1);
            let width = builder.omega.round(width_of(range));
            vec![(Primitive::Cmp { width }, 1)]
        }
        NodeView::Eq(..) => vec![(Primitive::Eq, 1)],
        NodeView::Not(_) => Vec::new(),
        NodeView::And(..) => vec![(Primitive::And, 1)],
        NodeView::Or(..) => vec![(Primitive::Or, 1)],
    }
}

/// `divmod_k(t)` of `../README.md` §7: the range on the quotient is taken on
/// `q − lo_q`, and the remainder's width is `⌈log2 k⌉`.
fn divmod(builder: &Builder<'_>, t: u32, k: i128) -> Primitive {
    let range = node_interval(builder, t);
    let magnitude = k.abs().max(1);
    let quotient = Interval::new(
        range.lo.map(|lo| crate::phase0::term::floor_div(lo, k)),
        range.hi.map(|hi| crate::phase0::term::floor_div(hi, k)),
    );
    let span = match (quotient.lo, quotient.hi) {
        (Some(lo), Some(hi)) => bits_for((hi - lo).abs() + 1),
        _ => 64,
    };
    Primitive::DivModConst {
        quotient_width: builder.omega.round(span),
        divisor_width: builder.omega.round(bits_for(magnitude)),
        power_of_two: magnitude.count_ones() == 1,
    }
}

/// Whether every width a list of primitives mentions is admitted
/// (`../README.md` §6.1, `Ω.max_width`).
pub fn widths_admitted(builder: &Builder<'_>, primitives: &[(Primitive, i128)]) -> bool {
    primitives.iter().all(|&(primitive, _)| match primitive {
        Primitive::Cmp { width }
        | Primitive::Range { width }
        | Primitive::DivModWire { width }
        | Primitive::Wrap { width, .. } => builder.omega.admits(width),
        Primitive::DivModConst { quotient_width, divisor_width, .. } => {
            builder.omega.admits(quotient_width) && builder.omega.admits(divisor_width)
        }
        _ => true,
    })
}
