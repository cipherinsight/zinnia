//! Rule group `reduction` (`revision/phase1/rules/reduction/README.md`).
//!
//! Every source operation that combines many elements of one or two arrays
//! into fewer: reductions over axes, contractions, and prefix scans. One
//! interface-driven rule, `EXPAND_FIBERS`, decides where the elements are
//! read from and lays them out as fibers of slots with a validity bit each;
//! `FOLD_GROUPS` and `CUMSUM` only combine slots and never look at a
//! representation, and `MATMUL` fetches its operands and enumerates its own
//! fibers.
//!
//! This module holds the definitions the files share: the axis set of a
//! reduction, read off the AST node, and the wrap-once policy of §2.

pub(crate) mod cumsum;
pub(crate) mod expand_fibers;
pub(crate) mod fold_groups;
pub(crate) mod matmul;

use crate::phase0::desc::{Dtype, ValueKey};

use crate::phase1::ast_index::AstNode;
use crate::phase1::build::Builder;
use crate::phase1::interface::Capacity;

/// `A`, the axes a reduction runs over (`fold_groups.md`, `source_pattern`):
/// every axis for `E-Reduce-Scalar` and for `axis=None`, and the normalised
/// axis set otherwise.
pub fn axes_of(node: &AstNode, rank: usize, position: usize, name: &str) -> Vec<usize> {
    let normalise = |value: i64| -> Option<usize> {
        let wrapped = if value < 0 { value + rank as i64 } else { value };
        (0..rank as i64).contains(&wrapped).then_some(wrapped as usize)
    };
    match node.parameter(position, name).and_then(|value| value.ints()) {
        Some(axes) => {
            let mut out: Vec<usize> = axes.into_iter().filter_map(normalise).collect();
            out.sort();
            out.dedup();
            if out.is_empty() {
                (0..rank).collect()
            } else {
                out
            }
        }
        None => (0..rank).collect(),
    }
}

/// *Wrap-once* of `README.md` §2: an integer sum whose unwrapped value fits
/// the field is taken in the field and wrapped once at the end.
///
/// The condition `W · 2^{w'} ≤ 2^{f−2}` is a policy checked at
/// instantiation, `f` the field's bit length, which the backend fixes; the
/// library's own bound is `Ω.max_width`.
pub fn wrap_once(builder: &Builder<'_>, width: Capacity, operand_width: u32) -> bool {
    if !builder.omega.field_accumulate {
        return false;
    }
    let field = builder.omega.max_width as i128;
    crate::phase1::build::bits_for(width.max(1)) as i128 + operand_width as i128 <= field - 2
}

/// The dtype a fold produces, read off Δ.
pub fn output_dtype(builder: &Builder<'_>, value: &ValueKey) -> Option<Dtype> {
    builder
        .array(value)
        .map(|array| array.dtype)
        .or_else(|| builder.scalar(value).map(|scalar| scalar.dtype))
}
