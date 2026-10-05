//! Rule group `sources` (`revision/phase1/rules/sources/README.md`).
//!
//! The rules that bring a value into the circuit without computing it from
//! another array: the circuit inputs, whose wires the front end binds, and
//! the constants, whose every element is a compile-time function of its
//! coordinate. `CONSTANT_SIZED` is the constant whose length is a data
//! scalar, sized by the maximum the source declared.

pub(crate) mod constant;
pub(crate) mod constant_sized;
pub(crate) mod input_scalar;
pub(crate) mod input_store;

use crate::phase0::desc::Dtype;
use crate::types::ZinniaType;

/// The visibility the front end wrote on a parameter, `Private` when absent
/// (`input_scalar.md`, `input_store.md`).
pub fn is_private(annotation: &crate::ast::ASTAnnotation) -> bool {
    !matches!(annotation.kind.as_deref(), Some("Public"))
}

/// The declared type of a parameter, with `PoseidonHashed` unwrapped and the
/// hashed flag alongside.
pub fn declared(annotation: &crate::ast::ASTAnnotation) -> Option<(ZinniaType, bool)> {
    let parsed: ZinniaType = serde_json::from_value(annotation.dt.clone()).ok()?;
    match parsed {
        ZinniaType::PoseidonHashed { dtype } => Some((*dtype, true)),
        other => Some((other, false)),
    }
}

/// Whether a dtype has a wire encoding (`rules/README.md` §4.3, §5).
///
/// Bool and the integer dtypes have the library's own: a field element with
/// a width, range-checked and wrapped by the gadgets of §7. Float has the
/// backend's: one wire whose well-formedness is the backend's encoding, with
/// no width, no `range_w` and no decomposition, and whose every operation is
/// an `fop⟨name⟩` (`crate::phase1::cost::FloatOp`). Complex and String have
/// none, and the group's `Open` list says so.
pub fn has_encoding(dtype: Dtype) -> bool {
    matches!(dtype, Dtype::Bool | Dtype::Idx | Dtype::I64 | Dtype::Float)
}

/// Whether the wire is the backend's float encoding rather than a field
/// element of a stated width, which is what decides between an integer
/// gadget and an `fop⟨name⟩` row.
pub fn is_float(dtype: Dtype) -> bool {
    matches!(dtype, Dtype::Float)
}

/// [`has_encoding`], with the note a row carries when it is false, so that a
/// rejection says which dtype the library has no gadget for rather than only
/// which rule did not match (`scalar_control/README.md` §4 item 1).
pub fn encoded(builder: &mut crate::phase1::build::Builder<'_>, rule: &str, dtype: Dtype) -> bool {
    if has_encoding(dtype) {
        return true;
    }
    let note = format!(
        "{} has no gadget for the dtype {}: the library states no encoding \
         for it and its primitives are the backend's",
        rule,
        dtype.as_str()
    );
    if !builder.missing.contains(&note) {
        builder.missing.push(note);
    }
    false
}
