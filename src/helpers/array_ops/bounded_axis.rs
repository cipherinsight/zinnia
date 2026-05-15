//! Bounded-axis stride dispatch and strict-mode validation for dynamic
//! ndarrays.
//!
//! Subscript-read into a dynamic ndarray must decide between two stride
//! layouts:
//! * Compile-time-constant logical strides (the default, zero-regression
//!   path).
//! * SSA-`Value` runtime strides (the "compact buffer" path, used when at
//!   least one axis is bounded and `total_bound < product(max_shape)`).
//!
//! [`select_stride_mode`] performs the dispatch; [`stride_value`]
//! materialises a stride as a `Value` for a given axis. The strict-mode
//! switch ([`bounded_axis_strict`]) layers per-axis `ir_assert` bounds
//! checks on top.

use crate::builder::IRBuilder;
use crate::types::{DynamicNDArrayData, ScalarValue, Value};

// ── Strict-mode env var (multi-dim Case B) ──────────────────────────────

/// `ZINNIA_BOUNDED_AXIS_STRICT=1` switches subscript-read and the compact
/// multi-D constructor to **strict mode**:
/// * Subscript-read into a dyn-ndarray with any bounded axis emits per-axis
///   `ir_assert` checks: `0 <= idx[axis] < runtime_shape[axis]`.
/// * The compact multi-D constructor (`np_fill` Tuple/List path that proved
///   `prod(runtime_shape) <= total_bound < prod(max_shape)`) emits an
///   `ir_assert` for `prod(runtime_shape) <= total_bound` via
///   `lower_precondition_to_ir`.
///
/// Default (lenient): the user's `@requires` facts are trusted. Reads
/// beyond `runtime_shape` quietly return the buffer's init value (typically
/// `0` from segment init); the in-circuit clamping in
/// `read_memory` prevents undefined behaviour, so the verifier sees no
/// inconsistency. Strict mode is opt-in for debugging and CI verification
/// builds; programs in production-shape don't pay the per-axis assertion
/// cost.
pub(crate) fn bounded_axis_strict() -> bool {
    std::env::var("ZINNIA_BOUNDED_AXIS_STRICT")
        .map(|s| matches!(s.as_str(), "1" | "true" | "TRUE"))
        .unwrap_or(false)
}

// ── Layout-dispatch helper (multi-dim Case B) ───────────────────────────

/// Stride layout for a dynamic-ndarray's subscript-read site.
///
/// Three-way dispatch from `select_stride_mode`:
///
/// 1. [`StrideMode::LiteralLogical`] — used when no axis is bounded
///    (today's behaviour for every existing dyn-ndarray, also the static-
///    shaped path), or when `total_bound == product(max_shape)` (the
///    "loose-bound" case where per-axis bounds are themselves tight).
///    The address arithmetic uses compile-time constant strides from
///    `data.meta.logical_strides`. **Zero regression** for existing programs.
///
/// 2. [`StrideMode::SymbolicRuntime`] — used when at least one axis is
///    bounded (its `runtime_shape[i]` is symbolic, i.e. `static_val !=
///    Some(logical_shape[i])`) AND `total_bound < product(max_shape)` (the
///    imbalanced-bound case — the compact-buffer mode). The address
///    arithmetic uses SSA-`Value` strides from `data.meta.runtime_strides`
///    via `ir_mul_i` per axis. Cost: ~N-1 `ir_mul_i` per subscript-read
///    (no constant folding). Sound because the compact constructor sets
///    `runtime_strides` to `prod(runtime_shape[k+1:])` for axis k via an
///    `ir_mul_i` chain at constructor time — addressing matches the slot
///    layout of the compact buffer.
pub(crate) enum StrideMode<'a> {
    LiteralLogical(&'a [usize]),
    SymbolicRuntime(&'a [ScalarValue<i64>]),
}

/// Select the appropriate stride layout for subscript-read into `data`.
///
/// See [`StrideMode`] for the three cases. The check is one linear pass
/// over `runtime_shape` plus one `product()` of `logical_shape`.
pub(crate) fn select_stride_mode<'a>(data: &'a DynamicNDArrayData) -> StrideMode<'a> {
    let any_bounded = data
        .meta
        .runtime_shape
        .iter()
        .enumerate()
        .any(|(i, s)| s.static_val != Some(data.meta.logical_shape[i] as i64));
    let max_product: usize = data.meta.logical_shape.iter().product();
    let total_bound_tighter = data.envelope.total_bound < max_product;
    if any_bounded && total_bound_tighter {
        StrideMode::SymbolicRuntime(&data.meta.runtime_strides)
    } else {
        StrideMode::LiteralLogical(&data.meta.logical_strides)
    }
}

/// Materialize a stride from [`StrideMode`] as an SSA `Value` for axis
/// `ax`. For [`StrideMode::LiteralLogical`] this is a compile-time
/// constant; for [`StrideMode::SymbolicRuntime`] this is the
/// `runtime_strides[ax]` SSA scalar (`static_val` may be `None`).
pub(crate) fn stride_value(b: &mut IRBuilder, mode: &StrideMode, ax: usize) -> Value {
    match mode {
        StrideMode::LiteralLogical(s) => b.ir_constant_int(s[ax] as i64),
        StrideMode::SymbolicRuntime(rs) => {
            let sv = &rs[ax];
            if let Some(v) = sv.static_val {
                b.ir_constant_int(v)
            } else if let Some(ptr) = sv.stmt_id {
                Value::Integer(ScalarValue::new(None, Some(ptr)))
            } else {
                // Unreachable for well-formed compact arrays: the compact
                // constructor always materializes a stmt_id-bearing
                // `runtime_strides[ax]` via the `ir_mul_i` chain.
                b.ir_constant_int(0)
            }
        }
    }
}
