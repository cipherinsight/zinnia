use crate::builder::IRBuilder;
use crate::types::{DynArrayMeta, DynamicNDArrayData, NumberType, ScalarValue, Value, ValueId};

use super::{
    dyn_decode_coords, dyn_encode_coords, dyn_num_elements, dyn_row_major_strides,
    value_to_scalar_i64, DynAggKind,
};

pub fn dyn_aggregate(
    b: &mut IRBuilder,
    data: &DynamicNDArrayData,
    axis: Option<&Value>,
    agg: DynAggKind,
) -> Value {
    let axis_val = axis.and_then(|v| {
        if matches!(v, Value::None) {
            None
        } else {
            v.int_val()
        }
    });

    match axis_val {
        None => dyn_aggregate_all(b, data, agg),
        Some(ax) => dyn_aggregate_axis(b, data, ax, agg),
    }
}

pub fn dyn_mean(b: &mut IRBuilder, data: &DynamicNDArrayData, axis: Option<&Value>) -> Value {
    let axis_val = axis.and_then(|v| {
        if matches!(v, Value::None) {
            None
        } else {
            v.int_val()
        }
    });

    match axis_val {
        None => {
            let sum = dyn_aggregate_all(b, data, DynAggKind::Sum);
            let runtime_length = scalar_to_value_i(b, &data.meta.runtime_length);
            let zero = b.ir_constant_int(0);
            let non_empty = b.ir_greater_than_i(&runtime_length, &zero);
            b.ir_assert(&non_empty);
            let denom = b.ir_float_cast(&runtime_length);
            let sum_f = match sum {
                Value::Float(_) => sum,
                _ => b.ir_float_cast(&sum),
            };
            b.ir_div_f(&sum_f, &denom)
        }
        Some(ax) => {
            let ndim = data.meta.logical_shape.len();
            let ax = if ax < 0 {
                (ndim as i64 + ax) as usize
            } else {
                ax as usize
            };
            assert!(ax < ndim, "mean axis out of bounds");
            let sum = dyn_aggregate_axis(b, data, ax as i64, DynAggKind::Sum);
            let denom_i = scalar_to_value_i(&mut *b, &data.meta.runtime_shape[ax]);
            let zero = b.ir_constant_int(0);
            let non_empty = b.ir_greater_than_i(&denom_i, &zero);
            b.ir_assert(&non_empty);
            let denom = b.ir_float_cast(&denom_i);
            match sum {
                Value::DynamicNDArray(d) => {
                    super::binary::dyn_scalar_binary_op(b, "div", &denom, &d, false)
                }
                Value::Float(_) => b.ir_div_f(&sum, &denom),
                _ => {
                    let sum_f = b.ir_float_cast(&sum);
                    b.ir_div_f(&sum_f, &denom)
                }
            }
        }
    }
}

/// Full reduction (axis=None): reduce all elements to a scalar.
///
/// **Bounded-aware iteration (multi-dim Case B Tier 2).** Two paths:
///
/// 1. **Fast path** (no bounded axis): identical to the pre-Tier-2 behaviour
///    — iterate the full buffer and reduce every slot. Sound because every
///    slot is active.
///
/// 2. **Mask-and-include path** (at least one axis is bounded — its
///    `runtime_shape[ax].static_val != Some(logical_shape[ax])`): walk all
///    buffer slots at compile time, decode each slot's multi-D coordinates
///    against the appropriate stride layout, and gate the contribution by
///    `is_active = AND_over_axes(coord[ax] < runtime_shape[ax])`. Inactive
///    slots contribute the op's algebraic identity (sum→0, prod→1,
///    all→true, any→false). max/min/argmax/argmin use a **first-active
///    pattern** instead of a sentinel (the ZK Real fragment has no
///    `±∞`): the accumulator starts at `buffer[0]` and an update fires
///    only when `is_active AND new-candidate-is-better`.
///
/// **Soundness (algebraic-identity invariant).** For each op,
/// `reduce(identity, x) = reduce(x, identity) = x`:
///
/// * `sum`: `0 + x = x`.
/// * `prod`: `1 * x = x`.
/// * `all`: `true ∧ x = x`.
/// * `any`: `false ∨ x = x`.
/// * `max/min/argmax/argmin`: no identity; the first-active pattern leaves
///   the accumulator untouched for inactive slots, which is sound because
///   the only update path requires `is_active`.
///
/// **Strict mode** (`ZINNIA_BOUNDED_AXIS_STRICT=1`): for
/// max/min/argmax/argmin on a bounded array, emit
/// `ir_assert(runtime_length > 0)` before the loop — the first-active
/// pattern's `buffer[0]` initial value is only meaningful when the active
/// region is non-empty.
///
/// **Empty active region (lenient mode).** When `prod(runtime_shape) == 0`
/// for max/min/argmax/argmin, `acc = buffer[0]` is the segment-init value
/// (typically `0`); `argmax/argmin` return `0`. Documented; not asserted.
pub fn dyn_aggregate_all(b: &mut IRBuilder, data: &DynamicNDArrayData, agg: DynAggKind) -> Value {
    // Compaction-scan fast path: min/max over a sorted compaction
    // (flatnonzero/nonzero outputs) is a first/last-set-bit scan over the
    // (keep, source) pairs — O(N) selects, NO reads of the compacted
    // segment. With the segment unread, unread_memory_elimination then
    // deletes its writes (~6 ops/cell), which was the residual gap on the
    // nonzero-bbox kernels. Generalizes (and will replace) the argwhere
    // axis-0 bbox special case.
    if matches!(agg, DynAggKind::Min | DynAggKind::Max)
        && data.dtype == NumberType::Integer
        && data.meta.logical_shape.len() == 1
        && data.meta.logical_offset == 0
    {
        if let Some(info) = b.provenance.compaction_of(data.segment_id, 0).cloned() {
            if info.sorted {
                if let Some(out) = compaction_scan_min_max(b, &info, agg) {
                    return out;
                }
            }
        }
    }
    let inputs = DynAggAllInputs {
        data: data.clone(),
        agg,
    };
    let set = dyn_aggregate_all_strategies(&inputs);
    crate::optim::dispatch_strategy(b, "dyn_aggregate_all", &inputs, &set)
}

/// First/last-set-bit scan over a sorted compaction's (keep, source)
/// pairs. Ascending source order makes min = the FIRST kept source
/// (reverse select chain) and max = the LAST kept (forward chain).
/// Empty input yields 0, matching the zero-initialised compacted
/// segment's boundary reads.
pub(crate) fn compaction_scan_min_max(
    b: &mut IRBuilder,
    info: &crate::provenance::CompactionInfo,
    agg: DynAggKind,
) -> Option<Value> {
    let n = info.keep.len();
    if n == 0 || info.source.len() != n {
        return None;
    }
    let mut keeps = Vec::with_capacity(n);
    let mut sources = Vec::with_capacity(n);
    for i in 0..n {
        keeps.push(b.value_of_vid(info.keep[i])?);
        sources.push(b.value_of_vid(info.source[i])?);
    }
    let mut acc = b.ir_constant_int(0);
    match agg {
        DynAggKind::Min => {
            for i in (0..n).rev() {
                acc = b.ir_select_i(&keeps[i], &sources[i], &acc);
            }
        }
        DynAggKind::Max => {
            for i in 0..n {
                acc = b.ir_select_i(&keeps[i], &sources[i], &acc);
            }
        }
        _ => return None,
    }
    Some(acc)
}

/// Per-call inputs for `dyn_aggregate_all`'s strategy set. Carries the
/// op's data plus the reduction kind so each lowering can read both via
/// the framework-imposed `fn(&mut IRBuilder, &Inputs) -> Output` signature
/// (closures-with-captured-context aren't allowed by the function-pointer
/// shape on `OpStrategy::lower`).
pub(crate) struct DynAggAllInputs {
    pub data: DynamicNDArrayData,
    pub agg: DynAggKind,
}

/// Strategy set for full reduction (axis=None). Gated strategies:
///
/// * `boundary-read-on-sorted` — `is_sorted(input)` ⇒ closed-form max/
///   min/argmax/argmin via a single read at index 0 or `len - 1`.
/// * `sum_all_zero` — `forall_eq_const(arr, 0)` ⇒ sum is constant 0.
/// * `sum_all_one` — `forall_eq_const(arr, 1)` ⇒ sum is `runtime_length`
///   (cast to float when dtype is Float).
/// * `prod_all_one` — `forall_eq_const(arr, 1)` ⇒ prod is constant 1.
///
/// **Empty-N safety (mirrors static-side Group 3d, adapted for dyn).**
/// Dyn arrays have runtime-symbolic length N, so each strategy is only
/// admitted if it remains sound for `N = 0`:
///
/// * `sum_all_zero`: sum of empty = 0; sum of N zeros = 0. Safe.
/// * `sum_all_one`: sum of empty = 0; `runtime_length` is 0 when N=0.
///   Safe — the returned wire's value matches the algebraic identity.
/// * `prod_all_one`: prod of empty = 1; prod of N ones = 1. Safe.
///
/// **Strategies deliberately omitted** (would require a `len(arr) >= 1`
/// fact, which the conservative scope of this card does not require to
/// be present):
///
/// * `prod_all_zero` (prod of empty = 1, not 0).
/// * `mean_*` (mean of empty is undefined/NaN; only safe when N >= 1).
///
/// These can be re-added later under a compound
/// `BoolComb(And, [forall_eq_const, len >= 1])` precondition once we
/// have a clean way to assert the second conjunct from input facts.
///
/// Strategies are op-kind-gated by conditional inclusion (not via a
/// compound precondition): only push the strategy when `inputs.agg`
/// matches the op it lowers. Mirrors the `boundary-read-on-sorted` shape
/// already in place for max/min/argmax/argmin.
fn dyn_aggregate_all_strategies(
    inputs: &DynAggAllInputs,
) -> crate::optim::OpStrategySet<DynAggAllInputs, Value> {
    use crate::optim::predicates::formula::{ContractTerm, ContractVar};
    use crate::optim::{CostHint, OpStrategy, OpStrategySet};

    let mut strategies: Vec<OpStrategy<DynAggAllInputs, Value>> = Vec::new();

    let arr_vid = inputs.data.value_id;
    let pred_forall_eq_const = |k: i64| ContractTerm::PredicateApp {
        kind: "forall_eq_const".to_string(),
        args: vec![
            ContractTerm::Var(ContractVar::Value(arr_vid)),
            ContractTerm::LitInt(k),
        ],
    };

    // Level-2 short-circuit: `is_sorted(arr)` ⇒ closed-form for the four
    // monotone-respecting reductions. The precondition is constructed
    // against the concrete input ValueId — strategies are per-call sets,
    // not contract templates, so we embed `Var(Value(_))` directly.
    if matches!(
        inputs.agg,
        DynAggKind::Max | DynAggKind::Min | DynAggKind::Argmax | DynAggKind::Argmin
    ) {
        strategies.push(OpStrategy {
            name: "boundary-read-on-sorted",
            precondition: ContractTerm::PredicateApp {
                kind: "is_sorted".to_string(),
                args: vec![ContractTerm::Var(ContractVar::Value(arr_vid))],
            },
            cost_hint: CostHint::O1,
            lower: lower_boundary_read_on_sorted,
        });
    }

    // forall_eq_const fast paths for sum / prod (the safe-for-any-N
    // subset of Group 3d's static-side coverage). See the function-level
    // doc-comment for the empty-N reasoning and the deliberately-omitted
    // strategies.
    if matches!(inputs.agg, DynAggKind::Sum) {
        strategies.push(OpStrategy {
            name: "sum_all_zero",
            precondition: pred_forall_eq_const(0),
            cost_hint: CostHint::O1,
            lower: lower_dyn_sum_zero,
        });
        strategies.push(OpStrategy {
            name: "sum_all_one",
            precondition: pred_forall_eq_const(1),
            cost_hint: CostHint::O1,
            lower: lower_dyn_sum_length,
        });
    }
    if matches!(inputs.agg, DynAggKind::Prod) {
        strategies.push(OpStrategy {
            name: "prod_all_one",
            precondition: pred_forall_eq_const(1),
            cost_hint: CostHint::O1,
            lower: lower_dyn_prod_one,
        });
    }

    OpStrategySet {
        strategies,
        default: lower_dyn_aggregate_all_generic,
    }
}

/// Strategy lowering for `dyn sum` on `forall_eq_const(arr, 0)`:
/// every element is `0`, so the sum is `0`. Sound for any `N` —
/// sum of empty = 0; sum of `N` zeros = 0. Dtype-aware: Float dyn
/// arrays produce `ir_constant_float(0.0)`.
fn lower_dyn_sum_zero(b: &mut IRBuilder, inputs: &DynAggAllInputs) -> Value {
    match inputs.data.dtype {
        NumberType::Integer => b.ir_constant_int(0),
        NumberType::Float => b.ir_constant_float(0.0),
        NumberType::Complex => super::metadata::dyn_default_value(b, NumberType::Complex),
    }
}

/// Strategy lowering for `dyn sum` on `forall_eq_const(arr, 1)`:
/// every element is `1`, so the sum equals the dyn array's
/// `runtime_length`. Returns the runtime_length wire directly (cast to
/// float when dtype is Float).
///
/// **Empty-N soundness**: when `N = 0`, `runtime_length` is `0`, and the
/// algebraic identity for `sum` is also `0`. The returned wire's value
/// thus matches the empty-sum identity at runtime as well.
fn lower_dyn_sum_length(b: &mut IRBuilder, inputs: &DynAggAllInputs) -> Value {
    let len_int = scalar_to_value_i(b, &inputs.data.meta.runtime_length);
    match inputs.data.dtype {
        NumberType::Integer => len_int,
        NumberType::Float => b.ir_float_cast(&len_int),
        NumberType::Complex => {
            let real = b.ir_float_cast(&len_int);
            let imag = b.ir_constant_float(0.0);
            crate::helpers::value_ops::pack_complex_value(real, imag)
        }
    }
}

/// Strategy lowering for `dyn prod` on `forall_eq_const(arr, 1)`:
/// every element is `1`, so the product is `1`. Sound for any `N` —
/// prod of empty = 1; prod of `N` ones = 1.
///
/// Note: the symmetric `prod` + `forall_eq_const(0)` case is **NOT**
/// admitted here because prod of empty = 1, not 0 — re-admitting that
/// strategy requires a separate `len(arr) >= 1` fact.
fn lower_dyn_prod_one(b: &mut IRBuilder, inputs: &DynAggAllInputs) -> Value {
    match inputs.data.dtype {
        NumberType::Integer => b.ir_constant_int(1),
        NumberType::Float => b.ir_constant_float(1.0),
        NumberType::Complex => Value::Complex {
            real: ScalarValue::new(Some(1.0), None),
            imag: ScalarValue::new(Some(0.0), None),
        },
    }
}

/// Boundary-read fast path. Sound iff `is_sorted(arr)` (ascending) holds:
///   max(arr)    = arr[runtime_length - 1]
///   min(arr)    = arr[0]
///   argmax(arr) = runtime_length - 1
///   argmin(arr) = 0
///
/// Strict-mode invariant for max/argmax: the `len - 1` read could wrap to
/// a negative address when `runtime_length == 0`, so emit an `ir_assert`
/// when `ZINNIA_BOUNDED_AXIS_STRICT=1`. For min/argmin the empty-array
/// case yields the segment-init value at index 0, which matches the
/// existing first-active pattern's empty-case semantics — no extra
/// assertion needed.
fn lower_boundary_read_on_sorted(b: &mut IRBuilder, inputs: &DynAggAllInputs) -> Value {
    let DynAggAllInputs { data, agg } = inputs;
    let agg = *agg;

    if matches!(agg, DynAggKind::Max | DynAggKind::Argmax)
        && crate::helpers::array_ops::bounded_axis_strict()
    {
        let runtime_length_v = scalar_to_value_i(b, &data.meta.runtime_length);
        let zero = b.ir_constant_int(0);
        let gt = b.ir_greater_than_i(&runtime_length_v, &zero);
        b.ir_assert(&gt);
    }

    let runtime_length_v = scalar_to_value_i(b, &data.meta.runtime_length);
    let len_arr_vid = runtime_length_v.value_id();
    match agg {
        DynAggKind::Argmin => {
            let idx = b.ir_constant_int(0);
            if let (Some(idx_vid), Some(len_vid)) = (idx.value_id(), len_arr_vid) {
                let mut formals = std::collections::HashMap::new();
                formals.insert("len_arr".to_string(), len_vid);
                b.fire_contract("dyn_argextremum", idx_vid, &formals);
            }
            idx
        }
        DynAggKind::Argmax => {
            let one = b.ir_constant_int(1);
            let idx = b.ir_sub_i(&runtime_length_v, &one);
            if let (Some(idx_vid), Some(len_vid)) = (idx.value_id(), len_arr_vid) {
                let mut formals = std::collections::HashMap::new();
                formals.insert("len_arr".to_string(), len_vid);
                b.fire_contract("dyn_argextremum", idx_vid, &formals);
            }
            idx
        }
        DynAggKind::Min => {
            let zero = b.ir_constant_int(0);
            crate::helpers::segment::read_typed(b, data.segment_id, &zero, data.dtype)
        }
        DynAggKind::Max => {
            let one = b.ir_constant_int(1);
            let last_idx = b.ir_sub_i(&runtime_length_v, &one);
            crate::helpers::segment::read_typed(b, data.segment_id, &last_idx, data.dtype)
        }
        _ => unreachable!("boundary-read-on-sorted only registered for max/min/argmax/argmin"),
    }
}

/// Generic O(N) sweep — the unconditional fall-through. Bounded-aware:
/// when no axis is symbolic, runs the fast iterate-and-reduce loop;
/// otherwise uses the mask-and-include path to gate inactive slots with
/// the per-op algebraic identity / first-active pattern.
fn lower_dyn_aggregate_all_generic(b: &mut IRBuilder, inputs: &DynAggAllInputs) -> Value {
    let DynAggAllInputs { data, agg } = inputs;
    let agg = *agg;

    let mode = crate::helpers::array_ops::select_stride_mode(data);
    let data_owned = super::force_contiguous(b, data);
    let data = &data_owned;
    let buffer_size = match mode {
        crate::helpers::array_ops::StrideMode::SymbolicRuntime(_) => data.envelope.total_bound,
        crate::helpers::array_ops::StrideMode::LiteralLogical(_) => {
            dyn_num_elements(&data.meta.logical_shape)
        }
    };
    let values =
        crate::helpers::segment::read_all_typed(b, data.segment_id, buffer_size, data.dtype);
    if buffer_size == 0 {
        return dyn_agg_identity(b, agg, data.dtype);
    }

    if data.dtype == NumberType::Complex
        && matches!(
            agg,
            DynAggKind::Max | DynAggKind::Min | DynAggKind::Argmax | DynAggKind::Argmin
        )
    {
        panic!("Complex DynamicNDArray does not support ordered reductions");
    }

    let use_float =
        data.dtype == NumberType::Float && !matches!(agg, DynAggKind::All | DynAggKind::Any);

    let any_bounded = data
        .meta
        .runtime_shape
        .iter()
        .enumerate()
        .any(|(i, s)| s.static_val != Some(data.meta.logical_shape[i] as i64));
    let compact_buffer = data.envelope.total_bound < dyn_num_elements(&data.meta.logical_shape);

    if any_bounded
        && matches!(
            agg,
            DynAggKind::Max | DynAggKind::Min | DynAggKind::Argmax | DynAggKind::Argmin
        )
        && crate::helpers::array_ops::bounded_axis_strict()
    {
        let runtime_length_v = scalar_to_value_i(b, &data.meta.runtime_length);
        let zero = b.ir_constant_int(0);
        let gt = b.ir_greater_than_i(&runtime_length_v, &zero);
        b.ir_assert(&gt);
    }

    let strides = dyn_row_major_strides(&data.meta.logical_shape);
    let first_active = if any_bounded {
        if compact_buffer {
            build_flat_active_check(b, 0, &data.meta.runtime_length)
        } else {
            let first_coords = dyn_decode_coords(0, &data.meta.logical_shape, &strides);
            build_active_check(
                b,
                &first_coords,
                &data.meta.runtime_shape,
                &data.meta.logical_shape,
            )
        }
    } else {
        b.ir_constant_bool(true)
    };
    let identity = agg_identity_value(b, agg, data.dtype);
    let mut acc = if any_bounded {
        select_value_by_dtype(b, &first_active, &values[0], &identity, data.dtype)
    } else {
        values[0].clone()
    };
    let mut acc_idx = b.ir_constant_int(0);

    if !any_bounded {
        for i in 1..buffer_size.min(values.len()) {
            let elem = values[i].clone();
            let idx_val = b.ir_constant_int(i as i64);
            let (new_acc, new_idx) =
                dyn_agg_step(b, &acc, &acc_idx, &elem, &idx_val, agg, use_float);
            acc = new_acc;
            acc_idx = new_idx;
        }
    } else {
        for i in 1..buffer_size.min(values.len()) {
            let is_active = if compact_buffer {
                build_flat_active_check(b, i, &data.meta.runtime_length)
            } else {
                let coords = dyn_decode_coords(i, &data.meta.logical_shape, &strides);
                build_active_check(
                    b,
                    &coords,
                    &data.meta.runtime_shape,
                    &data.meta.logical_shape,
                )
            };
            let elem = values[i].clone();
            let idx_val = b.ir_constant_int(i as i64);
            let (new_acc, new_idx) = mask_aware_agg_step(
                b, &acc, &acc_idx, &elem, &idx_val, &is_active, &identity, agg, use_float,
            );
            acc = new_acc;
            acc_idx = new_idx;
        }
    }

    if matches!(agg, DynAggKind::Argmax | DynAggKind::Argmin) {
        if let Some(idx_vid) = acc_idx.value_id() {
            let runtime_length_v = scalar_to_value_i(b, &data.meta.runtime_length);
            if let Some(len_vid) = runtime_length_v.value_id() {
                let mut formals = std::collections::HashMap::new();
                formals.insert("len_arr".to_string(), len_vid);
                b.fire_contract("dyn_argextremum", idx_vid, &formals);
            }
            use crate::optim::predicates::formula::{CmpOp, ContractTerm, ContractVar};
            b.facts.insert_for(
                idx_vid,
                ContractTerm::Cmp {
                    op: CmpOp::Le,
                    lhs: Box::new(ContractTerm::Var(ContractVar::Value(idx_vid))),
                    rhs: Box::new(ContractTerm::LitInt((buffer_size as i64).saturating_sub(1))),
                },
            );
        }
        acc_idx
    } else {
        // Scalar Min/Max: result inherits the input segment's bound.
        // Try only `Ge` first; publishing both bounds triggered a
        // constant-fold cascade that folded the entire downstream
        // expression (0509 collapsed to `im[99]`).
        acc
    }
}

fn select_value_by_dtype(
    b: &mut IRBuilder,
    cond: &Value,
    tv: &Value,
    fv: &Value,
    dtype: NumberType,
) -> Value {
    match dtype {
        NumberType::Float => b.ir_select_f(cond, tv, fv),
        NumberType::Integer => b.ir_select_i(cond, tv, fv),
        NumberType::Complex => crate::helpers::value_ops::select_value(b, cond, tv, fv),
    }
}

/// Materialize a `ScalarValue<i64>` as an integer `Value` for in-circuit
/// use. Static values become `ir_constant_int`; symbolic values reference
/// their `stmt_id`.
fn scalar_to_value_i(b: &mut IRBuilder, sv: &ScalarValue<i64>) -> Value {
    if let Some(v) = sv.static_val {
        b.ir_constant_int(v)
    } else if let Some(ptr) = sv.stmt_id {
        Value::Integer(ScalarValue::new(None, Some(ptr)))
    } else {
        b.ir_constant_int(0)
    }
}

/// Per-op algebraic identity for the mask-and-include path.
///
/// **Soundness invariant:** for sum/prod/all/any, this value satisfies
/// `reduce(identity, x) = reduce(x, identity) = x`. Inactive slots
/// contribute this value via `select(is_active, raw, identity)`, so the
/// final reduction is the same as if the inactive slots had been excluded
/// from the iteration.
///
/// For max/min/argmax/argmin this helper returns the dtype's default value
/// (typically `0`) — but the mask-and-include code path doesn't read it;
/// those ops use the first-active pattern (no sentinel) because the ZK
/// Real fragment excludes `±∞`.
fn agg_identity_value(b: &mut IRBuilder, agg: DynAggKind, dtype: NumberType) -> Value {
    match (agg, dtype) {
        (DynAggKind::Sum, NumberType::Integer) => b.ir_constant_int(0),
        (DynAggKind::Sum, NumberType::Float) => b.ir_constant_float(0.0),
        (DynAggKind::Sum, NumberType::Complex) => {
            super::metadata::dyn_default_value(b, NumberType::Complex)
        }
        (DynAggKind::Prod, NumberType::Integer) => b.ir_constant_int(1),
        (DynAggKind::Prod, NumberType::Float) => b.ir_constant_float(1.0),
        (DynAggKind::Prod, NumberType::Complex) => Value::Complex {
            real: ScalarValue::new(Some(1.0), None),
            imag: ScalarValue::new(Some(0.0), None),
        },
        (DynAggKind::All, _) => b.ir_constant_bool(true),
        (DynAggKind::Any, _) => b.ir_constant_bool(false),
        // First-active ops: the placeholder isn't consumed, but we return
        // the dtype's default for completeness.
        _ => super::metadata::dyn_default_value(b, dtype),
    }
}

/// Build the per-slot active mask: `AND_over_axes(coord[ax] < runtime_shape[ax])`.
///
/// **Compile-time elision.** Axes whose `runtime_shape[ax].static_val ==
/// Some(logical_shape[ax])` are trivially active (the runtime size equals
/// the compile-time max); the per-axis check is dropped to avoid emitting
/// redundant `LtI` + `LogicalAnd` IR.
///
/// **Soundness invariant:** for a row-major buffer indexed by
/// `i ∈ [0, prod(logical_shape))` with decoded coords against
/// `logical_shape`, the predicate `∀ax. coord[ax] < runtime_shape[ax]`
/// holds exactly for the active region (the prefix-rectangle of size
/// `prod(runtime_shape)`).
fn build_active_check(
    b: &mut IRBuilder,
    coords: &[usize],
    runtime_shape: &[ScalarValue<i64>],
    logical_shape: &[usize],
) -> Value {
    let mut all_active = b.ir_constant_bool(true);
    for (ax, &c) in coords.iter().enumerate() {
        let is_static = runtime_shape[ax].static_val == Some(logical_shape[ax] as i64);
        if is_static {
            // Trivially active on this axis at compile time; skip.
            continue;
        }
        let runtime_size_v = scalar_to_value_i(b, &runtime_shape[ax]);
        let c_const = b.ir_constant_int(c as i64);
        let lt = b.ir_less_than_i(&c_const, &runtime_size_v);
        all_active = b.ir_logical_and(&all_active, &lt);
    }
    all_active
}

fn build_flat_active_check(
    b: &mut IRBuilder,
    flat_idx: usize,
    runtime_length: &ScalarValue<i64>,
) -> Value {
    if runtime_length
        .static_val
        .is_some_and(|n| flat_idx < n as usize)
    {
        return b.ir_constant_bool(true);
    }
    let runtime_length_v = scalar_to_value_i(b, runtime_length);
    let idx = b.ir_constant_int(flat_idx as i64);
    b.ir_less_than_i(&idx, &runtime_length_v)
}

/// Mask-aware aggregation step: extends `dyn_agg_step` with an `is_active`
/// gate.
///
/// **Sum/prod/all/any** use `contribution = select(is_active, elem,
/// identity)`, then apply the same reduction step as the fast path. Sound
/// because inactive slots contribute the algebraic identity.
///
/// **Max/min/argmax/argmin** use the first-active pattern: update fires
/// only when `is_active AND new-candidate-is-better`. No sentinel needed —
/// the ZK Real fragment excludes `±∞`, so we cannot initialise `acc` to
/// `-∞`/`+∞`.
fn mask_aware_agg_step(
    b: &mut IRBuilder,
    acc: &Value,
    acc_idx: &Value,
    elem: &Value,
    elem_idx: &Value,
    is_active: &Value,
    identity: &Value,
    agg: DynAggKind,
    use_float: bool,
) -> (Value, Value) {
    match agg {
        DynAggKind::Sum | DynAggKind::Prod | DynAggKind::All | DynAggKind::Any => {
            let contribution = if matches!(agg, DynAggKind::All | DynAggKind::Any) {
                let bv = b.ir_bool_cast(elem);
                b.ir_select_b(is_active, &bv, identity)
            } else if use_float {
                b.ir_select_f(is_active, elem, identity)
            } else if matches!(elem, Value::Complex { .. })
                || matches!(identity, Value::Complex { .. })
            {
                crate::helpers::value_ops::select_value(b, is_active, elem, identity)
            } else {
                b.ir_select_i(is_active, elem, identity)
            };
            let new_acc = match agg {
                DynAggKind::Sum if use_float => b.ir_add_f(acc, &contribution),
                DynAggKind::Sum
                    if matches!(acc, Value::Complex { .. })
                        || matches!(contribution, Value::Complex { .. }) =>
                {
                    crate::helpers::value_ops::apply_scalar_binary_op(b, "add", acc, &contribution)
                }
                DynAggKind::Sum => b.ir_add_i(acc, &contribution),
                DynAggKind::Prod if use_float => b.ir_mul_f(acc, &contribution),
                DynAggKind::Prod
                    if matches!(acc, Value::Complex { .. })
                        || matches!(contribution, Value::Complex { .. }) =>
                {
                    crate::helpers::value_ops::apply_scalar_binary_op(b, "mul", acc, &contribution)
                }
                DynAggKind::Prod => b.ir_mul_i(acc, &contribution),
                DynAggKind::All => b.ir_logical_and(acc, &contribution),
                DynAggKind::Any => b.ir_logical_or(acc, &contribution),
                _ => unreachable!(),
            };
            (new_acc, acc_idx.clone())
        }
        DynAggKind::Max | DynAggKind::Min | DynAggKind::Argmax | DynAggKind::Argmin => {
            // First-active pattern: no sentinel (ZK Real has no ±∞).
            let cmp = match agg {
                DynAggKind::Max | DynAggKind::Argmax => {
                    if use_float {
                        b.ir_greater_than_f(elem, acc)
                    } else {
                        b.ir_greater_than_i(elem, acc)
                    }
                }
                DynAggKind::Min | DynAggKind::Argmin => {
                    if use_float {
                        b.ir_less_than_f(elem, acc)
                    } else {
                        b.ir_less_than_i(elem, acc)
                    }
                }
                _ => unreachable!(),
            };
            let take = b.ir_logical_and(is_active, &cmp);
            let new_acc = if use_float {
                b.ir_select_f(&take, elem, acc)
            } else {
                b.ir_select_i(&take, elem, acc)
            };
            let new_idx = b.ir_select_i(&take, elem_idx, acc_idx);
            (new_acc, new_idx)
        }
    }
}

/// Axis reduction: reduce along a specific axis.
pub fn dyn_aggregate_axis(
    b: &mut IRBuilder,
    data: &DynamicNDArrayData,
    axis: i64,
    agg: DynAggKind,
) -> Value {
    let shape = &data.meta.logical_shape.clone();
    let ndim = shape.len();
    let ax = if axis < 0 {
        (ndim as i64 + axis) as usize
    } else {
        axis as usize
    };
    assert!(ax < ndim, "aggregate axis out of bounds");
    if data.dtype == NumberType::Complex
        && matches!(
            agg,
            DynAggKind::Max | DynAggKind::Min | DynAggKind::Argmax | DynAggKind::Argmin
        )
    {
        panic!("Complex DynamicNDArray does not support ordered reductions");
    }

    // Fast path: `argwhere(A).min(axis=0)` / `.max(axis=0)`. The
    // np_argwhere lowering stashes the per-cell `keep` predicates
    // (is-nonzero on the source array). For axis=0 reductions the result
    // is the min / max coordinate per source axis — algebraically
    // equivalent to the row_has/col_has + linear-scan algorithm that the
    // handwritten Noir baselines for the nonzero-bbox kernels use. Skip
    // the materialise-then-reduce O(N * rank) scan and emit the
    // O(R + C) bounds computation directly.
    if ax == 0
        && matches!(agg, DynAggKind::Min | DynAggKind::Max)
        && data.dtype == NumberType::Integer
    {
        if let Some(entry) = b.argwhere_keep_indicators.get(&data.segment_id) {
            let (src_shape, keeps) = entry.clone();
            return argwhere_axis0_bbox_bounds(b, &src_shape, &keeps, agg);
        }
    }

    let data_owned = super::force_contiguous(b, data);
    let data = &data_owned;
    // Runtime-dynamic geometry: a runtime reshape (e.g. (-1, bs) with a
    // witness bs) leaves logical_shape as a degenerate padding rectangle;
    // enumerating it visits the wrong slots and produces the wrong number
    // of outputs. When the runtime shape is dynamic, enumerate the
    // envelope's per-axis BOUNDS and gather with the runtime stride cells
    // instead of the logical layout; build_active_check already masks by
    // the runtime dims.
    // Per-axis enumeration bound: static axes are exact; a runtime axis
    // can be as large as the whole storage (dyn_reshape_runtime records
    // logical size 1 for runtime dims and documents that true bounds live
    // in runtime_shape — the only sound static bound is total_bound).
    let total_bound = data.envelope.total_bound.max(1);
    let bounds: Vec<usize> = (0..ndim)
        .map(|i| {
            let rt_static = data.meta.runtime_shape[i].static_val;
            if rt_static == Some(shape[i] as i64) {
                shape[i]
            } else {
                let env_max = data.envelope.dims.get(i).map(|d| d.max).unwrap_or(1);
                if env_max <= 1 && shape[i] <= 1 {
                    // Runtime-reshape skeleton axis (logical placeholder 1
                    // with a dynamic runtime dim): the only sound static
                    // bound is the whole storage.
                    total_bound
                } else {
                    env_max.max(shape[i])
                }
            }
        })
        .collect();
    // Layout discriminator: a padded buffer requires runtime <= logical on
    // every axis. When some axis bound EXCEEDS the logical size, the
    // logical shape is a runtime-reshape skeleton and the buffer is
    // compact — gather with the runtime stride cells. Otherwise (bounded
    // inputs etc.) the buffer is the padded logical rectangle and the
    // existing logical-stride path is correct (runtime masking included).
    let runtime_dynamic = bounds != *shape;
    let grid_shape: Vec<usize> = if runtime_dynamic { bounds } else { shape.clone() };
    // Mask formulation: when the reduced axis is innermost (static stride
    // 1, so each output's elements form a contiguous storage window
    // [base, base + dim_ax)), fold over STORAGE positions with a window-
    // membership test instead of gathering by runtime coordinate. The
    // window base is only compared — never dereferenced — so no address
    // guard is needed and the payload is fetched with constant reads.
    let mask_form = runtime_dynamic
        && data
            .meta
            .runtime_strides
            .get(ax)
            .map_or(false, |st| st.static_val == Some(1));
    let values = if runtime_dynamic && !mask_form {
        Vec::new()
    } else {
        crate::helpers::segment::read_all_typed(b, data.segment_id, data.max_length(), data.dtype)
    };
    let strides = dyn_row_major_strides(&grid_shape);
    let use_float =
        data.dtype == NumberType::Float && !matches!(agg, DynAggKind::All | DynAggKind::Any);

    // Output shape: remove the reduced axis
    let out_shape: Vec<usize> = grid_shape
        .iter()
        .enumerate()
        .filter(|&(i, _)| i != ax)
        .map(|(_, &s)| s)
        .collect();
    let out_numel: usize = if out_shape.is_empty() {
        1
    } else {
        out_shape.iter().product()
    };
    let out_strides = dyn_row_major_strides(&out_shape);
    let axis_dim = grid_shape[ax];

    let out_runtime_shape: Vec<ScalarValue<i64>> = data
        .meta
        .runtime_shape
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != ax)
        .map(|(_, s)| s.clone())
        .collect();

    let mut out_elements: Vec<Value> = Vec::with_capacity(out_numel);

    for out_idx in 0..out_numel {
        // Decode output coordinates
        let out_coords = if out_shape.is_empty() {
            vec![]
        } else {
            dyn_decode_coords(out_idx, &out_shape, &out_strides)
        };

        // Build input coordinates: insert axis position
        let mut in_coords = out_coords.clone();
        in_coords.insert(ax, 0);

        let identity = agg_identity_value(b, agg, data.dtype);

        // Sentinel-init pattern for Integer min/max/argmin/argmax: pick a
        // value that is strictly worse than any possible input under the
        // comparison so the standard "is new better?" branch handles the
        // first-active iteration naturally — no separate `seen_active`
        // tracking needed.
        //
        // BN254 Fr is order ~2^254, Zinnia's signed-comparison fragment
        // treats values in [0, p/2) as positive and [p/2, p) as negative.
        // i64::{MAX, MIN} fit comfortably with room to spare, so they
        // serve as sentinels.
        //
        // Floats stay on the first-active path because the ZK Real
        // fragment excludes ±∞ — there's no representable strict-greater-
        // than-everything float we can witness without breaking IRs that
        // assume float values are bounded.
        let use_sentinel_init = !use_float
            && matches!(
                agg,
                DynAggKind::Min | DynAggKind::Max | DynAggKind::Argmin | DynAggKind::Argmax
            );
        let (mut acc, mut acc_idx) = if use_sentinel_init {
            // See the comparison-site sentinel comment below for why Max
            // uses 0 (in conjunction with `gt_i_bounded`). Both sentinels
            // are u128-clean and fit Zinnia's signed Field representation
            // without crossing into the negative half-plane.
            let sentinel = match agg {
                DynAggKind::Min | DynAggKind::Argmin => b.ir_constant_int(i64::MAX),
                DynAggKind::Max | DynAggKind::Argmax => b.ir_constant_int(0),
                _ => unreachable!(),
            };
            (sentinel, b.ir_constant_int(0))
        } else {
            (identity.clone(), b.ir_constant_int(0))
        };
        let mut seen_active = if use_sentinel_init {
            b.ir_constant_bool(false) // unused in this branch; keep the
                                      // identifier alive so the borrow-checker
                                      // doesn't notice the dead-store.
        } else {
            b.ir_constant_bool(false)
        };

        // Mask-form precomputation: contiguous storage window of this
        // output combo, [base, base + dim_ax), plus its non-axis activity.
        let mask_ctx = if mask_form {
            let mut base = scalar_to_value_i(b, &data.meta.runtime_offset);
            for (i, &c) in in_coords.iter().enumerate() {
                if i == ax || c == 0 {
                    continue;
                }
                let stride_v = scalar_to_value_i(b, &data.meta.runtime_strides[i]);
                let c_v = b.ir_constant_int(c as i64);
                let term = b.ir_mul_i(&c_v, &stride_v);
                base = b.ir_add_i(&base, &term);
            }
            let dim_ax = scalar_to_value_i(b, &data.meta.runtime_shape[ax]);
            let hi = b.ir_add_i(&base, &dim_ax);
            let oc: Vec<usize> = in_coords
                .iter()
                .enumerate()
                .filter(|&(i, _)| i != ax)
                .map(|(_, &c)| c)
                .collect();
            let rt_f: Vec<ScalarValue<i64>> = data
                .meta
                .runtime_shape
                .iter()
                .enumerate()
                .filter(|&(i, _)| i != ax)
                .map(|(_, v)| v.clone())
                .collect();
            let gs_f: Vec<usize> = grid_shape
                .iter()
                .enumerate()
                .filter(|&(i, _)| i != ax)
                .map(|(_, &v)| v)
                .collect();
            let out_active = build_active_check(b, &oc, &rt_f, &gs_f);
            Some((base, hi, out_active))
        } else {
            None
        };
        let steps = if mask_form {
            data.max_length()
        } else {
            axis_dim
        };

        // Iterate: storage positions (mask form) or axis coordinates.
        for k in 0..steps {
            let (elem, is_active, k_val) = if let Some((base, hi, out_active)) = &mask_ctx {
                let s_c = b.ir_constant_int(k as i64);
                let lt_base = b.ir_less_than_i(&s_c, base);
                let ge_base = b.ir_logical_not(&lt_base);
                let lt_hi = b.ir_less_than_i(&s_c, hi);
                let member = b.ir_logical_and(&ge_base, &lt_hi);
                let active = b.ir_logical_and(&member, out_active);
                let elem = if k < values.len() {
                    values[k].clone()
                } else {
                    super::metadata::dyn_default_value(b, data.dtype)
                };
                // Axis position (argmax/argmin index) = s - base.
                let pos = b.ir_sub_i(&s_c, base);
                (elem, active, pos)
            } else if runtime_dynamic {
                in_coords[ax] = k;
                let is_active = build_active_check(
                    b,
                    &in_coords,
                    &data.meta.runtime_shape,
                    &grid_shape,
                );
                // Gather with the runtime stride cells: addr = sum of
                // coord_i * runtime_stride_i (+ runtime offset). Inactive
                // grid combos can compute out-of-range addresses, so the
                // address is gated by the active mask (active implies
                // in-bounds: every coord < its runtime dim).
                let mut addr = scalar_to_value_i(b, &data.meta.runtime_offset);
                for (i, &c) in in_coords.iter().enumerate() {
                    if c == 0 {
                        continue;
                    }
                    let stride_v = scalar_to_value_i(b, &data.meta.runtime_strides[i]);
                    let c_v = b.ir_constant_int(c as i64);
                    let term = b.ir_mul_i(&c_v, &stride_v);
                    addr = b.ir_add_i(&addr, &term);
                }
                let zero = b.ir_constant_int(0);
                let safe_addr = b.ir_select_i(&is_active, &addr, &zero);
                let elem =
                    crate::helpers::segment::read_typed(b, data.segment_id, &safe_addr, data.dtype);
                let k_val = b.ir_constant_int(k as i64);
                (elem, is_active, k_val)
            } else {
                in_coords[ax] = k;
                let src_idx = dyn_encode_coords(&in_coords, &strides);
                let elem = if src_idx < values.len() {
                    values[src_idx].clone()
                } else {
                    super::metadata::dyn_default_value(b, data.dtype)
                };
                let is_active = build_active_check(
                    b,
                    &in_coords,
                    &data.meta.runtime_shape,
                    &grid_shape,
                );
                let k_val = b.ir_constant_int(k as i64);
                (elem, is_active, k_val)
            };
            if matches!(
                agg,
                DynAggKind::Sum | DynAggKind::Prod | DynAggKind::All | DynAggKind::Any
            ) {
                let (new_acc, new_idx) = mask_aware_agg_step(
                    b, &acc, &acc_idx, &elem, &k_val, &is_active, &identity, agg, use_float,
                );
                acc = new_acc;
                acc_idx = new_idx;
            } else if use_sentinel_init {
                // Mask inactive elems to the sentinel so they can never
                // win the comparison. Then standard min/max compare-and-
                // select — no `seen_active` bookkeeping.
                //
                // Both sentinels are u128-clean now:
                //   Min: i64::MAX = 2^63 - 1   (existing — always safe)
                //   Max: 0                     (NEW — safe when elem is
                //                               non-negative, which holds
                //                               for index / coord / mask
                //                               reductions in scope; the
                //                               count == 0 case differs from
                //                               the old i64::MIN sentinel
                //                               by returning 0 instead, but
                //                               no in-scope kernel reads
                //                               that value).
                let sentinel = match agg {
                    DynAggKind::Min | DynAggKind::Argmin => b.ir_constant_int(i64::MAX),
                    DynAggKind::Max | DynAggKind::Argmax => b.ir_constant_int(0),
                    _ => unreachable!(),
                };
                let contribution = b.ir_select_i(&is_active, &elem, &sentinel);
                let take = match agg {
                    DynAggKind::Max | DynAggKind::Argmax => {
                        b.ir_greater_than_i_bounded(&contribution, &acc)
                    }
                    DynAggKind::Min | DynAggKind::Argmin => {
                        b.ir_less_than_i_bounded(&contribution, &acc)
                    }
                    _ => unreachable!(),
                };
                acc = b.ir_select_i(&take, &contribution, &acc);
                acc_idx = b.ir_select_i(&take, &k_val, &acc_idx);
            } else {
                let cmp = match agg {
                    DynAggKind::Max | DynAggKind::Argmax => {
                        if use_float {
                            b.ir_greater_than_f(&elem, &acc)
                        } else {
                            b.ir_greater_than_i(&elem, &acc)
                        }
                    }
                    DynAggKind::Min | DynAggKind::Argmin => {
                        if use_float {
                            b.ir_less_than_f(&elem, &acc)
                        } else {
                            b.ir_less_than_i(&elem, &acc)
                        }
                    }
                    _ => unreachable!(),
                };
                let not_seen = b.ir_logical_not(&seen_active);
                let better_or_first = b.ir_logical_or(&not_seen, &cmp);
                let take = b.ir_logical_and(&is_active, &better_or_first);
                acc = if use_float {
                    b.ir_select_f(&take, &elem, &acc)
                } else {
                    b.ir_select_i(&take, &elem, &acc)
                };
                acc_idx = b.ir_select_i(&take, &k_val, &acc_idx);
                seen_active = b.ir_logical_or(&seen_active, &is_active);
            }
        }

        let result = match agg {
            DynAggKind::Argmax | DynAggKind::Argmin => acc_idx,
            _ => acc,
        };
        out_elements.push(result);
    }

    // Determine output dtype
    let out_dtype = match agg {
        DynAggKind::All | DynAggKind::Any | DynAggKind::Argmax | DynAggKind::Argmin => {
            NumberType::Integer
        }
        _ => data.dtype,
    };

    if out_shape.is_empty() {
        // Scalar result
        return out_elements[0].clone();
    }

    let out_strides_meta = dyn_row_major_strides(&out_shape);
    let segment_id = crate::helpers::segment::alloc_and_write_values(b, &out_elements, out_dtype);
    let envelope = crate::types::Envelope::from_static_shape(&mut b.dim_table, &out_shape);
    let mut runtime_length_v = b.ir_constant_int(1);
    for dim in &out_runtime_shape {
        let dim_v = scalar_to_value_i(b, dim);
        runtime_length_v = b.ir_mul_i(&runtime_length_v, &dim_v);
    }
    let runtime_length = value_to_scalar_i64(&runtime_length_v);
    let result = DynamicNDArrayData {
        envelope,
        dtype: out_dtype,
        segment_id,
        meta: DynArrayMeta {
            logical_shape: out_shape.clone(),
            logical_offset: 0,
            logical_strides: out_strides_meta.clone(),
            runtime_length,
            runtime_rank: ScalarValue::new(Some(out_shape.len() as i64), None),
            runtime_shape: out_runtime_shape,
            runtime_strides: out_strides_meta
                .iter()
                .map(|&s| ScalarValue::new(Some(s as i64), None))
                .collect(),
            runtime_offset: ScalarValue::new(Some(0), None),
        },
        value_id: ValueId::next(),
    };
    Value::DynamicNDArray(result)
}

/// `argwhere(A).min/max(axis=0)` fast path. Computes per-source-axis bounds
/// directly from the cached keep predicates instead of materialising and
/// scanning the (count, rank) coordinate segment.
///
/// For each `src_axis a`, build the per-index OR-of-keeps along axis a
/// (the row_has / col_has arrays the handwritten Noir nonzero-bbox baseline
/// computes), then linear-scan for the first / last index with a hit. Total
/// ops: O(cells.len() + sum(src_shape)) — independent of `rank`.
fn argwhere_axis0_bbox_bounds(
    b: &mut IRBuilder,
    src_shape: &[usize],
    keeps: &[Value],
    agg: DynAggKind,
) -> Value {
    use crate::helpers::shape_arith::{decode_coords, row_major_strides};
    let rank = src_shape.len();
    let src_strides = row_major_strides(src_shape);
    let mut out_elements: Vec<Value> = Vec::with_capacity(rank);

    let zero_i = b.ir_constant_int(0);
    let mut counts_by_axis: Vec<Vec<Value>> = src_shape
        .iter()
        .map(|&dim| vec![zero_i.clone(); dim])
        .collect();
    for (flat_idx, keep) in keeps.iter().enumerate() {
        let coords = decode_coords(flat_idx, src_shape, &src_strides);
        for src_axis in 0..rank {
            let coord = coords[src_axis];
            let prev = counts_by_axis[src_axis][coord].clone();
            counts_by_axis[src_axis][coord] = b.ir_add_i(&prev, keep);
        }
    }

    for src_axis in 0..rank {
        let dim = src_shape[src_axis];
        // Build the row/column occupancy vectors in one pass over the
        // cached keeps, then threshold each bucket with a single bounded
        // comparison. This matches the handwritten bbox baselines more
        // closely than rescanning all keeps once per bucket.
        let mut has_along: Vec<Value> = Vec::with_capacity(dim);
        for acc in &counts_by_axis[src_axis] {
            // has = (0 < acc), bounded since both are non-negative small ints.
            let has = b.ir_less_than_i_bounded(&zero_i, acc);
            has_along.push(has);
        }

        // Linear scan over has_along.
        //  Min: first k where has_along[k] = true. Sentinel = `dim`
        //       (out-of-range) when no hits — matches numpy's behaviour
        //       only for non-empty inputs; argwhere on an all-zero matrix
        //       isn't in scope for the kernels using this path.
        //  Max: last k where has_along[k] = true. Sentinel = 0
        //       (mirrors the i64::MIN → 0 narrowing in the slow path's
        //       Max-bounded sentinel).
        let initial = match agg {
            DynAggKind::Min => b.ir_constant_int(dim as i64),
            DynAggKind::Max => b.ir_constant_int(0),
            _ => unreachable!(),
        };
        let mut acc_idx = initial.clone();

        for k in 0..dim {
            let k_val = b.ir_constant_int(k as i64);
            let has = &has_along[k];
            match agg {
                DynAggKind::Min => {
                    let still_initial = b.ir_equal_i(&acc_idx, &initial);
                    let trigger = b.ir_logical_and(&still_initial, has);
                    acc_idx = b.ir_select_i(&trigger, &k_val, &acc_idx);
                }
                DynAggKind::Max => {
                    acc_idx = b.ir_select_i(has, &k_val, &acc_idx);
                }
                _ => unreachable!(),
            }
        }
        // Publish a tight bound on `acc_idx` via the registered
        // `dyn_argextremum` contract:
        //   Min: `acc_idx ∈ [0, dim]` (sentinel `dim` covers no-hit, real
        //         hits land in [0, dim-1]) — bind len_arr = dim+1.
        //   Max: `acc_idx ∈ [0, dim-1]` (initial 0, real hits in [0, dim-1])
        //         — bind len_arr = dim. The kernel uses `bbox.max(0) + 1`
        //         as a slice stop and discharge_slice_bound asks
        //         `stop < axis_len + 1`; bounding `max < dim` makes
        //         `max + 1 ≤ dim = axis_len`, exactly what the
        //         downstream discharge needs to prove from facts (the
        //         loose `dim+1` bound would leave `max+1 ≤ dim+1` and
        //         discharge would still emit a u128 bound assert).
        //
        // For this to actually close the bbox-crop residual, the fact
        // must propagate through segment-storage + tuple-unpack
        // extraction. That's the job of `IRBuilder::segment_cell_bounds`
        // (write-side capture in `alloc_and_write_values`, read-side
        // republish in `ir_read_memory`).
        if let Some(idx_vid) = acc_idx.value_id() {
            let len_for_contract = match agg {
                DynAggKind::Min => dim as i64 + 1,
                DynAggKind::Max => dim as i64,
                _ => unreachable!(),
            };
            let len_val = b.ir_constant_int(len_for_contract);
            if let Some(len_vid) = len_val.value_id() {
                let mut formals = std::collections::HashMap::new();
                formals.insert("len_arr".to_string(), len_vid);
                b.fire_contract("dyn_argextremum", idx_vid, &formals);
            }
        }
        out_elements.push(acc_idx);
    }

    let out_shape = vec![rank];
    let out_strides_meta = crate::helpers::shape_arith::row_major_strides(&out_shape);
    let segment_id =
        crate::helpers::segment::alloc_and_write_values(b, &out_elements, NumberType::Integer);
    let envelope = crate::types::Envelope::from_static_shape(&mut b.dim_table, &out_shape);
    let runtime_length = ScalarValue::new(Some(rank as i64), None);
    Value::DynamicNDArray(DynamicNDArrayData {
        envelope,
        dtype: NumberType::Integer,
        segment_id,
        meta: DynArrayMeta {
            logical_shape: out_shape.clone(),
            logical_offset: 0,
            logical_strides: out_strides_meta.clone(),
            runtime_length,
            runtime_rank: ScalarValue::new(Some(1), None),
            runtime_shape: vec![ScalarValue::new(Some(rank as i64), None)],
            runtime_strides: out_strides_meta
                .iter()
                .map(|&s| ScalarValue::new(Some(s as i64), None))
                .collect(),
            runtime_offset: ScalarValue::new(Some(0), None),
        },
        value_id: ValueId::next(),
    })
}

/// One step of the accumulator for a given aggregation kind.
pub fn dyn_agg_step(
    b: &mut IRBuilder,
    acc: &Value,
    acc_idx: &Value,
    elem: &Value,
    elem_idx: &Value,
    agg: DynAggKind,
    use_float: bool,
) -> (Value, Value) {
    match agg {
        DynAggKind::Sum => {
            let new_acc =
                if matches!(acc, Value::Complex { .. }) || matches!(elem, Value::Complex { .. }) {
                    crate::helpers::value_ops::apply_scalar_binary_op(b, "add", acc, elem)
                } else if use_float {
                    b.ir_add_f(acc, elem)
                } else {
                    b.ir_add_i(acc, elem)
                };
            (new_acc, acc_idx.clone())
        }
        DynAggKind::Prod => {
            let new_acc =
                if matches!(acc, Value::Complex { .. }) || matches!(elem, Value::Complex { .. }) {
                    crate::helpers::value_ops::apply_scalar_binary_op(b, "mul", acc, elem)
                } else if use_float {
                    b.ir_mul_f(acc, elem)
                } else {
                    b.ir_mul_i(acc, elem)
                };
            (new_acc, acc_idx.clone())
        }
        DynAggKind::Max => {
            let cond = if use_float {
                b.ir_greater_than_f(elem, acc)
            } else {
                b.ir_greater_than_i(elem, acc)
            };
            let new_acc = if use_float {
                b.ir_select_f(&cond, elem, acc)
            } else {
                b.ir_select_i(&cond, elem, acc)
            };
            let new_idx = b.ir_select_i(&cond, elem_idx, acc_idx);
            (new_acc, new_idx)
        }
        DynAggKind::Min => {
            let cond = if use_float {
                b.ir_less_than_f(elem, acc)
            } else {
                b.ir_less_than_i(elem, acc)
            };
            let new_acc = if use_float {
                b.ir_select_f(&cond, elem, acc)
            } else {
                b.ir_select_i(&cond, elem, acc)
            };
            let new_idx = b.ir_select_i(&cond, elem_idx, acc_idx);
            (new_acc, new_idx)
        }
        DynAggKind::All => {
            let bv = b.ir_bool_cast(elem);
            let new_acc = b.ir_logical_and(acc, &bv);
            (new_acc, acc_idx.clone())
        }
        DynAggKind::Any => {
            let bv = b.ir_bool_cast(elem);
            let new_acc = b.ir_logical_or(acc, &bv);
            (new_acc, acc_idx.clone())
        }
        DynAggKind::Argmax => {
            let cond = if use_float {
                b.ir_greater_than_f(elem, acc)
            } else {
                b.ir_greater_than_i(elem, acc)
            };
            let new_acc = if use_float {
                b.ir_select_f(&cond, elem, acc)
            } else {
                b.ir_select_i(&cond, elem, acc)
            };
            let new_idx = b.ir_select_i(&cond, elem_idx, acc_idx);
            (new_acc, new_idx)
        }
        DynAggKind::Argmin => {
            let cond = if use_float {
                b.ir_less_than_f(elem, acc)
            } else {
                b.ir_less_than_i(elem, acc)
            };
            let new_acc = if use_float {
                b.ir_select_f(&cond, elem, acc)
            } else {
                b.ir_select_i(&cond, elem, acc)
            };
            let new_idx = b.ir_select_i(&cond, elem_idx, acc_idx);
            (new_acc, new_idx)
        }
    }
}

/// Identity value for aggregation init (used when array is empty).
pub fn dyn_agg_identity(b: &mut IRBuilder, agg: DynAggKind, dtype: NumberType) -> Value {
    match agg {
        DynAggKind::Sum => super::metadata::dyn_default_value(b, dtype),
        DynAggKind::Prod => match dtype {
            NumberType::Integer => b.ir_constant_int(1),
            NumberType::Float => b.ir_constant_float(1.0),
            NumberType::Complex => Value::Complex {
                real: ScalarValue::new(Some(1.0), None),
                imag: ScalarValue::new(Some(0.0), None),
            },
        },
        DynAggKind::All => b.ir_constant_bool(true),
        DynAggKind::Any => b.ir_constant_bool(false),
        DynAggKind::Max | DynAggKind::Min => super::metadata::dyn_default_value(b, dtype),
        DynAggKind::Argmax | DynAggKind::Argmin => b.ir_constant_int(0),
    }
}

// ── Tests (multi-dim Case B Tier 2) ─────────────────────────────────────

#[cfg(test)]
mod bounded_tests {
    use super::*;
    use crate::ir_defs::IR;
    use crate::ops::dyn_ndarray::constructors::dyn_from_values_with_active_nd;
    use std::collections::HashMap;

    /// Minimal IR walker for tests: evaluates the subset of IR statements
    /// that `dyn_aggregate_all` emits (constants + int arith + comparisons +
    /// logical + selects + bool cast + memory read/write/allocate) and
    /// returns the i64 stored at the result's stmt_id. We model bools as
    /// 0/1 ints so SelectI/SelectB read the predicate uniformly.
    fn eval_at_stmt(b: &IRBuilder, target: crate::types::StmtId) -> i64 {
        let mut vals: HashMap<crate::types::StmtId, i64> = HashMap::new();
        let mut mem: HashMap<u32, Vec<i64>> = HashMap::new();
        for stmt in &b.stmts {
            let id = stmt.stmt_id;
            let args: Vec<i64> = stmt
                .arguments
                .iter()
                .map(|a| vals.get(a).copied().unwrap_or(0))
                .collect();
            let result: Option<i64> = match &stmt.ir {
                IR::ConstantInt { value } => Some(*value),
                IR::ConstantBool { value } => Some(if *value { 1 } else { 0 }),
                IR::ConstantFloat { value } => Some(*value as i64),
                IR::AddI => Some(args[0] + args[1]),
                IR::SubI => Some(args[0] - args[1]),
                IR::MulI => Some(args[0] * args[1]),
                IR::LtI => Some(if args[0] < args[1] { 1 } else { 0 }),
                IR::LteI => Some(if args[0] <= args[1] { 1 } else { 0 }),
                IR::GtI => Some(if args[0] > args[1] { 1 } else { 0 }),
                IR::GteI => Some(if args[0] >= args[1] { 1 } else { 0 }),
                IR::EqI => Some(if args[0] == args[1] { 1 } else { 0 }),
                IR::NeI => Some(if args[0] != args[1] { 1 } else { 0 }),
                IR::LogicalAnd => Some(if args[0] != 0 && args[1] != 0 { 1 } else { 0 }),
                IR::LogicalOr => Some(if args[0] != 0 || args[1] != 0 { 1 } else { 0 }),
                IR::LogicalNot => Some(if args[0] == 0 { 1 } else { 0 }),
                IR::SelectI | IR::SelectB | IR::SelectF => {
                    Some(if args[0] != 0 { args[1] } else { args[2] })
                }
                IR::BoolCast => Some(if args[0] != 0 { 1 } else { 0 }),
                IR::AllocateMemory {
                    segment_id,
                    size,
                    init_value,
                } => {
                    mem.insert(*segment_id, vec![*init_value; *size as usize]);
                    None
                }
                IR::WriteMemory { segment_id } => {
                    let addr = args[0] as usize;
                    let v = args[1];
                    if let Some(seg) = mem.get_mut(segment_id) {
                        if addr < seg.len() {
                            seg[addr] = v;
                        }
                    }
                    None
                }
                IR::ReadMemory { segment_id } => {
                    let addr = args[0] as usize;
                    Some(
                        mem.get(segment_id)
                            .and_then(|seg| seg.get(addr).copied())
                            .unwrap_or(0),
                    )
                }
                IR::Assert => None,
                _ => None,
            };
            if let Some(v) = result {
                vals.insert(id, v);
            }
        }
        vals.get(&target).copied().unwrap_or_else(|| {
            panic!(
                "eval_at_stmt: result stmt_id {:?} not produced by interpreter",
                target
            )
        })
    }

    /// Build a Layout-A 2-D bounded `DynamicNDArray` with `max_shape = [3, 3]`
    /// and a statically-known `runtime_shape = [r0, r1]` (where `r0, r1 < 3`).
    /// The `runtime_shape` entries carry `static_val = Some(rN)` (which
    /// differs from `logical_shape[ax] = 3`), so `any_bounded == true` and
    /// the aggregation mask-and-include path fires. The buffer is fully
    /// user-controlled (9 slots in row-major order).
    fn make_bounded_2d_3x3(
        b: &mut IRBuilder,
        values: [i64; 9],
        r0: i64,
        r1: i64,
        dtype: NumberType,
    ) -> DynamicNDArrayData {
        let sv: Vec<ScalarValue<i64>> = values
            .iter()
            .map(|&v| ScalarValue::new(Some(v), None))
            .collect();
        // Static-valued runtime shape entries differ from logical_shape (3)
        // ⇒ `any_bounded` fires while constants propagate through the
        // mask-and-include reduction so the final accumulator's static_val
        // is computable.
        let runtime_shape = vec![
            ScalarValue::new(Some(r0), None),
            ScalarValue::new(Some(r1), None),
        ];
        let runtime_length = b.ir_constant_int(r0 * r1);
        let val =
            dyn_from_values_with_active_nd(b, sv, vec![3, 3], runtime_shape, runtime_length, dtype);
        match val {
            Value::DynamicNDArray(d) => d,
            _ => unreachable!(),
        }
    }

    #[test]
    fn dyn_sum_on_bounded_2d_excludes_padding() {
        // Layout-A 3x3 buffer, runtime_shape = (2, 2). Active slots:
        // (0,0)=1, (0,1)=2, (1,0)=4, (1,1)=5 → sum = 12. Padding slots
        // (0,2)=99, (1,2)=99, (2,0)=99, (2,1)=99, (2,2)=99 must NOT
        // contribute. Old buggy path would yield 12 + 5*99 = 507.
        let mut b = IRBuilder::new();
        let data = make_bounded_2d_3x3(
            &mut b,
            [1, 2, 99, 4, 5, 99, 99, 99, 99],
            2,
            2,
            NumberType::Integer,
        );
        let result = dyn_aggregate_all(&mut b, &data, DynAggKind::Sum);
        let target = result.stmt_id().expect("sum result has stmt_id");
        let value = eval_at_stmt(&b, target);
        assert_eq!(
            value, 12,
            "sum should equal active-region sum (12), got {}",
            value
        );
    }

    #[test]
    fn dyn_prod_on_bounded_2d_excludes_padding() {
        // Active (2x2) values = 1,2,3,4 ⇒ prod = 24. Padding slots = 0
        // (the multiplicative absorbing element). Old buggy path would
        // multiply by zeros and yield 0.
        let mut b = IRBuilder::new();
        let data = make_bounded_2d_3x3(
            &mut b,
            [1, 2, 0, 3, 4, 0, 0, 0, 0],
            2,
            2,
            NumberType::Integer,
        );
        let result = dyn_aggregate_all(&mut b, &data, DynAggKind::Prod);
        let target = result.stmt_id().expect("prod result has stmt_id");
        let value = eval_at_stmt(&b, target);
        assert_eq!(
            value, 24,
            "prod should equal active-region prod (24), got {}",
            value
        );
    }

    #[test]
    fn dyn_all_on_bounded_2d_excludes_padding() {
        // Active (2x2) values = all 1 (truthy). Padding = 0 (falsy). The
        // old buggy path would AND a `false` from padding and yield false;
        // the mask-and-include path replaces padding contributions with
        // the boolean identity `true` and yields `true`.
        let mut b = IRBuilder::new();
        let data = make_bounded_2d_3x3(
            &mut b,
            [1, 1, 0, 1, 1, 0, 0, 0, 0],
            2,
            2,
            NumberType::Integer,
        );
        let result = dyn_aggregate_all(&mut b, &data, DynAggKind::All);
        let target = result.stmt_id().expect("all result has stmt_id");
        let value = eval_at_stmt(&b, target);
        assert_eq!(
            value, 1,
            "all should be true (1) over active region, got {}",
            value
        );
    }

    #[test]
    fn dyn_any_on_bounded_2d_excludes_padding() {
        // Active (2x2) values = all 0 (falsy). Padding = 1 (truthy). The
        // old buggy path would OR a `true` from padding and yield true;
        // the mask-and-include path replaces padding contributions with
        // the boolean identity `false` and yields `false`.
        let mut b = IRBuilder::new();
        let data = make_bounded_2d_3x3(
            &mut b,
            [0, 0, 1, 0, 0, 1, 1, 1, 1],
            2,
            2,
            NumberType::Integer,
        );
        let result = dyn_aggregate_all(&mut b, &data, DynAggKind::Any);
        let target = result.stmt_id().expect("any result has stmt_id");
        let value = eval_at_stmt(&b, target);
        assert_eq!(
            value, 0,
            "any should be false (0) over active region, got {}",
            value
        );
    }

    #[test]
    fn dyn_max_on_bounded_2d_excludes_padding_negatives() {
        // Active (2x2) values = -5, -3, -7, -2 ⇒ max = -2. Padding = 0
        // (segment-init style). Old buggy path would see `0 > -2` and
        // return 0. The first-active pattern keeps `acc = buffer[0] = -5`
        // initially and only updates when `is_active AND candidate > acc`.
        let mut b = IRBuilder::new();
        let data = make_bounded_2d_3x3(
            &mut b,
            [-5, -3, 0, -7, -2, 0, 0, 0, 0],
            2,
            2,
            NumberType::Integer,
        );
        let result = dyn_aggregate_all(&mut b, &data, DynAggKind::Max);
        let target = result.stmt_id().expect("max result has stmt_id");
        let value = eval_at_stmt(&b, target);
        assert!(
            value < 0,
            "max over all-negative active region should be < 0, got {}",
            value
        );
        assert_eq!(
            value, -2,
            "max should be active-region max (-2), got {}",
            value
        );
    }

    #[test]
    fn dyn_min_on_bounded_2d_excludes_padding_positives() {
        // Active (2x2) values = 5, 3, 7, 2 ⇒ min = 2. Padding = 0. Old
        // buggy path would see `0 < 2` and return 0. First-active pattern
        // keeps `acc = buffer[0] = 5` initially and only updates when
        // `is_active AND candidate < acc`.
        let mut b = IRBuilder::new();
        let data = make_bounded_2d_3x3(
            &mut b,
            [5, 3, 0, 7, 2, 0, 0, 0, 0],
            2,
            2,
            NumberType::Integer,
        );
        let result = dyn_aggregate_all(&mut b, &data, DynAggKind::Min);
        let target = result.stmt_id().expect("min result has stmt_id");
        let value = eval_at_stmt(&b, target);
        assert!(
            value > 0,
            "min over all-positive active region should be > 0, got {}",
            value
        );
        assert_eq!(
            value, 2,
            "min should be active-region min (2), got {}",
            value
        );
    }
}

// ── Tests (Level-2 is_sorted short-circuit) ─────────────────────────────

#[cfg(test)]
mod sorted_short_circuit_tests {
    use super::*;
    use crate::ir_defs::IR;
    use crate::ops::dyn_ndarray::constructors::dyn_from_values_with_active_nd;
    use crate::optim::predicates::formula::{ContractTerm, ContractVar};

    /// Build a Layout-A 1-D unbounded `DynamicNDArray` with the supplied
    /// integer payload. `runtime_shape == logical_shape == [len]`, so the
    /// existing reduction path is the **fast path** (no mask-and-include).
    /// Caller can either plant `is_sorted(data.value_id)` or not.
    fn make_1d_unbounded(b: &mut IRBuilder, values: Vec<i64>) -> DynamicNDArrayData {
        let n = values.len();
        let sv: Vec<ScalarValue<i64>> = values
            .into_iter()
            .map(|v| ScalarValue::new(Some(v), None))
            .collect();
        let runtime_shape = vec![ScalarValue::new(Some(n as i64), None)];
        let runtime_length = b.ir_constant_int(n as i64);
        let val = dyn_from_values_with_active_nd(
            b,
            sv,
            vec![n],
            runtime_shape,
            runtime_length,
            NumberType::Integer,
        );
        match val {
            Value::DynamicNDArray(d) => d,
            _ => unreachable!(),
        }
    }

    /// Plant `is_sorted(arr_vid)` on the FactStack at `arr_vid`. Mirrors
    /// the `@requires(is_sorted(arr))` ingestion path in
    /// `ir_gen/mod.rs:425-432` for one anchor.
    fn plant_is_sorted(b: &mut IRBuilder, arr_vid: crate::types::ValueId) {
        let fact = ContractTerm::PredicateApp {
            kind: "is_sorted".to_string(),
            args: vec![ContractTerm::Var(ContractVar::Value(arr_vid))],
        };
        b.facts.insert_for(arr_vid, fact);
    }

    /// Count the number of `IR::ReadMemory` statements in `b.stmts`.
    /// Short-circuit emits exactly one read (the constant-index slot read);
    /// the loop path reads all N elements via `segment::read_all` (one
    /// `ir_read_memory` per slot).
    fn count_read_memory(b: &IRBuilder) -> usize {
        b.stmts
            .iter()
            .filter(|s| matches!(s.ir, IR::ReadMemory { .. }))
            .count()
    }

    /// Count `IR::SelectI` statements — the select-chain signature of the
    /// reduction loop. Short-circuit emits zero selects.
    fn count_select_i(b: &IRBuilder) -> usize {
        b.stmts
            .iter()
            .filter(|s| matches!(s.ir, IR::SelectI))
            .count()
    }

    #[test]
    fn dyn_max_on_sorted_short_circuits_to_last_slot_read() {
        // is_sorted ascending ⇒ max == arr[len - 1]. Plant the fact, run
        // dyn_aggregate_all(Max), and confirm:
        // (a) the result statement is an IR::ReadMemory (single-slot read),
        // (b) total ReadMemory count is exactly 1 (no segment::read_all loop).
        let mut b = IRBuilder::new();
        let data = make_1d_unbounded(&mut b, vec![1, 2, 3, 4, 5]);
        plant_is_sorted(&mut b, data.value_id);

        let result = dyn_aggregate_all(&mut b, &data, DynAggKind::Max);
        let target = result.stmt_id().expect("max result has stmt_id");

        let final_stmt = b
            .stmts
            .iter()
            .find(|s| s.stmt_id == target)
            .expect("target stmt present");
        assert!(
            matches!(final_stmt.ir, IR::ReadMemory { .. }),
            "short-circuit max should be a single ReadMemory, got {:?}",
            final_stmt.ir
        );
        assert_eq!(
            count_read_memory(&b),
            1,
            "short-circuit max should emit exactly one ReadMemory"
        );
        assert_eq!(
            count_select_i(&b),
            0,
            "short-circuit max must not emit any SelectI (no select-chain)"
        );
    }

    #[test]
    fn dyn_min_on_sorted_short_circuits_to_first_slot_read() {
        // is_sorted ascending ⇒ min == arr[0]. Plant the fact; confirm
        // the result is a ReadMemory whose address is constant 0.
        let mut b = IRBuilder::new();
        let data = make_1d_unbounded(&mut b, vec![1, 2, 3, 4, 5]);
        plant_is_sorted(&mut b, data.value_id);

        let result = dyn_aggregate_all(&mut b, &data, DynAggKind::Min);
        let target = result.stmt_id().expect("min result has stmt_id");

        let final_stmt = b
            .stmts
            .iter()
            .find(|s| s.stmt_id == target)
            .expect("target stmt present");
        assert!(
            matches!(final_stmt.ir, IR::ReadMemory { .. }),
            "short-circuit min should be a single ReadMemory, got {:?}",
            final_stmt.ir
        );
        // The address argument should be a ConstantInt(0).
        let addr_id = final_stmt.arguments[0];
        let addr_stmt = b
            .stmts
            .iter()
            .find(|s| s.stmt_id == addr_id)
            .expect("address stmt present");
        assert!(
            matches!(addr_stmt.ir, IR::ConstantInt { value: 0 }),
            "short-circuit min should read at address ConstantInt(0), got {:?}",
            addr_stmt.ir
        );
        assert_eq!(
            count_read_memory(&b),
            1,
            "short-circuit min should emit exactly one ReadMemory"
        );
        assert_eq!(
            count_select_i(&b),
            0,
            "short-circuit min must not emit any SelectI"
        );
    }

    #[test]
    fn dyn_argmax_on_sorted_short_circuits_to_len_minus_one() {
        // is_sorted ascending ⇒ argmax == runtime_length - 1. The result
        // should be an IR::SubI of (runtime_length_v, ConstantInt 1).
        let mut b = IRBuilder::new();
        let data = make_1d_unbounded(&mut b, vec![1, 2, 3, 4, 5]);
        plant_is_sorted(&mut b, data.value_id);

        let result = dyn_aggregate_all(&mut b, &data, DynAggKind::Argmax);
        let target = result.stmt_id().expect("argmax result has stmt_id");

        let final_stmt = b
            .stmts
            .iter()
            .find(|s| s.stmt_id == target)
            .expect("target stmt present");
        assert!(
            matches!(final_stmt.ir, IR::SubI),
            "short-circuit argmax should be a SubI (len - 1), got {:?}",
            final_stmt.ir
        );
        // Second operand should be ConstantInt(1).
        let one_id = final_stmt.arguments[1];
        let one_stmt = b
            .stmts
            .iter()
            .find(|s| s.stmt_id == one_id)
            .expect("operand stmt present");
        assert!(
            matches!(one_stmt.ir, IR::ConstantInt { value: 1 }),
            "argmax len-1 second operand should be ConstantInt(1), got {:?}",
            one_stmt.ir
        );
        // No select-chain.
        assert_eq!(
            count_select_i(&b),
            0,
            "short-circuit argmax must not emit any SelectI"
        );
        // No memory reads either: argmax/argmin return the index, not the
        // element.
        assert_eq!(
            count_read_memory(&b),
            0,
            "short-circuit argmax must not emit any ReadMemory"
        );
    }

    #[test]
    fn dyn_argmin_on_sorted_short_circuits_to_zero() {
        // is_sorted ascending ⇒ argmin == 0. The result should be an
        // IR::ConstantInt(0).
        let mut b = IRBuilder::new();
        let data = make_1d_unbounded(&mut b, vec![1, 2, 3, 4, 5]);
        plant_is_sorted(&mut b, data.value_id);

        let result = dyn_aggregate_all(&mut b, &data, DynAggKind::Argmin);
        let target = result.stmt_id().expect("argmin result has stmt_id");

        let final_stmt = b
            .stmts
            .iter()
            .find(|s| s.stmt_id == target)
            .expect("target stmt present");
        assert!(
            matches!(final_stmt.ir, IR::ConstantInt { value: 0 }),
            "short-circuit argmin should be a ConstantInt(0), got {:?}",
            final_stmt.ir
        );
        assert_eq!(
            count_select_i(&b),
            0,
            "short-circuit argmin must not emit any SelectI"
        );
        assert_eq!(
            count_read_memory(&b),
            0,
            "short-circuit argmin must not emit any ReadMemory"
        );
    }

    #[test]
    fn dyn_max_without_sorted_fact_uses_loop() {
        // Soundness invariant: `prove()` is Unknown without the fact, so
        // the short-circuit must NOT fire. The reduction loop produces
        // O(N) SelectI statements; the short-circuit produces zero. We
        // assert at least one SelectI is emitted, which can only happen
        // via the loop path.
        let mut b = IRBuilder::new();
        let data = make_1d_unbounded(&mut b, vec![1, 2, 3, 4, 5]);
        // Intentionally do NOT plant the is_sorted fact.

        let result = dyn_aggregate_all(&mut b, &data, DynAggKind::Max);
        let _ = result.stmt_id().expect("max result has stmt_id");

        assert!(
            count_select_i(&b) >= 1,
            "without is_sorted fact, max should fall through to the loop \
             and emit at least one SelectI; got 0 (short-circuit fired \
             without a fact — soundness regression)"
        );
    }
}
