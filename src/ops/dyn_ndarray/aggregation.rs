use crate::builder::IRBuilder;
use crate::types::{ValueId, 
    DynArrayMeta, DynamicNDArrayData, NumberType, ScalarValue, Value,
};

use super::{
    dyn_decode_coords, dyn_encode_coords, dyn_num_elements, dyn_row_major_strides,
    scalar_i64_to_value, value_to_scalar_i64, DynAggKind,
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
    let inputs = DynAggAllInputs { data: data.clone(), agg };
    let set = dyn_aggregate_all_strategies(&inputs);
    crate::optim::dispatch_strategy(b, "dyn_aggregate_all", &inputs, &set)
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

/// Strategy set for full reduction (axis=None). Today's only gated
/// strategy is `boundary-read-on-sorted`: when `is_sorted(input)` is
/// provable and the op is max/min/argmax/argmin, the reduction collapses
/// to a constant-index closed form. Otherwise the default O(N) sweep
/// fires (bounded-aware iteration with mask-and-include).
fn dyn_aggregate_all_strategies(
    inputs: &DynAggAllInputs,
) -> crate::optim::OpStrategySet<DynAggAllInputs, Value> {
    use crate::optim::predicates::formula::{ContractTerm, ContractVar};
    use crate::optim::{CostHint, OpStrategy, OpStrategySet};

    let mut strategies: Vec<OpStrategy<DynAggAllInputs, Value>> = Vec::new();

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
                args: vec![ContractTerm::Var(ContractVar::Value(inputs.data.value_id))],
            },
            cost_hint: CostHint::O1,
            lower: lower_boundary_read_on_sorted,
        });
    }

    OpStrategySet {
        strategies,
        default: lower_dyn_aggregate_all_generic,
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
            let raw = b.ir_read_memory(data.segment_id, &zero);
            coerce_read_to_dtype(raw, data.dtype)
        }
        DynAggKind::Max => {
            let one = b.ir_constant_int(1);
            let last_idx = b.ir_sub_i(&runtime_length_v, &one);
            let raw = b.ir_read_memory(data.segment_id, &last_idx);
            coerce_read_to_dtype(raw, data.dtype)
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
    let buffer_size = match mode {
        crate::helpers::array_ops::StrideMode::SymbolicRuntime(_) => {
            data.envelope.total_bound
        }
        crate::helpers::array_ops::StrideMode::LiteralLogical(_) => {
            dyn_num_elements(&data.meta.logical_shape)
        }
    };
    let values = crate::helpers::segment::read_all(b, data.segment_id, buffer_size);
    if buffer_size == 0 {
        return dyn_agg_identity(b, agg, data.dtype);
    }

    let use_float = data.dtype == NumberType::Float
        && !matches!(agg, DynAggKind::All | DynAggKind::Any);

    let any_bounded = data
        .meta
        .runtime_shape
        .iter()
        .enumerate()
        .any(|(i, s)| s.static_val != Some(data.meta.logical_shape[i] as i64));

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

    let mut acc = values[0].clone();
    let mut acc_idx = b.ir_constant_int(0);

    let strides = dyn_row_major_strides(&data.meta.logical_shape);

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
        let identity = agg_identity_value(b, agg, data.dtype);
        for i in 1..buffer_size.min(values.len()) {
            let coords = dyn_decode_coords(i, &data.meta.logical_shape, &strides);
            let is_active = build_active_check(
                b,
                &coords,
                &data.meta.runtime_shape,
                &data.meta.logical_shape,
            );
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
        acc
    }
}

/// Re-type the integer-typed `Value` returned by `ir_read_memory` to the
/// dyn-ndarray's dtype. Mirrors the pattern used by `dyn_getitem_element`
/// in `helpers/array_ops/indexing.rs`: `read_memory` always returns an
/// `Integer` Value (slots are stored as i64), so for Float dyn-ndarrays
/// we wrap the integer-typed stmt_id in a `Value::Float` carrying an
/// optional compile-time `f64` projection of the static value.
fn coerce_read_to_dtype(raw: Value, dtype: NumberType) -> Value {
    match dtype {
        NumberType::Integer => raw,
        NumberType::Float => Value::Float(ScalarValue::new(
            raw.int_val().map(|v| v as f64),
            raw.stmt_id(),
        )),
        NumberType::Complex => panic!(
            "DynamicNDArray of Complex is not yet supported (compiler.complex-ndarray-ops scope)"
        ),
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
        (DynAggKind::Prod, NumberType::Integer) => b.ir_constant_int(1),
        (DynAggKind::Prod, NumberType::Float) => b.ir_constant_float(1.0),
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
            } else {
                b.ir_select_i(is_active, elem, identity)
            };
            let new_acc = match agg {
                DynAggKind::Sum if use_float => b.ir_add_f(acc, &contribution),
                DynAggKind::Sum => b.ir_add_i(acc, &contribution),
                DynAggKind::Prod if use_float => b.ir_mul_f(acc, &contribution),
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

    let values = crate::helpers::segment::read_all(b, data.segment_id, data.max_length());
    let strides = dyn_row_major_strides(shape);
    let use_float = data.dtype == NumberType::Float
        && !matches!(agg, DynAggKind::All | DynAggKind::Any);

    // Output shape: remove the reduced axis
    let out_shape: Vec<usize> = shape
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
    let axis_dim = shape[ax];

    let mut out_elements = Vec::with_capacity(out_numel);

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

        // Initialize accumulator with first element along axis
        let first_src_idx = dyn_encode_coords(&in_coords, &strides);
        let first_elem = if first_src_idx < values.len() {
            values[first_src_idx].clone()
        } else {
            super::metadata::dyn_default_value(b, data.dtype)
        };

        let mut acc = first_elem;
        let mut acc_idx = b.ir_constant_int(0);

        // Iterate along reduction axis
        for k in 1..axis_dim {
            in_coords[ax] = k;
            let src_idx = dyn_encode_coords(&in_coords, &strides);
            let elem = if src_idx < values.len() {
                values[src_idx].clone()
            } else {
                super::metadata::dyn_default_value(b, data.dtype)
            };
            let k_val = b.ir_constant_int(k as i64);
            let (new_acc, new_idx) =
                dyn_agg_step(b, &acc, &acc_idx, &elem, &k_val, agg, use_float);
            acc = new_acc;
            acc_idx = new_idx;
        }

        let result = match agg {
            DynAggKind::Argmax | DynAggKind::Argmin => acc_idx,
            _ => acc,
        };
        out_elements.push(value_to_scalar_i64(&result));
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
        return scalar_i64_to_value(&out_elements[0], out_dtype);
    }

    let out_strides_meta = dyn_row_major_strides(&out_shape);
    let _ = out_numel;
    let segment_id = crate::helpers::segment::alloc_and_write(b, &out_elements, out_dtype);
    let envelope = crate::types::Envelope::from_static_shape(&mut b.dim_table, &out_shape);
    let result = DynamicNDArrayData {
        envelope,
        dtype: out_dtype,
        segment_id,
        meta: DynArrayMeta {
            logical_shape: out_shape.clone(),
            logical_offset: 0,
            logical_strides: out_strides_meta.clone(),
            runtime_length: ScalarValue::new(Some(out_numel as i64), None),
            runtime_rank: ScalarValue::new(Some(out_shape.len() as i64), None),
            runtime_shape: out_shape
                .iter()
                .map(|&s| ScalarValue::new(Some(s as i64), None))
                .collect(),
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
            let new_acc = if use_float {
                b.ir_add_f(acc, elem)
            } else {
                b.ir_add_i(acc, elem)
            };
            (new_acc, acc_idx.clone())
        }
        DynAggKind::Prod => {
            let new_acc = if use_float {
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
            NumberType::Complex => panic!(
                "DynamicNDArray of Complex is not yet supported (compiler.complex-ndarray-ops scope)"
            ),
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
                IR::AllocateMemory { segment_id, size, init_value } => {
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
            panic!("eval_at_stmt: result stmt_id {:?} not produced by interpreter", target)
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
        let val = dyn_from_values_with_active_nd(
            b,
            sv,
            vec![3, 3],
            runtime_shape,
            runtime_length,
            dtype,
        );
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
