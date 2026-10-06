use std::collections::HashMap;

use crate::builder::IRBuilder;
use crate::types::{DynArrayMeta, DynamicNDArrayData, NumberType, ScalarValue, SliceIndex, Value, ValueId};

use super::{
    dyn_decode_coords, dyn_encode_coords, dyn_num_elements, dyn_row_major_strides,
    value_to_scalar_i64,
};

fn cast_to_dtype(b: &mut IRBuilder, value: &Value, dtype: NumberType) -> Value {
    match dtype {
        NumberType::Integer => b.ir_int_cast(value),
        NumberType::Float => b.ir_float_cast(value),
        NumberType::Complex => {
            let (real, imag) = crate::helpers::value_ops::unpack_value_to_complex_parts(b, value);
            crate::helpers::value_ops::pack_complex_value(real, imag)
        }
    }
}

/// Output of the shared boolean-mask compaction network ([`compact_by_mask`]).
pub(crate) struct Compaction {
    /// The compacted, default-padded buffer (`leading_total * trailing_total`
    /// elements): kept blocks packed to the front in source order, defaults
    /// behind.
    pub acc: Vec<Value>,
    /// Number of KEPT leading blocks (rows). This is the runtime row count,
    /// NOT multiplied by `trailing_total`.
    pub kept_rows: Value,
}

/// The single boolean-mask compaction network. `cast_cells` is the source
/// payload already cast to `default_val`'s dtype (`leading_total *
/// trailing_total` cells, row-major); `mask_cells` carries one mask cell per
/// leading block. Returns the compacted buffer plus the kept-row count.
///
/// Compaction via a one-hot shift network — no memory ops. `row[j]` one-hot
/// encodes the landing *row* of input block i (the prefix count of kept rows
/// before i): it starts as e_0 and shifts right by one whenever a row is
/// kept. Each input row select-writes its (masked) cells into the landing
/// row: kept rows deposit their values, non-kept rows deposit defaults at the
/// still-unowned next free row — exactly the previous write-pointer
/// semantics, so the output layout is bit-identical (compacted rows, default
/// padding). The row region is triangular (landing row <= i), so entries
/// beyond i+1 stay constant 0 and emit nothing.
///
/// With a zero default (every real dtype), slot `j` is accumulated as the
/// sum `sum_i row_i[j] * deposit_i` instead of a chain of selects: the rows
/// whose landing row is `j` are the run of non-kept rows after the j-th kept
/// row, which deposit 0, and the (j+1)-th kept row itself, so the sum is
/// that row's value, or 0 when fewer rows are kept, as the select chain
/// gives. The sum costs one product and one addition per term and needs no
/// bit check of the one-hot entry, which a select's condition does.
///
/// This is the SINGLE copy of the triangular shift loop: `dynamic_mask_read`
/// (StaticArray boolean-mask read, trailing tail support) and `dyn_filter`
/// (DynamicNDArray `arr[mask]` / `.filter`, trailing_total == 1) both call it.
pub(crate) fn compact_by_mask(
    b: &mut IRBuilder,
    cast_cells: &[Value],
    mask_cells: &[Value],
    default_val: &Value,
    trailing_total: usize,
) -> Compaction {
    let trailing_total = trailing_total.max(1);
    let leading_total = mask_cells.len();
    let total = leading_total * trailing_total;
    debug_assert_eq!(cast_cells.len(), total);

    let one = b.ir_constant_int(1);
    let zero = b.ir_constant_int(0);
    let mut row: Vec<Value> = (0..leading_total)
        .map(|j| if j == 0 { one.clone() } else { zero.clone() })
        .collect();
    let mut acc: Vec<Value> = vec![default_val.clone(); total];
    let mut kept_rows = b.ir_constant_int(0);

    let is_const_zero = |v: &Value| v.int_val() == Some(0);
    let zero_default = is_zero_scalar(default_val);
    // With a zero default, the `(one-hot entry, deposit)` terms of each slot,
    // summed after the network so that each slot's sum is one contiguous
    // multiply-add chain (backends chain consecutive multiply-adds).
    let mut terms: Vec<Vec<(Value, Value)>> = vec![Vec::new(); total];

    for i in 0..leading_total {
        let mask_bool = ensure_bool(b, &mask_cells[i]);
        let mut deposit: Vec<Value> = Vec::with_capacity(trailing_total);
        for t in 0..trailing_total {
            let src = &cast_cells[i * trailing_total + t];
            deposit.push(crate::helpers::value_ops::select_value(
                b,
                &mask_bool,
                src,
                default_val,
            ));
        }

        for j in 0..(i + 1).min(leading_total) {
            if is_const_zero(&row[j]) {
                continue;
            }
            for (t, dep) in deposit.iter().enumerate() {
                let slot = &acc[j * trailing_total + t];
                if zero_default {
                    terms[j * trailing_total + t].push((row[j].clone(), dep.clone()));
                } else {
                    acc[j * trailing_total + t] = crate::helpers::value_ops::select_value(b, &row[j], dep, slot);
                }
            }
        }

        // Shift the one-hot row right when this input row was kept.
        let mut new_row = row.clone();
        for j in (0..(i + 2).min(leading_total)).rev() {
            let prev = if j == 0 { &zero } else { &row[j - 1] };
            if is_const_zero(prev) && is_const_zero(&row[j]) {
                continue;
            }
            new_row[j] = b.ir_select_i(&mask_bool, prev, &row[j]);
        }
        row = new_row;

        let inc = b.ir_select_i(&mask_bool, &one, &zero);
        kept_rows = b.ir_add_i(&kept_rows, &inc);
    }
    for (slot, slot_terms) in terms.into_iter().enumerate() {
        for (hit, dep) in slot_terms {
            acc[slot] = add_weighted(b, &acc[slot], &hit, &dep);
        }
    }

    Compaction { acc, kept_rows }
}

/// Whether `v` is a real (not complex) compile-time zero.
fn is_zero_scalar(v: &Value) -> bool {
    v.int_val() == Some(0) || v.float_val() == Some(0.0)
}

/// `acc + hit * value` for a 0/1 `hit`, as a field product and sum (no
/// rescale: a Float `value` is scaled by 0 or 1), typed as `value`.
fn add_weighted(b: &mut IRBuilder, acc: &Value, hit: &Value, value: &Value) -> Value {
    let product = b.create_ir(&crate::ir_defs::IR::MulI, &[hit.clone(), value.clone()]);
    let term = match (product, value) {
        (Value::Integer(sv), Value::Float(_)) => Value::Float(ScalarValue {
            static_val: None,
            stmt_id: sv.stmt_id,
            value_id: sv.value_id,
        }),
        (product, _) => product,
    };
    match (acc, value) {
        _ if is_zero_scalar(acc) => term,
        (_, Value::Float(_)) => b.ir_add_f(acc, &term),
        _ => b.ir_add_i(acc, &term),
    }
}

fn ensure_bool(b: &mut IRBuilder, v: &Value) -> Value {
    match v {
        Value::Boolean(_) => v.clone(),
        _ => b.ir_bool_cast(v),
    }
}

pub fn dyn_filter(b: &mut IRBuilder, data: &DynamicNDArrayData, args: &[Value]) -> Value {
    let mask = args.first().expect("filter: requires a mask argument");
    let data_owned = super::force_contiguous(b, data);
    let data = &data_owned;
    let max_len = data.max_length();
    let target_shape = &data.meta.logical_shape;
    let elements: Vec<Value> =
        crate::helpers::segment::read_all_typed(b, data.segment_id, max_len, data.dtype);

    // Get mask elements — from segment if dynamic, from composite if static.
    let mask_elements: Vec<Value> = match mask {
        Value::DynamicNDArray(md) => {
            if md.meta.logical_shape != *target_shape {
                panic!(
                    "filter: array shape {:?} and mask shape {:?} must match",
                    target_shape, md.meta.logical_shape
                );
            }
            if md.max_length() != max_len {
                panic!(
                    "filter: array has {} elements but mask has {}",
                    max_len,
                    md.max_length()
                );
            }
            let md_owned = super::force_contiguous(b, md);
            crate::helpers::segment::read_all_typed(
                b,
                md_owned.segment_id,
                md_owned.max_length(),
                md_owned.dtype,
            )
        }
        Value::List(_) | Value::Tuple(_) | Value::StaticArray { .. } => {
            let shape = crate::helpers::composite::get_composite_shape(mask);
            if shape != *target_shape {
                panic!(
                    "filter: array shape {:?} and mask shape {:?} must match",
                    target_shape, shape
                );
            }
            let mask_total: usize = shape.iter().product();
            if mask_total != max_len {
                panic!(
                    "filter: array has {} elements but mask has {}",
                    max_len, mask_total
                );
            }
            crate::helpers::composite::flatten_composite_with_builder(b, mask)
        }
        _ => panic!("filter: mask must be array-like"),
    };
    if mask_elements.len() != max_len {
        panic!(
            "filter: array has {} elements but mask has {}",
            max_len,
            mask_elements.len()
        );
    }
    // Only slots holding elements of the runtime array may be kept: the
    // buffer's padding past the runtime shape is not data, but a mask
    // computed elementwise (`y[y < 5]` with `y` itself filtered) is true
    // on zero padding. (The filter of a filter used to keep it.)
    let mask_elements: Vec<Value> = match super::aggregation::active_slots(b, data) {
        None => mask_elements,
        Some(active) => mask_elements
            .iter()
            .zip(&active)
            .map(|(m, a)| {
                let keep = ensure_bool(b, m);
                b.ir_logical_and(&keep, a)
            })
            .collect(),
    };

    // Compact `elements` to the front under `mask_elements` via the shared
    // one-hot shift network (`compact_by_mask`). `elements` already carry
    // `data.dtype`, so they are their own cast sources (1-to-1 with source
    // positions) — exactly what the MaskReadView deposit below needs.
    let default_val = super::metadata::dyn_default_value(b, data.dtype);
    let Compaction {
        acc,
        kept_rows: write_ptr,
    } = compact_by_mask(b, &elements, &mask_elements, &default_val, 1);

    let segment_id = crate::helpers::segment::alloc_and_write_values(b, &acc, data.dtype);
    // The kept elements keep their dtype (the compaction's selects do not).
    if b.is_bool_segment(data.segment_id) {
        b.mark_bool_segment(segment_id);
    }

    // Deposit the mask-read view so an `a[mask] op= v` round trip can cancel
    // the shift network into per-element selects at the write (see
    // `MaskReadView` and `try_apply_boolean_mask_write` /
    // `dyn_setitem_mask`). 1-D whole-element masks only; deposit AFTER the
    // alloc writes — `on_segment_write` eager-drops the record on any later
    // write to this segment. The `count_stmt` pins the runtime-length
    // identity so a sliced/reshaped re-view is not consumed positionally.
    if data.dtype != NumberType::Complex && elements.len() == max_len {
        b.provenance.set_mask_read_view(
            segment_id,
            crate::provenance::MaskReadView {
                mask_cells: mask_elements.clone(),
                sources: elements.clone(),
                count_stmt: write_ptr.stmt_id(),
            },
        );
    }

    // After filter, the runtime length is unknown (depends on the mask) but
    // bounded by the input's max. total_bound conserved from source (§3.8).
    let envelope = crate::types::Envelope::new_with_bound(
        vec![crate::types::Dim::new_dynamic(&mut b.dim_table, 0, max_len)],
        data.envelope.total_bound,
    );
    let runtime_length_sv = value_to_scalar_i64(&write_ptr);
    let result = DynamicNDArrayData {
        envelope,
        dtype: data.dtype,
        segment_id,
        meta: DynArrayMeta {
            logical_shape: vec![max_len],
            logical_offset: 0,
            logical_strides: vec![1],
            runtime_length: runtime_length_sv.clone(),
            runtime_rank: ScalarValue::new(Some(1), None),
            runtime_shape: vec![runtime_length_sv.clone()],
            runtime_strides: vec![ScalarValue::new(Some(1), None)],
            runtime_offset: ScalarValue::new(Some(0), None),
        },
        value_id: ValueId::next(),
    };

    // Op contract for `dyn_filter` (compiler.op-contract-content):
    // (a) registry-driven postcondition `Var(Output) >= 0`, and
    // (b) call-site-local upper-bound fact `Var(Output) <= max_len`.
    // The static upper bound depends on the call-site value of `max_len`,
    // so it can't live in the pure registry template — we splice it in
    // directly.
    {
        let len_vid = runtime_length_sv.value_id;
        b.fire_contract("dyn_filter", len_vid, &HashMap::new());
        use crate::optim::predicates::formula::{CmpOp, ContractTerm, ContractVar};
        let upper = ContractTerm::Cmp {
            op: CmpOp::Le,
            lhs: Box::new(ContractTerm::Var(ContractVar::Value(len_vid))),
            rhs: Box::new(ContractTerm::LitInt(max_len as i64)),
        };
        b.facts.insert_for(len_vid, upper);
    }

    Value::DynamicNDArray(result)
}

/// DynamicNDArray.repeat(repeats, axis=...)
pub fn dyn_repeat(
    b: &mut IRBuilder,
    data: &DynamicNDArrayData,
    args: &[Value],
    kwargs: &HashMap<String, Value>,
) -> Value {
    let repeats = args
        .first()
        .and_then(|v| v.int_val())
        .expect("repeat: repeats must be a constant integer") as usize;
    let axis = kwargs
        .get("axis")
        .or_else(|| args.get(1))
        .and_then(|v| v.int_val());

    let data_owned = super::force_contiguous(b, data);
    let data = &data_owned;
    let values =
        crate::helpers::segment::read_all_typed(b, data.segment_id, data.max_length(), data.dtype);
    let numel = dyn_num_elements(&data.meta.logical_shape);

    let result = if let Some(_ax) = axis {
        // For axis-specific repeat, flatten first, repeat, return 1D
        // (full axis support would need coordinate transforms)
        let mut new_elements = Vec::new();
        for val in values.iter().take(numel) {
            for _ in 0..repeats {
                new_elements.push(val.clone());
            }
        }
        let new_len = new_elements.len();
        let segment_id =
            crate::helpers::segment::alloc_and_write_values(b, &new_elements, data.dtype);
        let envelope = crate::types::Envelope::from_static_shape(&mut b.dim_table, &[new_len]);
        DynamicNDArrayData {
            envelope,
            dtype: data.dtype,
            segment_id,
            meta: DynArrayMeta {
                logical_shape: vec![new_len],
                logical_offset: 0,
                logical_strides: vec![1],
                runtime_length: ScalarValue::new(Some(new_len as i64), None),
                runtime_rank: ScalarValue::new(Some(1), None),
                runtime_shape: vec![ScalarValue::new(Some(new_len as i64), None)],
                runtime_strides: vec![ScalarValue::new(Some(1), None)],
                runtime_offset: ScalarValue::new(Some(0), None),
            },
            value_id: ValueId::next(),
        }
    } else {
        // No axis: flatten then repeat each element
        let mut new_elements = Vec::new();
        for val in values.iter().take(numel) {
            for _ in 0..repeats {
                new_elements.push(val.clone());
            }
        }
        let new_len = new_elements.len();
        let segment_id =
            crate::helpers::segment::alloc_and_write_values(b, &new_elements, data.dtype);
        let envelope = crate::types::Envelope::from_static_shape(&mut b.dim_table, &[new_len]);
        DynamicNDArrayData {
            envelope,
            dtype: data.dtype,
            segment_id,
            meta: DynArrayMeta {
                logical_shape: vec![new_len],
                logical_offset: 0,
                logical_strides: vec![1],
                runtime_length: ScalarValue::new(Some(new_len as i64), None),
                runtime_rank: ScalarValue::new(Some(1), None),
                runtime_shape: vec![ScalarValue::new(Some(new_len as i64), None)],
                runtime_strides: vec![ScalarValue::new(Some(1), None)],
                runtime_offset: ScalarValue::new(Some(0), None),
            },
            value_id: ValueId::next(),
        }
    };
    let out_vid = result.value_id;
    let out = Value::DynamicNDArray(result);
    crate::optim::resolver::relay_forall_eq_const_from_input(b, data.value_id, out_vid);
    out
}

/// Coerce a Value to a 1-D DynamicNDArrayData for concat/stack scalar support.
/// Scalars become length-1 arrays; existing DynamicNDArrays pass through.
fn coerce_to_dyn(b: &mut IRBuilder, v: &Value) -> DynamicNDArrayData {
    match v {
        Value::DynamicNDArray(d) => d.clone(),
        Value::Integer(_) | Value::Float(_) | Value::Boolean(_) | Value::Complex { .. } => {
            let dtype = if matches!(v, Value::Complex { .. }) {
                NumberType::Complex
            } else if matches!(v, Value::Float(_)) {
                NumberType::Float
            } else {
                NumberType::Integer
            };
            let segment_id =
                crate::helpers::segment::alloc_and_write_values(b, &[v.clone()], dtype);
            let envelope = crate::types::Envelope::from_static_shape(&mut b.dim_table, &[1]);
            DynamicNDArrayData {
                envelope,
                dtype,
                segment_id,
                meta: DynArrayMeta {
                    logical_shape: vec![1],
                    logical_offset: 0,
                    logical_strides: vec![1],
                    runtime_length: ScalarValue::new(Some(1), None),
                    runtime_rank: ScalarValue::new(Some(1), None),
                    runtime_shape: vec![ScalarValue::new(Some(1), None)],
                    runtime_strides: vec![ScalarValue::new(Some(1), None)],
                    runtime_offset: ScalarValue::new(Some(0), None),
                },
                value_id: ValueId::next(),
            }
        }
        _ => panic!(
            "concatenate/stack: unsupported element type {:?}",
            v.zinnia_type()
        ),
    }
}

/// Resolve output dtype: float wins over integer (NumPy promotion rule).
fn resolve_dtype(arrays: &[DynamicNDArrayData]) -> NumberType {
    if arrays.iter().any(|a| a.dtype == NumberType::Complex) {
        NumberType::Complex
    } else if arrays.iter().any(|a| a.dtype == NumberType::Float) {
        NumberType::Float
    } else {
        NumberType::Integer
    }
}

/// Concatenate dynamic arrays along an existing axis.
///
/// Supports: dtype promotion, axis=None (flatten then concat), scalar inputs.
/// Output envelope: non-concat dims from first array, concat axis = sum.
/// total_bound = sum of input total_bounds.
pub fn dyn_concatenate(
    b: &mut IRBuilder,
    args: &[Value],
    kwargs: &HashMap<String, Value>,
) -> Value {
    let arrays_val = args.first().expect("concatenate: requires arrays argument");
    let raw_values: Vec<Value> = match arrays_val {
        Value::List(cd) | Value::Tuple(cd) => cd.values.clone(),
        _ => panic!("concatenate: first arg must be list/tuple of arrays"),
    };
    assert!(
        !raw_values.is_empty(),
        "concatenate: need at least one array"
    );

    // Check axis=None (flatten all then concatenate as 1D).
    let axis_val = kwargs.get("axis").or_else(|| args.get(1));
    let axis_is_none = matches!(axis_val, Some(Value::None));

    // Coerce scalars to 1-d arrays.
    let mut arrays: Vec<DynamicNDArrayData> =
        raw_values.iter().map(|v| coerce_to_dyn(b, v)).collect();

    // If axis=None, flatten all arrays to 1-d first.
    if axis_is_none {
        arrays = arrays
            .iter()
            .map(|a| {
                let total: usize = a.meta.logical_shape.iter().product();
                if a.meta.logical_shape.len() == 1 && a.meta.logical_shape[0] == total {
                    a.clone()
                } else {
                    // Flatten: reinterpret as 1-d (always contiguous).
                    let strides = vec![1];
                    let envelope = crate::types::Envelope::new_with_bound(
                        vec![crate::types::Dim::new_static(&mut b.dim_table, total)],
                        a.envelope.total_bound,
                    );
                    DynamicNDArrayData {
                        envelope,
                        dtype: a.dtype,
                        segment_id: a.segment_id,
                        meta: DynArrayMeta {
                            logical_shape: vec![total],
                            logical_offset: 0,
                            logical_strides: strides,
                            runtime_length: a.meta.runtime_length.clone(),
                            runtime_rank: ScalarValue::new(Some(1), None),
                            runtime_shape: vec![a.meta.runtime_length.clone()],
                            runtime_strides: vec![ScalarValue::new(Some(1), None)],
                            runtime_offset: ScalarValue::new(Some(0), None),
                        },
                        value_id: ValueId::next(),
                    }
                }
            })
            .collect();
    }

    let axis = if axis_is_none {
        0i64
    } else {
        axis_val.and_then(|v| v.int_val()).unwrap_or(0)
    };
    let ndim = arrays[0].meta.logical_shape.len();
    let ax = if axis < 0 {
        (ndim as i64 + axis) as usize
    } else {
        axis as usize
    };
    assert!(
        ax < ndim,
        "concatenate: axis {} out of bounds for rank {}",
        ax,
        ndim
    );

    // Validate non-concat axes match.
    let base_shape = &arrays[0].meta.logical_shape;
    for (i, arr) in arrays.iter().enumerate().skip(1) {
        assert_eq!(
            arr.meta.logical_shape.len(),
            ndim,
            "concatenate: array {} has rank {} but expected {}",
            i,
            arr.meta.logical_shape.len(),
            ndim
        );
        for d in 0..ndim {
            if d != ax {
                assert_eq!(
                    arr.meta.logical_shape[d], base_shape[d],
                    "concatenate: array {} has size {} on axis {} but expected {}",
                    i, arr.meta.logical_shape[d], d, base_shape[d]
                );
            }
        }
    }

    // Inputs are made contiguous; element (c_0, ..., c_{n-1}) of an input is
    // then at its row-major position in the buffer shape (a padded layout),
    // or at `sum_d c_d * runtime_stride_d` (a compact multi-dimensional
    // buffer), see the read below.
    let arrays: Vec<DynamicNDArrayData> =
        arrays.iter().map(|a| super::force_contiguous(b, a)).collect();
    // NumPy requires equal extents on every axis but the concatenation
    // axis; the output takes the first operand's runtime extents there.
    let other_extents: Vec<(usize, ScalarValue<i64>)> = (0..ndim)
        .filter(|&d| d != ax)
        .map(|d| (d, runtime_extent(&arrays[0], d)))
        .collect();
    for arr in arrays.iter().skip(1) {
        for (d, first) in &other_extents {
            require_equal_extent(b, first, &runtime_extent(arr, *d), *d);
        }
    }

    // Max output shape: sum of max axis dims.
    let max_concat_dim: usize = arrays.iter().map(|a| a.meta.logical_shape[ax]).sum();
    let mut out_shape = base_shape.clone();
    out_shape[ax] = max_concat_dim;
    let out_numel: usize = out_shape.iter().product();
    let out_strides = dyn_row_major_strides(&out_shape);
    let out_dtype = resolve_dtype(&arrays);
    let out_ax_stride = out_strides[ax] as i64;

    // Check if any input has a dynamic concat axis.
    let has_dynamic_axis = arrays
        .iter()
        .any(|a| a.envelope.dims[ax].min != a.envelope.dims[ax].max);

    // Build envelope.
    let total_bound: usize = arrays.iter().map(|a| a.envelope.total_bound).sum();
    let mut out_dims: Vec<crate::types::Dim> = arrays[0].envelope.dims.clone();
    if has_dynamic_axis {
        out_dims[ax] = crate::types::Dim::new_dynamic(&mut b.dim_table, 0, max_concat_dim);
    } else {
        out_dims[ax] = crate::types::Dim::new_static(&mut b.dim_table, max_concat_dim);
    }
    let envelope = crate::types::Envelope::new_with_bound(out_dims, total_bound);

    // Non-concat axes: product for per-axis-position element count.
    let other_product: usize = out_shape
        .iter()
        .enumerate()
        .filter(|&(d, _)| d != ax)
        .map(|(_, &s)| s)
        .product::<usize>()
        .max(1);
    let mut other_dims: Vec<usize> = Vec::new();
    for (d, &s) in base_shape.iter().enumerate() {
        if d != ax {
            other_dims.push(s);
        }
    }
    let other_dim_strides = dyn_row_major_strides(&other_dims);

    // Allocate output segment with defaults.
    let default_val = super::metadata::dyn_default_value(b, out_dtype);
    let init_elements = vec![default_val.clone(); out_numel];
    let segment_id = crate::helpers::segment::alloc_and_write_values(b, &init_elements, out_dtype);
    let max_out_addr = (out_numel as i64 - 1).max(0);

    // Single pass: iterate per input, write to output at cumulative_offset.
    // For each input k, position i on concat axis (0..max_axis_k):
    //   source_addr = i * src_stride[ax] + other_offset
    //   output_addr = (cumulative_offset + i) * out_stride[ax] + other_offset
    //   in_bounds = i < runtime_shape_k[ax]
    //   write selected value or default
    // After input k: cumulative_offset += runtime_shape_k[ax]
    //
    // Out-of-bounds writes from input k land at positions that will be
    // overwritten by input k+1 (since cumulative_offset advances by
    // runtime length, not max length).

    let mut cumulative_offset = b.ir_constant_int(0);
    let mut runtime_concat_len = b.ir_constant_int(0);
    // For the multi-formal contract upgrade (compiler.op-contracts-non-walkable-batch):
    // track each input's runtime axis length as a ContractTerm so we can
    // emit `len(out) == sum(arg_lens) * other_product` after the loop.
    let mut arg_len_terms: Vec<crate::optim::predicates::formula::ContractTerm> =
        Vec::with_capacity(arrays.len());

    for (_k, arr) in arrays.iter().enumerate() {
        let src_shape = &arr.meta.logical_shape;
        let src_strides_k = dyn_row_major_strides(src_shape);
        let src_ax_stride = src_strides_k[ax] as i64;
        let src_max_ax = src_shape[ax];
        let src_max_len = arr.max_length() as i64;

        // Runtime axis length for this input (may be dynamic).
        let runtime_ax_len: Value = if let Some(sv) = arr.meta.runtime_shape.get(ax) {
            if let Some(s) = sv.static_val {
                b.ir_constant_int(s)
            } else if let Some(ptr) = sv.stmt_id {
                Value::Integer(ScalarValue::new(None, Some(ptr)))
            } else {
                b.ir_constant_int(src_max_ax as i64)
            }
        } else {
            b.ir_constant_int(src_max_ax as i64)
        };
        // Capture this input's runtime-length term for the contract fact.
        {
            use crate::optim::predicates::formula::{ContractTerm, ContractVar};
            let term = if let Some(s) = runtime_ax_len.int_val() {
                ContractTerm::LitInt(s)
            } else if let Some(vid) = runtime_ax_len.value_id() {
                ContractTerm::Var(ContractVar::Value(vid))
            } else {
                ContractTerm::LitInt(src_max_ax as i64)
            };
            arg_len_terms.push(term);
        }

        let padded = super::has_padded_layout(arr);
        let runtime_strides: Vec<Value> = (0..ndim)
            .map(|d| crate::ops::dyn_ndarray::scalar_i64_to_value(
                &arr.meta.runtime_strides[d],
                NumberType::Integer,
            ))
            .collect();
        for i in 0..src_max_ax {
            let i_val = b.ir_constant_int(i as i64);
            let along_ax = b.ir_less_than_i(&i_val, &runtime_ax_len);

            // Source address: i * src_stride[ax] (base for this axis position).
            let src_ax_base = i as i64 * src_ax_stride;

            // Output axis position = cumulative_offset + i (dynamic).
            let out_ax_pos = b.ir_add_i(&cumulative_offset, &i_val);
            let out_ax_stride_val = b.ir_constant_int(out_ax_stride);
            let out_ax_base = b.ir_mul_i(&out_ax_pos, &out_ax_stride_val);

            for other_flat in 0..other_product {
                let other_coords = dyn_decode_coords(other_flat, &other_dims, &other_dim_strides);

                // Compute non-concat address contributions (static for both src and out).
                let mut src_other_addr: i64 = 0;
                let mut out_other_addr: i64 = 0;
                let mut oc_idx = 0;
                for d in 0..ndim {
                    if d != ax {
                        src_other_addr += other_coords[oc_idx] as i64 * src_strides_k[d] as i64;
                        out_other_addr += other_coords[oc_idx] as i64 * out_strides[d] as i64;
                        oc_idx += 1;
                    }
                }

                // Element (i, other coordinates) holds data iff every
                // coordinate is below the input's runtime extent; the
                // others take the default (no input cell, private or
                // padding, reaches the output's padding).
                let mut in_bounds = along_ax.clone();
                for (oc, (d, _)) in other_coords.iter().zip(&other_extents) {
                    let extent = runtime_extent(arr, *d);
                    if extent.static_val.is_some_and(|n| (*oc as i64) < n) {
                        continue;
                    }
                    let coord = b.ir_constant_int(*oc as i64);
                    let extent_v =
                        crate::ops::dyn_ndarray::scalar_i64_to_value(&extent, NumberType::Integer);
                    let inside = b.ir_less_than_i(&coord, &extent_v);
                    in_bounds = b.ir_logical_and(&in_bounds, &inside);
                }

                // Source address: the row-major position in a padded
                // buffer, or the runtime-stride position in a compact one
                // (clamped into the buffer; out-of-range positions are
                // masked above).
                let src_addr_val = if padded {
                    let src_addr_static = src_ax_base + src_other_addr;
                    b.ir_constant_int(src_addr_static.max(0).min(src_max_len - 1))
                } else {
                    let mut addr = b.ir_mul_i(&i_val, &runtime_strides[ax]);
                    let mut oc_idx = 0;
                    for d in 0..ndim {
                        if d != ax {
                            let c = b.ir_constant_int(other_coords[oc_idx] as i64);
                            let part = b.ir_mul_i(&c, &runtime_strides[d]);
                            addr = b.ir_add_i(&addr, &part);
                            oc_idx += 1;
                        }
                    }
                    let zero_v = b.ir_constant_int(0);
                    let max_v = b.ir_constant_int(src_max_len - 1);
                    let below = b.ir_less_than_i(&addr, &zero_v);
                    let low = b.ir_select_i(&below, &zero_v, &addr);
                    let over = b.ir_greater_than_i(&low, &max_v);
                    b.ir_select_i(&over, &max_v, &low)
                };
                let src_val = crate::helpers::segment::read_typed(
                    b,
                    arr.segment_id,
                    &src_addr_val,
                    arr.dtype,
                );

                // Cast dtype if needed.
                let casted = if arr.dtype == out_dtype {
                    src_val
                } else {
                    cast_to_dtype(b, &src_val, out_dtype)
                };

                // Output address = out_ax_base + out_other_addr (dynamic + static).
                let out_other_val = b.ir_constant_int(out_other_addr);
                let out_addr = b.ir_add_i(&out_ax_base, &out_other_val);

                // Clamp output address for safety.
                let max_addr_val = b.ir_constant_int(max_out_addr);
                let zero_val = b.ir_constant_int(0);
                let is_neg = b.ir_less_than_i(&out_addr, &zero_val);
                let clamped_lo = b.ir_select_i(&is_neg, &zero_val, &out_addr);
                let is_over = b.ir_greater_than_i(&clamped_lo, &max_addr_val);
                let clamped_out = b.ir_select_i(&is_over, &max_addr_val, &clamped_lo);

                // Select value: in_bounds → casted source, else → default.
                let write_val =
                    crate::helpers::value_ops::select_value(b, &in_bounds, &casted, &default_val);

                crate::helpers::segment::write_typed(
                    b,
                    segment_id,
                    &clamped_out,
                    &write_val,
                    out_dtype,
                );
            }
        }

        // Advance cumulative offset by this input's runtime axis length.
        cumulative_offset = b.ir_add_i(&cumulative_offset, &runtime_ax_len);
        runtime_concat_len = b.ir_add_i(&runtime_concat_len, &runtime_ax_len);
    }

    // Runtime metadata: the concatenated extent on `ax`, the first
    // operand's (checked equal) runtime extents elsewhere.
    let mut runtime_shape: Vec<ScalarValue<i64>> = vec![ScalarValue::new(Some(0), None); ndim];
    runtime_shape[ax] = value_to_scalar_i64(&runtime_concat_len);
    let mut runtime_length = runtime_concat_len.clone();
    for (d, extent) in &other_extents {
        runtime_shape[*d] = extent.clone();
        let extent_v = crate::ops::dyn_ndarray::scalar_i64_to_value(extent, NumberType::Integer);
        runtime_length = b.ir_mul_i(&runtime_length, &extent_v);
    }

    let runtime_length_sv = value_to_scalar_i64(&runtime_length);

    // Op contract for `dyn_concatenate` (compiler.op-contract-content
    // base + compiler.op-contracts-non-walkable-batch upgrade):
    //
    // (a) registry-driven postcondition `Var(Output) >= 0`;
    // (b) call-site-built multi-formal fact
    //     `len(out) == (arg0_len + arg1_len + ... + argN-1_len) * other_product`.
    //
    // The base fact stays in the registry (template-shape, fixed). The
    // length-sum fact's arity depends on the number of inputs at the
    // call site, so the term is built here over the per-input runtime
    // axis-length ptrs/literals captured during the write loop. Anchored
    // on the runtime-length SSA ptr.
    //
    // Soundness for the length-sum equality: the IR computes
    //   runtime_length = (Σ runtime_ax_len_k) * other_product
    // (see `runtime_concat_len` accumulator + the final `b.ir_mul_i`
    // above). Each `runtime_ax_len_k` is the k-th input's runtime axis
    // length on the concatenated axis. The contract fact mirrors that
    // arithmetic identity exactly; consumers reading it avoid paying the
    // full IR walk to rediscover the same relation.
    {
        let len_vid = runtime_length_sv.value_id;
        b.fire_contract("dyn_concatenate", len_vid, &HashMap::new());

        use crate::optim::predicates::formula::{ArithOp, CmpOp, ContractTerm, ContractVar};
        let sum_term: ContractTerm = if arg_len_terms.is_empty() {
            ContractTerm::LitInt(0)
        } else {
            let mut iter = arg_len_terms.into_iter();
            let first = iter.next().unwrap();
            iter.fold(first, |acc, t| ContractTerm::Arith {
                op: ArithOp::Add,
                lhs: Box::new(acc),
                rhs: Box::new(t),
            })
        };
        let rhs_term = ContractTerm::Arith {
            op: ArithOp::Mul,
            lhs: Box::new(sum_term),
            rhs: Box::new(ContractTerm::LitInt(other_product as i64)),
        };
        let equality = ContractTerm::Cmp {
            op: CmpOp::Eq,
            lhs: Box::new(ContractTerm::Var(ContractVar::Value(len_vid))),
            rhs: Box::new(rhs_term),
        };
        b.facts.insert_for(len_vid, equality);
    }

    let out_value_id = ValueId::next();
    let input_vids: Vec<crate::types::ValueId> =
        raw_values.iter().filter_map(|v| v.value_id()).collect();
    if input_vids.len() == raw_values.len() {
        crate::optim::resolver::relay_forall_eq_const_from_all_inputs(b, &input_vids, out_value_id);
    }

    Value::DynamicNDArray(DynamicNDArrayData {
        envelope,
        dtype: out_dtype,
        segment_id,
        meta: DynArrayMeta {
            logical_shape: out_shape.clone(),
            logical_offset: 0,
            logical_strides: out_strides.clone(),
            runtime_length: runtime_length_sv,
            runtime_rank: ScalarValue::new(Some(ndim as i64), None),
            runtime_shape,
            runtime_strides: out_strides
                .iter()
                .map(|&s| ScalarValue::new(Some(s as i64), None))
                .collect(),
            runtime_offset: ScalarValue::new(Some(0), None),
        },
        value_id: out_value_id,
    })
}

/// The runtime extent of axis `d` (the buffer extent when none is recorded).
fn runtime_extent(a: &DynamicNDArrayData, d: usize) -> ScalarValue<i64> {
    a.meta
        .runtime_shape
        .get(d)
        .cloned()
        .unwrap_or_else(|| ScalarValue::new(Some(a.meta.logical_shape[d] as i64), None))
}

/// Requires two runtime extents of axis `d` to be equal: a compile error
/// for two different constants, an in-circuit check otherwise.
fn require_equal_extent(b: &mut IRBuilder, x: &ScalarValue<i64>, y: &ScalarValue<i64>, d: usize) {
    if let (Some(p), Some(q)) = (x.static_val, y.static_val) {
        if p != q {
            panic!("concatenate: operands have sizes {p} and {q} on axis {d}");
        }
        return;
    }
    let xv = crate::ops::dyn_ndarray::scalar_i64_to_value(x, NumberType::Integer);
    let yv = crate::ops::dyn_ndarray::scalar_i64_to_value(y, NumberType::Integer);
    let equal = b.ir_equal_i(&xv, &yv);
    b.ir_assert(&equal);
}

/// Stack dynamic arrays along a new axis.
///
/// Supports: dtype promotion, scalar inputs (treated as 0-d → stacked as 1-d).
/// Output envelope: inserts new static dim (= num_arrays), other dims from first.
/// total_bound = N * T_input.
pub fn dyn_stack(b: &mut IRBuilder, args: &[Value], kwargs: &HashMap<String, Value>) -> Value {
    let arrays_val = args.first().expect("stack: requires arrays argument");
    let raw_values: Vec<Value> = match arrays_val {
        Value::List(cd) | Value::Tuple(cd) => cd.values.clone(),
        _ => panic!("stack: first arg must be list/tuple of arrays"),
    };
    assert!(!raw_values.is_empty(), "stack: need at least one array");

    let arrays: Vec<DynamicNDArrayData> = raw_values.iter().map(|v| coerce_to_dyn(b, v)).collect();

    let axis = kwargs
        .get("axis")
        .or_else(|| args.get(1))
        .and_then(|v| v.int_val())
        .unwrap_or(0);
    let base_shape = &arrays[0].meta.logical_shape;
    let ndim = base_shape.len();
    let ax = if axis < 0 {
        (ndim as i64 + 1 + axis) as usize
    } else {
        axis as usize
    };
    assert!(
        ax <= ndim,
        "stack: axis {} out of bounds for rank {}",
        ax,
        ndim
    );

    // Validate all arrays have the same shape.
    for (i, arr) in arrays.iter().enumerate().skip(1) {
        assert_eq!(
            &arr.meta.logical_shape, base_shape,
            "stack: array {} has shape {:?} but expected {:?}",
            i, arr.meta.logical_shape, base_shape
        );
    }

    let num_arrays = arrays.len();
    let mut out_shape = base_shape.clone();
    out_shape.insert(ax, num_arrays);
    let out_numel: usize = out_shape.iter().product();
    let out_strides = dyn_row_major_strides(&out_shape);
    let out_dtype = resolve_dtype(&arrays);

    // Build envelope.
    let total_bound = num_arrays.saturating_mul(arrays[0].envelope.total_bound);
    let mut out_dims = arrays[0].envelope.dims.clone();
    out_dims.insert(
        ax,
        crate::types::Dim::new_static(&mut b.dim_table, num_arrays),
    );
    let envelope = crate::types::Envelope::new_with_bound(out_dims, total_bound);

    // Check if any input has dynamic element count.
    let _has_dynamic_input = arrays
        .iter()
        .any(|a| a.envelope.dims.iter().any(|d| d.min != d.max));

    // Allocate output segment with defaults.
    let default_val = super::metadata::dyn_default_value(b, out_dtype);
    let init_elements = vec![default_val.clone(); out_numel];
    let segment_id = crate::helpers::segment::alloc_and_write_values(b, &init_elements, out_dtype);
    let max_out_addr = (out_numel as i64 - 1).max(0);

    // Per-input pass: for each input k, write its elements into the output
    // at stack axis position k. Use ir_read_memory for source access and
    // in_bounds checks based on runtime_length for dynamic inputs.
    let src_max_numel: usize = base_shape.iter().product();
    let src_strides_base = dyn_row_major_strides(base_shape);

    for (k, arr) in arrays.iter().enumerate() {
        let src_numel = arr.max_length();

        // Runtime length for this input (may be dynamic).
        let runtime_len: Value = if let Some(sv) = &arr.meta.runtime_length.static_val {
            b.ir_constant_int(*sv)
        } else if let Some(ptr) = arr.meta.runtime_length.stmt_id {
            Value::Integer(ScalarValue::new(None, Some(ptr)))
        } else {
            b.ir_constant_int(src_numel as i64)
        };

        for src_flat in 0..src_max_numel {
            let src_flat_val = b.ir_constant_int(src_flat as i64);
            let in_bounds = b.ir_less_than_i(&src_flat_val, &runtime_len);

            // Read from source via ir_read_memory.
            let clamped_src = (src_flat as i64).min(src_numel as i64 - 1).max(0);
            let src_addr = b.ir_constant_int(clamped_src);
            let src_val =
                crate::helpers::segment::read_typed(b, arr.segment_id, &src_addr, arr.dtype);

            // Cast dtype if needed.
            let casted = if arr.dtype == out_dtype {
                src_val
            } else {
                cast_to_dtype(b, &src_val, out_dtype)
            };

            // Compute output address: decode src_flat to source coords,
            // insert k at stack axis, encode with output strides.
            let src_coords = dyn_decode_coords(src_flat, base_shape, &src_strides_base);
            let mut out_coords = src_coords.clone();
            out_coords.insert(ax, k);
            let out_flat = dyn_encode_coords(&out_coords, &out_strides) as i64;
            let out_addr = b.ir_constant_int(out_flat.min(max_out_addr).max(0));

            let write_val =
                crate::helpers::value_ops::select_value(b, &in_bounds, &casted, &default_val);

            crate::helpers::segment::write_typed(b, segment_id, &out_addr, &write_val, out_dtype);
        }
    }

    // Runtime metadata: stacked axis is always static (num_arrays).
    // Other dims may be dynamic — use first input's runtime_shape.
    let mut runtime_shape: Vec<ScalarValue<i64>> = Vec::new();
    let mut src_dim_idx = 0;
    for d in 0..out_shape.len() {
        if d == ax {
            runtime_shape.push(ScalarValue::new(Some(num_arrays as i64), None));
        } else {
            if src_dim_idx < arrays[0].meta.runtime_shape.len() {
                runtime_shape.push(arrays[0].meta.runtime_shape[src_dim_idx].clone());
            } else {
                runtime_shape.push(ScalarValue::new(Some(out_shape[d] as i64), None));
            }
            src_dim_idx += 1;
        }
    }

    // runtime_length = num_arrays * input_runtime_length.
    let num_arr_val = b.ir_constant_int(num_arrays as i64);
    let input_runtime_len = if let Some(sv) = &arrays[0].meta.runtime_length.static_val {
        b.ir_constant_int(*sv)
    } else if let Some(ptr) = arrays[0].meta.runtime_length.stmt_id {
        Value::Integer(ScalarValue::new(None, Some(ptr)))
    } else {
        b.ir_constant_int(src_max_numel as i64)
    };
    let runtime_length = b.ir_mul_i(&num_arr_val, &input_runtime_len);

    let out_value_id = ValueId::next();
    let input_vids: Vec<crate::types::ValueId> =
        raw_values.iter().filter_map(|v| v.value_id()).collect();
    if input_vids.len() == raw_values.len() {
        crate::optim::resolver::relay_forall_eq_const_from_all_inputs(b, &input_vids, out_value_id);
    }

    Value::DynamicNDArray(DynamicNDArrayData {
        envelope,
        dtype: out_dtype,
        segment_id,
        meta: DynArrayMeta {
            logical_shape: out_shape.clone(),
            logical_offset: 0,
            logical_strides: out_strides.clone(),
            runtime_length: value_to_scalar_i64(&runtime_length),
            runtime_rank: ScalarValue::new(Some(out_shape.len() as i64), None),
            runtime_shape,
            runtime_strides: out_strides
                .iter()
                .map(|&s| ScalarValue::new(Some(s as i64), None))
                .collect(),
            runtime_offset: ScalarValue::new(Some(0), None),
        },
        value_id: out_value_id,
    })
}

/// Split a dynamic array into parts along an axis.
///
/// `sections`: either an integer N (equal division) or a list of split indices.
/// Split indices may be static (compile-time) or dynamic (circuit wires).
/// `allow_unequal`: if true, integer N allows unequal chunks (array_split).
///
/// Static indices → O(N) fast path. Dynamic indices → O(K*axis_len) pad-and-mask.
/// Returns a List of DynamicNDArrays.
pub fn dyn_split(
    b: &mut IRBuilder,
    data: &DynamicNDArrayData,
    args: &[Value],
    kwargs: &HashMap<String, Value>,
) -> Value {
    dyn_split_impl(b, data, args, kwargs, false)
}

/// array_split: like split but allows unequal chunks for integer N.
pub fn dyn_array_split(
    b: &mut IRBuilder,
    data: &DynamicNDArrayData,
    args: &[Value],
    kwargs: &HashMap<String, Value>,
) -> Value {
    dyn_split_impl(b, data, args, kwargs, true)
}

fn dyn_split_impl(
    b: &mut IRBuilder,
    data: &DynamicNDArrayData,
    args: &[Value],
    kwargs: &HashMap<String, Value>,
    allow_unequal: bool,
) -> Value {
    let sections_val = args.get(0).expect("split: requires sections argument");
    let axis = kwargs
        .get("axis")
        .or_else(|| args.get(1))
        .and_then(|v| v.int_val())
        .unwrap_or(0);
    let ndim = data.meta.logical_shape.len();
    let ax = if axis < 0 {
        (ndim as i64 + axis) as usize
    } else {
        axis as usize
    };
    assert!(
        ax < ndim,
        "split: axis {} out of bounds for rank {}",
        ax,
        ndim
    );

    let axis_len = data.meta.logical_shape[ax];

    // Parse sections into split-point Values (may be static or dynamic).
    // Also track whether all are static for the fast path.
    let split_values: Vec<Value>;
    let num_chunks: usize;

    if let Some(n) = sections_val.int_val() {
        let n = n as usize;
        assert!(n > 0, "split: number of sections must be positive");
        if !allow_unequal {
            assert!(
                axis_len % n == 0,
                "split: array of size {} on axis {} cannot be split into {} equal parts",
                axis_len,
                ax,
                n
            );
        }
        // Compute split points for equal/unequal division.
        // array_split: first (axis_len % n) chunks get ceil(axis_len/n),
        // remaining get floor(axis_len/n).
        let mut points = Vec::new();
        let base = axis_len / n;
        let extra = axis_len % n;
        let mut pos = 0usize;
        for i in 0..n - 1 {
            pos += base + if i < extra { 1 } else { 0 };
            points.push(Value::Integer(ScalarValue::new(Some(pos as i64), None)));
        }
        split_values = points;
        num_chunks = n;
    } else {
        match sections_val {
            Value::List(cd) | Value::Tuple(cd) => {
                split_values = cd.values.clone();
                num_chunks = cd.values.len() + 1;
            }
            _ => panic!("split: sections must be int or list of ints"),
        }
    }

    // NumPy: chunk k is `a[d_k : d_{k+1}]` along the axis, with
    // `d = [0, *split_points, len]` and Python's slice semantics (negative
    // points count from the end, everything clamps into [0, len], a
    // decreasing pair gives an empty chunk), `len` being the axis's
    // runtime extent.
    let shape_is_static = data
        .meta
        .runtime_shape
        .iter()
        .zip(&data.meta.logical_shape)
        .all(|(rt, &n)| rt.static_val == Some(n as i64));
    let result_arrays: Vec<Value> = if shape_is_static
        && split_values.iter().all(|v| v.int_val().is_some())
    {
        let points: Vec<i64> = split_values.iter().map(|v| v.int_val().unwrap()).collect();
        static_split_chunks(b, data, ax, &points)
    } else {
        let split_values = match sections_val.int_val() {
            Some(n) if !shape_is_static => {
                runtime_section_points(b, data, ax, n as usize, allow_unequal)
            }
            _ => split_values,
        };
        (0..num_chunks)
            .map(|k| {
                let start = if k == 0 { Some(b.ir_constant_int(0)) } else { Some(split_values[k - 1].clone()) };
                let stop = split_values.get(k).cloned();
                let indices: Vec<SliceIndex> = (0..ndim)
                    .map(|d| {
                        if d == ax {
                            SliceIndex::Range(start.clone(), stop.clone(), None)
                        } else {
                            SliceIndex::Range(None, None, None)
                        }
                    })
                    .collect();
                crate::helpers::array_ops::dyn_subscript(b, data, &indices)
            })
            .collect()
    };

    let types = result_arrays.iter().map(|v| v.zinnia_type()).collect();
    Value::List(crate::types::CompositeData {
        elements_type: types,
        values: result_arrays,

        value_id: ValueId::next(),
    })
}

/// The split points of `n` sections of axis `ax` with a runtime extent `L`
/// (NumPy): `np.split` requires `L % n == 0` (else `ValueError`: no proof)
/// and cuts at `k * (L // n)`; `np.array_split` gives the first `L % n`
/// sections one extra element, cutting at `k * (L // n) + min(k, L % n)`.
fn runtime_section_points(
    b: &mut IRBuilder,
    data: &DynamicNDArrayData,
    ax: usize,
    n: usize,
    allow_unequal: bool,
) -> Vec<Value> {
    let extent = crate::ops::dyn_ndarray::scalar_i64_to_value(
        &data.meta.runtime_shape[ax],
        NumberType::Integer,
    );
    let n_v = b.ir_constant_int(n as i64);
    let size = b.ir_floor_div_i(&extent, &n_v);
    let covered = b.ir_mul_i(&size, &n_v);
    let rest = b.ir_sub_i(&extent, &covered);
    if !allow_unequal {
        let zero = b.ir_constant_int(0);
        let even = b.ir_equal_i(&rest, &zero);
        b.ir_assert(&even);
    }
    (1..n)
        .map(|k| {
            let k_v = b.ir_constant_int(k as i64);
            let base = b.ir_mul_i(&k_v, &size);
            if allow_unequal {
                let k_below = b.ir_less_than_i(&k_v, &rest);
                let extra = b.ir_select_i(&k_below, &k_v, &rest);
                b.ir_add_i(&base, &extra)
            } else {
                base
            }
        })
        .collect()
}

/// The chunks of a static-shape array split at compile-time points along
/// axis `ax`: Python slice semantics for each `a[d_k : d_{k+1}]` (module of
/// `dyn_split_impl`), copied cell by cell.
fn static_split_chunks(
    b: &mut IRBuilder,
    data: &DynamicNDArrayData,
    ax: usize,
    points: &[i64],
) -> Vec<Value> {
    let ndim = data.meta.logical_shape.len();
    let axis_len = data.meta.logical_shape[ax] as i64;
    let wrap = |p: i64| (if p < 0 { p + axis_len } else { p }).clamp(0, axis_len) as usize;
    let mut bounds: Vec<usize> = vec![0];
    bounds.extend(points.iter().map(|&p| wrap(p)));
    bounds.push(axis_len as usize);

    let data = super::force_contiguous(b, data);
    let src_vals =
        crate::helpers::segment::read_all_typed(b, data.segment_id, data.max_length(), data.dtype);
    let src_strides = dyn_row_major_strides(&data.meta.logical_shape);
    bounds
        .windows(2)
        .map(|w| {
            let (start, end) = (w[0], w[1].max(w[0]));
            let mut chunk_shape = data.meta.logical_shape.clone();
            chunk_shape[ax] = end - start;
            let chunk_numel: usize = chunk_shape.iter().product();
            let chunk_strides = dyn_row_major_strides(&chunk_shape);
            let chunk_elements: Vec<Value> = (0..chunk_numel)
                .map(|flat| {
                    let mut coords = dyn_decode_coords(flat, &chunk_shape, &chunk_strides);
                    coords[ax] += start;
                    src_vals[dyn_encode_coords(&coords, &src_strides)].clone()
                })
                .collect();
            let segment_id =
                crate::helpers::segment::alloc_and_write_values(b, &chunk_elements, data.dtype);
            let mut chunk_dims = data.envelope.dims.clone();
            chunk_dims[ax] = crate::types::Dim::new_static(&mut b.dim_table, end - start);
            let envelope = crate::types::Envelope::new_with_bound(chunk_dims, chunk_numel);
            let constant = |v: usize| ScalarValue::new(Some(v as i64), None);
            Value::DynamicNDArray(DynamicNDArrayData {
                envelope,
                dtype: data.dtype,
                segment_id,
                meta: DynArrayMeta {
                    logical_shape: chunk_shape.clone(),
                    logical_offset: 0,
                    logical_strides: chunk_strides.clone(),
                    runtime_length: constant(chunk_numel),
                    runtime_rank: constant(ndim),
                    runtime_shape: chunk_shape.iter().map(|&n| constant(n)).collect(),
                    runtime_strides: chunk_strides.iter().map(|&n| constant(n)).collect(),
                    runtime_offset: constant(0),
                },
                value_id: ValueId::next(),
            })
        })
        .collect()
}
