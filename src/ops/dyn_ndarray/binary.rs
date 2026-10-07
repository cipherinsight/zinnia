//! Element-wise binary operations for dynamic ndarrays.
//!
//! Implements Phase B from ROADMAP/04-lazy-views.md: reads both operands
//! from their ZKRAM segments, applies a scalar op element-wise (with
//! broadcasting), and writes the result to a fresh segment.

use crate::builder::IRBuilder;
use crate::helpers::shape_arith::{decode_coords, encode_coords, row_major_strides};
use crate::helpers::value_ops::{apply_scalar_binary_op, binary_op_result_dtype};
use crate::types::{
    broadcast_envelopes, DynArrayMeta, DynamicNDArrayData, NumberType, ScalarValue, Value, ValueId,
};

use super::value_to_scalar_i64;

/// Element-wise binary op on two `DynamicNDArrayData` values.
///
/// Handles broadcasting via coordinate remapping. The output envelope and
/// `total_bound` are computed by `broadcast_envelopes`; the bound rules
/// from §3.3/§3.4 are enforced here.
pub fn dyn_binary_op(
    b: &mut IRBuilder,
    op: &str,
    lhs: &DynamicNDArrayData,
    rhs: &DynamicNDArrayData,
) -> Value {
    // 1. Compute output envelope (includes total_bound propagation).
    let out_envelope = broadcast_envelopes(&mut b.dim_table, &lhs.envelope, &rhs.envelope)
        .unwrap_or_else(|e| panic!("dynamic binary op: broadcast failed: {}", e));

    // 1b. Refusal check (§3.4 / §11.2): if the broadcast produces a bound
    // that's larger than both inputs' bounds AND the increase comes from
    // dynamic dims, the static type system gave us nothing — refuse.
    check_broadcast_bound(&lhs.envelope, &rhs.envelope, &out_envelope);

    // 2. Output dtype (shared NumPy promotion rule). Between two bool arrays
    // `+` is a logical or (`helpers::dtype::operator_on_bools`).
    let both_bool = b.is_bool_segment(lhs.segment_id) && b.is_bool_segment(rhs.segment_id);
    let op = if both_bool { crate::helpers::dtype::operator_on_bools(op) } else { op };
    let out_dtype = binary_op_result_dtype(op, lhs.dtype, rhs.dtype);

    // 3. Pre-read source segments. Force contiguity so the row-major
    // indexing below sees cells in logical order even if the source is
    // a stride-permuted view (e.g. from a lazy `dyn_transpose`).
    let lhs_owned = super::force_contiguous(b, lhs);
    let rhs_owned = super::force_contiguous(b, rhs);
    let lhs = &lhs_owned;
    let rhs = &rhs_owned;
    let lhs_vals =
        crate::helpers::segment::read_all_typed(b, lhs.segment_id, lhs.max_length(), lhs.dtype);
    let rhs_vals =
        crate::helpers::segment::read_all_typed(b, rhs.segment_id, rhs.max_length(), rhs.dtype);

    // 4. Compute shapes and strides for coordinate remapping.
    let lhs_max_shape: Vec<usize> = lhs.envelope.dims.iter().map(|d| d.max).collect();
    let rhs_max_shape: Vec<usize> = rhs.envelope.dims.iter().map(|d| d.max).collect();
    let out_max_shape: Vec<usize> = out_envelope.dims.iter().map(|d| d.max).collect();
    let out_total = out_envelope.max_total();

    let lhs_strides = row_major_strides(&lhs_max_shape);
    let rhs_strides = row_major_strides(&rhs_max_shape);
    let out_strides = row_major_strides(&out_max_shape);

    // 5. Element-wise computation with broadcast coordinate remapping.
    let mut out_elements: Vec<Value> = Vec::with_capacity(out_total);

    for flat_idx in 0..out_total {
        let out_coords = decode_coords(flat_idx, &out_max_shape, &out_strides);

        let lhs_flat = remap_and_encode(&out_coords, &lhs_max_shape, &lhs_strides);
        let rhs_flat = remap_and_encode(&out_coords, &rhs_max_shape, &rhs_strides);

        let lhs_val = &lhs_vals[lhs_flat];
        let rhs_val = &rhs_vals[rhs_flat];

        let result_val = apply_scalar_binary_op(b, op, lhs_val, rhs_val);
        out_elements.push(result_val);
    }

    // 6. Write to fresh segment. `+` and `*` of two bool arrays are bool in
    // NumPy (the cells here are integers; see `elementwise::keeps_bool_dtype`).
    let segment_id = crate::helpers::segment::alloc_and_write_values(b, &out_elements, out_dtype);
    if both_bool && matches!(op, "mul" | "or") {
        b.mark_bool_segment(segment_id);
    }

    // 7. Build runtime metadata.
    require_shared_axes_equal(b, &out_envelope, lhs, rhs);
    let runtime_shape = build_runtime_shape(&out_envelope, lhs, rhs);
    let runtime_length = compute_runtime_length(b, &runtime_shape);

    Value::DynamicNDArray(DynamicNDArrayData {
        envelope: out_envelope,
        dtype: out_dtype,
        segment_id,
        meta: DynArrayMeta {
            logical_shape: out_max_shape.clone(),
            logical_offset: 0,
            logical_strides: out_strides.clone(),
            runtime_length,
            runtime_rank: ScalarValue::new(Some(out_max_shape.len() as i64), None),
            runtime_shape,
            runtime_strides: out_strides
                .iter()
                .map(|&s| ScalarValue::new(Some(s as i64), None))
                .collect(),
            runtime_offset: ScalarValue::new(Some(0), None),
        },
        value_id: ValueId::next(),
    })
}

/// Element-wise unary op on a dynamic array (negation, abs, etc.).
/// `apply_fn` takes the builder and one element value, returns the result.
pub fn dyn_unary_op(
    b: &mut IRBuilder,
    arr: &DynamicNDArrayData,
    out_dtype: NumberType,
    apply_fn: impl Fn(&mut IRBuilder, &Value) -> Value,
) -> Value {
    let arr_owned = super::force_contiguous(b, arr);
    let arr = &arr_owned;
    let arr_vals =
        crate::helpers::segment::read_all_typed(b, arr.segment_id, arr.max_length(), arr.dtype);
    let mut out_elements: Vec<Value> = Vec::with_capacity(arr_vals.len());

    for val in &arr_vals {
        let result = apply_fn(b, val);
        out_elements.push(result);
    }

    let segment_id = crate::helpers::segment::alloc_and_write_values(b, &out_elements, out_dtype);

    Value::DynamicNDArray(DynamicNDArrayData {
        envelope: arr.envelope.clone(),
        dtype: out_dtype,
        segment_id,
        meta: arr.meta.clone(),
        value_id: ValueId::next(),
    })
}

/// Element-wise binary op: scalar (lhs) against a dynamic array (rhs).
pub fn dyn_scalar_binary_op(
    b: &mut IRBuilder,
    op: &str,
    scalar: &Value,
    arr: &DynamicNDArrayData,
    scalar_is_lhs: bool,
) -> Value {
    let scalar_dtype = match scalar {
        Value::Float(_) => NumberType::Float,
        Value::Complex { .. } => NumberType::Complex,
        _ => NumberType::Integer,
    };
    let both_bool = b.is_bool_cell(scalar) && b.is_bool_segment(arr.segment_id);
    let op = if both_bool { crate::helpers::dtype::operator_on_bools(op) } else { op };
    let out_dtype = binary_op_result_dtype(op, scalar_dtype, arr.dtype);

    let arr_owned = super::force_contiguous(b, arr);
    let arr = &arr_owned;
    let arr_vals =
        crate::helpers::segment::read_all_typed(b, arr.segment_id, arr.max_length(), arr.dtype);
    let mut out_elements: Vec<Value> = Vec::with_capacity(arr_vals.len());

    for arr_val in &arr_vals {
        let result = if scalar_is_lhs {
            apply_scalar_binary_op(b, op, scalar, arr_val)
        } else {
            apply_scalar_binary_op(b, op, arr_val, scalar)
        };
        out_elements.push(result);
    }

    let segment_id = crate::helpers::segment::alloc_and_write_values(b, &out_elements, out_dtype);

    // Mask-read view propagation: lane k of the output is `op(lane k)`,
    // so the packed-kept-sources invariant survives with the same op
    // applied to each source. Restricted to add/sub (plain field ops —
    // harmless if the transformed view is never consumed) plus integer
    // mul; float mul/div would duplicate expensive rescales.
    let cheap = matches!(op, "add" | "sub")
        || (op == "mul" && out_dtype == NumberType::Integer);
    if cheap {
        if let Some(view) = b.provenance.mask_read_view(arr.segment_id).cloned() {
            let new_sources: Vec<Value> = view
                .sources
                .iter()
                .map(|src| {
                    if scalar_is_lhs {
                        apply_scalar_binary_op(b, op, scalar, src)
                    } else {
                        apply_scalar_binary_op(b, op, src, scalar)
                    }
                })
                .collect();
            b.provenance.set_mask_read_view(
                segment_id,
                crate::provenance::MaskReadView {
                    mask_cells: view.mask_cells,
                    sources: new_sources,
                    count_stmt: view.count_stmt,
                },
            );
        }
    }

    // Scalar broadcast: output envelope = input array's envelope (§3.2).
    let out_val = Value::DynamicNDArray(DynamicNDArrayData {
        envelope: arr.envelope.clone(),
        dtype: out_dtype,
        segment_id,
        meta: arr.meta.clone(),
        value_id: ValueId::next(),
    });

    // Element-bound propagation (compiler.op-contract-inplace-mutation):
    // for arithmetic ops (`add`, `sub`, `mul`) on integer arrays, derive
    // new element bounds from the input array's bounds combined with
    // the scalar's bounds. Skip comparisons (output dtype = Integer 0/1
    // is trivially bounded but it's not informative downstream) and
    // skip Float/Complex (no integer bound representation today).
    if matches!(op, "add" | "sub" | "mul") && out_dtype == NumberType::Integer {
        let scalar_arr = Value::DynamicNDArray(arr.clone());
        let arr_bounds =
            crate::helpers::static_array::reductions::read_array_element_bounds(b, &scalar_arr);
        let scalar_bounds = if let Some(n) = scalar.int_val() {
            Some((n, n))
        } else {
            b.ask_bounds(scalar)
        };
        if let (Some((alo, ahi)), Some((slo, shi))) = (arr_bounds, scalar_bounds) {
            let (xlo, xhi, ylo, yhi) = if scalar_is_lhs {
                (slo, shi, alo, ahi)
            } else {
                (alo, ahi, slo, shi)
            };
            let computed = match op {
                "add" => Some((xlo.checked_add(ylo), xhi.checked_add(yhi))),
                "sub" => Some((xlo.checked_sub(yhi), xhi.checked_sub(ylo))),
                "mul" => {
                    let c1 = xlo.checked_mul(ylo);
                    let c2 = xlo.checked_mul(yhi);
                    let c3 = xhi.checked_mul(ylo);
                    let c4 = xhi.checked_mul(yhi);
                    match (c1, c2, c3, c4) {
                        (Some(a), Some(b), Some(c), Some(d)) => {
                            Some((Some(a.min(b).min(c).min(d)), Some(a.max(b).max(c).max(d))))
                        }
                        _ => None,
                    }
                }
                _ => None,
            };
            if let Some((Some(new_lo), Some(new_hi))) = computed {
                crate::helpers::static_array::reductions::deposit_array_in_range_fact(
                    b, &out_val, new_lo, new_hi,
                );
            }
        }
    }

    out_val
}

// ────────────────────────────────────────────────────────────────────────
// Helpers
// ────────────────────────────────────────────────────────────────────────

/// Check that the broadcast bound is derivable from static knowledge.
///
/// Allowed: same-shape (T_out ≤ max(T_a, T_b)), or broadcast where all
/// a-only/b-only dims are statically known (predictable multiplication).
///
/// Refused: broadcast where BOTH the a-only factor and the b-only factor
/// involve dynamic dims — the product of two independent unknowns causes
/// multiplicative explosion (e.g., `[D1, 1] + [1, D2]` → T = D1 × D2).
///
/// A single dynamic factor is safe: `[D, 1] + [1, 3]` → T = 3 × D_max,
/// bounded by a static multiplier on the largest input.
fn check_broadcast_bound(
    a: &crate::types::Envelope,
    b: &crate::types::Envelope,
    out: &crate::types::Envelope,
) {
    let t_max_input = a.total_bound.max(b.total_bound);

    // If output bound doesn't exceed either input, no explosion — allow.
    if out.total_bound <= t_max_input {
        return;
    }

    // Output is larger than both inputs. Check if the increase comes from
    // dynamic factors on BOTH sides (disjoint dynamic broadcast).
    //
    // a-only factor (multiplies T_a): product of b's dims where a has 1.
    // b-only factor (multiplies T_b): product of a's dims where b has 1.
    // Refuse only when BOTH factors involve dynamic dims.
    let rank = out.rank();
    let mut a_factor_has_dynamic = false; // a-only factor involves a dynamic dim
    let mut b_factor_has_dynamic = false; // b-only factor involves a dynamic dim

    for i in 0..rank {
        let a_axis = if i < a.rank() {
            Some(a.rank() - 1 - (rank - 1 - i))
        } else {
            None
        };
        let b_axis = if i < b.rank() {
            Some(b.rank() - 1 - (rank - 1 - i))
        } else {
            None
        };

        match (a_axis, b_axis) {
            (Some(ai), None) => {
                // a extends beyond b's rank → b implicitly 1 → b broadcasts.
                // b-only factor gets a.dims[ai].
                if a.dims[ai].is_static().is_none() {
                    b_factor_has_dynamic = true;
                }
            }
            (None, Some(bi)) => {
                // b extends beyond a's rank → a implicitly 1 → a broadcasts.
                // a-only factor gets b.dims[bi].
                if b.dims[bi].is_static().is_none() {
                    a_factor_has_dynamic = true;
                }
            }
            (Some(ai), Some(bi)) => {
                let x = a.dims[ai];
                let y = b.dims[bi];
                if x.is_static() == Some(1) && y.is_static() != Some(1) {
                    // a has 1 → a broadcasts. a-only factor gets b's dim.
                    if y.is_static().is_none() {
                        a_factor_has_dynamic = true;
                    }
                }
                if y.is_static() == Some(1) && x.is_static() != Some(1) {
                    // b has 1 → b broadcasts. b-only factor gets a's dim.
                    if x.is_static().is_none() {
                        b_factor_has_dynamic = true;
                    }
                }
            }
            (None, None) => unreachable!(),
        }
    }

    if a_factor_has_dynamic && b_factor_has_dynamic {
        panic!(
            "dynamic broadcast refused: output bound {} exceeds both inputs' bounds \
             (T_a={}, T_b={}) due to dynamic broadcast factors on both sides. \
             The static type system cannot derive a tight bound for this broadcast. \
             To proceed, supply a #[bound(total ≤ N)] annotation (not yet implemented).",
            out.total_bound, a.total_bound, b.total_bound,
        );
    }
}

/// Remap output coordinates to a source's coordinate space (handling
/// broadcast and rank difference), then encode to a flat index.
fn remap_and_encode(out_coords: &[usize], src_max_shape: &[usize], src_strides: &[usize]) -> usize {
    if src_max_shape.is_empty() {
        return 0; // scalar source
    }
    let rank_diff = out_coords.len() - src_max_shape.len();
    let src_coords: Vec<usize> = src_max_shape
        .iter()
        .enumerate()
        .map(|(i, &dim_max)| {
            if dim_max == 1 {
                0 // broadcast dim
            } else {
                out_coords[i + rank_diff]
            }
        })
        .collect();
    encode_coords(&src_coords, src_strides)
}

/// Build runtime_shape for the output by picking from whichever input
/// provides each axis (shared → either input, a-only → a, b-only → b).
/// Require equal runtime extents on every axis the two operands share
/// (neither side's envelope extent is a static 1).
///
/// The element-wise loop pairs slot `k` of one operand with slot `k` of the
/// other and the result takes the left operand's extent, so it computes
/// NumPy's result only when the extents agree. NumPy raises when they
/// differ (unless one is 1; broadcasting a runtime extent of 1 is not
/// lowered, so such inputs are rejected too). Without this check
/// `a + b` with `len(a) = 8`, `len(b) = 5` proved a value, and the result's
/// extent could exceed its envelope. Equal values (the same IR value or the
/// same constant) need no check; otherwise the equality is checked where
/// the code runs.
fn require_shared_axes_equal(
    b: &mut IRBuilder,
    out_envelope: &crate::types::Envelope,
    lhs: &DynamicNDArrayData,
    rhs: &DynamicNDArrayData,
) {
    let out_rank = out_envelope.rank();
    let (lhs_rank, rhs_rank) = (lhs.envelope.rank(), rhs.envelope.rank());
    for from_back in 0..out_rank.min(lhs_rank).min(rhs_rank) {
        let (la, ra) = (lhs_rank - 1 - from_back, rhs_rank - 1 - from_back);
        if lhs.envelope.dims[la].is_static() == Some(1) || rhs.envelope.dims[ra].is_static() == Some(1) {
            continue;
        }
        let (l, r) = (&lhs.meta.runtime_shape[la], &rhs.meta.runtime_shape[ra]);
        let same_value = (l.stmt_id.is_some() && l.stmt_id == r.stmt_id)
            || (l.static_val.is_some() && l.static_val == r.static_val);
        if same_value {
            continue;
        }
        if let (Some(x), Some(y)) = (l.static_val, r.static_val) {
            panic!(
                "operands could not be broadcast together: axis extents {x} and {y} differ"
            );
        }
        let (l, r) = (extent_value(b, l), extent_value(b, r));
        let equal = b.ir_equal_i(&l, &r);
        b.ir_assert(&equal);
    }
}

/// A runtime extent as an IR value.
fn extent_value(b: &mut IRBuilder, sv: &ScalarValue<i64>) -> Value {
    match sv.static_val {
        Some(c) => b.ir_constant_int(c),
        None => Value::Integer(sv.clone()),
    }
}

fn build_runtime_shape(
    out_envelope: &crate::types::Envelope,
    lhs: &DynamicNDArrayData,
    rhs: &DynamicNDArrayData,
) -> Vec<ScalarValue<i64>> {
    let out_rank = out_envelope.rank();
    let lhs_rank = lhs.envelope.rank();
    let rhs_rank = rhs.envelope.rank();

    (0..out_rank)
        .map(|i| {
            // Right-aligned source axis indices.
            let from_back = out_rank - 1 - i;
            let lhs_axis = if from_back < lhs_rank {
                Some(lhs_rank - 1 - from_back)
            } else {
                None
            };
            let rhs_axis = if from_back < rhs_rank {
                Some(rhs_rank - 1 - from_back)
            } else {
                None
            };

            match (lhs_axis, rhs_axis) {
                (Some(la), None) => lhs.meta.runtime_shape[la].clone(),
                (None, Some(ra)) => rhs.meta.runtime_shape[ra].clone(),
                (Some(la), Some(ra)) => {
                    let l_dim = lhs.envelope.dims[la];
                    let r_dim = rhs.envelope.dims[ra];
                    if l_dim.is_static() == Some(1) {
                        // a broadcasts → use b's runtime dim
                        rhs.meta.runtime_shape[ra].clone()
                    } else if r_dim.is_static() == Some(1) {
                        // b broadcasts → use a's runtime dim
                        lhs.meta.runtime_shape[la].clone()
                    } else {
                        // shared axis (unified) → use either
                        lhs.meta.runtime_shape[la].clone()
                    }
                }
                (None, None) => unreachable!(),
            }
        })
        .collect()
}

/// Compute runtime_length as the product of runtime_shape entries.
/// If all entries have static values, produce a static result.
fn compute_runtime_length(
    b: &mut IRBuilder,
    runtime_shape: &[ScalarValue<i64>],
) -> ScalarValue<i64> {
    if runtime_shape.is_empty() {
        return ScalarValue::new(Some(1), None);
    }
    // Check if all static.
    let all_static: Option<i64> = runtime_shape
        .iter()
        .try_fold(1i64, |acc, sv| sv.static_val.map(|v| acc * v));
    if let Some(total) = all_static {
        return ScalarValue::new(Some(total), None);
    }
    // Dynamic: multiply via IR.
    let mut acc = if let Some(v) = runtime_shape[0].static_val {
        b.ir_constant_int(v)
    } else if let Some(ptr) = runtime_shape[0].stmt_id {
        Value::Integer(ScalarValue::new(None, Some(ptr)))
    } else {
        b.ir_constant_int(0)
    };
    for sv in &runtime_shape[1..] {
        let val = if let Some(v) = sv.static_val {
            b.ir_constant_int(v)
        } else if let Some(ptr) = sv.stmt_id {
            Value::Integer(ScalarValue::new(None, Some(ptr)))
        } else {
            b.ir_constant_int(0)
        };
        acc = b.ir_mul_i(&acc, &val);
    }
    value_to_scalar_i64(&acc)
}

// ────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::helpers::promote::promote_static_to_dynamic;
    use crate::helpers::value_ops::apply_binary_op;
    use crate::types::CompositeData;

    fn list_of(values: Vec<Value>) -> Value {
        let types = values.iter().map(|v| v.zinnia_type()).collect();
        Value::List(CompositeData {
            elements_type: types,
            values,

            value_id: ValueId::next(),
        })
    }

    fn extract_dyn(v: &Value) -> &DynamicNDArrayData {
        match v {
            Value::DynamicNDArray(d) => d,
            _ => panic!("expected DynamicNDArray, got {:?}", v),
        }
    }

    /// Build a static list, promote to dynamic — avoids borrow issues by
    /// constructing the list before passing `&mut b`.
    fn make_int_dyn(b: &mut IRBuilder, vals: &[i64]) -> Value {
        let elems: Vec<Value> = vals.iter().map(|&v| b.ir_constant_int(v)).collect();
        let lst = list_of(elems);
        promote_static_to_dynamic(b, &lst)
    }

    fn make_2d_int_dyn(b: &mut IRBuilder, rows: &[&[i64]]) -> Value {
        let row_vals: Vec<Value> = rows
            .iter()
            .map(|row| {
                let elems: Vec<Value> = row.iter().map(|&v| b.ir_constant_int(v)).collect();
                list_of(elems)
            })
            .collect();
        let mat = list_of(row_vals);
        promote_static_to_dynamic(b, &mat)
    }

    #[test]
    fn add_same_shape_1d() {
        let mut b = IRBuilder::new();
        let a = make_int_dyn(&mut b, &[1, 2, 3, 4]);
        let bv = make_int_dyn(&mut b, &[10, 11, 12, 13]);
        let result = apply_binary_op(&mut b, "add", &a, &bv);
        let d = extract_dyn(&result);
        assert_eq!(d.envelope.max_total(), 4);
        assert_eq!(d.envelope.total_bound, 4);
        assert_eq!(d.dtype, NumberType::Integer);
        assert_eq!(d.meta.logical_shape, vec![4]);
        assert_eq!(d.meta.runtime_length.static_val, Some(4));
    }

    #[test]
    fn mul_dtype_promotion() {
        let mut b = IRBuilder::new();
        let a = make_int_dyn(&mut b, &[2, 3]);
        let float_elems: Vec<Value> = vec![b.ir_constant_float(1.5), b.ir_constant_float(2.0)];
        let float_list = list_of(float_elems);
        let bv = promote_static_to_dynamic(&mut b, &float_list);
        let result = apply_binary_op(&mut b, "mul", &a, &bv);
        let d = extract_dyn(&result);
        assert_eq!(d.dtype, NumberType::Float);
        assert_eq!(d.max_length(), 2);
    }

    #[test]
    fn scalar_add_dynamic() {
        let mut b = IRBuilder::new();
        let arr = make_int_dyn(&mut b, &[0, 1, 2]);
        let scalar = b.ir_constant_int(10);
        let result = apply_binary_op(&mut b, "add", &arr, &scalar);
        let d = extract_dyn(&result);
        assert_eq!(d.max_length(), 3);
        assert_eq!(d.envelope.total_bound, 3);
        assert_eq!(d.dtype, NumberType::Integer);
    }

    #[test]
    fn comparison_produces_integer_dtype() {
        let mut b = IRBuilder::new();
        let a = make_int_dyn(&mut b, &[1, 2, 3]);
        let bv = make_int_dyn(&mut b, &[1, 5, 3]);
        let result = apply_binary_op(&mut b, "eq", &a, &bv);
        let d = extract_dyn(&result);
        assert_eq!(d.dtype, NumberType::Integer);
        assert_eq!(d.max_length(), 3);
    }

    #[test]
    fn broadcast_1d_against_2d() {
        let mut b = IRBuilder::new();
        let a = make_int_dyn(&mut b, &[1, 2, 3]);
        let bv = make_2d_int_dyn(&mut b, &[&[10, 20, 30], &[40, 50, 60]]);
        let result = apply_binary_op(&mut b, "add", &a, &bv);
        let d = extract_dyn(&result);
        assert_eq!(d.envelope.rank(), 2);
        assert_eq!(d.envelope.max_total(), 6);
        assert_eq!(d.meta.logical_shape, vec![2, 3]);
        assert_eq!(d.meta.runtime_length.static_val, Some(6));
    }

    #[test]
    fn total_bound_same_shape() {
        let mut b = IRBuilder::new();
        let a = make_int_dyn(&mut b, &[0, 1, 2, 3]);
        let bv = make_int_dyn(&mut b, &[0, 1, 2, 3]);
        let result = apply_binary_op(&mut b, "add", &a, &bv);
        let d = extract_dyn(&result);
        assert_eq!(d.envelope.total_bound, 4);
    }

    #[test]
    fn static_composite_plus_dynamic() {
        let mut b = IRBuilder::new();
        let static_elems: Vec<Value> = vec![
            b.ir_constant_int(1),
            b.ir_constant_int(2),
            b.ir_constant_int(3),
        ];
        let static_list = list_of(static_elems);
        let dynamic = make_int_dyn(&mut b, &[10, 20, 30]);
        let result = apply_binary_op(&mut b, "add", &static_list, &dynamic);
        let d = extract_dyn(&result);
        assert_eq!(d.max_length(), 3);
        // Result is DynamicNDArray (static was promoted)
        assert_eq!(d.envelope.total_bound, 3);
    }

    #[test]
    fn broadcast_static_factor_allowed() {
        // [D, 1] + [1, 3] → [D, 3]. Static factor 3, should be allowed.
        let mut b = IRBuilder::new();
        // Build [D, 1] as a column vector: [[1], [2], [3]]
        let col_rows: Vec<Value> = (1..=3)
            .map(|i| list_of(vec![b.ir_constant_int(i)]))
            .collect();
        let col = promote_static_to_dynamic(&mut b, &list_of(col_rows));
        // Build [1, 3] as a row: [[10, 20, 30]]
        let row = list_of(vec![list_of(vec![
            b.ir_constant_int(10),
            b.ir_constant_int(20),
            b.ir_constant_int(30),
        ])]);
        let row_dyn = promote_static_to_dynamic(&mut b, &row);
        let result = apply_binary_op(&mut b, "add", &col, &row_dyn);
        let d = extract_dyn(&result);
        assert_eq!(d.envelope.rank(), 2);
        assert_eq!(d.envelope.max_total(), 9); // 3 * 3
    }

    #[test]
    #[should_panic(expected = "dynamic broadcast refused")]
    fn refuse_dynamic_disjoint_broadcast() {
        // Two arrays with dynamic dims at disjoint positions:
        // [D1, 1] + [1, D2] → would produce D1*D2 elements. Must refuse.
        let mut b = IRBuilder::new();

        // Simulate arrays with dynamic envelopes (not from promotion).
        // Create them directly with dynamic dims.
        let lhs_envelope = crate::types::Envelope::new_with_bound(
            vec![
                crate::types::Dim::new_dynamic(&mut b.dim_table, 0, 10),
                crate::types::Dim::new_static(&mut b.dim_table, 1),
            ],
            10,
        );
        let rhs_envelope = crate::types::Envelope::new_with_bound(
            vec![
                crate::types::Dim::new_static(&mut b.dim_table, 1),
                crate::types::Dim::new_dynamic(&mut b.dim_table, 0, 10),
            ],
            10,
        );
        let dummy_elements = vec![ScalarValue::new(Some(0), None); 10];
        let lhs_seg =
            crate::helpers::segment::alloc_and_write(&mut b, &dummy_elements, NumberType::Integer);
        let rhs_seg =
            crate::helpers::segment::alloc_and_write(&mut b, &dummy_elements, NumberType::Integer);

        let lhs_data = DynamicNDArrayData {
            envelope: lhs_envelope,
            dtype: NumberType::Integer,
            segment_id: lhs_seg,
            meta: DynArrayMeta {
                logical_shape: vec![10, 1],
                logical_offset: 0,
                logical_strides: vec![1, 1],
                runtime_length: ScalarValue::new(None, None),
                runtime_rank: ScalarValue::new(Some(2), None),
                runtime_shape: vec![
                    ScalarValue::new(None, None),
                    ScalarValue::new(Some(1), None),
                ],
                runtime_strides: vec![
                    ScalarValue::new(Some(1), None),
                    ScalarValue::new(Some(1), None),
                ],
                runtime_offset: ScalarValue::new(Some(0), None),
            },
            value_id: ValueId::next(),
        };
        let rhs_data = DynamicNDArrayData {
            envelope: rhs_envelope,
            dtype: NumberType::Integer,
            segment_id: rhs_seg,
            meta: DynArrayMeta {
                logical_shape: vec![1, 10],
                logical_offset: 0,
                logical_strides: vec![10, 1],
                runtime_length: ScalarValue::new(None, None),
                runtime_rank: ScalarValue::new(Some(2), None),
                runtime_shape: vec![
                    ScalarValue::new(Some(1), None),
                    ScalarValue::new(None, None),
                ],
                runtime_strides: vec![
                    ScalarValue::new(Some(10), None),
                    ScalarValue::new(Some(1), None),
                ],
                runtime_offset: ScalarValue::new(Some(0), None),
            },
            value_id: ValueId::next(),
        };

        // This should panic with "dynamic broadcast refused"
        dyn_binary_op(&mut b, "add", &lhs_data, &rhs_data);
    }

    #[test]
    fn sub_and_div_ops() {
        let mut b = IRBuilder::new();
        let a = make_int_dyn(&mut b, &[10, 20, 30]);
        let bv = make_int_dyn(&mut b, &[1, 2, 3]);
        let sub_result = apply_binary_op(&mut b, "sub", &a, &bv);
        assert_eq!(extract_dyn(&sub_result).max_length(), 3);
        let div_result = apply_binary_op(&mut b, "div", &a, &bv);
        assert_eq!(extract_dyn(&div_result).max_length(), 3);
    }
}
