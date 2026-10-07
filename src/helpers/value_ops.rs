//! Free-function versions of IRBuilder value operations.
//!
//! These functions implement binary operations, conditional selects,
//! dynamic list indexing, and other value-level operations, taking
//! `&mut IRBuilder` as an explicit parameter instead of `&mut self`.

use crate::builder::IRBuilder;
use crate::types::{CompositeData, NumberType, Value, ValueId};

/// `true` for element-wise operators whose result is a boolean
/// (comparisons and the logical connectives); booleans are stored as
/// Integer cells.
pub fn op_yields_boolean(op: &str) -> bool {
    matches!(op, "eq" | "ne" | "lt" | "lte" | "gt" | "gte" | "and" | "or")
}

/// Result dtype of the element-wise binary operator `op` on operands of
/// dtypes `lhs` and `rhs`, following NumPy's promotion restricted to
/// Zinnia's number types:
///
/// - comparisons and logical connectives yield booleans (Integer cells);
/// - true division `/` (`"div"`) always yields Float, as `np.true_divide`
///   does for integer operands (`np.array([1]) / 2` is `[0.5]`);
/// - otherwise Complex dominates Float, and Float dominates Integer.
///
/// This is the single dtype rule shared by the scalar, static-array and
/// dynamic-array element-wise paths.
pub fn binary_op_result_dtype(op: &str, lhs: NumberType, rhs: NumberType) -> NumberType {
    if op_yields_boolean(op) {
        return NumberType::Integer;
    }
    if lhs == NumberType::Complex || rhs == NumberType::Complex {
        return NumberType::Complex;
    }
    if op == "div" || lhs == NumberType::Float || rhs == NumberType::Float {
        return NumberType::Float;
    }
    NumberType::Integer
}

/// Elementwise numeric binary op for INTERNAL array math on composites.
///
/// `apply_binary_op` implements user-facing semantics, where List/Tuple
/// are Python sequences (`+` concatenates, `*` repeats, `==` is a single
/// Boolean). Lowerings that hold rows/cells in Lists and need ndarray
/// arithmetic must use this instead: it recurses structurally on
/// composites (zipping equal-length List/Tuple sides, broadcasting a
/// scalar side) and only delegates scalars to `apply_binary_op`.
pub fn elementwise_binary_numeric(b: &mut IRBuilder, op: &str, lhs: &Value, rhs: &Value) -> Value {
    let lhs_seq = matches!(lhs, Value::List(_) | Value::Tuple(_) | Value::StaticArray { .. });
    let rhs_seq = matches!(rhs, Value::List(_) | Value::Tuple(_) | Value::StaticArray { .. });
    if !lhs_seq && !rhs_seq {
        return apply_binary_op(b, op, lhs, rhs);
    }
    // StaticArray sides already have native elementwise dispatch.
    if matches!(lhs, Value::StaticArray { .. }) || matches!(rhs, Value::StaticArray { .. }) {
        if let Some(out) = crate::helpers::static_array::elementwise::try_apply_binary_op(b, op, lhs, rhs)
        {
            return out;
        }
    }
    let mat = |b: &mut IRBuilder, v: &Value| -> Value {
        if matches!(v, Value::StaticArray { .. }) {
            crate::helpers::static_array::base::to_value_list(b, v)
        } else {
            v.clone()
        }
    };
    let l = mat(b, lhs);
    let r = mat(b, rhs);
    match (&l, &r) {
        (Value::List(ld) | Value::Tuple(ld), Value::List(rd) | Value::Tuple(rd)) => {
            assert_eq!(
                ld.values.len(),
                rd.values.len(),
                "elementwise_binary_numeric: length mismatch ({} vs {})",
                ld.values.len(),
                rd.values.len()
            );
            let out: Vec<Value> = ld
                .values
                .iter()
                .zip(rd.values.iter())
                .map(|(a, c)| elementwise_binary_numeric(b, op, a, c))
                .collect();
            let types = out.iter().map(|v| v.zinnia_type()).collect();
            Value::List(CompositeData {
                elements_type: types,
                values: out,
                value_id: ValueId::next(),
            })
        }
        (Value::List(ld) | Value::Tuple(ld), _) => {
            let out: Vec<Value> = ld
                .values
                .iter()
                .map(|a| elementwise_binary_numeric(b, op, a, &r))
                .collect();
            let types = out.iter().map(|v| v.zinnia_type()).collect();
            Value::List(CompositeData {
                elements_type: types,
                values: out,
                value_id: ValueId::next(),
            })
        }
        (_, Value::List(rd) | Value::Tuple(rd)) => {
            let out: Vec<Value> = rd
                .values
                .iter()
                .map(|c| elementwise_binary_numeric(b, op, &l, c))
                .collect();
            let types = out.iter().map(|v| v.zinnia_type()).collect();
            Value::List(CompositeData {
                elements_type: types,
                values: out,
                value_id: ValueId::next(),
            })
        }
        _ => apply_binary_op(b, op, &l, &r),
    }
}

/// Whether a runtime-shaped array is statically empty (its runtime length
/// is the constant 0): the contents of its buffer are not data.
fn is_statically_empty(d: &crate::types::DynamicNDArrayData) -> bool {
    d.meta.runtime_length.static_val == Some(0)
}

/// `cond ? t : f` for two runtime-shaped arrays (the value of a variable
/// assigned in both arms of a runtime `if`).
///
/// The merged buffer must hold each element where the selected arm's
/// metadata says it is, for every consumer (aggregations, indexing,
/// elementwise ops, outputs). Both arms are made contiguous and laid out in
/// one common padded buffer: the shape is the per-axis maximum of the arms'
/// buffer shapes (an arm that is statically empty contributes nothing),
/// element `c` of an arm goes to the row-major position of `c` in that
/// shape, and every slot outside the selected arm's data holds the default
/// value (never the other arm's data, which would leak into a public
/// output's padding). The runtime length, rank and shape select on the
/// condition; the strides are the common buffer's.
///
/// Arms with other layouts (compact 2-D buffers with runtime strides, or
/// different ranks with data on both sides) are rejected: the merge used to
/// give them strides `[1; rank]` and a flat buffer, and aggregations and
/// indexing then read the wrong elements.
fn merge_dynamic_arrays(
    b: &mut IRBuilder,
    cond: &Value,
    t: &crate::types::DynamicNDArrayData,
    f: &crate::types::DynamicNDArrayData,
) -> Value {
    use crate::types::{DynArrayMeta, DynamicNDArrayData, NumberType, ScalarValue};
    let td = crate::ops::dyn_ndarray::force_contiguous(b, t);
    let fd = crate::ops::dyn_ndarray::force_contiguous(b, f);
    assert!(
        td.dtype != NumberType::Complex && fd.dtype != NumberType::Complex,
        "select_value: Complex dynamic-array merge is not supported"
    );
    let out_dtype = if td.dtype == NumberType::Float || fd.dtype == NumberType::Float {
        NumberType::Float
    } else {
        NumberType::Integer
    };
    let data_arms: Vec<&DynamicNDArrayData> =
        [&td, &fd].into_iter().filter(|d| !is_statically_empty(d)).collect();
    let rank = data_arms.first().map_or(td.meta.logical_shape.len(), |d| d.meta.logical_shape.len());
    if data_arms.iter().any(|d| d.meta.logical_shape.len() != rank || !crate::ops::dyn_ndarray::has_padded_layout(d)) {
        panic!(
            "a variable assigned runtime-shaped arrays of different ranks or with a compact \
             multi-dimensional layout in the two arms of a runtime `if` is not supported"
        );
    }
    let shape: Vec<usize> = (0..rank)
        .map(|ax| data_arms.iter().map(|d| d.meta.logical_shape[ax]).max().unwrap_or(0))
        .collect();
    let strides = crate::helpers::shape_arith::row_major_strides(&shape);
    let total: usize = shape.iter().product();

    let default_val = crate::ops::dyn_ndarray::metadata::dyn_default_value(b, out_dtype);
    let cell_of = |b: &mut IRBuilder, d: &DynamicNDArrayData, coords: &[usize]| -> Value {
        let inside = !is_statically_empty(d)
            && coords.iter().zip(&d.meta.logical_shape).all(|(c, n)| c < n);
        if !inside {
            return default_val.clone();
        }
        let d_strides = crate::helpers::shape_arith::row_major_strides(&d.meta.logical_shape);
        let pos: usize = coords.iter().zip(&d_strides).map(|(c, s)| c * s).sum();
        let addr = b.ir_constant_int(pos as i64);
        crate::helpers::segment::read_typed(b, d.segment_id, &addr, d.dtype)
    };
    let mut cells: Vec<Value> = Vec::with_capacity(total);
    for flat in 0..total {
        let coords = crate::helpers::shape_arith::decode_coords(flat, &shape, &strides);
        let t_elem = cell_of(b, &td, &coords);
        let f_elem = cell_of(b, &fd, &coords);
        // Scalar recursion promotes mixed int/float to float.
        cells.push(select_value(b, cond, &t_elem, &f_elem));
    }
    let segment_id = crate::helpers::segment::alloc_and_write_values(b, &cells, out_dtype);

    let mut select_scalar = |t: &ScalarValue<i64>, f: &ScalarValue<i64>| -> ScalarValue<i64> {
        let merged = select_value(b, cond, &Value::Integer(t.clone()), &Value::Integer(f.clone()));
        crate::ops::dyn_ndarray::value_to_scalar_i64(&merged)
    };
    let zero = ScalarValue::new(Some(0), None);
    let runtime_length = select_scalar(&td.meta.runtime_length, &fd.meta.runtime_length);
    let runtime_rank = select_scalar(&td.meta.runtime_rank, &fd.meta.runtime_rank);
    let runtime_shape: Vec<ScalarValue<i64>> = (0..rank)
        .map(|ax| {
            let t_dim = td.meta.runtime_shape.get(ax).unwrap_or(&zero).clone();
            let f_dim = fd.meta.runtime_shape.get(ax).unwrap_or(&zero).clone();
            select_scalar(&t_dim, &f_dim)
        })
        .collect();
    let dims = shape
        .iter()
        .map(|&n| crate::types::Dim::new_dynamic(&mut b.dim_table, 0, n))
        .collect();
    let constant = |v: usize| ScalarValue::new(Some(v as i64), None);
    Value::DynamicNDArray(DynamicNDArrayData {
        envelope: crate::types::Envelope::new_with_bound(dims, total),
        dtype: out_dtype,
        segment_id,
        meta: DynArrayMeta {
            logical_shape: shape,
            logical_offset: 0,
            logical_strides: strides.clone(),
            runtime_length,
            runtime_rank,
            runtime_shape,
            runtime_strides: strides.iter().map(|&s| constant(s)).collect(),
            runtime_offset: constant(0),
        },
        value_id: ValueId::next(),
    })
}

/// Conditional select: if cond { tv } else { fv }, with element-wise support.
pub fn select_value(b: &mut IRBuilder, cond: &Value, tv: &Value, fv: &Value) -> Value {
    // A stale operand (an array that no longer holds what NumPy would show,
    // `builder/store.rs`) is read here without an error, and its staleness
    // carried over to the matching part of the result: reading the result
    // is the error (the merge may never be read).
    if !b.has_stale(tv) && !b.has_stale(fv) {
        return select_value_of_fresh(b, cond, tv, fv);
    }
    let out = b.with_stale_reads(|b| select_value_of_fresh(b, cond, tv, fv));
    carry_staleness(b, cond, tv, fv, &out);
    out
}

/// Mark the parts of `out` (a merge of `tv` and `fv`) stale where the
/// matching part of the operand it takes is: element by element through
/// lists and tuples of one length, and the whole of any other part.
fn carry_staleness(b: &mut IRBuilder, cond: &Value, tv: &Value, fv: &Value, out: &Value) {
    match (tv, fv, out) {
        (Value::List(t) | Value::Tuple(t), Value::List(f) | Value::Tuple(f), Value::List(o) | Value::Tuple(o))
            if t.values.len() == o.values.len() && f.values.len() == o.values.len() =>
        {
            for ((t, f), o) in t.values.iter().zip(&f.values).zip(&o.values) {
                carry_staleness(b, cond, t, f, o);
            }
        }
        _ => b.carry_merge_staleness(cond, tv, fv, out),
    }
}

fn select_value_of_fresh(b: &mut IRBuilder, cond: &Value, tv: &Value, fv: &Value) -> Value {
    if std::env::var("ZINNIA_DEBUG_SELECT_VALUE").is_ok() {
        eprintln!("[select_value] tv={:?} fv={:?}", tv.zinnia_type(), fv.zinnia_type());
    }
    // Static condition: pick the branch at compile time, no IR. Keeps
    // statically-resolved `if`s free even though the branch-merge path
    // runs for every conditional assignment.
    if let Some(c) = cond.bool_val().or_else(|| cond.int_val().map(|v| v != 0)) {
        return if c { tv.clone() } else { fv.clone() };
    }
    // P4a: native StaticArray + StaticArray select. Both sides same shape
    // and same dtype → per-cell select into a fresh segment-backed result.
    // Mismatched shapes / dtypes / one-sided StaticArray fall through to
    // the legacy materialised path.
    if let (
        Value::StaticArray {
            dtype: tdt,
            shape: ts,
            ..
        },
        Value::StaticArray {
            dtype: fdt,
            shape: fs,
            ..
        },
    ) = (tv, fv)
    {
        if ts == fs && tdt == fdt && !matches!(tdt, crate::types::NumberType::Complex) {
            let shape = ts.clone();
            let dtype = *tdt;
            let t_cells = crate::helpers::static_array::elementwise::payload_cells(b, tv);
            let f_cells = crate::helpers::static_array::elementwise::payload_cells(b, fv);
            let total: usize = shape.iter().product();
            let mut out: Vec<Value> = Vec::with_capacity(total);
            for i in 0..total {
                out.push(select_value(b, cond, &t_cells[i], &f_cells[i]));
            }
            return crate::helpers::static_array::base::build_static_array_from_flat(
                b, out, shape, dtype,
            );
        }
    }
    if (matches!(tv, Value::StaticArray { .. }) || matches!(fv, Value::StaticArray { .. }))
        && !matches!(tv, Value::DynamicNDArray(_))
        && !matches!(fv, Value::DynamicNDArray(_))
    {
        // A StaticArray side carrying runtime-extents provenance (a
        // dynamic-bounds slice) must keep its segment identity through
        // the merge — materialising to a List would discard the runtime
        // shape and the output contract would leak envelope dims.
        // Promote it (consuming the record) and take the dynamic arm.
        let has_extents = |b: &IRBuilder, v: &Value| {
            matches!(v, Value::StaticArray { segment_id, .. }
                if b.provenance.runtime_extents_of(*segment_id).is_some())
        };
        if has_extents(b, tv) {
            let t2 = crate::helpers::promote::promote_static_to_dynamic_with_extents(b, tv);
            return select_value(b, cond, &t2, fv);
        }
        if has_extents(b, fv) {
            let f2 = crate::helpers::promote::promote_static_to_dynamic_with_extents(b, fv);
            return select_value(b, cond, tv, &f2);
        }
        // Two multi-dimensional arrays of different shapes would merge into
        // a list of rows, which loses the array's semantics: `m[i, j]` reads
        // row `i` and `m[i, j] = v` writes nothing. (1-D arrays of different
        // lengths merge into a runtime-shaped array below.)
        if let (Value::StaticArray { shape: ts, .. }, Value::StaticArray { shape: fs, .. }) = (tv, fv) {
            if ts != fs && (ts.len() > 1 || fs.len() > 1) {
                panic!(
                    "a variable bound to arrays of different shapes on different paths (a runtime \
                     condition, or a loop that may stop early) is not supported for \
                     multi-dimensional arrays ({ts:?} and {fs:?})"
                );
            }
        }
        let tv_n = crate::helpers::static_array::base::to_value_list(b, tv);
        let fv_n = crate::helpers::static_array::base::to_value_list(b, fv);
        return select_value(b, cond, &tv_n, &fv_n);
    }
    match (tv, fv) {
        (Value::List(td), Value::List(fd)) if td.values.len() == fd.values.len() => {
            let results: Vec<Value> = td
                .values
                .iter()
                .zip(fd.values.iter())
                .map(|(t, f)| select_value(b, cond, t, f))
                .collect();
            let types = results.iter().map(|v| v.zinnia_type()).collect();
            Value::List(CompositeData {
                elements_type: types,
                values: results,
                value_id: ValueId::next(),
            })
        }
        (Value::Tuple(td), Value::Tuple(fd)) if td.values.len() == fd.values.len() => {
            let results: Vec<Value> = td
                .values
                .iter()
                .zip(fd.values.iter())
                .map(|(t, f)| select_value(b, cond, t, f))
                .collect();
            let types = results.iter().map(|v| v.zinnia_type()).collect();
            Value::Tuple(CompositeData {
                elements_type: types,
                values: results,
                value_id: ValueId::next(),
            })
        }
        // Dynamic arrays: promote the other side (static array / list /
        // scalar) to dynamic, then merge payload cells and runtime metadata
        // under the condition (`merge_dynamic_arrays`).
        (Value::DynamicNDArray(_), _) | (_, Value::DynamicNDArray(_)) => {
            let to_dyn = |b: &mut IRBuilder, v: &Value| -> crate::types::DynamicNDArrayData {
                match v {
                    Value::DynamicNDArray(d) => d.clone(),
                    other => match crate::helpers::promote::promote_static_to_dynamic(b, other) {
                        Value::DynamicNDArray(d) => d,
                        _ => panic!(
                            "select_value: cannot merge {:?} with a dynamic array",
                            other.zinnia_type()
                        ),
                    },
                }
            };
            let td = to_dyn(b, tv);
            let fd = to_dyn(b, fv);
            merge_dynamic_arrays(b, cond, &td, &fd)
        }
        // Composites with mismatched structure (e.g. an empty np.array([])
        // branch merging with a non-empty result): the honest merged value
        // is a *dynamic* array whose runtime length/shape select on the
        // condition. Promote both sides and re-enter the dynamic-merge arm.
        (
            Value::List(_) | Value::Tuple(_),
            Value::List(_) | Value::Tuple(_),
        ) => {
            let tdyn = crate::helpers::promote::promote_static_to_dynamic(b, tv);
            let fdyn = crate::helpers::promote::promote_static_to_dynamic(b, fv);
            select_value(b, cond, &tdyn, &fdyn)
        }
        // A sequence (an array's elements) against a number: the two values
        // have different shapes (`m = x[::2]` in a loop that may break before
        // it, with `m` a 2-D array), which no merged value represents. It
        // used to take the true side on every path.
        (Value::List(_) | Value::Tuple(_), Value::Integer(_) | Value::Float(_) | Value::Boolean(_) | Value::Complex { .. })
        | (Value::Integer(_) | Value::Float(_) | Value::Boolean(_) | Value::Complex { .. }, Value::List(_) | Value::Tuple(_)) => {
            panic!(
                "a variable bound to arrays of different shapes on different paths (a runtime \
                 condition, or a loop that may stop early) is not supported"
            )
        }
        // Composite vs another kind of value: no meaningful merge; keep the
        // legacy true-side fallback.
        (Value::List(_) | Value::Tuple(_), _) | (_, Value::List(_) | Value::Tuple(_)) => tv.clone(),
        // Complex (2-cell) values: select per component. Promote any
        // numeric scalar on either side to Complex first.
        (Value::Complex { .. }, _) | (_, Value::Complex { .. }) => {
            let (tr, ti) = unpack_to_complex_parts(b, tv);
            let (fr, fi) = unpack_to_complex_parts(b, fv);
            let real = b.ir_select_f(cond, &tr, &fr);
            let imag = b.ir_select_f(cond, &ti, &fi);
            pack_complex(real, imag)
        }
        _ => {
            let tv_is_float = matches!(tv, Value::Float(_));
            let fv_is_float = matches!(fv, Value::Float(_));
            if tv_is_float || fv_is_float {
                let tf = ensure_float(b, tv);
                let ff = ensure_float(b, fv);
                b.ir_select_f(cond, &tf, &ff)
            } else {
                let out = b.ir_select_i(cond, tv, fv);
                // A choice between two NumPy bools is a bool (`np.where(c, m1, m2)`).
                if b.is_bool_cell(tv) && b.is_bool_cell(fv) {
                    b.record_bool_cell(&out);
                }
                out
            }
        }
    }
}

/// Public re-export of `unpack_to_complex_parts` for use by the segment-
/// native Complex paths (P5a). Same semantics: returns a (real_f, imag_f)
/// pair of `Value::Float` ScalarValues for any of {Complex, Float, Integer,
/// Boolean}.
pub fn unpack_value_to_complex_parts(b: &mut IRBuilder, val: &Value) -> (Value, Value) {
    unpack_to_complex_parts(b, val)
}

/// Public re-export of `pack_complex` for the segment-native Complex paths.
pub fn pack_complex_value(real: Value, imag: Value) -> Value {
    pack_complex(real, imag)
}

/// Promote any of {Complex, Float, Integer, Boolean} into a (real_f, imag_f)
/// pair of `Value::Float` ScalarValues that can feed into ir_*_f gates.
fn unpack_to_complex_parts(b: &mut IRBuilder, val: &Value) -> (Value, Value) {
    match val {
        Value::Complex { real, imag } => (Value::Float(real.clone()), Value::Float(imag.clone())),
        Value::Float(_) => (val.clone(), b.ir_constant_float(0.0)),
        Value::Integer(_) | Value::Boolean(_) => {
            let real = b.ir_float_cast(val);
            let imag = b.ir_constant_float(0.0);
            (real, imag)
        }
        _ => panic!(
            "Cannot promote value of type {} to Complex",
            val.zinnia_type()
        ),
    }
}

fn pack_complex(real: Value, imag: Value) -> Value {
    let r = match real {
        Value::Float(s) => s,
        _ => panic!("complex real part must be a Float ScalarValue"),
    };
    let i = match imag {
        Value::Float(s) => s,
        _ => panic!("complex imag part must be a Float ScalarValue"),
    };
    Value::Complex { real: r, imag: i }
}

/// Component-wise binary arithmetic on Complex operands. Each operand is
/// promoted to a 2-tuple of Floats; the result is repackaged as
/// `Value::Complex` (or, for `eq` / `ne`, a single boolean cell).
pub fn apply_complex_binary_op(b: &mut IRBuilder, op: &str, lhs: &Value, rhs: &Value) -> Value {
    let (la, lb) = unpack_to_complex_parts(b, lhs);
    let (ra, rb) = unpack_to_complex_parts(b, rhs);

    match op {
        "add" => {
            let real = b.ir_add_f(&la, &ra);
            let imag = b.ir_add_f(&lb, &rb);
            pack_complex(real, imag)
        }
        "sub" => {
            let real = b.ir_sub_f(&la, &ra);
            let imag = b.ir_sub_f(&lb, &rb);
            pack_complex(real, imag)
        }
        "mul" => {
            // (a + bi)(c + di) = (ac − bd) + (ad + bc)i
            let ac = b.ir_mul_f(&la, &ra);
            let bd = b.ir_mul_f(&lb, &rb);
            let ad = b.ir_mul_f(&la, &rb);
            let bc = b.ir_mul_f(&lb, &ra);
            let real = b.ir_sub_f(&ac, &bd);
            let imag = b.ir_add_f(&ad, &bc);
            pack_complex(real, imag)
        }
        "div" => {
            // (a + bi)/(c + di) = ((ac + bd) + (bc − ad)i) / (c² + d²)
            let ac = b.ir_mul_f(&la, &ra);
            let bd = b.ir_mul_f(&lb, &rb);
            let ad = b.ir_mul_f(&la, &rb);
            let bc = b.ir_mul_f(&lb, &ra);
            let real_num = b.ir_add_f(&ac, &bd);
            let imag_num = b.ir_sub_f(&bc, &ad);
            let cc = b.ir_mul_f(&ra, &ra);
            let dd = b.ir_mul_f(&rb, &rb);
            let denom = b.ir_add_f(&cc, &dd);
            let real = b.ir_div_f(&real_num, &denom);
            let imag = b.ir_div_f(&imag_num, &denom);
            pack_complex(real, imag)
        }
        "eq" => {
            let r_eq = b.ir_equal_f(&la, &ra);
            let i_eq = b.ir_equal_f(&lb, &rb);
            b.ir_logical_and(&r_eq, &i_eq)
        }
        "ne" => {
            let r_ne = b.ir_not_equal_f(&la, &ra);
            let i_ne = b.ir_not_equal_f(&lb, &rb);
            b.ir_logical_or(&r_ne, &i_ne)
        }
        "pow" => {
            // Integer-exponent power via repeated multiplication; arbitrary
            // exponents (transcendental) are tracked in
            // compiler.complex-transcendentals.
            let exp = match rhs.int_val() {
                Some(n) if n >= 0 => n as u32,
                Some(_) => panic!("Complex pow with negative integer exponent not yet supported"),
                None => panic!(
                    "Complex pow requires a compile-time integer exponent; \
                     arbitrary complex powers are tracked in compiler.complex-transcendentals"
                ),
            };
            // Start with 1 + 0i.
            let one = b.ir_constant_float(1.0);
            let zero = b.ir_constant_float(0.0);
            let mut acc = pack_complex(one, zero);
            for _ in 0..exp {
                acc = apply_complex_binary_op(b, "mul", &acc, lhs);
            }
            acc
        }
        "lt" | "lte" | "gt" | "gte" => panic!(
            "Ordering comparison `{}` is not defined on Complex (numpy raises TypeError)",
            op
        ),
        _ => panic!("Unsupported complex binary operator: {}", op),
    }
}

/// Convert a value to a scalar boolean, reducing composites via AND.
pub fn to_scalar_bool(b: &mut IRBuilder, val: &Value) -> Value {
    match val {
        Value::List(data) | Value::Tuple(data) => {
            if data.values.is_empty() {
                return b.ir_constant_bool(true);
            }
            let mut acc = to_scalar_bool(b, &data.values[0]);
            for elem in &data.values[1..] {
                let elem_bool = to_scalar_bool(b, elem);
                acc = b.ir_logical_and(&acc, &elem_bool);
            }
            acc
        }
        // P4a: a `Value::StaticArray` reaches here when an elementwise
        // comparison was native and the result is being used as a
        // condition / assertion. Reduce via AND over the cached payload —
        // matches the legacy List path's semantics.
        Value::StaticArray { .. } => {
            let cells = crate::helpers::static_array::elementwise::payload_cells(b, val);
            if cells.is_empty() {
                return b.ir_constant_bool(true);
            }
            let mut acc = to_scalar_bool(b, &cells[0]);
            for elem in &cells[1..] {
                let elem_bool = to_scalar_bool(b, elem);
                acc = b.ir_logical_and(&acc, &elem_bool);
            }
            acc
        }
        Value::Boolean(_) => val.clone(),
        _ => b.ir_bool_cast(val),
    }
}

/// Apply a binary operation, with element-wise support for composite types.
pub fn apply_binary_op(b: &mut IRBuilder, op: &str, lhs: &Value, rhs: &Value) -> Value {
    if matches!(op, "in" | "not_in") {
        return apply_membership_op(b, op, lhs, rhs);
    }

    // ── StaticArray native dispatch (P4a) ────────────────────────────
    // For two StaticArrays (or one StaticArray + numeric scalar), apply
    // the elementwise op on the segment-backed payload. Falls back to
    // `None` for ops we haven't migrated (e.g. `mat_mul`) or for residual
    // cases (StaticArray + List, Complex on either side, …).
    if matches!(lhs, Value::StaticArray { .. }) || matches!(rhs, Value::StaticArray { .. }) {
        if let Some(out) =
            crate::helpers::static_array::elementwise::try_apply_binary_op(b, op, lhs, rhs)
        {
            return out;
        }
        // NumPy dominance: ndarray ⊕ numeric list applies ndarray
        // semantics (`np.asarray([1,2]) + [10, 20]` is elementwise, not
        // concatenation) — promote the list side and retry the native
        // elementwise path.
        if matches!(lhs, Value::List(_) | Value::Tuple(_)) && is_numeric_composite(lhs) {
            if let Some(l_sa) = crate::helpers::static_array::base::to_static_array(b, lhs) {
                if let Some(out) = crate::helpers::static_array::elementwise::try_apply_binary_op(
                    b, op, &l_sa, rhs,
                ) {
                    return out;
                }
            }
        }
        if matches!(rhs, Value::List(_) | Value::Tuple(_)) && is_numeric_composite(rhs) {
            if let Some(r_sa) = crate::helpers::static_array::base::to_static_array(b, rhs) {
                if let Some(out) = crate::helpers::static_array::elementwise::try_apply_binary_op(
                    b, op, lhs, &r_sa,
                ) {
                    return out;
                }
            }
        }
        // Broadcast fallback with ndarray semantics: the native path
        // doesn't cover this shape pair, but ndarray operands must NEVER
        // pick up the Python sequence semantics below (a broadcast
        // comparison like `isclose(a, a[0])` would collapse to one
        // Boolean). Materialise both sides to the broadcast shape and
        // apply the op structurally.
        if op != "mat_mul" {
            let lhs_n = crate::helpers::static_array::base::to_value_list(b, lhs);
            let rhs_n = crate::helpers::static_array::base::to_value_list(b, rhs);
            let l_ok = is_numeric_composite(&lhs_n) || lhs_n.is_number();
            let r_ok = is_numeric_composite(&rhs_n) || rhs_n.is_number();
            if l_ok && r_ok {
                let ls = crate::helpers::composite::get_composite_shape(&lhs_n);
                let rs = crate::helpers::composite::get_composite_shape(&rhs_n);
                if let Some(out_shape) = crate::helpers::broadcast::broadcast_shapes(&ls, &rs) {
                    let lm = if ls == out_shape {
                        lhs_n.clone()
                    } else {
                        crate::helpers::broadcast::materialize_to_shape(&lhs_n, &out_shape)
                    };
                    let rm = if rs == out_shape {
                        rhs_n.clone()
                    } else {
                        crate::helpers::broadcast::materialize_to_shape(&rhs_n, &out_shape)
                    };
                    return elementwise_binary_numeric(b, op, &lm, &rm);
                }
            }
            return apply_binary_op(b, op, &lhs_n, &rhs_n);
        }
        let ls = crate::helpers::composite::get_composite_shape(lhs);
        let rs = crate::helpers::composite::get_composite_shape(rhs);
        if ls.len() > 2 || rs.len() > 2 || ls.contains(&0) || rs.contains(&0) {
            return crate::helpers::static_array::matmul::matmul(b, lhs, rhs);
        }
        // Residual: materialise to nested List and re-dispatch through the
        // legacy path (the matmul route).
        let lhs_n = crate::helpers::static_array::base::to_value_list(b, lhs);
        let rhs_n = crate::helpers::static_array::base::to_value_list(b, rhs);
        return apply_binary_op(b, op, &lhs_n, &rhs_n);
    }

    // ── Complex dispatch ─────────────────────────────────────────────
    // If either operand is Complex (and the other is a numeric scalar or
    // also Complex), route to component-wise complex arithmetic.
    if matches!(lhs, Value::Complex { .. }) || matches!(rhs, Value::Complex { .. }) {
        let lhs_promotable = matches!(
            lhs,
            Value::Complex { .. } | Value::Float(_) | Value::Integer(_) | Value::Boolean(_)
        );
        let rhs_promotable = matches!(
            rhs,
            Value::Complex { .. } | Value::Float(_) | Value::Integer(_) | Value::Boolean(_)
        );
        if lhs_promotable && rhs_promotable {
            return apply_complex_binary_op(b, op, lhs, rhs);
        }
    }

    // ── DynamicNDArray dispatch ──────────────────────────────────────
    // When either operand is a DynamicNDArray, promote the other and
    // dispatch to the dynamic binary op.
    if matches!(op, "mat_mul")
        && (matches!(lhs, Value::DynamicNDArray(_)) || matches!(rhs, Value::DynamicNDArray(_)))
    {
        panic!(
            "DynamicNDArray matmul is not yet supported for bounded-dynamic arrays; use static arrays or keep matmul outside the bounded-dynamic surface"
        );
    }
    match (lhs, rhs) {
        (Value::DynamicNDArray(ld), Value::DynamicNDArray(rd)) => {
            return crate::ops::dyn_ndarray::binary::dyn_binary_op(b, op, ld, rd);
        }
        (Value::DynamicNDArray(d), other) | (other, Value::DynamicNDArray(d))
            if other.is_number() =>
        {
            let scalar_is_lhs =
                matches!(lhs, Value::Integer(_) | Value::Float(_) | Value::Boolean(_));
            return crate::ops::dyn_ndarray::binary::dyn_scalar_binary_op(
                b,
                op,
                other,
                d,
                scalar_is_lhs,
            );
        }
        (Value::DynamicNDArray(d), other) | (other, Value::DynamicNDArray(d)) => {
            // Static composite + dynamic: promote static to dynamic first.
            let promoted = crate::helpers::promote::promote_static_to_dynamic(b, other);
            let pd = match &promoted {
                Value::DynamicNDArray(pd) => pd,
                _ => unreachable!("promote always returns DynamicNDArray"),
            };
            let lhs_is_dyn = matches!(lhs, Value::DynamicNDArray(_));
            return if lhs_is_dyn {
                crate::ops::dyn_ndarray::binary::dyn_binary_op(b, op, d, pd)
            } else {
                crate::ops::dyn_ndarray::binary::dyn_binary_op(b, op, pd, d)
            };
        }
        _ => {}
    }

    // Python semantics: `+` on lists/tuples is CONCATENATION. NumPy
    // elementwise addition lives on `StaticArray` operands (handled
    // above); since list literals stay `Value::List`, a List here is a
    // Python list, not an ndarray. (The old same-length-elementwise and
    // broadcast carve-outs silently gave `[1, 2] + [3, 4] == [4, 6]`.)
    if op == "add" {
        match (lhs, rhs) {
            (Value::List(ld), Value::List(rd))
            | (Value::Tuple(ld), Value::List(rd))
            | (Value::List(ld), Value::Tuple(rd))
            | (Value::Tuple(ld), Value::Tuple(rd)) => {
                let mut values = ld.values.clone();
                values.extend(rd.values.clone());
                let types = values.iter().map(|v| v.zinnia_type()).collect();
                let is_tuple = matches!(lhs, Value::Tuple(_));
                return if is_tuple {
                    Value::Tuple(CompositeData {
                        elements_type: types,
                        values,
                        value_id: ValueId::next(),
                    })
                } else {
                    Value::List(CompositeData {
                        elements_type: types,
                        values,
                        value_id: ValueId::next(),
                    })
                };
            }
            _ => {}
        }
    }
    // List/tuple repetition via `*` (Python `[1,2] * 3` → `[1,2,1,2,1,2]`).
    // Only fires for *non-numeric* composites — numeric composites are
    // ndarrays, where `scalar * arr` means elementwise multiplication and
    // is handled by the broadcasting arm below.
    if op == "mul" {
        match (lhs, rhs) {
            (Value::List(ld), _) | (Value::Tuple(ld), _) if rhs.int_val().is_some() =>
            {
                let n = rhs.int_val().unwrap().max(0) as usize;
                let mut values = Vec::new();
                for _ in 0..n {
                    values.extend(ld.values.clone());
                }
                let types = values.iter().map(|v| v.zinnia_type()).collect();
                let is_tuple = matches!(lhs, Value::Tuple(_));
                return if is_tuple {
                    Value::Tuple(CompositeData {
                        elements_type: types,
                        values,
                        value_id: ValueId::next(),
                    })
                } else {
                    Value::List(CompositeData {
                        elements_type: types,
                        values,
                        value_id: ValueId::next(),
                    })
                };
            }
            (_, Value::List(rd)) | (_, Value::Tuple(rd)) if lhs.int_val().is_some() =>
            {
                let n = lhs.int_val().unwrap().max(0) as usize;
                let mut values = Vec::new();
                for _ in 0..n {
                    values.extend(rd.values.clone());
                }
                let types = values.iter().map(|v| v.zinnia_type()).collect();
                let is_tuple = matches!(rhs, Value::Tuple(_));
                return if is_tuple {
                    Value::Tuple(CompositeData {
                        elements_type: types,
                        values,
                        value_id: ValueId::next(),
                    })
                } else {
                    Value::List(CompositeData {
                        elements_type: types,
                        values,
                        value_id: ValueId::next(),
                    })
                };
            }
            // As for `+` above, a List here is a Python list: `lst * n`
            // repeats it, and with a runtime `n` its length is unknown.
            // (The broadcasting arm below used to multiply every element,
            // so `[0] * (1 + n)` became `[0]`.)
            (Value::List(_) | Value::Tuple(_), Value::Integer(_))
            | (Value::Integer(_), Value::List(_) | Value::Tuple(_)) => {
                panic!(
                    "sequence repetition `seq * n` with a runtime count `n` is not supported: \
                     a list's length must be known at compile time (use np.zeros/np.full with \
                     a bounded shape, or a compile-time count)"
                );
            }
            _ => {}
        }
    }
    // Composite comparison: handle eq/ne/lt/lte/gt/gte for composites.
    //
    // NumPy-style element-wise comparison fires when both sides are
    // *numeric* composites (purely numeric leaves) with broadcast-compatible
    // shapes — in that case the result is an ndarray of booleans, not a
    // single scalar bool. This is the same heuristic used by the binary
    // arithmetic broadcast path so the two stay consistent.
    //
    // For non-numeric composites (heterogeneous Python lists, lists of
    // strings, tuples of mixed types, etc.) we keep the existing
    // lexicographic-and-reduce semantics handled by `composite_comparison`,
    // which is closer to Python's `list == list` behaviour.
    if matches!(op, "eq" | "ne" | "lt" | "lte" | "gt" | "gte") {
        match (lhs, rhs) {
            (Value::List(ld), Value::List(rd))
            | (Value::Tuple(ld), Value::List(rd))
            | (Value::List(ld), Value::Tuple(rd))
            | (Value::Tuple(ld), Value::Tuple(rd)) => {
                // Python sequence comparison: `==` is a single Boolean
                // (AND of elementwise, False on length mismatch), `<` is
                // lexicographic. NumPy elementwise comparison stays on
                // StaticArray operands.
                return composite_comparison(b, op, ld, rd);
            }
            _ => {}
        }
    }
    // Element-wise: both composite with matching length (for arithmetic ops).
    // `mat_mul` is excluded — even when the two operands happen to share a
    // top-level length they have matmul shape semantics, not element-wise.
    // (E.g. (8,) @ (8, 7) — both have outermost length 8 but the result is
    // (7,), not 8 element-wise scalar matmuls.)
    match (lhs, rhs) {
        (Value::List(ld), Value::List(rd))
        | (Value::Tuple(ld), Value::List(rd))
        | (Value::List(ld), Value::Tuple(rd))
        | (Value::Tuple(ld), Value::Tuple(rd))
            if ld.values.len() == rd.values.len() && op != "mat_mul" =>
        {
            let results: Vec<Value> = ld
                .values
                .iter()
                .zip(rd.values.iter())
                .map(|(l, r)| apply_binary_op(b, op, l, r))
                .collect();
            let types = results.iter().map(|v| v.zinnia_type()).collect();
            return Value::List(CompositeData {
                elements_type: types,
                values: results,
                value_id: ValueId::next(),
            });
        }
        // Shape-level broadcasting: both composites, mismatched shapes that are
        // broadcast-compatible (NumPy semantics). Materialize both operands to
        // the broadcast shape, then recurse — the equal-length arm above will
        // handle the actual element-wise op.
        //
        // This deliberately fires only for *arithmetic* ops on composite ↔
        // composite. Pure list/tuple `+` concatenation and `*` repetition were
        // already handled near the top of this function, so by the time we get
        // here those cases are out of scope. Composite comparison (`eq`/`lt`/…)
        // is dispatched even earlier and is intentionally left untouched —
        // Python lexicographic comparison semantics are preserved.
        // `mat_mul` is excluded — matmul has its own shape rules and error
        // reporting in `static_ndarray_ops::matmul`, reached via the scalar
        // fallback below.
        (Value::List(_) | Value::Tuple(_), Value::List(_) | Value::Tuple(_)) if op != "mat_mul" => {
            let lshape = crate::helpers::composite::get_composite_shape(lhs);
            let rshape = crate::helpers::composite::get_composite_shape(rhs);
            if let Some(out_shape) = crate::helpers::broadcast::broadcast_shapes(&lshape, &rshape) {
                if out_shape != lshape || out_shape != rshape {
                    let l_mat = crate::helpers::broadcast::materialize_to_shape(lhs, &out_shape);
                    let r_mat = crate::helpers::broadcast::materialize_to_shape(rhs, &out_shape);
                    return apply_binary_op(b, op, &l_mat, &r_mat);
                }
                // Same shape but different lengths? Shouldn't happen — fall through.
            } else {
                panic!(
                    "operands could not be broadcast together with shapes {:?} {:?}",
                    lshape, rshape
                );
            }
        }
        // Broadcasting: scalar op composite
        (_, Value::List(rd)) | (_, Value::Tuple(rd)) if lhs.is_number() => {
            let results: Vec<Value> = rd
                .values
                .iter()
                .map(|r| apply_binary_op(b, op, lhs, r))
                .collect();
            let types = results.iter().map(|v| v.zinnia_type()).collect();
            return Value::List(CompositeData {
                elements_type: types,
                values: results,
                value_id: ValueId::next(),
            });
        }
        (Value::List(ld), _) | (Value::Tuple(ld), _) if rhs.is_number() => {
            let results: Vec<Value> = ld
                .values
                .iter()
                .map(|l| apply_binary_op(b, op, l, rhs))
                .collect();
            let types = results.iter().map(|v| v.zinnia_type()).collect();
            return Value::List(CompositeData {
                elements_type: types,
                values: results,
                value_id: ValueId::next(),
            });
        }
        _ => {}
    }
    // Type and dtype comparisons (`x.dtype == np.uint8`) are compile-time:
    // two types, a type and a dtype string (how the fixed-width integer
    // types are represented, see `helpers::dtype`), or two strings.
    let equality = matches!(op, "eq" | "ne");
    let compile_time_equal = match (lhs, rhs) {
        (Value::Class(lt), Value::Class(rt)) => Some(lt == rt),
        (Value::String(ls), Value::String(rs)) if equality => Some(ls.val == rs.val),
        (Value::Class(_), Value::String(_)) | (Value::String(_), Value::Class(_)) if equality => {
            use crate::helpers::dtype::Dtype;
            Some(Dtype::from_value(lhs).is_some() && Dtype::from_value(lhs) == Dtype::from_value(rhs))
        }
        _ => None,
    };
    if let Some(equal) = compile_time_equal {
        return match op {
            "eq" => b.ir_constant_bool(equal),
            "ne" => b.ir_constant_bool(!equal),
            _ => b.ir_constant_bool(false),
        };
    }
    // Scalar operation
    apply_scalar_binary_op(b, op, lhs, rhs)
}

fn apply_membership_op(b: &mut IRBuilder, op: &str, lhs: &Value, rhs: &Value) -> Value {
    let candidates = match rhs {
        Value::List(data) | Value::Tuple(data) => data.values.clone(),
        Value::StaticArray { .. } => {
            crate::helpers::composite::flatten_composite_with_builder(b, rhs)
        }
        Value::DynamicNDArray(_) => {
            panic!(
                "membership over DynamicNDArray is not supported; use a static List, Tuple, or NDArray RHS"
            )
        }
        _ => panic!("membership operator requires a static List, Tuple, or NDArray RHS"),
    };

    let mut found = b.ir_constant_bool(false);
    for candidate in candidates {
        let eq = apply_binary_op(b, "eq", lhs, &candidate);
        let eq_bool = to_scalar_bool(b, &eq);
        if eq_bool.bool_val() == Some(true) {
            found = eq_bool;
            break;
        }
        found = b.ir_logical_or(&found, &eq_bool);
    }

    match op {
        "in" => found,
        "not_in" => b.ir_logical_not(&found),
        _ => unreachable!("membership helper only accepts in/not_in"),
    }
}

pub fn apply_scalar_binary_op(b: &mut IRBuilder, op: &str, lhs: &Value, rhs: &Value) -> Value {
    // Bitwise operators are defined only on integers / booleans. Reject
    // float operands with a clear panic before the float-promotion branch
    // (Python's & / ^ / | on float raises TypeError).
    if matches!(op, "bit_and" | "bit_or" | "bit_xor" | "shl" | "shr") {
        if matches!(lhs, Value::Float(_)) || matches!(rhs, Value::Float(_)) {
            panic!(
                "Bitwise operator `{}` requires integer/boolean operands, got float",
                op
            );
        }
        // NumPy: `&`, `|`, `^` on two booleans are the logical
        // connectives (and stay boolean, e.g. for mask indexing); on
        // integers they are bitwise. (Array operands used to be mapped
        // to the logical connectives wholesale, so `a & 4` on an integer
        // array computed `a != 0 and 4 != 0`.) On operands known to be 0
        // or 1 the two agree, so the logical form is used there too (a
        // `dtype=bool` array's cells may be stored as 0/1 integers).
        let both_boolean = is_zero_or_one(b, lhs) && is_zero_or_one(b, rhs);
        return match op {
            "bit_and" if both_boolean => b.ir_logical_and(lhs, rhs),
            "bit_or" if both_boolean => b.ir_logical_or(lhs, rhs),
            "bit_xor" if both_boolean => b.ir_not_equal_i(lhs, rhs),
            "bit_and" => b.ir_bit_and_i(lhs, rhs),
            "bit_or" => b.ir_bit_or_i(lhs, rhs),
            "bit_xor" => b.ir_bit_xor_i(lhs, rhs),
            "shl" => b.ir_shl_i(lhs, rhs),
            "shr" => b.ir_shr_i(lhs, rhs),
            _ => unreachable!(),
        };
    }

    // True division always computes in Float, even on integer operands
    // (Python and NumPy `/`); see `binary_op_result_dtype`.
    let use_float =
        op == "div" || matches!(lhs, Value::Float(_)) || matches!(rhs, Value::Float(_));
    if use_float {
        // Float base ^ Integer exponent: stay in PowFI so backends can lower
        // to integer-exponent square-and-multiply instead of qexp(b*qlog(a)).
        // Detect BEFORE the implicit float cast erases the type.
        if op == "pow" && matches!(lhs, Value::Float(_)) && matches!(rhs, Value::Integer(_)) {
            return b.ir_pow_f_int(lhs, rhs);
        }
        // Implicit cast: ensure both operands are float
        let lf = ensure_float(b, lhs);
        let rf = ensure_float(b, rhs);
        // Float operations
        match op {
            "add" => b.ir_add_f(&lf, &rf),
            "sub" => b.ir_sub_f(&lf, &rf),
            "mul" => b.ir_mul_f(&lf, &rf),
            "div" => b.ir_div_f(&lf, &rf),
            "pow" => b.ir_pow_f(&lf, &rf),
            "floor_div" => b.ir_floor_div_f(&lf, &rf),
            "mod" => b.ir_mod_f(&lf, &rf),
            "eq" => b.ir_equal_f(&lf, &rf),
            "ne" => {
                let eq = b.ir_equal_f(&lf, &rf);
                b.ir_logical_not(&eq)
            }
            "lt" => b.ir_less_than_f(&lf, &rf),
            "lte" => b.ir_less_than_or_equal_f(&lf, &rf),
            "gt" => b.ir_greater_than_f(&lf, &rf),
            "gte" => b.ir_greater_than_or_equal_f(&lf, &rf),
            "and" => b.ir_logical_and(&lf, &rf),
            "or" => b.ir_logical_or(&lf, &rf),
            "mat_mul" => crate::ops::static_ndarray_ops::matmul(b, &lf, &rf),
            _ => panic!("Unknown binary operator: {}", op),
        }
    } else {
        // Integer operations
        match op {
            "add" => b.ir_add_i(lhs, rhs),
            "sub" => b.ir_sub_i(lhs, rhs),
            "mul" => b.ir_mul_i(lhs, rhs),
            "floor_div" => b.ir_floor_div_i(lhs, rhs),
            "mod" => b.ir_mod_i(lhs, rhs),
            "pow" => b.ir_pow_i(lhs, rhs),
            "eq" => b.ir_equal_i(lhs, rhs),
            "ne" => b.ir_not_equal_i(lhs, rhs),
            "lt" => b.ir_less_than_i(lhs, rhs),
            "lte" => b.ir_less_than_or_equal_i(lhs, rhs),
            "gt" => b.ir_greater_than_i(lhs, rhs),
            "gte" => b.ir_greater_than_or_equal_i(lhs, rhs),
            "and" => b.ir_logical_and(lhs, rhs),
            "or" => b.ir_logical_or(lhs, rhs),
            "mat_mul" => crate::ops::static_ndarray_ops::matmul(b, lhs, rhs),
            _ => panic!("Unknown binary operator: {}", op),
        }
    }
}

/// `v` is a boolean, or an integer the facts and range analysis place in
/// {0, 1}.
fn is_zero_or_one(b: &mut IRBuilder, v: &Value) -> bool {
    match v {
        Value::Boolean(_) => true,
        Value::Integer(_) => b.ask_bounds(v).is_some_and(|(min, max)| min >= 0 && max <= 1),
        _ => false,
    }
}

/// Composite comparison (eq, ne, lt, lte, gt, gte) — element-wise then reduce.
pub fn composite_comparison(
    b: &mut IRBuilder,
    op: &str,
    ld: &CompositeData,
    rd: &CompositeData,
) -> Value {
    let min_len = ld.values.len().min(rd.values.len());
    match op {
        "eq" => {
            if ld.values.len() != rd.values.len() {
                return b.ir_constant_bool(false);
            }
            let mut result = b.ir_constant_bool(true);
            for i in 0..min_len {
                let cmp = apply_binary_op(b, "eq", &ld.values[i], &rd.values[i]);
                let cmp_bool = to_scalar_bool(b, &cmp);
                result = b.ir_logical_and(&result, &cmp_bool);
            }
            result
        }
        "ne" => {
            let eq = composite_comparison(b, "eq", ld, rd);
            b.ir_logical_not(&eq)
        }
        "lt" | "lte" | "gt" | "gte" => {
            // Lexicographic comparison
            // For simplicity, compare element by element
            if min_len == 0 {
                return match op {
                    "lt" => b.ir_constant_bool(ld.values.len() < rd.values.len()),
                    "lte" => b.ir_constant_bool(ld.values.len() <= rd.values.len()),
                    "gt" => b.ir_constant_bool(ld.values.len() > rd.values.len()),
                    "gte" => b.ir_constant_bool(ld.values.len() >= rd.values.len()),
                    _ => unreachable!(),
                };
            }
            // Compare first elements
            let mut result = apply_binary_op(b, op, &ld.values[0], &rd.values[0]);
            for i in 1..min_len {
                // If previous elements were equal, compare this element
                let prev_eq = apply_binary_op(b, "eq", &ld.values[i - 1], &rd.values[i - 1]);
                let prev_eq_bool = to_scalar_bool(b, &prev_eq);
                let this_cmp = apply_binary_op(b, op, &ld.values[i], &rd.values[i]);
                let this_cmp_bool = to_scalar_bool(b, &this_cmp);
                // result = prev_eq ? this_cmp : result
                result = b.ir_select_i(&prev_eq_bool, &this_cmp_bool, &result);
            }
            // Handle different lengths: if all common elements are equal
            if ld.values.len() != rd.values.len() {
                let all_eq = composite_comparison(
                    b,
                    "eq",
                    &CompositeData {
                        elements_type: ld.elements_type[..min_len].to_vec(),
                        values: ld.values[..min_len].to_vec(),
                        value_id: ValueId::next(),
                    },
                    &CompositeData {
                        elements_type: rd.elements_type[..min_len].to_vec(),
                        values: rd.values[..min_len].to_vec(),
                        value_id: ValueId::next(),
                    },
                );
                let all_eq_bool = to_scalar_bool(b, &all_eq);
                let len_result = match op {
                    "lt" | "lte" => b.ir_constant_bool(ld.values.len() < rd.values.len()),
                    "gt" | "gte" => b.ir_constant_bool(ld.values.len() > rd.values.len()),
                    _ => unreachable!(),
                };
                result = b.ir_select_i(&all_eq_bool, &len_result, &result);
            }
            result
        }
        _ => b.ir_constant_bool(false),
    }
}

/// The position the compile-time index `idx` selects in a Python sequence
/// of length `n` (a negative index counts from the end). Python raises
/// `IndexError` for an index outside `-n..n` where the access runs: on a
/// path that always runs this is a compile error; under a runtime execution
/// predicate (a loop item past a runtime bound, a runtime branch) the
/// circuit asserts that the access does not run, and the access uses
/// position 0, observed nowhere. `None` when the access cannot run and the
/// sequence is empty (there is no position to use).
pub fn checked_constant_list_position(b: &mut IRBuilder, idx: i64, n: usize) -> Option<usize> {
    let len = n as i64;
    if -len <= idx && idx < len {
        return Some(if idx < 0 { (idx + len) as usize } else { idx as usize });
    }
    if b.assert_never_runs() {
        return (n > 0).then_some(0);
    }
    panic!("IndexError: index {idx} is out of range for a sequence of length {n}");
}

/// `data[idx]` for a Python list/tuple and a runtime `idx`.
///
/// Python accepts `-n <= idx < n` (negative indices count from the end)
/// and raises `IndexError` otherwise, so the index is wrapped and
/// bounds-checked first (`checked_axis_index`); the element is then
/// picked with a select chain over the `n` positions. Without the check
/// an out-of-range index would silently return the last element.
pub fn dynamic_list_subscript(b: &mut IRBuilder, data: &CompositeData, idx: &Value) -> Value {
    let n = data.values.len();
    let idx = checked_list_index(b, idx, n);
    let mut result = data.values[n - 1].clone();
    for i in (0..n - 1).rev() {
        let const_i = b.ir_constant_int(i as i64);
        let cmp = b.ir_equal_i(&idx, &const_i);
        result = select_value(b, &cmp, &data.values[i], &result);
    }
    result
}

/// `data[idx] = value` for a Python list and a runtime `idx`; returns the
/// updated list. The index is wrapped and bounds-checked as in
/// [`dynamic_list_subscript`], so an out-of-range index is rejected
/// instead of leaving the list unchanged.
pub fn dynamic_list_set_item(
    b: &mut IRBuilder,
    data: &CompositeData,
    idx: &Value,
    value: &Value,
) -> Value {
    let n = data.values.len();
    let idx = checked_list_index(b, idx, n);
    let mut new_values = Vec::with_capacity(n);
    for (i, old) in data.values.iter().enumerate() {
        let const_i = b.ir_constant_int(i as i64);
        let cmp = b.ir_equal_i(&idx, &const_i);
        new_values.push(select_value(b, &cmp, value, old));
    }
    Value::List(CompositeData {
        elements_type: new_values.iter().map(|v| v.zinnia_type()).collect(),
        values: new_values,
        value_id: ValueId::next(),
    })
}

/// Wrap and bounds-check a runtime index into a sequence of length `n`
/// (Python semantics; see `checked_axis_index`). An empty sequence has no
/// valid index at all.
fn checked_list_index(b: &mut IRBuilder, idx: &Value, n: usize) -> Value {
    if n == 0 {
        panic!("IndexError: index into an empty list or tuple");
    }
    crate::helpers::static_array::base::checked_axis_index(b, idx, 0, n, "list_subscript")
}

/// Returns true if `val` is a composite (or scalar) whose every leaf is a
/// numeric value (Integer/Float/Boolean). Used to distinguish "ndarray-like"
/// composites from heterogeneous Python lists, so we can pick NumPy-style
/// broadcasting vs. Python list concatenation in the binary-op dispatcher.
pub fn is_numeric_composite(val: &Value) -> bool {
    match val {
        Value::List(data) | Value::Tuple(data) => data.values.iter().all(is_numeric_composite),
        Value::Integer(_) | Value::Float(_) | Value::Boolean(_) => true,
        _ => false,
    }
}

/// Ensure a value is a float, casting if necessary.
pub fn ensure_float(b: &mut IRBuilder, val: &Value) -> Value {
    match val {
        Value::Float(_) => val.clone(),
        _ => b.ir_float_cast(val),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ZinniaType;

    /// Regression for the `select_value` float-typing bug
    /// (compiler.fix-prover-kernel-float-sqrt — actual fix is in select_value).
    ///
    /// Before the fix, two Float operands fell through to the scalar
    /// `ir_select_i` branch, producing a Value typed as Integer but carrying
    /// Q32-quantised float bits. A subsequent `apply_binary_op("sub", ...)`
    /// against a Float rhs then ran `ensure_float` on the mislabelled lhs,
    /// multiplying its already-quantised cell by another 2^32 and producing a
    /// ~2^32 scaled garbage value rather than the intended difference.
    #[test]
    fn select_value_float_branches_return_float_typed_result() {
        let mut b = IRBuilder::new();
        let cond_true = b.ir_constant_bool(true);
        let cond_false = b.ir_constant_bool(false);
        let tv = b.ir_constant_float(4.0);
        let fv = b.ir_constant_float(1.0);

        let r_true = select_value(&mut b, &cond_true, &tv, &fv);
        let r_false = select_value(&mut b, &cond_false, &tv, &fv);

        // Result must be typed as Float, not Integer. This is the bit the
        // pre-fix code got wrong.
        assert!(matches!(r_true, Value::Float(_)));
        assert!(matches!(r_false, Value::Float(_)));
        assert_eq!(r_true.zinnia_type(), ZinniaType::Float);
        assert_eq!(r_false.zinnia_type(), ZinniaType::Float);

        // Constant folding should pick the right branch end-to-end.
        assert_eq!(r_true.float_val(), Some(4.0));
        assert_eq!(r_false.float_val(), Some(1.0));

        // The downstream effect: `r_true - 4.0` should be ~0.0, not the
        // 2^32-scaled garbage the old path produced.
        let four = b.ir_constant_float(4.0);
        let diff = apply_binary_op(&mut b, "sub", &r_true, &four);
        assert!(matches!(diff, Value::Float(_)));
        let dv = diff.float_val().expect("constant-folded float");
        assert!(dv.abs() < 1e-9, "expected ~0.0, got {dv}");
    }

    #[test]
    fn membership_static_list_or_folds_equalities() {
        let mut b = IRBuilder::new();
        let needle = b.ir_constant_int(3);
        let vals = vec![
            b.ir_constant_int(1),
            b.ir_constant_int(3),
            b.ir_constant_int(5),
        ];
        let rhs = Value::List(CompositeData {
            elements_type: vals.iter().map(|v| v.zinnia_type()).collect(),
            values: vals,
            value_id: ValueId::next(),
        });

        let result = apply_binary_op(&mut b, "in", &needle, &rhs);

        assert_eq!(result.bool_val(), Some(true));
    }

    #[test]
    fn membership_empty_composite_constants() {
        let mut b = IRBuilder::new();
        let needle = b.ir_constant_int(3);
        let rhs = Value::Tuple(CompositeData {
            elements_type: vec![],
            values: vec![],
            value_id: ValueId::next(),
        });

        let is_in = apply_binary_op(&mut b, "in", &needle, &rhs);
        let not_in = apply_binary_op(&mut b, "not_in", &needle, &rhs);

        assert_eq!(is_in.bool_val(), Some(false));
        assert_eq!(not_in.bool_val(), Some(true));
    }
}
