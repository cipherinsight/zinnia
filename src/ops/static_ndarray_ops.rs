use std::collections::HashMap;

use crate::builder::IRBuilder;
use crate::types::{CompositeData, Value, ZinniaType};

pub fn matmul(b: &mut IRBuilder, lhs: &Value, rhs: &Value) -> Value {
    let lhs_shape = crate::helpers::composite::get_composite_shape(lhs);
    let rhs_shape = crate::helpers::composite::get_composite_shape(rhs);

    // Scalar case
    if lhs_shape.is_empty() || rhs_shape.is_empty() {
        return crate::helpers::value_ops::apply_binary_op(b, "mul", lhs, rhs);
    }

    let lhs_cols = *lhs_shape.last().unwrap();
    let rhs_rows = if rhs_shape.len() >= 1 { rhs_shape[0] } else { 1 };

    if lhs_cols != rhs_rows {
        panic!("their shapes are not multiply compatible: ({}) and ({})",
            lhs_shape.iter().map(|s| s.to_string()).collect::<Vec<_>>().join(", "),
            rhs_shape.iter().map(|s| s.to_string()).collect::<Vec<_>>().join(", "),
        );
    }

    // Determine if we should use float ops (if either operand has float elements)
    let lhs_flat = crate::helpers::composite::flatten_composite(lhs);
    let rhs_flat = crate::helpers::composite::flatten_composite(rhs);
    let use_float = lhs_flat.iter().any(|v| matches!(v, Value::Float(_)))
        || rhs_flat.iter().any(|v| matches!(v, Value::Float(_)));

    if let (Value::List(ld), Value::List(rd)) = (lhs, rhs) {
        if rhs_shape.len() == 1 {
            // Matrix @ vector or vector @ vector
            if lhs_shape.len() == 1 {
                // 1D @ 1D: dot product → scalar
                return matmul_dot(b, &ld.values, &rd.values, use_float);
            }
            // 2D @ 1D: each row dot product with vector → 1D
            let mut results = Vec::new();
            for row in &ld.values {
                if let Value::List(row_data) | Value::Tuple(row_data) = row {
                    results.push(matmul_dot(b, &row_data.values, &rd.values, use_float));
                }
            }
            let types = results.iter().map(|v| v.zinnia_type()).collect();
            return Value::List(CompositeData { elements_type: types, values: results });
        }

        if lhs_shape.len() == 2 && rhs_shape.len() == 2 {
            // 2D @ 2D: full matrix multiply
            let m = lhs_shape[0]; // rows of lhs
            let k = lhs_shape[1]; // cols of lhs = rows of rhs
            let n = rhs_shape[1]; // cols of rhs

            let mut rows = Vec::new();
            for i in 0..m {
                let lhs_row = match &ld.values[i] {
                    Value::List(r) | Value::Tuple(r) => &r.values,
                    _ => panic!("matmul: expected 2D array"),
                };
                let mut row_vals = Vec::new();
                for j in 0..n {
                    // Compute dot product of lhs row i with rhs column j
                    let zero = if use_float {
                        b.ir_constant_float(0.0)
                    } else {
                        b.ir_constant_int(0)
                    };
                    let mut acc = zero;
                    for kk in 0..k {
                        let rhs_row = match &rd.values[kk] {
                            Value::List(r) | Value::Tuple(r) => &r.values,
                            _ => panic!("matmul: expected 2D array"),
                        };
                        let prod = if use_float {
                            let a = crate::helpers::value_ops::ensure_float(b, &lhs_row[kk]);
                            let b_val = crate::helpers::value_ops::ensure_float(b, &rhs_row[j]);
                            b.ir_mul_f(&a, &b_val)
                        } else {
                            b.ir_mul_i(&lhs_row[kk], &rhs_row[j])
                        };
                        acc = if use_float {
                            b.ir_add_f(&acc, &prod)
                        } else {
                            b.ir_add_i(&acc, &prod)
                        };
                    }
                    row_vals.push(acc);
                }
                let rtypes = row_vals.iter().map(|v| v.zinnia_type()).collect();
                rows.push(Value::List(CompositeData { elements_type: rtypes, values: row_vals }));
            }
            let row_types = rows.iter().map(|v| v.zinnia_type()).collect();
            return Value::List(CompositeData { elements_type: row_types, values: rows });
        }
    }

    // Fallback: scalar multiply
    crate::helpers::value_ops::apply_binary_op(b, "mul", lhs, rhs)
}

/// Dot product helper for matmul.
pub fn matmul_dot(b: &mut IRBuilder, a: &[Value], bv: &[Value], use_float: bool) -> Value {
    let zero = if use_float {
        b.ir_constant_float(0.0)
    } else {
        b.ir_constant_int(0)
    };
    let mut acc = zero;
    for (x, y) in a.iter().zip(bv.iter()) {
        let prod = if use_float {
            let xf = crate::helpers::value_ops::ensure_float(b, x);
            let yf = crate::helpers::value_ops::ensure_float(b, y);
            b.ir_mul_f(&xf, &yf)
        } else {
            b.ir_mul_i(x, y)
        };
        acc = if use_float {
            b.ir_add_f(&acc, &prod)
        } else {
            b.ir_add_i(&acc, &prod)
        };
    }
    acc
}

/// Element-wise binary operation on two composites (bypasses list concatenation).
pub fn elementwise_binary(b: &mut IRBuilder, op: &str, a: &Value, bv: &Value) -> Value {
    match (a, bv) {
        (Value::List(ad), Value::List(bd)) | (Value::Tuple(ad), Value::List(bd))
        | (Value::List(ad), Value::Tuple(bd)) | (Value::Tuple(ad), Value::Tuple(bd))
            if ad.values.len() == bd.values.len() => {
            let results: Vec<Value> = ad.values.iter().zip(bd.values.iter())
                .map(|(x, y)| elementwise_binary(b, op, x, y))
                .collect();
            let types = results.iter().map(|v| v.zinnia_type()).collect();
            Value::List(CompositeData { elements_type: types, values: results })
        }
        _ => crate::helpers::value_ops::apply_scalar_binary_op(b, op, a, bv),
    }
}

/// Element-wise min or max of two composites.
pub fn elementwise_minmax(b: &mut IRBuilder, a: &Value, bv: &Value, is_max: bool) -> Value {
    match (a, bv) {
        (Value::List(ad), Value::List(bd)) | (Value::Tuple(ad), Value::List(bd))
        | (Value::List(ad), Value::Tuple(bd)) | (Value::Tuple(ad), Value::Tuple(bd))
            if ad.values.len() == bd.values.len() => {
            let results: Vec<Value> = ad.values.iter().zip(bd.values.iter())
                .map(|(x, y)| elementwise_minmax(b, x, y, is_max))
                .collect();
            let types = results.iter().map(|v| v.zinnia_type()).collect();
            Value::List(CompositeData { elements_type: types, values: results })
        }
        _ => {
            let cond = if is_max {
                b.ir_greater_than_i(a, bv)
            } else {
                b.ir_less_than_i(a, bv)
            };
            b.ir_select_i(&cond, a, bv)
        }
    }
}

/// Reduce along a specific axis.
/// For a 2D array with axis=0: reduce columns (result is 1D with same ncols)
/// For a 2D array with axis=1: reduce rows (result is 1D with same nrows)
pub fn reduce_with_axis(b: &mut IRBuilder, op: &str, val: &Value, axis: i64) -> Value {
    if let Value::List(outer) | Value::Tuple(outer) = val {
        let ndim = crate::helpers::composite::get_composite_shape(val).len();
        let axis = if axis < 0 { (ndim as i64 + axis) as usize } else { axis as usize };

        if axis == 0 {
            // Reduce along axis 0: for each column position, reduce across rows
            if outer.values.is_empty() { return Value::None; }
            // Get the number of columns from the first row
            if let Value::List(first_row) | Value::Tuple(first_row) = &outer.values[0] {
                let ncols = first_row.values.len();
                let mut results = Vec::new();
                for col in 0..ncols {
                    // Collect all values in this column
                    let mut col_vals = Vec::new();
                    for row in &outer.values {
                        if let Value::List(rd) | Value::Tuple(rd) = row {
                            if col < rd.values.len() {
                                col_vals.push(rd.values[col].clone());
                            }
                        }
                    }
                    let col_list = Value::List(CompositeData {
                        elements_type: col_vals.iter().map(|v| v.zinnia_type()).collect(),
                        values: col_vals,
                    });
                    results.push(crate::helpers::ndarray::builtin_reduce(b, op, &col_list));
                }
                let types = results.iter().map(|v| v.zinnia_type()).collect();
                Value::List(CompositeData { elements_type: types, values: results })
            } else {
                // If first element is scalar, just reduce the whole thing
                crate::helpers::ndarray::builtin_reduce(b, op, val)
            }
        } else if axis == 1 {
            // Reduce along axis 1: for each row, reduce to scalar
            let mut results = Vec::new();
            for row in &outer.values {
                results.push(crate::helpers::ndarray::builtin_reduce(b, op, row));
            }
            let types = results.iter().map(|v| v.zinnia_type()).collect();
            Value::List(CompositeData { elements_type: types, values: results })
        } else {
            crate::helpers::ndarray::builtin_reduce(b, op, val)
        }
    } else {
        crate::helpers::ndarray::builtin_reduce(b, op, val)
    }
}

// ── Numpy-like helpers ────────────────────────────────────────────

pub fn np_fill(b: &mut IRBuilder, args: &[Value], kwargs: &HashMap<String, Value>, fill_value: i64) -> Value {
    // np.zeros(shape, dtype=...) / np.ones(shape, dtype=...)
    let shape = if let Some(arg) = args.first() {
        match arg {
            Value::Integer(_) => vec![arg.int_val().unwrap_or(0) as usize],
            Value::Tuple(data) => data.values.iter().map(|v| v.int_val().unwrap_or(0) as usize).collect(),
            Value::List(data) => data.values.iter().map(|v| v.int_val().unwrap_or(0) as usize).collect(),
            _ => vec![0],
        }
    } else {
        return Value::None;
    };
    let use_float = matches!(kwargs.get("dtype"), Some(Value::Class(ZinniaType::Float)));
    let total: usize = shape.iter().product();
    let (fill, elem_type) = if use_float {
        (b.ir_constant_float(fill_value as f64), ZinniaType::Float)
    } else {
        (b.ir_constant_int(fill_value), ZinniaType::Integer)
    };
    let values = vec![fill; total];
    let types = vec![elem_type; total];
    build_ndarray_from_flat(b, values, types, &shape)
}

pub fn np_identity(b: &mut IRBuilder, args: &[Value]) -> Value {
    let n = args.first().and_then(|a| a.int_val()).unwrap_or(0) as usize;
    let zero = b.ir_constant_int(0);
    let one = b.ir_constant_int(1);
    let mut rows = Vec::new();
    for i in 0..n {
        let mut row_vals = Vec::new();
        for j in 0..n {
            row_vals.push(if i == j { one.clone() } else { zero.clone() });
        }
        let row_types = vec![ZinniaType::Integer; n];
        rows.push(Value::List(CompositeData { elements_type: row_types, values: row_vals }));
    }
    let types = rows.iter().map(|v| v.zinnia_type()).collect();
    Value::List(CompositeData { elements_type: types, values: rows })
}

pub fn np_arange(b: &mut IRBuilder, args: &[Value]) -> Value {
    builtin_range(b, args)
}

pub fn np_linspace(b: &mut IRBuilder, args: &[Value], kwargs: &HashMap<String, Value>) -> Value {
    if args.len() < 2 { return Value::None; }
    let start = args[0].float_val().or_else(|| args[0].int_val().map(|v| v as f64)).unwrap_or(0.0);
    let stop = args[1].float_val().or_else(|| args[1].int_val().map(|v| v as f64)).unwrap_or(0.0);
    let num = args.get(2).and_then(|v| v.int_val()).or_else(|| kwargs.get("num").and_then(|v| v.int_val())).unwrap_or(50) as usize;
    let endpoint = kwargs.get("endpoint").and_then(|v| v.bool_val()).unwrap_or(true);
    let use_int = matches!(kwargs.get("dtype"), Some(Value::Class(ZinniaType::Integer)));

    if num == 0 {
        return Value::List(CompositeData { elements_type: vec![], values: vec![] });
    }
    if num == 1 {
        let v = if use_int { b.ir_constant_int(start as i64) } else { b.ir_constant_float(start) };
        let t = if use_int { ZinniaType::Integer } else { ZinniaType::Float };
        return Value::List(CompositeData { elements_type: vec![t], values: vec![v] });
    }

    let divisor = if endpoint { (num - 1) as f64 } else { num as f64 };
    let step = (stop - start) / divisor;
    let mut values = Vec::new();
    for i in 0..num {
        let fval = start + step * i as f64;
        if use_int {
            values.push(b.ir_constant_int(fval as i64));
        } else {
            values.push(b.ir_constant_float(fval));
        }
    }
    let elem_type = if use_int { ZinniaType::Integer } else { ZinniaType::Float };
    let types = vec![elem_type; values.len()];
    Value::List(CompositeData { elements_type: types, values })
}

pub fn np_allclose(b: &mut IRBuilder, args: &[Value], kwargs: &HashMap<String, Value>) -> Value {
    // np.allclose(a, b, rtol=1e-05, atol=1e-08) — check |a - b| <= atol + rtol * |b|
    if args.len() < 2 { return Value::None; }

    // Validate argument types
    fn is_valid_allclose_arg(val: &Value) -> bool {
        matches!(val, Value::Integer(_) | Value::Float(_) | Value::Boolean(_) | Value::List(_) | Value::Tuple(_))
    }
    if !is_valid_allclose_arg(&args[0]) {
        panic!("Unsupported argument type for `lhs`");
    }
    if !is_valid_allclose_arg(&args[1]) {
        panic!("Unsupported argument type for `rhs`");
    }
    if let Some(atol) = kwargs.get("atol").or_else(|| args.get(3)) {
        if !is_valid_allclose_arg(atol) {
            panic!("Unsupported argument type for `atol`");
        }
    }
    if let Some(rtol) = kwargs.get("rtol").or_else(|| args.get(2)) {
        if !is_valid_allclose_arg(rtol) {
            panic!("Unsupported argument type for `rtol`");
        }
    }

    // Extract atol and rtol from kwargs or positional args
    let default_rtol = 1e-5_f64;
    let default_atol = 1e-8_f64;

    fn extract_scalar_float(val: &Value) -> Option<f64> {
        match val {
            Value::Float(s) => s.static_val,
            Value::Integer(s) => s.static_val.map(|i| i as f64),
            Value::List(d) | Value::Tuple(d) if d.values.len() == 1 => {
                extract_scalar_float(&d.values[0])
            }
            _ => None,
        }
    }

    let rtol = kwargs.get("rtol")
        .and_then(|v| extract_scalar_float(v))
        .or_else(|| args.get(2).and_then(|v| extract_scalar_float(v)))
        .unwrap_or(default_rtol);
    let atol = kwargs.get("atol")
        .and_then(|v| extract_scalar_float(v))
        .or_else(|| args.get(3).and_then(|v| extract_scalar_float(v)))
        .unwrap_or(default_atol);

    let a_flat = crate::helpers::composite::flatten_composite(&args[0]);
    let b_flat = crate::helpers::composite::flatten_composite(&args[1]);

    // Handle broadcasting: scalar vs array
    let (a_elems, b_elems) = if a_flat.len() == 1 && b_flat.len() > 1 {
        (vec![a_flat[0].clone(); b_flat.len()], b_flat)
    } else if b_flat.len() == 1 && a_flat.len() > 1 {
        let b_repeated = vec![b_flat[0].clone(); a_flat.len()];
        (a_flat, b_repeated)
    } else {
        (a_flat, b_flat)
    };

    if a_elems.len() != b_elems.len() {
        return b.ir_constant_bool(false);
    }

    // For each element: |a - b| <= atol + rtol * |b|
    // Since we're in ZK, use static evaluation for compile-time known values
    let mut result = b.ir_constant_bool(true);
    for (a_val, b_val) in a_elems.iter().zip(b_elems.iter()) {
        let a_f = a_val.float_val().or_else(|| a_val.int_val().map(|i| i as f64));
        let b_f = b_val.float_val().or_else(|| b_val.int_val().map(|i| i as f64));

        if let (Some(av), Some(bv)) = (a_f, b_f) {
            let diff = (av - bv).abs();
            let threshold = atol + rtol * bv.abs();
            let close = diff <= threshold;
            let close_val = b.ir_constant_bool(close);
            result = b.ir_logical_and(&result, &close_val);
        } else {
            // Dynamic: fall back to exact equality
            let eq = b.ir_equal_i(a_val, b_val);
            result = b.ir_logical_and(&result, &eq);
        }
    }
    result
}

pub fn np_concatenate(_b: &mut IRBuilder, args: &[Value], kwargs: &HashMap<String, Value>) -> Value {
    let axis = kwargs.get("axis")
        .or_else(|| args.get(1))
        .and_then(|v| v.int_val())
        .unwrap_or(0);

    if let Some(Value::List(data) | Value::Tuple(data)) = args.first() {
        // Validate axis bounds
        if !data.values.is_empty() {
            let ndim = crate::helpers::composite::get_composite_shape(&data.values[0]).len();
            let resolved_axis = if axis < 0 { ndim as i64 + axis } else { axis };
            if resolved_axis < 0 || resolved_axis >= ndim as i64 {
                panic!("axis {} is out of bounds for array with {} dimensions", axis, ndim);
            }
        }
        if axis == 0 {
            // Concatenate along axis 0: just flatten one level
            let mut all_values = Vec::new();
            let mut all_types = Vec::new();
            for arr in &data.values {
                match arr {
                    Value::List(d) | Value::Tuple(d) => {
                        all_values.extend(d.values.clone());
                        all_types.extend(d.elements_type.clone());
                    }
                    _ => { all_values.push(arr.clone()); all_types.push(arr.zinnia_type()); }
                }
            }
            Value::List(CompositeData { elements_type: all_types, values: all_values })
        } else if axis == 1 {
            // Concatenate along axis 1: merge inner rows
            // [[1,2],[3,4]] + [[5,6],[7,8]] axis=1 → [[1,2,5,6],[3,4,7,8]]
            if data.values.is_empty() { return Value::None; }
            let num_arrays = data.values.len();
            // Get number of rows from first array
            let first = &data.values[0];
            if let Value::List(first_data) | Value::Tuple(first_data) = first {
                let nrows = first_data.values.len();
                let mut result_rows = Vec::new();
                for row_idx in 0..nrows {
                    let mut row_values = Vec::new();
                    let mut row_types = Vec::new();
                    for arr_idx in 0..num_arrays {
                        if let Value::List(arr_data) | Value::Tuple(arr_data) = &data.values[arr_idx] {
                            if row_idx < arr_data.values.len() {
                                match &arr_data.values[row_idx] {
                                    Value::List(rd) | Value::Tuple(rd) => {
                                        row_values.extend(rd.values.clone());
                                        row_types.extend(rd.elements_type.clone());
                                    }
                                    v => { row_values.push(v.clone()); row_types.push(v.zinnia_type()); }
                                }
                            }
                        }
                    }
                    result_rows.push(Value::List(CompositeData { elements_type: row_types, values: row_values }));
                }
                let types = result_rows.iter().map(|v| v.zinnia_type()).collect();
                Value::List(CompositeData { elements_type: types, values: result_rows })
            } else {
                Value::None
            }
        } else {
            Value::None
        }
    } else {
        Value::None
    }
}

pub fn np_stack(_b: &mut IRBuilder, args: &[Value], kwargs: &HashMap<String, Value>) -> Value {
    // np.stack(arrays, axis=0) — stack arrays along a new axis
    let axis = kwargs.get("axis")
        .or_else(|| args.get(1))
        .and_then(|v| v.int_val())
        .unwrap_or(0);

    if let Some(Value::List(data) | Value::Tuple(data)) = args.first() {
        // Validate axis bounds
        if !data.values.is_empty() {
            let ndim = crate::helpers::composite::get_composite_shape(&data.values[0]).len() + 1;
            let resolved_axis = if axis < 0 { ndim as i64 + axis } else { axis };
            if resolved_axis < 0 || resolved_axis >= ndim as i64 {
                panic!("axis {} is out of bounds for array of dimension {}", axis, ndim - 1);
            }
        }
        if axis == 0 {
            // Stack along axis 0: just wrap arrays as rows
            let types = data.values.iter().map(|v| v.zinnia_type()).collect();
            Value::List(CompositeData { elements_type: types, values: data.values.clone() })
        } else if axis == 1 {
            // Stack along axis 1: transpose-like — zip elements from each array
            // e.g., stack([[1,2,3], [4,5,6]], axis=1) = [[1,4],[2,5],[3,6]]
            if let Some(Value::List(first) | Value::Tuple(first)) = data.values.first() {
                let n_elements = first.values.len();
                let mut result = Vec::new();
                for i in 0..n_elements {
                    let mut row = Vec::new();
                    for arr in &data.values {
                        if let Value::List(d) | Value::Tuple(d) = arr {
                            if i < d.values.len() {
                                row.push(d.values[i].clone());
                            }
                        }
                    }
                    let types = row.iter().map(|v| v.zinnia_type()).collect();
                    result.push(Value::List(CompositeData { elements_type: types, values: row }));
                }
                let types = result.iter().map(|v| v.zinnia_type()).collect();
                Value::List(CompositeData { elements_type: types, values: result })
            } else {
                Value::None
            }
        } else {
            // Higher axes — not common, fall back to axis=0
            let types = data.values.iter().map(|v| v.zinnia_type()).collect();
            Value::List(CompositeData { elements_type: types, values: data.values.clone() })
        }
    } else {
        Value::None
    }
}

pub fn build_ndarray_from_flat(b: &mut IRBuilder, values: Vec<Value>, types: Vec<ZinniaType>, shape: &[usize]) -> Value {
    if shape.len() == 1 {
        Value::List(CompositeData { elements_type: types, values })
    } else {
        // Build nested structure
        let inner_size: usize = shape[1..].iter().product();
        let mut rows = Vec::new();
        for chunk in values.chunks(inner_size) {
            let chunk_types = chunk.iter().map(|v| v.zinnia_type()).collect();
            rows.push(build_ndarray_from_flat(b, chunk.to_vec(), chunk_types, &shape[1..]));
        }
        let row_types = rows.iter().map(|v| v.zinnia_type()).collect();
        Value::List(CompositeData { elements_type: row_types, values: rows })
    }
}

// ── NDArray helpers ───────────────────────────────────────────────

/// NDArray reshape: flatten, then rebuild with new shape.
pub fn ndarray_reshape(b: &mut IRBuilder, val: &Value, args: &[Value]) -> Value {
    let flat = crate::helpers::composite::flatten_composite(val);
    let total = flat.len();

    // Parse new shape from args — single tuple arg or multiple int args
    let new_shape: Vec<usize> = if args.len() == 1 {
        match &args[0] {
            Value::Tuple(data) | Value::List(data) => {
                data.values.iter().map(|v| v.int_val().expect("reshape: shape elements must be constant ints") as usize).collect()
            }
            Value::Integer(_) => vec![args[0].int_val().unwrap() as usize],
            _ => panic!("reshape: invalid shape argument"),
        }
    } else {
        args.iter().map(|v| v.int_val().expect("reshape: shape elements must be constant ints") as usize).collect()
    };

    // Handle -1 (infer one dimension)
    let neg_count = new_shape.iter().filter(|&&s| s == usize::MAX).count();
    let final_shape: Vec<usize> = if neg_count == 1 {
        let known_product: usize = new_shape.iter().filter(|&&s| s != usize::MAX).product();
        assert!(known_product > 0 && total % known_product == 0, "reshape: cannot infer dimension");
        new_shape.iter().map(|&s| if s == usize::MAX { total / known_product } else { s }).collect()
    } else {
        // Also handle -1 encoded as a large number from i64 cast
        let has_neg = args.iter().any(|a| a.int_val() == Some(-1));
        if has_neg {
            let known: Vec<usize> = new_shape.iter().copied().collect();
            let known_product: usize = known.iter().filter(|&&s| s < usize::MAX / 2).product();
            assert!(known_product > 0 && total % known_product == 0, "reshape: cannot infer dimension");
            known.iter().map(|&s| if s >= usize::MAX / 2 { total / known_product } else { s }).collect()
        } else {
            new_shape
        }
    };

    let shape_product: usize = final_shape.iter().product();
    assert_eq!(total, shape_product, "reshape: total size mismatch ({} vs {})", total, shape_product);

    let types = flat.iter().map(|v| v.zinnia_type()).collect();
    build_ndarray_from_flat(b, flat, types, &final_shape)
}

/// NDArray moveaxis: reorder axes by moving source axis to destination.
pub fn ndarray_moveaxis(b: &mut IRBuilder, val: &Value, args: &[Value]) -> Value {
    let shape = crate::helpers::composite::get_composite_shape(val);
    let ndim = shape.len();
    assert!(args.len() >= 2, "moveaxis: requires source and destination arguments");

    let src = {
        let s = args[0].int_val().expect("moveaxis: source must be constant int");
        if s < 0 { (ndim as i64 + s) as usize } else { s as usize }
    };
    let dst = {
        let d = args[1].int_val().expect("moveaxis: destination must be constant int");
        if d < 0 { (ndim as i64 + d) as usize } else { d as usize }
    };
    assert!(src < ndim && dst < ndim, "moveaxis: axis out of bounds");

    // Build permutation: remove src, insert at dst
    let mut order: Vec<usize> = (0..ndim).filter(|&i| i != src).collect();
    order.insert(dst, src);

    let axes_val: Vec<Value> = order.iter()
        .map(|&a| Value::Integer(crate::types::ScalarValue::new(Some(a as i64), None)))
        .collect();
    let axes_tuple = Value::Tuple(CompositeData {
        elements_type: vec![ZinniaType::Integer; order.len()],
        values: axes_val,
    });
    crate::helpers::ndarray::ndarray_transpose(b, val, &[axes_tuple])
}

/// NDArray repeat: repeat array elements along an axis.
pub fn ndarray_repeat(b: &mut IRBuilder, val: &Value, args: &[Value], kwargs: &HashMap<String, Value>) -> Value {
    let repeats = args.first()
        .and_then(|v| v.int_val())
        .expect("repeat: repeats must be a constant integer");
    let axis = kwargs.get("axis")
        .or_else(|| args.get(1))
        .and_then(|v| v.int_val());

    if let Some(ax) = axis {
        // Repeat along specific axis
        let shape = crate::helpers::composite::get_composite_shape(val);
        let ndim = shape.len();
        let ax = if ax < 0 { (ndim as i64 + ax) as usize } else { ax as usize };

        if ax == 0 {
            if let Value::List(data) | Value::Tuple(data) = val {
                let mut new_vals = Vec::new();
                for v in &data.values {
                    for _ in 0..repeats {
                        new_vals.push(v.clone());
                    }
                }
                let types = new_vals.iter().map(|v| v.zinnia_type()).collect();
                return Value::List(CompositeData { elements_type: types, values: new_vals });
            }
        }
        // For other axes, transpose so target axis is first, repeat, transpose back
        let mut fwd: Vec<usize> = (0..ndim).collect();
        fwd.swap(0, ax);
        let fwd_vals: Vec<Value> = fwd.iter().map(|&a| Value::Integer(crate::types::ScalarValue::new(Some(a as i64), None))).collect();
        let fwd_tuple = Value::Tuple(CompositeData { elements_type: vec![ZinniaType::Integer; ndim], values: fwd_vals });
        let transposed = crate::helpers::ndarray::ndarray_transpose(b, val, &[fwd_tuple.clone()]);
        let repeated = ndarray_repeat(b, &transposed, args, &HashMap::new());
        crate::helpers::ndarray::ndarray_transpose(b, &repeated, &[fwd_tuple])
    } else {
        // No axis: flatten, then repeat each element
        let flat = crate::helpers::composite::flatten_composite(val);
        let mut new_vals = Vec::new();
        for v in &flat {
            for _ in 0..repeats {
                new_vals.push(v.clone());
            }
        }
        let types = new_vals.iter().map(|v| v.zinnia_type()).collect();
        Value::List(CompositeData { elements_type: types, values: new_vals })
    }
}

/// NDArray filter: select elements where mask is true.
pub fn ndarray_filter(_b: &mut IRBuilder, val: &Value, args: &[Value]) -> Value {
    let mask = args.first().expect("filter: requires a mask argument");
    let elements = crate::helpers::composite::flatten_composite(val);
    let mask_elements = crate::helpers::composite::flatten_composite(mask);
    assert_eq!(elements.len(), mask_elements.len(), "filter: array and mask must have same size");

    // For static arrays, we can build a filtered result at compile time
    // by using select chains. The result length depends on the mask values.
    // If mask values are all statically known, produce a fixed-size result.
    let mut static_result = Vec::new();
    let mut all_static = true;
    for (elem, m) in elements.iter().zip(mask_elements.iter()) {
        match m.int_val().or_else(|| if matches!(m, Value::Boolean(bv) if bv.static_val == Some(true)) { Some(1) } else if matches!(m, Value::Boolean(bv) if bv.static_val == Some(false)) { Some(0) } else { None }) {
            Some(v) if v != 0 => static_result.push(elem.clone()),
            Some(_) => {} // masked out
            None => { all_static = false; break; }
        }
    }

    if all_static {
        let types = static_result.iter().map(|v| v.zinnia_type()).collect();
        Value::List(CompositeData { elements_type: types, values: static_result })
    } else {
        panic!("filter: dynamic masks require DynamicNDArray (not yet supported in Rust backend)");
    }
}

pub fn ndarray_argmax_argmin_with_axis(b: &mut IRBuilder, val: &Value, axis: i64, is_max: bool) -> Value {
    if let Value::List(outer) | Value::Tuple(outer) = val {
        let ndim = crate::helpers::composite::get_composite_shape(val).len();
        let axis = if axis < 0 { (ndim as i64 + axis) as usize } else { axis as usize };

        if axis == 0 {
            // argmax along axis 0: for each column, find row with max/min
            if let Some(Value::List(first_row) | Value::Tuple(first_row)) = outer.values.first() {
                let ncols = first_row.values.len();
                let mut results = Vec::new();
                for col in 0..ncols {
                    let mut best_idx = b.ir_constant_int(0);
                    let mut best_val_opt: Option<Value> = None;
                    for (row_idx, row) in outer.values.iter().enumerate() {
                        if let Value::List(rd) | Value::Tuple(rd) = row {
                            if col < rd.values.len() {
                                if let Some(ref best_val) = best_val_opt {
                                    let cond = if is_max {
                                        b.ir_greater_than_i(&rd.values[col], best_val)
                                    } else {
                                        b.ir_less_than_i(&rd.values[col], best_val)
                                    };
                                    let idx_val = b.ir_constant_int(row_idx as i64);
                                    best_idx = b.ir_select_i(&cond, &idx_val, &best_idx);
                                    best_val_opt = Some(b.ir_select_i(&cond, &rd.values[col], best_val));
                                } else {
                                    best_val_opt = Some(rd.values[col].clone());
                                }
                            }
                        }
                    }
                    results.push(best_idx);
                }
                let types = vec![ZinniaType::Integer; results.len()];
                return Value::List(CompositeData { elements_type: types, values: results });
            }
        } else if axis == 1 {
            // argmax along axis 1: for each row, find column index of max/min
            let mut results = Vec::new();
            for row in &outer.values {
                results.push(crate::helpers::ndarray::ndarray_argmax_argmin(b, row, &[], is_max));
            }
            let types = vec![ZinniaType::Integer; results.len()];
            return Value::List(CompositeData { elements_type: types, values: results });
        }
    }
    crate::helpers::ndarray::ndarray_argmax_argmin(b, val, &[], is_max)
}


pub fn ndarray_shape(val: &Value) -> Value {
    // Return the shape as a tuple of constants
    match val {
        Value::List(data) => {
            // For a list, shape is (len,)
            let len_val = Value::Integer(crate::types::ScalarValue::new(Some(data.values.len() as i64), None));
            Value::Tuple(CompositeData {
                elements_type: vec![ZinniaType::Integer],
                values: vec![len_val],
            })
        }
        Value::NDArray(nd) => {
            let shape_vals: Vec<Value> = nd.shape.iter()
                .map(|&s| Value::Integer(crate::types::ScalarValue::new(Some(s as i64), None)))
                .collect();
            let types = shape_vals.iter().map(|_| ZinniaType::Integer).collect();
            Value::Tuple(CompositeData {
                elements_type: types,
                values: shape_vals,
            })
        }
        _ => Value::None,
    }
}

// ── Builtin helpers (single-use) ─────────────────────────────────────

pub fn builtin_range(b: &mut IRBuilder, args: &[Value]) -> Value {
    let (start, stop, step) = match args.len() {
        1 => (0i64, args[0].int_val().unwrap_or(0), 1i64),
        2 => (args[0].int_val().unwrap_or(0), args[1].int_val().unwrap_or(0), 1i64),
        3 => (args[0].int_val().unwrap_or(0), args[1].int_val().unwrap_or(0), args[2].int_val().unwrap_or(1)),
        _ => return Value::None,
    };
    if step == 0 { return Value::None; }
    let mut values = Vec::new();
    let mut i = start;
    while (step > 0 && i < stop) || (step < 0 && i > stop) {
        values.push(b.ir_constant_int(i));
        i += step;
    }
    let types = vec![ZinniaType::Integer; values.len()];
    Value::List(CompositeData { elements_type: types, values })
}

pub fn builtin_len(b: &mut IRBuilder, args: &[Value]) -> Value {
    if let Some(val) = args.first() {
        match val {
            Value::List(data) | Value::Tuple(data) => {
                b.ir_constant_int(data.values.len() as i64)
            }
            _ => b.ir_constant_int(0),
        }
    } else {
        b.ir_constant_int(0)
    }
}

pub fn builtin_enumerate(b: &mut IRBuilder, iter_val: &Value) -> Value {
    match iter_val {
        Value::List(data) | Value::Tuple(data) => {
            let mut result = Vec::new();
            for (i, elem) in data.values.iter().enumerate() {
                let idx = b.ir_constant_int(i as i64);
                result.push(Value::Tuple(CompositeData {
                    elements_type: vec![ZinniaType::Integer, elem.zinnia_type()],
                    values: vec![idx, elem.clone()],
                }));
            }
            let types = result.iter().map(|v| v.zinnia_type()).collect();
            Value::List(CompositeData { elements_type: types, values: result })
        }
        _ => Value::None,
    }
}
