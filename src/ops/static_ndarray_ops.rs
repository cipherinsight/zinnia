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

/// Recursive concatenation along an arbitrary axis. At axis 0 we splice the
/// outer lists; at axis k > 0 we recurse into each outer position with
/// axis k − 1. All input arrays must have matching shapes except along the
/// concatenation axis (caller is expected to validate).
fn concat_recursive(arrays: &[Value], axis: usize) -> Value {
    if axis == 0 {
        let mut all_values = Vec::new();
        let mut all_types = Vec::new();
        for arr in arrays {
            match arr {
                Value::List(d) | Value::Tuple(d) => {
                    all_values.extend(d.values.clone());
                    all_types.extend(d.elements_type.clone());
                }
                v => {
                    all_values.push(v.clone());
                    all_types.push(v.zinnia_type());
                }
            }
        }
        return Value::List(CompositeData { elements_type: all_types, values: all_values });
    }
    let first = match &arrays[0] {
        Value::List(d) | Value::Tuple(d) => d,
        _ => panic!("concatenate: cannot apply axis > 0 to a 0-D value"),
    };
    let outer_len = first.values.len();
    let mut rows = Vec::with_capacity(outer_len);
    for i in 0..outer_len {
        let inner: Vec<Value> = arrays
            .iter()
            .map(|a| match a {
                Value::List(d) | Value::Tuple(d) => d.values[i].clone(),
                _ => panic!("concatenate: arrays must have matching ranks"),
            })
            .collect();
        rows.push(concat_recursive(&inner, axis - 1));
    }
    let types = rows.iter().map(|v| v.zinnia_type()).collect();
    Value::List(CompositeData { elements_type: types, values: rows })
}

pub fn np_concatenate(_b: &mut IRBuilder, args: &[Value], kwargs: &HashMap<String, Value>) -> Value {
    let raw_axis = kwargs
        .get("axis")
        .or_else(|| args.get(1))
        .and_then(|v| v.int_val())
        .unwrap_or(0);

    let data = match args.first() {
        Some(Value::List(d)) | Some(Value::Tuple(d)) => d,
        _ => return Value::None,
    };
    if data.values.is_empty() {
        return Value::None;
    }

    let ndim = crate::helpers::composite::get_composite_shape(&data.values[0]).len();
    let resolved = if raw_axis < 0 { ndim as i64 + raw_axis } else { raw_axis };
    if resolved < 0 || resolved >= ndim as i64 {
        panic!(
            "axis {} is out of bounds for array with {} dimensions",
            raw_axis, ndim
        );
    }

    concat_recursive(&data.values, resolved as usize)
}

/// Recursive stack along an arbitrary new axis. At axis 0 we wrap the input
/// arrays as the outer list directly. At axis k > 0 we walk the *first* axis
/// of the input arrays (which all have matching shapes) and recurse with
/// axis k − 1. The recursion bottoms out either in axis-0 wrap or in scalars.
fn stack_recursive(arrays: &[Value], axis: usize) -> Value {
    if axis == 0 {
        let types = arrays.iter().map(|v| v.zinnia_type()).collect();
        return Value::List(CompositeData {
            elements_type: types,
            values: arrays.to_vec(),
        });
    }
    let first = match &arrays[0] {
        Value::List(d) | Value::Tuple(d) => d,
        _ => panic!("stack: cannot stack at axis > input rank"),
    };
    let outer_len = first.values.len();
    let mut rows = Vec::with_capacity(outer_len);
    for i in 0..outer_len {
        let inner: Vec<Value> = arrays
            .iter()
            .map(|a| match a {
                Value::List(d) | Value::Tuple(d) => d.values[i].clone(),
                _ => panic!("stack: all input arrays must have the same rank"),
            })
            .collect();
        rows.push(stack_recursive(&inner, axis - 1));
    }
    let types = rows.iter().map(|v| v.zinnia_type()).collect();
    Value::List(CompositeData { elements_type: types, values: rows })
}

pub fn np_stack(_b: &mut IRBuilder, args: &[Value], kwargs: &HashMap<String, Value>) -> Value {
    let raw_axis = kwargs
        .get("axis")
        .or_else(|| args.get(1))
        .and_then(|v| v.int_val())
        .unwrap_or(0);

    let data = match args.first() {
        Some(Value::List(d)) | Some(Value::Tuple(d)) => d,
        _ => return Value::None,
    };
    if data.values.is_empty() {
        return Value::None;
    }

    // Stack inserts a new axis, so the result rank is input_rank + 1.
    let ndim = crate::helpers::composite::get_composite_shape(&data.values[0]).len() + 1;
    let resolved = if raw_axis < 0 { ndim as i64 + raw_axis } else { raw_axis };
    if resolved < 0 || resolved >= ndim as i64 {
        panic!(
            "axis {} is out of bounds for array of dimension {}",
            raw_axis,
            ndim - 1
        );
    }

    stack_recursive(&data.values, resolved as usize)
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

// ────────────────────────────────────────────────────────────────────────
// Shape-manipulation helpers (swapaxes / flip / squeeze / expand_dims /
// broadcast_to / atleast_Nd / tile / vstack / hstack / dstack /
// column_stack / row_stack)
// ────────────────────────────────────────────────────────────────────────

// `resolve_axis` lives in `helpers::shape_arith`. Re-export under the local
// path so the rest of this file can call it unqualified, exactly as before.
use crate::helpers::shape_arith::resolve_axis;

/// Build a constant Integer Value from a usize.
fn const_int(b: &mut IRBuilder, n: usize) -> Value {
    b.ir_constant_int(n as i64)
}

/// `np.swapaxes(arr, a1, a2)` — swap two axes.
pub fn ndarray_swapaxes(b: &mut IRBuilder, val: &Value, args: &[Value]) -> Value {
    let shape = crate::helpers::composite::get_composite_shape(val);
    let ndim = shape.len();
    assert!(args.len() >= 2, "swapaxes: requires two axis arguments");
    let a1 = resolve_axis(
        args[0].int_val().expect("swapaxes: axis must be a constant int"),
        ndim,
        "swapaxes",
    );
    let a2 = resolve_axis(
        args[1].int_val().expect("swapaxes: axis must be a constant int"),
        ndim,
        "swapaxes",
    );
    let mut order: Vec<usize> = (0..ndim).collect();
    order.swap(a1, a2);
    let axes_vals: Vec<Value> = order.iter().map(|&a| const_int(b, a)).collect();
    let axes_tuple = Value::Tuple(CompositeData {
        elements_type: vec![ZinniaType::Integer; ndim],
        values: axes_vals,
    });
    crate::helpers::ndarray::ndarray_transpose(b, val, &[axes_tuple])
}

/// Reverse `val`'s elements along axis `axis`. Other axes keep order.
fn flip_along(val: &Value, axis: usize) -> Value {
    if axis == 0 {
        match val {
            Value::List(d) | Value::Tuple(d) => {
                let new_vals: Vec<Value> = d.values.iter().rev().cloned().collect();
                let new_types = new_vals.iter().map(|v| v.zinnia_type()).collect();
                Value::List(CompositeData {
                    elements_type: new_types,
                    values: new_vals,
                })
            }
            _ => val.clone(),
        }
    } else {
        match val {
            Value::List(d) | Value::Tuple(d) => {
                let new_vals: Vec<Value> =
                    d.values.iter().map(|v| flip_along(v, axis - 1)).collect();
                let new_types = new_vals.iter().map(|v| v.zinnia_type()).collect();
                Value::List(CompositeData {
                    elements_type: new_types,
                    values: new_vals,
                })
            }
            _ => val.clone(),
        }
    }
}

/// `np.flip(arr, axis=None)` — reverse along the given axis (or all axes
/// when no axis is specified).
pub fn np_flip(_b: &mut IRBuilder, args: &[Value], kwargs: &HashMap<String, Value>) -> Value {
    let val = args.first().expect("flip: requires an array argument");
    let shape = crate::helpers::composite::get_composite_shape(val);
    let ndim = shape.len();
    let axis_val = kwargs.get("axis").or_else(|| args.get(1));
    let axes: Vec<usize> = match axis_val {
        Some(Value::None) | None => (0..ndim).collect(),
        Some(Value::List(d)) | Some(Value::Tuple(d)) => d
            .values
            .iter()
            .map(|v| {
                resolve_axis(
                    v.int_val().expect("flip: axis values must be constant ints"),
                    ndim,
                    "flip",
                )
            })
            .collect(),
        Some(a) => vec![resolve_axis(
            a.int_val().expect("flip: axis must be a constant int"),
            ndim,
            "flip",
        )],
    };
    let mut out = val.clone();
    for ax in axes {
        out = flip_along(&out, ax);
    }
    out
}

/// `np.flipud(arr)` — flip along axis 0.
pub fn np_flipud(_b: &mut IRBuilder, args: &[Value]) -> Value {
    let val = args.first().expect("flipud: requires an array argument");
    let shape = crate::helpers::composite::get_composite_shape(val);
    if shape.is_empty() {
        panic!("flipud: input must be at least 1-D");
    }
    flip_along(val, 0)
}

/// `np.fliplr(arr)` — flip along axis 1 (requires ndim ≥ 2).
pub fn np_fliplr(_b: &mut IRBuilder, args: &[Value]) -> Value {
    let val = args.first().expect("fliplr: requires an array argument");
    let shape = crate::helpers::composite::get_composite_shape(val);
    if shape.len() < 2 {
        panic!("fliplr: input must be at least 2-D");
    }
    flip_along(val, 1)
}

/// `np.rot90(arr, k=1, axes=(0, 1))` — rotate 90° counter-clockwise k times
/// in the plane spanned by `axes`. Each k=1 rotation is `flip(axes[1])`
/// then `swapaxes(axes[0], axes[1])`, matching NumPy's reference.
pub fn np_rot90(b: &mut IRBuilder, args: &[Value], kwargs: &HashMap<String, Value>) -> Value {
    let val = args.first().expect("rot90: requires an array argument");
    let shape = crate::helpers::composite::get_composite_shape(val);
    let ndim = shape.len();
    if ndim < 2 {
        panic!("rot90: input must be at least 2-D");
    }
    let k = kwargs
        .get("k")
        .or_else(|| args.get(1))
        .and_then(|v| v.int_val())
        .unwrap_or(1);
    let axes_arg = kwargs.get("axes").or_else(|| args.get(2));
    let (a0, a1) = match axes_arg {
        Some(Value::Tuple(d)) | Some(Value::List(d)) if d.values.len() == 2 => {
            let a = resolve_axis(d.values[0].int_val().unwrap_or(0), ndim, "rot90");
            let bb = resolve_axis(d.values[1].int_val().unwrap_or(1), ndim, "rot90");
            (a, bb)
        }
        _ => (0usize, 1usize),
    };
    if a0 == a1 {
        panic!("rot90: axes must be different");
    }
    let k = ((k % 4) + 4) % 4;
    let mut out = val.clone();
    for _ in 0..k {
        out = flip_along(&out, a1);
        let mut order: Vec<usize> = (0..ndim).collect();
        order.swap(a0, a1);
        let axes_vals: Vec<Value> = order.iter().map(|&a| const_int(b, a)).collect();
        let axes_tuple = Value::Tuple(CompositeData {
            elements_type: vec![ZinniaType::Integer; ndim],
            values: axes_vals,
        });
        out = crate::helpers::ndarray::ndarray_transpose(b, &out, &[axes_tuple]);
    }
    out
}

/// `np.squeeze(arr, axis=None)` — drop axes of length 1.
pub fn np_squeeze(_b: &mut IRBuilder, args: &[Value], kwargs: &HashMap<String, Value>) -> Value {
    let val = args.first().expect("squeeze: requires an array argument");
    let shape = crate::helpers::composite::get_composite_shape(val);
    let ndim = shape.len();
    let axis_val = kwargs.get("axis").or_else(|| args.get(1));

    let target_axes: Vec<usize> = match axis_val {
        Some(Value::None) | None => shape
            .iter()
            .enumerate()
            .filter_map(|(i, &d)| if d == 1 { Some(i) } else { None })
            .collect(),
        Some(Value::Tuple(d)) | Some(Value::List(d)) => d
            .values
            .iter()
            .map(|v| {
                resolve_axis(
                    v.int_val().expect("squeeze: axis must be a constant int"),
                    ndim,
                    "squeeze",
                )
            })
            .collect(),
        Some(a) => vec![resolve_axis(
            a.int_val().expect("squeeze: axis must be a constant int"),
            ndim,
            "squeeze",
        )],
    };
    for &ax in &target_axes {
        if shape[ax] != 1 {
            panic!(
                "squeeze: cannot select an axis to squeeze out which has size not equal to one (axis {})",
                ax
            );
        }
    }
    if target_axes.is_empty() {
        return val.clone();
    }
    let new_shape: Vec<usize> = shape
        .iter()
        .enumerate()
        .filter_map(|(i, &d)| if target_axes.contains(&i) { None } else { Some(d) })
        .collect();
    let flat = crate::helpers::composite::flatten_composite(val);
    if new_shape.is_empty() {
        return flat.into_iter().next().unwrap_or(Value::None);
    }
    let types = flat.iter().map(|v| v.zinnia_type()).collect();
    crate::helpers::composite::build_nested_value(flat, types, &new_shape)
}

/// `np.expand_dims(arr, axis)` — insert a new axis of length 1 at `axis`.
pub fn np_expand_dims(_b: &mut IRBuilder, args: &[Value]) -> Value {
    let val = args.first().expect("expand_dims: requires an array argument");
    let shape = crate::helpers::composite::get_composite_shape(val);
    let ndim = shape.len();
    let axis = args
        .get(1)
        .and_then(|v| v.int_val())
        .expect("expand_dims: axis must be a constant int");
    let new_ndim = ndim + 1;
    let resolved = if axis < 0 { new_ndim as i64 + axis } else { axis };
    if resolved < 0 || resolved >= new_ndim as i64 {
        panic!(
            "expand_dims: axis {} is out of bounds for array of rank {}",
            axis, new_ndim
        );
    }
    let pos = resolved as usize;
    fn insert_at(val: &Value, pos: usize) -> Value {
        if pos == 0 {
            Value::List(CompositeData {
                elements_type: vec![val.zinnia_type()],
                values: vec![val.clone()],
            })
        } else {
            match val {
                Value::List(d) | Value::Tuple(d) => {
                    let new_vals: Vec<Value> =
                        d.values.iter().map(|v| insert_at(v, pos - 1)).collect();
                    let new_types = new_vals.iter().map(|v| v.zinnia_type()).collect();
                    Value::List(CompositeData {
                        elements_type: new_types,
                        values: new_vals,
                    })
                }
                _ => val.clone(),
            }
        }
    }
    insert_at(val, pos)
}

/// `np.broadcast_to(arr, shape)` — materialize the broadcast to a target
/// shape. Thin wrapper around the broadcasting helper.
pub fn np_broadcast_to(_b: &mut IRBuilder, args: &[Value]) -> Value {
    let val = args.first().expect("broadcast_to: requires an array argument");
    let shape_arg = args.get(1).expect("broadcast_to: requires a shape argument");
    let target: Vec<usize> = match shape_arg {
        Value::Tuple(d) | Value::List(d) => d
            .values
            .iter()
            .map(|v| {
                v.int_val()
                    .expect("broadcast_to: shape elements must be constant ints")
                    as usize
            })
            .collect(),
        Value::Integer(_) => vec![shape_arg.int_val().unwrap() as usize],
        _ => panic!("broadcast_to: invalid shape argument"),
    };
    let src_shape = crate::helpers::composite::get_composite_shape(val);
    match crate::helpers::broadcast::broadcast_shapes(&src_shape, &target) {
        Some(s) if s == target => {}
        _ => panic!(
            "broadcast_to: shape {:?} cannot be broadcast to {:?}",
            src_shape, target
        ),
    }
    crate::helpers::broadcast::materialize_to_shape(val, &target)
}

/// `np.atleast_1d/2d/3d(arr)` — prepend unit axes until rank ≥ n.
pub fn np_atleast_nd(_b: &mut IRBuilder, args: &[Value], n: usize) -> Value {
    let val = args.first().expect("atleast_Nd: requires an array argument");
    let shape = crate::helpers::composite::get_composite_shape(val);
    if shape.len() >= n {
        return val.clone();
    }
    let mut out = val.clone();
    for _ in 0..(n - shape.len()) {
        out = Value::List(CompositeData {
            elements_type: vec![out.zinnia_type()],
            values: vec![out],
        });
    }
    out
}

/// `np.tile(arr, reps)` — repeat `arr` according to `reps`.
pub fn np_tile(_b: &mut IRBuilder, args: &[Value]) -> Value {
    let val = args.first().expect("tile: requires an array argument");
    let reps_arg = args.get(1).expect("tile: requires a reps argument");
    let reps: Vec<usize> = match reps_arg {
        Value::Tuple(d) | Value::List(d) => d
            .values
            .iter()
            .map(|v| {
                v.int_val()
                    .expect("tile: reps elements must be constant ints")
                    .max(0) as usize
            })
            .collect(),
        Value::Integer(_) => vec![reps_arg.int_val().unwrap().max(0) as usize],
        _ => panic!("tile: invalid reps argument"),
    };
    let src_shape = crate::helpers::composite::get_composite_shape(val);
    let rank = src_shape.len().max(reps.len());
    let mut padded_shape = vec![1usize; rank - src_shape.len()];
    padded_shape.extend_from_slice(&src_shape);
    let mut padded_reps = vec![1usize; rank - reps.len()];
    padded_reps.extend_from_slice(&reps);

    // Promote val to padded rank by prepending unit axes if needed.
    let mut promoted = val.clone();
    for _ in 0..(rank - src_shape.len()) {
        promoted = Value::List(CompositeData {
            elements_type: vec![promoted.zinnia_type()],
            values: vec![promoted],
        });
    }

    let target_shape: Vec<usize> = padded_shape
        .iter()
        .zip(padded_reps.iter())
        .map(|(s, r)| s * r)
        .collect();

    let total: usize = target_shape.iter().product();
    let mut out_strides = vec![1usize; rank];
    for i in (0..rank.saturating_sub(1)).rev() {
        out_strides[i] = out_strides[i + 1] * target_shape[i + 1];
    }
    let mut src_strides = vec![1usize; rank];
    for i in (0..rank.saturating_sub(1)).rev() {
        src_strides[i] = src_strides[i + 1] * padded_shape[i + 1];
    }
    let flat_src = crate::helpers::composite::flatten_composite(&promoted);
    let mut out_flat: Vec<Value> = Vec::with_capacity(total);
    for out_idx in 0..total {
        let mut remainder = out_idx;
        let mut src_flat = 0usize;
        for d in 0..rank {
            let coord = remainder / out_strides[d];
            remainder %= out_strides[d];
            let src_coord = coord % padded_shape[d];
            src_flat += src_coord * src_strides[d];
        }
        out_flat.push(flat_src[src_flat].clone());
    }
    let types = out_flat.iter().map(|v| v.zinnia_type()).collect();
    crate::helpers::composite::build_nested_value(out_flat, types, &target_shape)
}

// ── stack convenience wrappers ─────────────────────────────────────────

/// Promote a 1-D array to a 2-D row (`(N,)` → `(1, N)`); leave higher-rank
/// arrays untouched. Used by vstack/row_stack.
fn promote_to_row(val: &Value) -> Value {
    let shape = crate::helpers::composite::get_composite_shape(val);
    if shape.len() < 2 {
        Value::List(CompositeData {
            elements_type: vec![val.zinnia_type()],
            values: vec![val.clone()],
        })
    } else {
        val.clone()
    }
}

/// Promote a 1-D array to a 2-D column (`(N,)` → `(N, 1)`).
fn promote_to_column(val: &Value) -> Value {
    let shape = crate::helpers::composite::get_composite_shape(val);
    if shape.len() == 1 {
        if let Value::List(d) | Value::Tuple(d) = val {
            let new_vals: Vec<Value> = d
                .values
                .iter()
                .map(|v| {
                    Value::List(CompositeData {
                        elements_type: vec![v.zinnia_type()],
                        values: vec![v.clone()],
                    })
                })
                .collect();
            let types = new_vals.iter().map(|v| v.zinnia_type()).collect();
            return Value::List(CompositeData {
                elements_type: types,
                values: new_vals,
            });
        }
    }
    val.clone()
}

/// `np.vstack(arrays)` — stack along axis 0, promoting 1-D inputs to rows.
pub fn np_vstack(_b: &mut IRBuilder, args: &[Value]) -> Value {
    let arrays = match args.first() {
        Some(Value::List(d)) | Some(Value::Tuple(d)) => d,
        _ => return Value::None,
    };
    let promoted: Vec<Value> = arrays.values.iter().map(promote_to_row).collect();
    concat_recursive(&promoted, 0)
}

/// `np.hstack(arrays)` — concatenate along axis 1 for ≥2-D inputs, or along
/// axis 0 for 1-D inputs (NumPy convention).
pub fn np_hstack(_b: &mut IRBuilder, args: &[Value]) -> Value {
    let arrays = match args.first() {
        Some(Value::List(d)) | Some(Value::Tuple(d)) => d,
        _ => return Value::None,
    };
    let any_multi = arrays
        .values
        .iter()
        .any(|v| crate::helpers::composite::get_composite_shape(v).len() >= 2);
    let axis = if any_multi { 1 } else { 0 };
    concat_recursive(&arrays.values, axis)
}

/// `np.dstack(arrays)` — stack along axis 2, promoting lower-rank inputs.
pub fn np_dstack(b: &mut IRBuilder, args: &[Value]) -> Value {
    let arrays = match args.first() {
        Some(Value::List(d)) | Some(Value::Tuple(d)) => d,
        _ => return Value::None,
    };
    let promoted: Vec<Value> = arrays
        .values
        .iter()
        .map(|v| {
            let shape = crate::helpers::composite::get_composite_shape(v);
            match shape.len() {
                1 => {
                    // (N,) -> (1, N) -> (1, N, 1)
                    let row = Value::List(CompositeData {
                        elements_type: vec![v.zinnia_type()],
                        values: vec![v.clone()],
                    });
                    let two = b.ir_constant_int(2);
                    np_expand_dims(b, &[row, two])
                }
                2 => {
                    let two = b.ir_constant_int(2);
                    np_expand_dims(b, &[v.clone(), two])
                }
                _ => v.clone(),
            }
        })
        .collect();
    concat_recursive(&promoted, 2)
}

/// `np.column_stack(arrays)` — 1-D arrays become columns of a 2-D output.
pub fn np_column_stack(_b: &mut IRBuilder, args: &[Value]) -> Value {
    let arrays = match args.first() {
        Some(Value::List(d)) | Some(Value::Tuple(d)) => d,
        _ => return Value::None,
    };
    let promoted: Vec<Value> = arrays.values.iter().map(promote_to_column).collect();
    concat_recursive(&promoted, 1)
}

/// `np.row_stack(arrays)` — alias of vstack.
pub fn np_row_stack(b: &mut IRBuilder, args: &[Value]) -> Value {
    np_vstack(b, args)
}

// ────────────────────────────────────────────────────────────────────────
// Element-wise math: round / floor / ceil / trunc / reciprocal / where /
// clip. None of these have a dedicated IR primitive yet, so they are
// expressed in terms of existing ops (floor_div, sign, select, etc).
// ────────────────────────────────────────────────────────────────────────

/// Recursively apply `scalar` to every leaf in `val`. Used by all the
/// element-wise wrappers below — keeps the leaf-walking boilerplate in
/// one place.
fn vectorize_unary<F: FnMut(&mut IRBuilder, &Value) -> Value>(
    b: &mut IRBuilder,
    val: &Value,
    f: &mut F,
) -> Value {
    match val {
        Value::List(d) | Value::Tuple(d) => {
            let vals: Vec<Value> =
                d.values.iter().map(|v| vectorize_unary(b, v, f)).collect();
            let types = vals.iter().map(|v| v.zinnia_type()).collect();
            Value::List(CompositeData {
                elements_type: types,
                values: vals,
            })
        }
        _ => f(b, val),
    }
}

/// `np.floor(x)` — round towards negative infinity. For floats this is
/// `floor_div(x, 1.0)`; integers and booleans pass through unchanged.
pub fn np_floor(b: &mut IRBuilder, args: &[Value]) -> Value {
    let val = args.first().expect("floor: requires an argument");
    vectorize_unary(b, val, &mut |b, x| match x {
        Value::Float(_) => {
            let one = b.ir_constant_float(1.0);
            b.ir_floor_div_f(x, &one)
        }
        Value::Integer(_) | Value::Boolean(_) => x.clone(),
        _ => panic!("floor: unsupported type {:?}", x.zinnia_type()),
    })
}

/// `np.ceil(x)` — round towards positive infinity. Implemented as
/// `-floor(-x)`.
pub fn np_ceil(b: &mut IRBuilder, args: &[Value]) -> Value {
    let val = args.first().expect("ceil: requires an argument");
    vectorize_unary(b, val, &mut |b, x| match x {
        Value::Float(_) => {
            let zero = b.ir_constant_float(0.0);
            let one = b.ir_constant_float(1.0);
            let neg = b.ir_sub_f(&zero, x);
            let floored = b.ir_floor_div_f(&neg, &one);
            b.ir_sub_f(&zero, &floored)
        }
        Value::Integer(_) | Value::Boolean(_) => x.clone(),
        _ => panic!("ceil: unsupported type {:?}", x.zinnia_type()),
    })
}

/// `np.trunc(x)` — round towards zero. Implemented as `select(x >= 0,
/// floor(x), ceil(x))`.
pub fn np_trunc(b: &mut IRBuilder, args: &[Value]) -> Value {
    let val = args.first().expect("trunc: requires an argument");
    vectorize_unary(b, val, &mut |b, x| match x {
        Value::Float(_) => {
            let zero = b.ir_constant_float(0.0);
            let one = b.ir_constant_float(1.0);
            // floor branch
            let floor_x = b.ir_floor_div_f(x, &one);
            // ceil branch (= -floor(-x))
            let neg = b.ir_sub_f(&zero, x);
            let neg_floor = b.ir_floor_div_f(&neg, &one);
            let ceil_x = b.ir_sub_f(&zero, &neg_floor);
            let nonneg = b.ir_greater_than_or_equal_f(x, &zero);
            b.ir_select_f(&nonneg, &floor_x, &ceil_x)
        }
        Value::Integer(_) | Value::Boolean(_) => x.clone(),
        _ => panic!("trunc: unsupported type {:?}", x.zinnia_type()),
    })
}

/// `np.round(x)` — half-away-from-zero (NumPy uses banker's rounding which
/// requires extra primitives we don't have; half-away-from-zero is a
/// reasonable common-case substitute). Implemented as `floor(x + 0.5)` for
/// non-negative x and `-floor(-x + 0.5)` for negative x.
pub fn np_round(b: &mut IRBuilder, args: &[Value]) -> Value {
    let val = args.first().expect("round: requires an argument");
    vectorize_unary(b, val, &mut |b, x| match x {
        Value::Float(_) => {
            let zero = b.ir_constant_float(0.0);
            let half = b.ir_constant_float(0.5);
            let one = b.ir_constant_float(1.0);
            // pos branch: floor(x + 0.5)
            let pos_in = b.ir_add_f(x, &half);
            let pos_out = b.ir_floor_div_f(&pos_in, &one);
            // neg branch: -floor(-x + 0.5)
            let neg = b.ir_sub_f(&zero, x);
            let neg_in = b.ir_add_f(&neg, &half);
            let neg_floor = b.ir_floor_div_f(&neg_in, &one);
            let neg_out = b.ir_sub_f(&zero, &neg_floor);
            let nonneg = b.ir_greater_than_or_equal_f(x, &zero);
            b.ir_select_f(&nonneg, &pos_out, &neg_out)
        }
        Value::Integer(_) | Value::Boolean(_) => x.clone(),
        _ => panic!("round: unsupported type {:?}", x.zinnia_type()),
    })
}

/// `np.reciprocal(x)` — `1 / x`. Result is always a float.
pub fn np_reciprocal(b: &mut IRBuilder, args: &[Value]) -> Value {
    let val = args.first().expect("reciprocal: requires an argument");
    vectorize_unary(b, val, &mut |b, x| {
        let xf = match x {
            Value::Float(_) => x.clone(),
            _ => b.ir_float_cast(x),
        };
        let one = b.ir_constant_float(1.0);
        b.ir_div_f(&one, &xf)
    })
}

/// `np.where(cond, x, y)` — element-wise ternary select. All three args
/// are broadcast to a common shape, then a per-element select fires.
pub fn np_where(b: &mut IRBuilder, args: &[Value]) -> Value {
    if args.len() < 3 {
        panic!("where: requires three arguments (cond, x, y)");
    }
    let cond = &args[0];
    let x = &args[1];
    let y = &args[2];
    let cs = crate::helpers::composite::get_composite_shape(cond);
    let xs = crate::helpers::composite::get_composite_shape(x);
    let ys = crate::helpers::composite::get_composite_shape(y);
    // Broadcast cond/x first, then that result with y.
    let cx = crate::helpers::broadcast::broadcast_shapes(&cs, &xs).unwrap_or_else(|| {
        panic!("where: shapes {:?} and {:?} not broadcast compatible", cs, xs)
    });
    let target = crate::helpers::broadcast::broadcast_shapes(&cx, &ys).unwrap_or_else(|| {
        panic!("where: shapes {:?} and {:?} not broadcast compatible", cx, ys)
    });
    let cond_b = crate::helpers::broadcast::materialize_to_shape(cond, &target);
    let x_b = crate::helpers::broadcast::materialize_to_shape(x, &target);
    let y_b = crate::helpers::broadcast::materialize_to_shape(y, &target);
    fn rec(b: &mut IRBuilder, c: &Value, x: &Value, y: &Value) -> Value {
        match (c, x, y) {
            (
                Value::List(cd) | Value::Tuple(cd),
                Value::List(xd) | Value::Tuple(xd),
                Value::List(yd) | Value::Tuple(yd),
            ) => {
                let vals: Vec<Value> = cd
                    .values
                    .iter()
                    .zip(xd.values.iter())
                    .zip(yd.values.iter())
                    .map(|((cv, xv), yv)| rec(b, cv, xv, yv))
                    .collect();
                let types = vals.iter().map(|v| v.zinnia_type()).collect();
                Value::List(CompositeData {
                    elements_type: types,
                    values: vals,
                })
            }
            _ => crate::helpers::value_ops::select_value(b, c, x, y),
        }
    }
    rec(b, &cond_b, &x_b, &y_b)
}

/// `np.clip(arr, lo, hi)` — element-wise clamp. Implemented as
/// `where(arr < lo, lo, where(arr > hi, hi, arr))`. lo / hi may be scalars
/// or broadcast-compatible arrays.
pub fn np_clip(b: &mut IRBuilder, args: &[Value]) -> Value {
    if args.len() < 3 {
        panic!("clip: requires three arguments (arr, a_min, a_max)");
    }
    let arr = &args[0];
    let lo = &args[1];
    let hi = &args[2];
    // arr.clip(lo, hi) ≡ minimum(maximum(arr, lo), hi)
    let lower = crate::helpers::value_ops::apply_binary_op(b, "lt", arr, lo);
    let after_lower = np_where(b, &[lower, lo.clone(), arr.clone()]);
    let upper = crate::helpers::value_ops::apply_binary_op(b, "gt", &after_lower, hi);
    np_where(b, &[upper, hi.clone(), after_lower])
}

// ────────────────────────────────────────────────────────────────────────
// Reductions: mean / var / std / cumsum / cumprod (with axis support)
// ────────────────────────────────────────────────────────────────────────

/// Reduce along axis 0 of `items` — i.e. given N input arrays of the same
/// shape, walk them in lockstep and apply `op` element-wise across the N at
/// every leaf position. The result has the inner shape (one rank lower than
/// the outer collection).
fn reduce_along_axis_0(b: &mut IRBuilder, op: &str, items: &[Value]) -> Value {
    if items.is_empty() {
        return Value::None;
    }
    let first = &items[0];
    match first {
        Value::List(d) | Value::Tuple(d) => {
            let inner_len = d.values.len();
            let mut out = Vec::with_capacity(inner_len);
            for i in 0..inner_len {
                let mut inner_items: Vec<Value> = Vec::with_capacity(items.len());
                for it in items {
                    if let Value::List(dd) | Value::Tuple(dd) = it {
                        inner_items.push(dd.values[i].clone());
                    }
                }
                out.push(reduce_along_axis_0(b, op, &inner_items));
            }
            let types = out.iter().map(|v| v.zinnia_type()).collect();
            Value::List(CompositeData {
                elements_type: types,
                values: out,
            })
        }
        _ => {
            let lst = Value::List(CompositeData {
                elements_type: items.iter().map(|v| v.zinnia_type()).collect(),
                values: items.to_vec(),
            });
            crate::helpers::ndarray::builtin_reduce(b, op, &lst)
        }
    }
}

/// General axis-aware reduction for arbitrary axis. Replaces the old
/// hard-coded axis 0/1 logic.
fn reduce_axis_general(b: &mut IRBuilder, op: &str, val: &Value, axis: usize) -> Value {
    if axis == 0 {
        if let Value::List(d) | Value::Tuple(d) = val {
            return reduce_along_axis_0(b, op, &d.values);
        }
        return crate::helpers::ndarray::builtin_reduce(b, op, val);
    }
    match val {
        Value::List(d) | Value::Tuple(d) => {
            let new_vals: Vec<Value> = d
                .values
                .iter()
                .map(|v| reduce_axis_general(b, op, v, axis - 1))
                .collect();
            let types = new_vals.iter().map(|v| v.zinnia_type()).collect();
            Value::List(CompositeData {
                elements_type: types,
                values: new_vals,
            })
        }
        _ => crate::helpers::ndarray::builtin_reduce(b, op, val),
    }
}

/// Public entry point preserving the old `reduce_with_axis` name. Resolves
/// negative axes here so callers don't have to.
pub fn reduce_with_axis_general(b: &mut IRBuilder, op: &str, val: &Value, axis: i64) -> Value {
    let ndim = crate::helpers::composite::get_composite_shape(val).len();
    if ndim == 0 {
        return crate::helpers::ndarray::builtin_reduce(b, op, val);
    }
    let resolved = if axis < 0 { ndim as i64 + axis } else { axis };
    if resolved < 0 || resolved >= ndim as i64 {
        panic!(
            "reduce: axis {} is out of bounds for array of rank {}",
            axis, ndim
        );
    }
    reduce_axis_general(b, op, val, resolved as usize)
}

/// Cast a scalar value to float, leaving floats untouched.
fn ensure_scalar_float(b: &mut IRBuilder, v: &Value) -> Value {
    match v {
        Value::Float(_) => v.clone(),
        _ => b.ir_float_cast(v),
    }
}

/// `np.mean(arr, axis=None)` — element-wise mean. With no axis, the result
/// is a scalar; with an axis, the result has rank one less.
pub fn np_mean(b: &mut IRBuilder, args: &[Value], kwargs: &HashMap<String, Value>) -> Value {
    let val = args.first().expect("mean: requires an argument");
    let axis = kwargs
        .get("axis")
        .or_else(|| args.get(1))
        .and_then(|v| v.int_val());

    let shape = crate::helpers::composite::get_composite_shape(val);
    if let Some(ax) = axis {
        let ndim = shape.len();
        let resolved = if ax < 0 { ndim as i64 + ax } else { ax };
        if resolved < 0 || resolved >= ndim as i64 {
            panic!("mean: axis {} is out of bounds for array of rank {}", ax, ndim);
        }
        let n = shape[resolved as usize];
        let summed = reduce_axis_general(b, "sum", val, resolved as usize);
        let n_val = b.ir_constant_float(n as f64);
        // Vectorized division
        vectorize_unary(b, &summed, &mut |b, x| {
            let xf = ensure_scalar_float(b, x);
            b.ir_div_f(&xf, &n_val)
        })
    } else {
        let total: usize = shape.iter().product::<usize>().max(1);
        let total_sum = crate::helpers::ndarray::builtin_reduce(b, "sum", val);
        let total_f = ensure_scalar_float(b, &total_sum);
        let n_val = b.ir_constant_float(total as f64);
        b.ir_div_f(&total_f, &n_val)
    }
}

/// Compute element-wise `(x - m) ** 2` where `m` is broadcast against `x`.
fn squared_deviation(b: &mut IRBuilder, x: &Value, m: &Value) -> Value {
    let diff = crate::helpers::value_ops::apply_binary_op(b, "sub", x, m);
    crate::helpers::value_ops::apply_binary_op(b, "mul", &diff, &diff)
}

/// `np.var(arr, axis=None)` — population variance (ddof=0). NumPy supports
/// `ddof` but we keep things simple for now and pin ddof=0.
pub fn np_var(b: &mut IRBuilder, args: &[Value], kwargs: &HashMap<String, Value>) -> Value {
    let val = args.first().expect("var: requires an argument");
    let axis = kwargs
        .get("axis")
        .or_else(|| args.get(1))
        .and_then(|v| v.int_val());

    let mean = np_mean(b, args, kwargs);
    if let Some(ax) = axis {
        // Need to reinsert the reduced axis as length 1 so the broadcast
        // arithmetic works. Easiest: expand_dims at the resolved axis.
        let shape = crate::helpers::composite::get_composite_shape(val);
        let ndim = shape.len();
        let resolved = if ax < 0 { ndim as i64 + ax } else { ax };
        let axis_const = b.ir_constant_int(resolved);
        let mean_expanded = np_expand_dims(b, &[mean.clone(), axis_const]);
        let sq = squared_deviation(b, val, &mean_expanded);
        let sq_sum_args = vec![sq.clone()];
        let mut sq_sum_kwargs = HashMap::new();
        sq_sum_kwargs.insert(
            "axis".to_string(),
            b.ir_constant_int(resolved),
        );
        np_mean(b, &sq_sum_args, &sq_sum_kwargs)
    } else {
        let sq = squared_deviation(b, val, &mean);
        np_mean(b, &[sq], &HashMap::new())
    }
}

/// `np.std(arr, axis=None)` — population standard deviation = sqrt(var).
pub fn np_std(b: &mut IRBuilder, args: &[Value], kwargs: &HashMap<String, Value>) -> Value {
    let v = np_var(b, args, kwargs);
    vectorize_unary(b, &v, &mut |b, x| {
        let xf = ensure_scalar_float(b, x);
        b.ir_sqrt_f(&xf)
    })
}

/// Inclusive prefix scan along axis 0 of `val`, applying `op` (`add` or
/// `mul`). Used by cumsum/cumprod.
fn cumulative_axis_0(b: &mut IRBuilder, op: &str, val: &Value) -> Value {
    let outer = match val {
        Value::List(d) | Value::Tuple(d) => d,
        _ => return val.clone(),
    };
    if outer.values.is_empty() {
        return val.clone();
    }
    let mut out: Vec<Value> = Vec::with_capacity(outer.values.len());
    out.push(outer.values[0].clone());
    for i in 1..outer.values.len() {
        let prev = out.last().cloned().unwrap();
        let next = crate::helpers::value_ops::apply_binary_op(b, op, &prev, &outer.values[i]);
        out.push(next);
    }
    let types = out.iter().map(|v| v.zinnia_type()).collect();
    Value::List(CompositeData {
        elements_type: types,
        values: out,
    })
}

/// Recursive scan along an arbitrary axis. At axis 0 we run the prefix
/// scan; at axis > 0 we recurse into each outer child.
fn cumulative_axis_general(b: &mut IRBuilder, op: &str, val: &Value, axis: usize) -> Value {
    if axis == 0 {
        return cumulative_axis_0(b, op, val);
    }
    match val {
        Value::List(d) | Value::Tuple(d) => {
            let new_vals: Vec<Value> = d
                .values
                .iter()
                .map(|v| cumulative_axis_general(b, op, v, axis - 1))
                .collect();
            let types = new_vals.iter().map(|v| v.zinnia_type()).collect();
            Value::List(CompositeData {
                elements_type: types,
                values: new_vals,
            })
        }
        _ => val.clone(),
    }
}

/// `np.cumsum(arr, axis=None)` / `np.cumprod(arr, axis=None)`. Without an
/// axis, NumPy flattens first then scans, returning a 1-D result.
pub fn np_cumulative(
    b: &mut IRBuilder,
    args: &[Value],
    kwargs: &HashMap<String, Value>,
    op: &str,
) -> Value {
    let val = args.first().expect("cumulative: requires an argument");
    let axis = kwargs
        .get("axis")
        .or_else(|| args.get(1))
        .and_then(|v| v.int_val());

    if let Some(ax) = axis {
        let ndim = crate::helpers::composite::get_composite_shape(val).len();
        let resolved = if ax < 0 { ndim as i64 + ax } else { ax };
        if resolved < 0 || resolved >= ndim as i64 {
            panic!(
                "{}: axis {} is out of bounds for array of rank {}",
                op, ax, ndim
            );
        }
        cumulative_axis_general(b, op, val, resolved as usize)
    } else {
        // Flatten then scan along the new axis 0.
        let flat = crate::helpers::composite::flatten_composite(val);
        let types: Vec<crate::types::ZinniaType> = flat.iter().map(|v| v.zinnia_type()).collect();
        let flat_val = Value::List(CompositeData {
            elements_type: types,
            values: flat,
        });
        cumulative_axis_0(b, op, &flat_val)
    }
}

pub fn np_cumsum(b: &mut IRBuilder, args: &[Value], kwargs: &HashMap<String, Value>) -> Value {
    np_cumulative(b, args, kwargs, "add")
}

pub fn np_cumprod(b: &mut IRBuilder, args: &[Value], kwargs: &HashMap<String, Value>) -> Value {
    np_cumulative(b, args, kwargs, "mul")
}

// ────────────────────────────────────────────────────────────────────────
// Splitting family (split / array_split / hsplit / vsplit / dsplit)
// ────────────────────────────────────────────────────────────────────────

/// Take the slice `start..stop` of `val` along axis `axis`. Other axes are
/// kept intact. Used by all the splitting helpers below.
fn slice_along_axis(val: &Value, axis: usize, start: usize, stop: usize) -> Value {
    if axis == 0 {
        match val {
            Value::List(d) | Value::Tuple(d) => {
                let new_vals: Vec<Value> = d.values[start..stop].to_vec();
                let new_types = new_vals.iter().map(|v| v.zinnia_type()).collect();
                Value::List(CompositeData {
                    elements_type: new_types,
                    values: new_vals,
                })
            }
            _ => val.clone(),
        }
    } else {
        match val {
            Value::List(d) | Value::Tuple(d) => {
                let new_vals: Vec<Value> = d
                    .values
                    .iter()
                    .map(|v| slice_along_axis(v, axis - 1, start, stop))
                    .collect();
                let new_types = new_vals.iter().map(|v| v.zinnia_type()).collect();
                Value::List(CompositeData {
                    elements_type: new_types,
                    values: new_vals,
                })
            }
            _ => val.clone(),
        }
    }
}

/// Compute the section boundaries (cumulative sizes) for `np.split` and
/// `np.array_split`. For `array_split`, when `n` does not evenly divide
/// `length`, the first `length % n` sections get one extra element each
/// (matching NumPy).
fn compute_split_boundaries(
    length: usize,
    sections: &Value,
    allow_uneven: bool,
) -> Vec<(usize, usize)> {
    match sections {
        Value::Integer(_) => {
            let n = sections
                .int_val()
                .expect("split: sections must be a constant int")
                as usize;
            if n == 0 {
                panic!("split: number of sections must be > 0");
            }
            if !allow_uneven && length % n != 0 {
                panic!(
                    "split: array of length {} cannot be split into {} equal sections",
                    length, n
                );
            }
            let base = length / n;
            let extras = length % n;
            let mut out = Vec::with_capacity(n);
            let mut cursor = 0usize;
            for i in 0..n {
                let sz = base + if i < extras { 1 } else { 0 };
                out.push((cursor, cursor + sz));
                cursor += sz;
            }
            out
        }
        Value::List(d) | Value::Tuple(d) => {
            // Index list: split *at* these indices.
            let mut indices: Vec<usize> = d
                .values
                .iter()
                .map(|v| {
                    let i = v
                        .int_val()
                        .expect("split: index entries must be constant ints");
                    i.max(0).min(length as i64) as usize
                })
                .collect();
            indices.push(length);
            let mut out = Vec::with_capacity(indices.len());
            let mut prev = 0usize;
            for &i in &indices {
                out.push((prev, i.max(prev)));
                prev = i.max(prev);
            }
            out
        }
        _ => panic!("split: sections must be an int or a list of indices"),
    }
}

/// Shared body for `np.split` / `np.array_split` along an explicit axis.
fn split_impl(
    val: &Value,
    sections: &Value,
    axis: i64,
    allow_uneven: bool,
    op: &str,
) -> Value {
    let shape = crate::helpers::composite::get_composite_shape(val);
    let ndim = shape.len();
    if ndim == 0 {
        panic!("{}: cannot split a 0-D value", op);
    }
    let ax = resolve_axis(axis, ndim, op);
    let length = shape[ax];
    let bounds = compute_split_boundaries(length, sections, allow_uneven);
    let parts: Vec<Value> = bounds
        .into_iter()
        .map(|(s, e)| slice_along_axis(val, ax, s, e))
        .collect();
    let types = parts.iter().map(|v| v.zinnia_type()).collect();
    Value::List(CompositeData {
        elements_type: types,
        values: parts,
    })
}

/// `np.split(arr, sections, axis=0)` — equal-section split (errors if the
/// sections don't divide evenly).
pub fn np_split(_b: &mut IRBuilder, args: &[Value], kwargs: &HashMap<String, Value>) -> Value {
    let val = args.first().expect("split: requires an array argument");
    let sections = args.get(1).expect("split: requires a sections argument");
    let axis = kwargs
        .get("axis")
        .or_else(|| args.get(2))
        .and_then(|v| v.int_val())
        .unwrap_or(0);
    split_impl(val, sections, axis, false, "split")
}

/// `np.array_split(arr, sections, axis=0)` — like split but allows uneven
/// sections; the first `length % n` sections get one extra element.
pub fn np_array_split(
    _b: &mut IRBuilder,
    args: &[Value],
    kwargs: &HashMap<String, Value>,
) -> Value {
    let val = args.first().expect("array_split: requires an array argument");
    let sections = args.get(1).expect("array_split: requires a sections argument");
    let axis = kwargs
        .get("axis")
        .or_else(|| args.get(2))
        .and_then(|v| v.int_val())
        .unwrap_or(0);
    split_impl(val, sections, axis, true, "array_split")
}

/// `np.hsplit(arr, sections)` — split along axis 1 for ≥2-D, axis 0 for 1-D.
pub fn np_hsplit(_b: &mut IRBuilder, args: &[Value]) -> Value {
    let val = args.first().expect("hsplit: requires an array argument");
    let sections = args.get(1).expect("hsplit: requires a sections argument");
    let shape = crate::helpers::composite::get_composite_shape(val);
    let axis = if shape.len() >= 2 { 1 } else { 0 };
    split_impl(val, sections, axis, false, "hsplit")
}

/// `np.vsplit(arr, sections)` — split along axis 0 (requires ≥2-D).
pub fn np_vsplit(_b: &mut IRBuilder, args: &[Value]) -> Value {
    let val = args.first().expect("vsplit: requires an array argument");
    let sections = args.get(1).expect("vsplit: requires a sections argument");
    let shape = crate::helpers::composite::get_composite_shape(val);
    if shape.len() < 2 {
        panic!("vsplit: input must be at least 2-D");
    }
    split_impl(val, sections, 0, false, "vsplit")
}

/// `np.dsplit(arr, sections)` — split along axis 2 (requires ≥3-D).
pub fn np_dsplit(_b: &mut IRBuilder, args: &[Value]) -> Value {
    let val = args.first().expect("dsplit: requires an array argument");
    let sections = args.get(1).expect("dsplit: requires a sections argument");
    let shape = crate::helpers::composite::get_composite_shape(val);
    if shape.len() < 3 {
        panic!("dsplit: input must be at least 3-D");
    }
    split_impl(val, sections, 2, false, "dsplit")
}

// ────────────────────────────────────────────────────────────────────────
// np.block — recursive nested concatenation
// ────────────────────────────────────────────────────────────────────────

/// Recursive worker for `np.block`. At each block level (going from the
/// outermost level inward), we recurse on each child with `block_depth − 1`,
/// then concat the results along the appropriate axis.
///
/// The axis follows NumPy's "negative axis from the result rank" rule: at
/// the outermost level we concat along axis `result_ndim − block_depth`; at
/// the innermost block level we concat along axis `result_ndim − 1`.
fn block_recursive(val: &Value, block_depth: usize, result_ndim: usize) -> Value {
    if block_depth == 0 {
        return val.clone();
    }
    let children: Vec<Value> = match val {
        Value::List(d) | Value::Tuple(d) => d
            .values
            .iter()
            .map(|c| block_recursive(c, block_depth - 1, result_ndim))
            .collect(),
        _ => return val.clone(),
    };
    let axis = result_ndim - block_depth;
    concat_recursive(&children, axis)
}

/// Walk every leaf in the nested block structure at the given depth. Used to
/// validate that all leaves share a rank.
fn collect_block_leaves(val: &Value, block_depth: usize) -> Vec<Value> {
    if block_depth == 0 {
        return vec![val.clone()];
    }
    match val {
        Value::List(d) | Value::Tuple(d) => {
            let mut all = Vec::new();
            for child in &d.values {
                all.extend(collect_block_leaves(child, block_depth - 1));
            }
            all
        }
        _ => vec![val.clone()],
    }
}

/// `np.block(arrays, block_depth)` — recursive nested concatenation. The
/// block depth must be supplied by the caller (typically computed from the
/// AST nesting in `ir_gen/named_attr.rs`, since after a Python list literal
/// has been visited into a `Value::List` we can no longer distinguish "a
/// nested block of arrays" from "a single high-rank ndarray").
///
/// All leaf arrays must share the same rank, and that rank must be ≥
/// `block_depth`. NumPy auto-promotes mixed-rank leaves via `atleast_Nd`;
/// that case is currently out of scope and produces a hard error.
pub fn np_block_with_depth(val: &Value, block_depth: usize) -> Value {
    if block_depth == 0 {
        return val.clone();
    }

    let leaves = collect_block_leaves(val, block_depth);
    if leaves.is_empty() {
        return val.clone();
    }
    let first_rank = crate::helpers::composite::get_composite_shape(&leaves[0]).len();
    for leaf in &leaves {
        let r = crate::helpers::composite::get_composite_shape(leaf).len();
        if r != first_rank {
            panic!(
                "block: all leaf arrays must currently have the same rank \
                 (got {} and {}). Mixed-rank block (NumPy auto-promotes via \
                 atleast_Nd) is not yet supported on static ndarrays.",
                first_rank, r
            );
        }
    }
    if first_rank < block_depth {
        panic!(
            "block: leaf arrays of rank {} cannot be combined into a block \
             of nesting depth {}. Promote them with np.atleast_Nd first, or \
             use np.stack / np.concatenate directly.",
            first_rank, block_depth
        );
    }

    let result_ndim = first_rank.max(block_depth);
    block_recursive(val, block_depth, result_ndim)
}
