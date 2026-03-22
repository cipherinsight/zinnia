use crate::builder::IRBuilder;
use crate::types::{CompositeData, SliceIndex, Value, ZinniaType};
use super::composite;

pub fn ndarray_transpose(b: &mut IRBuilder, val: &Value, args: &[Value]) -> Value {
    // Determine the shape of the input
    let shape = composite::get_composite_shape(val);
    let ndim = shape.len();
    if ndim <= 1 { return val.clone(); }

    // Determine axis permutation — check length before validating individual values
    let raw_axes: Vec<i64> = if args.is_empty() || matches!(args.first(), Some(Value::None)) {
        (0..ndim as i64).rev().collect()
    } else if let Some(Value::Tuple(perm_data)) | Some(Value::List(perm_data)) = args.first() {
        perm_data.values.iter().map(|v| v.int_val().unwrap_or(0)).collect()
    } else {
        args.iter().map(|v| v.int_val().unwrap_or(0)).collect()
    };

    // Check length first (before resolving individual values)
    if raw_axes.len() != ndim {
        panic!("Length of `axes` should be equal to the number of dimensions of the array (expected {}, got {})", ndim, raw_axes.len());
    }

    let axes: Vec<usize> = raw_axes.iter().map(|&a| {
        let resolved = if a < 0 { ndim as i64 + a } else { a };
        if resolved < 0 || resolved >= ndim as i64 {
            panic!("Invalid axis value: {} is out of bounds for array with {} dimensions", a, ndim);
        }
        resolved as usize
    }).collect();
    // Check for invalid axis values
    for &a in &axes {
        if a >= ndim {
            panic!("Invalid axis value: {} is out of bounds for array with {} dimensions", a, ndim);
        }
    }
    // Check for valid permutation (no duplicates)
    let mut seen = vec![false; ndim];
    for &a in &axes {
        if seen[a] {
            panic!("axes should be a permutation of 0 to {}", ndim - 1);
        }
        seen[a] = true;
    }

    // Calculate output shape
    let out_shape: Vec<usize> = axes.iter().map(|&a| shape[a]).collect();

    // Flatten the input, then reassemble in transposed order
    let flat = composite::flatten_composite(val);
    if flat.is_empty() { return val.clone(); }

    // Compute strides for input
    let mut in_strides = vec![1usize; ndim];
    for i in (0..ndim - 1).rev() {
        in_strides[i] = in_strides[i + 1] * shape[i + 1];
    }
    // Compute strides for output
    let mut out_strides = vec![1usize; ndim];
    for i in (0..ndim - 1).rev() {
        out_strides[i] = out_strides[i + 1] * out_shape[i + 1];
    }

    let total: usize = shape.iter().product();
    let mut out_flat = vec![Value::None; total];

    // For each element in the flat array, compute its input index tuple,
    // permute it, and write to the output position
    for flat_idx in 0..total {
        // Compute input multi-index
        let mut remainder = flat_idx;
        let mut in_idx = vec![0usize; ndim];
        for d in 0..ndim {
            in_idx[d] = remainder / in_strides[d];
            remainder %= in_strides[d];
        }
        // Permute to get output multi-index
        let mut out_idx = vec![0usize; ndim];
        for d in 0..ndim {
            out_idx[d] = in_idx[axes[d]];
        }
        // Compute output flat index
        let mut out_flat_idx = 0;
        for d in 0..ndim {
            out_flat_idx += out_idx[d] * out_strides[d];
        }
        out_flat[out_flat_idx] = flat[flat_idx].clone();
    }

    // Rebuild nested structure from output shape
    let types = out_flat.iter().map(|v| v.zinnia_type()).collect();
    composite::build_nested_value(out_flat, types, &out_shape)
}

pub fn ndarray_argmax_argmin(b: &mut IRBuilder, val: &Value, _args: &[Value], is_max: bool) -> Value {
    let elements = composite::flatten_composite(val);
    if elements.is_empty() { return b.ir_constant_int(0); }
    let mut best_idx = b.ir_constant_int(0);
    let mut best_val = elements[0].clone();
    for (i, elem) in elements.iter().enumerate().skip(1) {
        let cond = if is_max {
            b.ir_greater_than_i(elem, &best_val)
        } else {
            b.ir_less_than_i(elem, &best_val)
        };
        let idx_val = b.ir_constant_int(i as i64);
        best_idx = b.ir_select_i(&cond, &idx_val, &best_idx);
        best_val = b.ir_select_i(&cond, elem, &best_val);
    }
    best_idx
}

pub fn multidim_subscript(b: &mut IRBuilder, data: &CompositeData, indices: &[SliceIndex]) -> Value {
    if indices.is_empty() {
        return Value::List(data.clone());
    }

    match &indices[0] {
        SliceIndex::Single(idx_value) => {
            if let Some(idx) = idx_value.int_val() {
                let i = if idx < 0 { (data.values.len() as i64 + idx) as usize } else { idx as usize };
                if i >= data.values.len() {
                    return Value::None;
                }
                if indices.len() == 1 {
                    return data.values[i].clone();
                }
                // Recurse into the selected element
                match &data.values[i] {
                    Value::List(inner) | Value::Tuple(inner) => {
                        multidim_subscript(b, inner, &indices[1..])
                    }
                    _ => data.values[i].clone(),
                }
            } else {
                // Dynamic index
                if indices.len() == 1 {
                    return crate::helpers::value_ops::dynamic_list_subscript(b, data, idx_value);
                }
                // Dynamic index with further dimensions: select from each possible row
                // For each possible index value, apply the remaining indices
                let mut results: Vec<Value> = Vec::new();
                for elem in &data.values {
                    if let Value::List(inner) | Value::Tuple(inner) = elem {
                        results.push(multidim_subscript(b, inner, &indices[1..]));
                    } else {
                        results.push(elem.clone());
                    }
                }
                // Now select from results using the dynamic index
                let result_data = CompositeData {
                    elements_type: results.iter().map(|v| v.zinnia_type()).collect(),
                    values: results,
                };
                crate::helpers::value_ops::dynamic_list_subscript(b, &result_data, idx_value)
            }
        }
        SliceIndex::Range(start, stop, step) => {
            let len = data.values.len() as i64;
            let s = start.as_ref().and_then(|v| v.int_val()).unwrap_or(0);
            let e = stop.as_ref().and_then(|v| v.int_val()).unwrap_or(len);
            let st = step.as_ref().and_then(|v| v.int_val()).unwrap_or(1);
            let s = if s < 0 { (len + s).max(0) } else { s.min(len) } as usize;
            let e = if e < 0 { (len + e).max(0) } else { e.min(len) } as usize;

            let mut selected = Vec::new();
            let mut i = s;
            while (st > 0 && i < e) || (st < 0 && i > e) {
                if i < data.values.len() {
                    if indices.len() == 1 {
                        selected.push(data.values[i].clone());
                    } else {
                        // Apply remaining indices to each selected element
                        match &data.values[i] {
                            Value::List(inner) | Value::Tuple(inner) => {
                                selected.push(multidim_subscript(b, inner, &indices[1..]));
                            }
                            _ => selected.push(data.values[i].clone()),
                        }
                    }
                }
                i = (i as i64 + st) as usize;
            }
            let types = selected.iter().map(|v| v.zinnia_type()).collect();
            Value::List(CompositeData { elements_type: types, values: selected })
        }
    }
}

pub fn builtin_reduce(b: &mut IRBuilder, op: &str, val: &Value) -> Value {
    let elements = composite::flatten_composite(val);
    if elements.is_empty() {
        return match op {
            "sum" => b.ir_constant_int(0),
            "any" => b.ir_constant_bool(false),
            "all" => b.ir_constant_bool(true),
            "prod" => b.ir_constant_int(1),
            "min" | "max" => Value::None,
            _ => Value::None,
        };
    }
    match op {
        "sum" => {
            let mut acc = elements[0].clone();
            for elem in &elements[1..] {
                acc = b.ir_add_i(&acc, elem);
            }
            acc
        }
        "any" => {
            let mut acc = crate::helpers::value_ops::to_scalar_bool(b, &elements[0]);
            for elem in &elements[1..] {
                let bool_val = crate::helpers::value_ops::to_scalar_bool(b, elem);
                acc = b.ir_logical_or(&acc, &bool_val);
            }
            acc
        }
        "all" => {
            let mut acc = crate::helpers::value_ops::to_scalar_bool(b, &elements[0]);
            for elem in &elements[1..] {
                let bool_val = crate::helpers::value_ops::to_scalar_bool(b, elem);
                acc = b.ir_logical_and(&acc, &bool_val);
            }
            acc
        }
        "min" => {
            let mut acc = elements[0].clone();
            for elem in &elements[1..] {
                let cond = b.ir_less_than_i(&acc, elem);
                acc = b.ir_select_i(&cond, &acc, elem);
            }
            acc
        }
        "max" => {
            let mut acc = elements[0].clone();
            for elem in &elements[1..] {
                let cond = b.ir_greater_than_i(&acc, elem);
                acc = b.ir_select_i(&cond, &acc, elem);
            }
            acc
        }
        "prod" => {
            let mut acc = elements[0].clone();
            for elem in &elements[1..] {
                acc = b.ir_mul_i(&acc, elem);
            }
            acc
        }
        _ => Value::None,
    }
}
