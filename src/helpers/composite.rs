use crate::types::{CompositeData, Value, ValueId, ZinniaType};

/// Recursively flatten a composite (List/Tuple) into a flat vector of leaf values.
///
/// `Value::StaticArray` is *not* flattened here — flattening would require an
/// `IRBuilder` to read from the segment. Use [`flatten_composite_with_builder`]
/// when a StaticArray may appear, or convert it to a list first via
/// `helpers::static_array::base::to_value_list`.
///
/// A `Value::DynamicNDArray` (an array with a runtime shape) is rejected:
/// the static-shape operations that flatten their operands here would
/// otherwise treat it as one scalar leaf and compute a different function
/// (e.g. `np.cumsum(x[x > 0])` returned its input).
pub fn flatten_composite(val: &Value) -> Vec<Value> {
    match val {
        Value::List(data) | Value::Tuple(data) => {
            let mut flat = Vec::new();
            for v in &data.values {
                flat.extend(flatten_composite(v));
            }
            flat
        }
        Value::DynamicNDArray(_) => reject_runtime_shaped_operand(),
        other => vec![other.clone()],
    }
}

/// Compile error for a static-shape operation given an array whose shape
/// is only known at runtime.
pub fn reject_runtime_shaped_operand() -> ! {
    panic!(
        "this operation is not supported on an array with a runtime shape (the result of a \
         boolean mask, nonzero/argwhere/unique, or a slice with runtime bounds)"
    )
}

/// Variant of [`flatten_composite`] that accepts `Value::StaticArray` by
/// materialising the segment payload first. Use at op boundaries that may
/// see segment-backed numeric arrays — recurses through `List` / `Tuple`
/// nesting.
pub fn flatten_composite_with_builder(
    b: &mut crate::builder::IRBuilder,
    val: &Value,
) -> Vec<Value> {
    match val {
        Value::StaticArray { .. } => {
            let lst = crate::helpers::static_array::base::to_value_list(b, val);
            flatten_composite_with_builder(b, &lst)
        }
        Value::List(data) | Value::Tuple(data) => {
            let mut flat = Vec::new();
            for v in &data.values {
                flat.extend(flatten_composite_with_builder(b, v));
            }
            flat
        }
        Value::DynamicNDArray(_) => reject_runtime_shaped_operand(),
        other => vec![other.clone()],
    }
}

/// Return the shape of a composite value (e.g. `[3, 4]` for a 3×4 nested list).
pub fn get_composite_shape(val: &Value) -> Vec<usize> {
    match val {
        Value::List(data) | Value::Tuple(data) => {
            if data.values.is_empty() {
                return vec![0];
            }
            let mut shape = vec![data.values.len()];
            // Recurse into first element to get inner dimensions
            let inner_shape = get_composite_shape(&data.values[0]);
            if inner_shape.len() > 0
                && !matches!(
                    &data.values[0],
                    Value::Integer(_)
                        | Value::Float(_)
                        | Value::Boolean(_)
                        | Value::String(_)
                        | Value::None
                        | Value::Class(_)
                )
            {
                shape.extend(inner_shape);
            }
            shape
        }
        Value::NDArray(nd) => nd.shape.clone(),
        Value::StaticArray { shape, .. } => shape.clone(),
        // A runtime-shaped array has no static shape; reporting `[]`
        // made static-shape operations treat it as a scalar.
        Value::DynamicNDArray(_) => reject_runtime_shaped_operand(),
        _ => vec![],
    }
}

/// Rebuild a nested `Value::List` structure from a flat vector of values and a shape.
pub fn build_nested_value(flat: Vec<Value>, flat_types: Vec<ZinniaType>, shape: &[usize]) -> Value {
    if shape.len() <= 1 {
        return Value::List(CompositeData {
            elements_type: flat_types,
            values: flat,

            value_id: ValueId::next(),
        });
    }
    let inner_size: usize = shape[1..].iter().product();
    let mut rows = Vec::new();
    for chunk in flat.chunks(inner_size) {
        let chunk_types = chunk.iter().map(|v| v.zinnia_type()).collect();
        rows.push(build_nested_value(chunk.to_vec(), chunk_types, &shape[1..]));
    }
    let row_types = rows.iter().map(|v| v.zinnia_type()).collect();
    Value::List(CompositeData {
        elements_type: row_types,
        values: rows,

        value_id: ValueId::next(),
    })
}
