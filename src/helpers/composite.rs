use crate::types::{CompositeData, Value, ZinniaType};

/// Recursively flatten a composite (List/Tuple) into a flat vector of leaf values.
pub fn flatten_composite(val: &Value) -> Vec<Value> {
    match val {
        Value::List(data) | Value::Tuple(data) => {
            let mut flat = Vec::new();
            for v in &data.values {
                flat.extend(flatten_composite(v));
            }
            flat
        }
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
        _ => vec![],
    }
}

/// Rebuild a nested `Value::List` structure from a flat vector of values and a shape.
pub fn build_nested_value(flat: Vec<Value>, flat_types: Vec<ZinniaType>, shape: &[usize]) -> Value {
    if shape.len() <= 1 {
        return Value::List(CompositeData {
            elements_type: flat_types,
            values: flat,
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
    })
}
