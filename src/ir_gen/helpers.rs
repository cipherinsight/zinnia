use crate::types::{CompositeData, Value, ZinniaType, DTDescriptorDict};

use super::{IRGenerator, SliceIndex};

impl IRGenerator {
    pub(crate) fn cast_composite(&mut self, val: &Value, to_float: bool) -> Value {
        match val {
            Value::List(data) | Value::Tuple(data) => {
                let is_tuple = matches!(val, Value::Tuple(_));
                let new_values: Vec<Value> = data.values.iter()
                    .map(|v| self.cast_composite(v, to_float))
                    .collect();
                let new_types = new_values.iter().map(|v| v.zinnia_type()).collect();
                if is_tuple {
                    Value::Tuple(CompositeData { elements_type: new_types, values: new_values })
                } else {
                    Value::List(CompositeData { elements_type: new_types, values: new_values })
                }
            }
            Value::Integer(_) | Value::Boolean(_) if to_float => {
                self.builder.ir_float_cast(val)
            }
            Value::Float(_) if !to_float => {
                self.builder.ir_int_cast(val)
            }
            _ => val.clone(),
        }
    }

    /// Set a nested value in a composite structure using slice indices.
    /// Cast a value to match the target element dtype if they differ.
    pub(crate) fn cast_value_to_match(&mut self, value: Value, target_type: &ZinniaType) -> Value {
        let vt = value.zinnia_type();
        if vt == *target_type { return value; }
        match (target_type, &value) {
            (ZinniaType::Integer, Value::Float(_)) => self.builder.ir_int_cast(&value),
            (ZinniaType::Float, Value::Integer(_)) => self.builder.ir_float_cast(&value),
            (ZinniaType::Float, Value::Boolean(_)) => self.builder.ir_float_cast(&value),
            (ZinniaType::Integer, Value::Boolean(_)) => self.builder.ir_bool_cast(&value),
            _ => value,
        }
    }

    pub(crate) fn set_nested_value(&mut self, current: Value, indices: &[SliceIndex], value: Value) -> Value {
        if indices.is_empty() {
            // At leaf: cast value to match current's type if they differ
            return self.cast_value_to_match(value, &current.zinnia_type());
        }
        match &current {
            Value::List(data) | Value::Tuple(data) => {
                let is_tuple = matches!(&current, Value::Tuple(_));
                if let SliceIndex::Single(idx_val) = &indices[0] {
                    if let Some(idx) = idx_val.int_val() {
                        // Static index
                        let idx = if idx < 0 { (data.values.len() as i64 + idx) as usize } else { idx as usize };
                        if idx < data.values.len() {
                            let mut new_values = data.values.clone();
                            let mut new_types = data.elements_type.clone();
                            if indices.len() == 1 {
                                // Cast value to match target element type
                                let target_et = &data.elements_type[idx];
                                new_values[idx] = self.cast_value_to_match(value, target_et);
                                new_types[idx] = new_values[idx].zinnia_type();
                            } else {
                                new_values[idx] = self.set_nested_value(new_values[idx].clone(), &indices[1..], value);
                                new_types[idx] = new_values[idx].zinnia_type();
                            }
                            return if is_tuple {
                                Value::Tuple(CompositeData { elements_type: new_types, values: new_values })
                            } else {
                                Value::List(CompositeData { elements_type: new_types, values: new_values })
                            };
                        }
                    } else {
                        // Dynamic index — use mux/memory path
                        if indices.len() == 1 {
                            // Single dynamic index on flat list
                            return crate::helpers::value_ops::dynamic_list_set_item(&mut self.builder, data, idx_val, &value);
                        } else {
                            // Multi-dim dynamic index: compute linear address
                            // For array[x, y] where array is [[...], [...], ...]:
                            // Flatten array, compute addr = x * ncols + y, set at addr
                            let shape = crate::helpers::composite::get_composite_shape(&current);
                            let flat = crate::helpers::composite::flatten_composite(&current);
                            if flat.is_empty() { return current; }

                            // Compute linear address from multi-dim indices
                            let mut strides = vec![1usize; shape.len()];
                            for i in (0..shape.len() - 1).rev() {
                                strides[i] = strides[i + 1] * shape[i + 1];
                            }

                            let mut linear_addr = self.builder.ir_constant_int(0);
                            // Process all indices (current + remaining)
                            let all_idx_vals: Vec<&Value> = std::iter::once(idx_val)
                                .chain(indices[1..].iter().filter_map(|si| {
                                    if let SliceIndex::Single(v) = si { Some(v) } else { None }
                                }))
                                .collect();

                            for (dim, &iv) in all_idx_vals.iter().enumerate() {
                                if dim < strides.len() {
                                    let stride_const = self.builder.ir_constant_int(strides[dim] as i64);
                                    let term = self.builder.ir_mul_i(iv, &stride_const);
                                    linear_addr = self.builder.ir_add_i(&linear_addr, &term);
                                }
                            }

                            let flat_data = CompositeData {
                                elements_type: flat.iter().map(|v| v.zinnia_type()).collect(),
                                values: flat,
                            };
                            let updated_flat = crate::helpers::value_ops::dynamic_list_set_item(&mut self.builder, &flat_data, &linear_addr, &value);

                            // Rebuild nested structure from flat
                            if let Value::List(uf) = &updated_flat {
                                let rebuilt = crate::helpers::composite::build_nested_value(uf.values.clone(), uf.elements_type.clone(), &shape);
                                return rebuilt;
                            }
                            return updated_flat;
                        }
                    }
                }
                // For range slicing assignment
                if let SliceIndex::Range(start, stop, step) = &indices[0] {
                    let len = data.values.len() as i64;
                    let start_idx = start.as_ref().and_then(|v| v.int_val()).unwrap_or(0);
                    let stop_idx = stop.as_ref().and_then(|v| v.int_val()).unwrap_or(len);
                    let step_val = step.as_ref().and_then(|v| v.int_val()).unwrap_or(1);
                    let start_idx = if start_idx < 0 { (len + start_idx).max(0) } else { start_idx.min(len) } as usize;
                    let stop_idx = if stop_idx < 0 { (len + stop_idx).max(0) } else { stop_idx.min(len) } as usize;

                    let mut new_values = data.values.clone();
                    let mut new_types = data.elements_type.clone();
                    if indices.len() > 1 {
                        // Multi-dim range assignment: array[:, col] = values
                        // Apply remaining indices to each selected element
                        if let Value::List(rhs_data) | Value::Tuple(rhs_data) = &value {
                            let mut rhs_idx = 0;
                            let mut i = start_idx;
                            while i < stop_idx && rhs_idx < rhs_data.values.len() {
                                new_values[i] = self.set_nested_value(
                                    new_values[i].clone(),
                                    &indices[1..],
                                    rhs_data.values[rhs_idx].clone(),
                                );
                                new_types[i] = new_values[i].zinnia_type();
                                rhs_idx += 1;
                                i += step_val as usize;
                            }
                        }
                    } else {
                        // Single-dim range assignment: array[start:stop] = values or scalar
                        if let Value::List(rhs_data) | Value::Tuple(rhs_data) = &value {
                            let mut rhs_idx = 0;
                            let mut i = start_idx;
                            while i < stop_idx && rhs_idx < rhs_data.values.len() {
                                let target_et = &data.elements_type[i];
                                new_values[i] = self.cast_value_to_match(rhs_data.values[rhs_idx].clone(), target_et);
                                new_types[i] = new_values[i].zinnia_type();
                                rhs_idx += 1;
                                i += step_val as usize;
                            }
                        } else {
                            // Scalar broadcasting: assign scalar to all positions in range
                            let mut i = start_idx;
                            while i < stop_idx {
                                let target_et = &data.elements_type[i];
                                new_values[i] = self.cast_value_to_match(value.clone(), target_et);
                                new_types[i] = new_values[i].zinnia_type();
                                i += step_val as usize;
                            }
                        }
                    }
                    return if is_tuple {
                        Value::Tuple(CompositeData { elements_type: new_types, values: new_values })
                    } else {
                        Value::List(CompositeData { elements_type: new_types, values: new_values })
                    };
                }
                current
            }
            _ => current,
        }
    }

    pub(crate) fn read_input_value(&mut self, dt: &ZinniaType, indices: Vec<u32>, is_public: bool) -> Value {
        match dt {
            ZinniaType::Integer | ZinniaType::Boolean => {
                self.builder.ir_read_integer(indices, is_public)
            }
            ZinniaType::Float => {
                self.builder.ir_read_float(indices, is_public)
            }
            ZinniaType::PoseidonHashed { .. } => {
                self.builder.ir_read_hash(indices, is_public)
            }
            ZinniaType::NDArray { shape, dtype } => {
                let total: usize = shape.iter().product();
                let inner_dt = match dtype {
                    crate::types::NumberType::Integer => ZinniaType::Integer,
                    crate::types::NumberType::Float => ZinniaType::Float,
                };
                let mut values = Vec::new();
                for flat_idx in 0..total {
                    let mut sub_indices = indices.clone();
                    sub_indices.push(flat_idx as u32);
                    values.push(self.read_input_value(&inner_dt, sub_indices, is_public));
                }
                // Build nested structure from flat values
                let types = values.iter().map(|v| v.zinnia_type()).collect();
                crate::helpers::composite::build_nested_value(values, types, shape)
            }
            ZinniaType::List { elements } => {
                let mut values = Vec::new();
                let mut types = Vec::new();
                for (j, elem_dt) in elements.iter().enumerate() {
                    let mut sub_indices = indices.clone();
                    sub_indices.push(j as u32);
                    let val = self.read_input_value(elem_dt, sub_indices, is_public);
                    types.push(val.zinnia_type());
                    values.push(val);
                }
                Value::List(CompositeData { elements_type: types, values })
            }
            ZinniaType::Tuple { elements } => {
                let mut values = Vec::new();
                let mut types = Vec::new();
                for (j, elem_dt) in elements.iter().enumerate() {
                    let mut sub_indices = indices.clone();
                    sub_indices.push(j as u32);
                    let val = self.read_input_value(elem_dt, sub_indices, is_public);
                    types.push(val.zinnia_type());
                    values.push(val);
                }
                Value::Tuple(CompositeData { elements_type: types, values })
            }
            ZinniaType::DynamicNDArray { dtype, max_length, max_rank } => {
                let inner_dt = match dtype {
                    crate::types::NumberType::Integer => ZinniaType::Integer,
                    crate::types::NumberType::Float => ZinniaType::Float,
                };
                // Read flat payload elements
                let mut elements = Vec::new();
                for flat_idx in 0..*max_length {
                    let mut sub_indices = indices.clone();
                    sub_indices.push(flat_idx as u32);
                    let val = self.read_input_value(&inner_dt, sub_indices, is_public);
                    elements.push(crate::ops::dyn_ndarray::value_to_scalar_i64(&val));
                }

                // Emit metadata allocation
                let arr_id = self.builder.alloc_array_id();
                let dtype_name = match dtype {
                    crate::types::NumberType::Integer => "int".to_string(),
                    crate::types::NumberType::Float => "float".to_string(),
                };
                self.builder.ir_allocate_dynamic_ndarray_meta(
                    arr_id, dtype_name, *max_length as u32, *max_rank as u32,
                );

                let strides = crate::ops::dyn_ndarray::dyn_row_major_strides(&[*max_length]);
                Value::DynamicNDArray(crate::types::DynamicNDArrayData {
                    max_length: *max_length,
                    max_rank: *max_rank,
                    dtype: *dtype,
                    elements,
                    meta: crate::types::DynArrayMeta {
                        logical_shape: vec![*max_length],
                        logical_offset: 0,
                        logical_strides: strides,
                        runtime_length: crate::types::ScalarValue::new(None, None),
                        runtime_rank: crate::types::ScalarValue::new(None, None),
                        runtime_shape: (0..*max_rank)
                            .map(|_| crate::types::ScalarValue::new(None, None))
                            .collect(),
                        runtime_strides: (0..*max_rank)
                            .map(|_| crate::types::ScalarValue::new(None, None))
                            .collect(),
                        runtime_offset: crate::types::ScalarValue::new(Some(0), None),
                    },
                })
            }
            _ => {
                self.builder.ir_read_integer(indices, is_public)
            }
        }
    }


    pub(crate) fn register_global_datatypes(&mut self) {
        // Register Float and Integer as class values
        let float_class = Value::Class(ZinniaType::Float);
        let int_class = Value::Class(ZinniaType::Integer);
        for name in &["Float", "float"] {
            self.ctx.set(name, float_class.clone());
        }
        for name in &["Integer", "int", "Int", "integer", "Boolean", "bool", "Bool", "boolean"] {
            self.ctx.set(name, int_class.clone());
        }
    }

    pub(crate) fn parse_dt_descriptor(&self, dt_json: &serde_json::Value) -> ZinniaType {
        // Try full DTDescriptorDict format: {"__class__": "...", "dt_data": {...}}
        if let Ok(dict) = serde_json::from_value::<DTDescriptorDict>(dt_json.clone()) {
            return ZinniaType::from_dt_dict(&dict).unwrap_or(ZinniaType::Integer);
        }
        // Fallback: bare dt_data without class wrapper (old format)
        ZinniaType::Integer
    }
}
