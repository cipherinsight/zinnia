use crate::types::{CompositeData, Value, ZinniaType};

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
                        }
                        // Multi-dim assignment with a dynamic outer index and
                        // *arbitrary* remaining indices (Single or Range,
                        // possibly mixed). The previous implementation tried
                        // to linearize the addresses, which dropped Range
                        // indices on the floor — `arr[x, :] = vec` would
                        // silently update only one element.
                        //
                        // Approach: for each possible position `i` of the
                        // dynamic outer index, recursively compute "what
                        // row `i` would look like if x == i", then mux
                        // between that and the original row using
                        // (idx_val == i). The recursion delegates back to
                        // this same function so any mix of Single/Range in
                        // `indices[1..]` is handled the same way it would be
                        // handled for a static outer index.
                        let outer_len = data.values.len();
                        let mut new_values = Vec::with_capacity(outer_len);
                        let mut new_types = Vec::with_capacity(outer_len);
                        for i in 0..outer_len {
                            let updated_row = self.set_nested_value(
                                data.values[i].clone(),
                                &indices[1..],
                                value.clone(),
                            );
                            let const_i = self.builder.ir_constant_int(i as i64);
                            let cmp = self.builder.ir_equal_i(idx_val, &const_i);
                            let blended = crate::helpers::value_ops::select_value(
                                &mut self.builder,
                                &cmp,
                                &updated_row,
                                &data.values[i],
                            );
                            new_types.push(blended.zinnia_type());
                            new_values.push(blended);
                        }
                        return if is_tuple {
                            Value::Tuple(CompositeData { elements_type: new_types, values: new_values })
                        } else {
                            Value::List(CompositeData { elements_type: new_types, values: new_values })
                        };
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

    pub(crate) fn read_input_value(
        &mut self,
        dt: &ZinniaType,
        param_name: &str,
        segments: Vec<crate::circuit_input::PathSegment>,
        is_public: bool,
    ) -> Value {
        use crate::circuit_input::{InputPath, PathSegment};

        match dt {
            ZinniaType::Integer | ZinniaType::Boolean => {
                let path = InputPath::new(param_name, segments);
                self.builder.ir_read_integer(path, is_public)
            }
            ZinniaType::Float => {
                let path = InputPath::new(param_name, segments);
                self.builder.ir_read_float(path, is_public)
            }
            ZinniaType::PoseidonHashed { dtype } => {
                // Read the hash value at segments ++ [Hash]
                let mut hash_segs = segments.clone();
                hash_segs.push(PathSegment::Hash);
                let hash_path = InputPath::new(param_name, hash_segs);
                let hash_val = self.builder.ir_read_hash(hash_path, is_public);

                // Read the inner value at segments ++ [Inner, ...]
                let mut inner_segs = segments;
                inner_segs.push(PathSegment::Inner);
                let inner_val = self.read_input_value(dtype, param_name, inner_segs, is_public);

                // Compute hash of flattened inner values and assert equality
                let flat = crate::helpers::composite::flatten_composite(&inner_val);
                let computed_hash = self.builder.ir_poseidon_hash(&flat);
                let eq = self.builder.ir_equal_hash(&computed_hash, &hash_val);
                let bc = self.builder.ir_bool_cast(&eq);
                self.builder.ir_assert(&bc);

                // Return the inner value for circuit use
                inner_val
            }
            ZinniaType::NDArray { shape, dtype } => {
                let total: usize = shape.iter().product();
                let inner_dt = match dtype {
                    crate::types::NumberType::Integer => ZinniaType::Integer,
                    crate::types::NumberType::Float => ZinniaType::Float,
                };
                let mut values = Vec::new();
                for flat_idx in 0..total {
                    let mut sub_segs = segments.clone();
                    sub_segs.push(PathSegment::Index(flat_idx as u32));
                    values.push(self.read_input_value(&inner_dt, param_name, sub_segs, is_public));
                }
                // Build nested structure from flat values
                let types = values.iter().map(|v| v.zinnia_type()).collect();
                crate::helpers::composite::build_nested_value(values, types, shape)
            }
            ZinniaType::List { elements } => {
                let mut values = Vec::new();
                let mut types = Vec::new();
                for (j, elem_dt) in elements.iter().enumerate() {
                    let mut sub_segs = segments.clone();
                    sub_segs.push(PathSegment::Index(j as u32));
                    let val = self.read_input_value(elem_dt, param_name, sub_segs, is_public);
                    types.push(val.zinnia_type());
                    values.push(val);
                }
                Value::List(CompositeData { elements_type: types, values })
            }
            ZinniaType::Tuple { elements } => {
                let mut values = Vec::new();
                let mut types = Vec::new();
                for (j, elem_dt) in elements.iter().enumerate() {
                    let mut sub_segs = segments.clone();
                    sub_segs.push(PathSegment::Index(j as u32));
                    let val = self.read_input_value(elem_dt, param_name, sub_segs, is_public);
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
                    let mut sub_segs = segments.clone();
                    sub_segs.push(PathSegment::Index(flat_idx as u32));
                    let val = self.read_input_value(&inner_dt, param_name, sub_segs, is_public);
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
                // Inputs come in with a single dynamic dim of unknown
                // length 0..=max_length. The richer per-axis annotation
                // syntax that would let users express tighter bounds is
                // future work; for now we degrade to one Dynamic dim.
                let envelope = crate::types::Envelope::new(vec![crate::types::Dim::new_dynamic(
                    &mut self.builder.dim_table,
                    0,
                    *max_length,
                )]);
                let _ = max_rank;
                Value::DynamicNDArray(crate::types::DynamicNDArrayData {
                    envelope,
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
                let path = InputPath::new(param_name, segments);
                self.builder.ir_read_integer(path, is_public)
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
        // Direct ZinniaType serde format: "Integer", {"NDArray": {...}}, etc.
        serde_json::from_value::<ZinniaType>(dt_json.clone()).unwrap_or(ZinniaType::Integer)
    }
}
