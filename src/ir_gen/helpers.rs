use crate::types::{CompositeData, Value, ValueId, ZinniaType};

use super::{IRGenerator, SliceIndex};

impl IRGenerator {
    pub(crate) fn cast_composite(&mut self, val: &Value, to_float: bool) -> Value {
        match val {
            Value::List(data) | Value::Tuple(data) => {
                let is_tuple = matches!(val, Value::Tuple(_));
                let new_values: Vec<Value> = data
                    .values
                    .iter()
                    .map(|v| self.cast_composite(v, to_float))
                    .collect();
                let new_types = new_values.iter().map(|v| v.zinnia_type()).collect();
                if is_tuple {
                    Value::Tuple(CompositeData {
                        elements_type: new_types,
                        values: new_values,
                        value_id: ValueId::next(),
                    })
                } else {
                    Value::List(CompositeData {
                        elements_type: new_types,
                        values: new_values,
                        value_id: ValueId::next(),
                    })
                }
            }
            Value::Integer(_) | Value::Boolean(_) if to_float => self.builder.ir_float_cast(val),
            Value::Float(_) if !to_float => self.builder.ir_int_cast(val),
            _ => val.clone(),
        }
    }

    /// Set a nested value in a composite structure using slice indices.
    /// Cast a value to match the target element dtype if they differ.
    pub(crate) fn try_cast_value_to_match(
        &mut self,
        value: Value,
        target_type: &ZinniaType,
    ) -> Value {
        let vt = value.zinnia_type();
        if vt == *target_type {
            return value;
        }
        match (target_type, &value) {
            (ZinniaType::Integer, Value::Float(_)) => self.builder.ir_int_cast(&value),
            (ZinniaType::Float, Value::Integer(_)) => self.builder.ir_float_cast(&value),
            (ZinniaType::Float, Value::Boolean(_)) => self.builder.ir_float_cast(&value),
            (ZinniaType::Integer, Value::Boolean(_)) => self.builder.ir_bool_cast(&value),
            _ => value,
        }
    }

    pub(crate) fn cast_value_to_match(&mut self, value: Value, target_type: &ZinniaType) -> Value {
        let value = self.try_cast_value_to_match(value, target_type);
        let vt = value.zinnia_type();
        if vt != *target_type {
            panic!("Type mismatch: expected {}, got {}", target_type, vt);
        }
        value
    }

    pub(crate) fn set_nested_value(
        &mut self,
        current: Value,
        indices: &[SliceIndex],
        value: Value,
    ) -> Value {
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
                        let idx = if idx < 0 {
                            (data.values.len() as i64 + idx) as usize
                        } else {
                            idx as usize
                        };
                        if idx < data.values.len() {
                            let mut new_values = data.values.clone();
                            let mut new_types = data.elements_type.clone();
                            if indices.len() == 1 {
                                // Cast value to match target element type
                                let target_et = &data.elements_type[idx];
                                new_values[idx] = self.cast_value_to_match(value, target_et);
                                new_types[idx] = new_values[idx].zinnia_type();
                            } else {
                                new_values[idx] = self.set_nested_value(
                                    new_values[idx].clone(),
                                    &indices[1..],
                                    value,
                                );
                                new_types[idx] = new_values[idx].zinnia_type();
                            }
                            return if is_tuple {
                                Value::Tuple(CompositeData {
                                    elements_type: new_types,
                                    values: new_values,
                                    value_id: ValueId::next(),
                                })
                            } else {
                                Value::List(CompositeData {
                                    elements_type: new_types,
                                    values: new_values,
                                    value_id: ValueId::next(),
                                })
                            };
                        }
                    } else {
                        // Dynamic index — use mux/memory path
                        if indices.len() == 1 {
                            // Single dynamic index on flat list
                            return crate::helpers::value_ops::dynamic_list_set_item(
                                &mut self.builder,
                                data,
                                idx_val,
                                &value,
                            );
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
                            Value::Tuple(CompositeData {
                                elements_type: new_types,
                                values: new_values,
                                value_id: ValueId::next(),
                            })
                        } else {
                            Value::List(CompositeData {
                                elements_type: new_types,
                                values: new_values,
                                value_id: ValueId::next(),
                            })
                        };
                    }
                }
                // For range slicing assignment
                if let SliceIndex::Range(start, stop, step) = &indices[0] {
                    let len = data.values.len() as i64;
                    let start_idx = start.as_ref().and_then(|v| v.int_val()).unwrap_or(0);
                    let stop_idx = stop.as_ref().and_then(|v| v.int_val()).unwrap_or(len);
                    let step_val = step.as_ref().and_then(|v| v.int_val()).unwrap_or(1);
                    let start_idx = if start_idx < 0 {
                        (len + start_idx).max(0)
                    } else {
                        start_idx.min(len)
                    } as usize;
                    let stop_idx = if stop_idx < 0 {
                        (len + stop_idx).max(0)
                    } else {
                        stop_idx.min(len)
                    } as usize;

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
                                new_values[i] = self.cast_value_to_match(
                                    rhs_data.values[rhs_idx].clone(),
                                    target_et,
                                );
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
                        Value::Tuple(CompositeData {
                            elements_type: new_types,
                            values: new_values,
                            value_id: ValueId::next(),
                        })
                    } else {
                        Value::List(CompositeData {
                            elements_type: new_types,
                            values: new_values,
                            value_id: ValueId::next(),
                        })
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
            ZinniaType::Complex => {
                // Encode a complex input as a 2-element list [real, imag]
                // at the same input path. The runtime supplies pairs.
                let mut real_segs = segments.clone();
                real_segs.push(PathSegment::Index(0));
                let real_val = self
                    .builder
                    .ir_read_float(InputPath::new(param_name, real_segs), is_public);
                let mut imag_segs = segments;
                imag_segs.push(PathSegment::Index(1));
                let imag_val = self
                    .builder
                    .ir_read_float(InputPath::new(param_name, imag_segs), is_public);
                let r = match real_val {
                    Value::Float(s) => s,
                    _ => unreachable!("ir_read_float returns Value::Float"),
                };
                let i = match imag_val {
                    Value::Float(s) => s,
                    _ => unreachable!("ir_read_float returns Value::Float"),
                };
                Value::Complex { real: r, imag: i }
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

                // Compute hash of flattened inner values and assert equality.
                // P1 segarr: use the builder-aware flattener so a StaticArray
                // inner value (e.g. PoseidonHashed[NDArray[...]]) materialises
                // via the segment payload before hashing.
                let flat = crate::helpers::composite::flatten_composite_with_builder(
                    &mut self.builder,
                    &inner_val,
                );
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
                    crate::types::NumberType::Complex => ZinniaType::Complex,
                };
                let mut values = Vec::new();
                for flat_idx in 0..total {
                    let mut sub_segs = segments.clone();
                    sub_segs.push(PathSegment::Index(flat_idx as u32));
                    values.push(self.read_input_value(&inner_dt, param_name, sub_segs, is_public));
                }
                // P1 segarr-foundation: numeric NDArray params become
                // segment-backed `Value::StaticArray`. P5a adds Complex via a
                // dual-segment StaticArray.
                match dtype {
                    crate::types::NumberType::Integer | crate::types::NumberType::Float => {
                        crate::helpers::static_array::base::build_static_array_from_flat(
                            &mut self.builder,
                            values,
                            shape.clone(),
                            *dtype,
                        )
                    }
                    crate::types::NumberType::Complex => {
                        // Split each Complex element into (real, imag) Floats.
                        let mut reals = Vec::with_capacity(values.len());
                        let mut imags = Vec::with_capacity(values.len());
                        for v in values {
                            match v {
                                Value::Complex { real, imag } => {
                                    reals.push(Value::Float(real));
                                    imags.push(Value::Float(imag));
                                }
                                _ => unreachable!(
                                    "Complex NDArray param: leaf must be Value::Complex"
                                ),
                            }
                        }
                        crate::helpers::static_array::base::build_static_array_from_flat_complex(
                            &mut self.builder,
                            reals,
                            imags,
                            shape.clone(),
                        )
                    }
                }
            }
            ZinniaType::SymbolicNDArray {
                shape,
                dtype,
                bindings,
            } => {
                if let Ok(concrete_shape) = crate::types::concretize_shape_dims(shape, bindings) {
                    let concrete = ZinniaType::NDArray {
                        shape: concrete_shape,
                        dtype: *dtype,
                    };
                    return self.read_input_value(&concrete, param_name, segments, is_public);
                }
                panic!(
                    "symbolic shapes must be resolved before IR generation: SymbolicNDArray[{}, {}]",
                    dtype,
                    shape
                        .iter()
                        .map(|d| d.to_string())
                        .collect::<Vec<_>>()
                        .join(", ")
                )
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
                Value::List(CompositeData {
                    elements_type: types,
                    values,
                    value_id: ValueId::next(),
                })
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
                Value::Tuple(CompositeData {
                    elements_type: types,
                    values,
                    value_id: ValueId::next(),
                })
            }
            ZinniaType::DynamicNDArray {
                dtype,
                max_length,
                max_rank,
                max_shape,
            } => {
                let inner_dt = match dtype {
                    crate::types::NumberType::Integer => ZinniaType::Integer,
                    crate::types::NumberType::Float => ZinniaType::Float,
                    crate::types::NumberType::Complex => ZinniaType::Complex,
                };
                // Read flat payload elements
                let mut elements = Vec::new();
                for flat_idx in 0..*max_length {
                    let mut sub_segs = segments.clone();
                    sub_segs.push(PathSegment::Index(flat_idx as u32));
                    let val = self.read_input_value(&inner_dt, param_name, sub_segs, is_public);
                    elements.push(val);
                }

                // Emit metadata allocation
                let arr_id = self.builder.alloc_array_id();
                let dtype_name = match dtype {
                    crate::types::NumberType::Integer => "int".to_string(),
                    crate::types::NumberType::Float => "float".to_string(),
                    crate::types::NumberType::Complex => "complex".to_string(),
                };
                self.builder.ir_allocate_dynamic_ndarray_meta(
                    arr_id,
                    dtype_name,
                    *max_length as u32,
                    *max_rank as u32,
                    max_shape
                        .clone()
                        .unwrap_or_else(|| vec![*max_length; *max_rank])
                        .into_iter()
                        .map(|n| n as u32)
                        .collect(),
                );

                let segment_id = crate::helpers::segment::alloc_and_write_values(
                    &mut self.builder,
                    &elements,
                    *dtype,
                );

                let max_shape = max_shape
                    .clone()
                    .unwrap_or_else(|| vec![*max_length; *max_rank]);
                let strides = crate::ops::dyn_ndarray::dyn_row_major_strides(&max_shape);
                // Dynamic inputs use a compact flat buffer of max_length
                // payload slots. Runtime shape metadata carries the actual
                // rank-width axes; product(runtime_shape) is asserted against
                // the compact bound by dynamic metadata consumers.
                let envelope = crate::types::Envelope::new_with_bound(
                    max_shape
                        .iter()
                        .map(|axis_bound| {
                            crate::types::Dim::new_dynamic(
                                &mut self.builder.dim_table,
                                0,
                                *axis_bound,
                            )
                        })
                        .collect(),
                    *max_length,
                );
                let runtime_shape_values: Vec<Value> = (0..*max_rank)
                    .map(|axis| {
                        self.builder.ir_read_integer(
                            InputPath::new(
                                param_name,
                                vec![PathSegment::RuntimeShape(axis as u32)],
                            ),
                            true,
                        )
                    })
                    .collect();
                self.builder.ir_witness_dynamic_ndarray_meta(
                    arr_id,
                    *max_rank as u32,
                    param_name.to_string(),
                    &runtime_shape_values,
                );
                let runtime_length_value = if runtime_shape_values.is_empty() {
                    self.builder.ir_constant_int(1)
                } else {
                    let mut acc = runtime_shape_values[0].clone();
                    for axis_len in runtime_shape_values.iter().skip(1) {
                        acc = self.builder.ir_mul_i(&acc, axis_len);
                    }
                    acc
                };
                let bound_value = self.builder.ir_constant_int(*max_length as i64);
                let in_bound = self
                    .builder
                    .ir_less_than_or_equal_i(&runtime_length_value, &bound_value);
                self.builder.ir_assert(&in_bound);
                let runtime_length =
                    crate::ops::dyn_ndarray::value_to_scalar_i64(&runtime_length_value);
                let runtime_rank = self.builder.ir_constant_int(*max_rank as i64);
                let runtime_rank = crate::ops::dyn_ndarray::value_to_scalar_i64(&runtime_rank);
                let runtime_shape: Vec<crate::types::ScalarValue<i64>> = runtime_shape_values
                    .iter()
                    .map(crate::ops::dyn_ndarray::value_to_scalar_i64)
                    .collect();
                let mut runtime_strides: Vec<crate::types::ScalarValue<i64>> =
                    vec![crate::types::ScalarValue::new(Some(1), None); *max_rank];
                for axis in (0..max_rank.saturating_sub(1)).rev() {
                    let next_stride = if let Some(n) = runtime_strides[axis + 1].static_val {
                        self.builder.ir_constant_int(n)
                    } else if let Some(stmt_id) = runtime_strides[axis + 1].stmt_id {
                        Value::Integer(crate::types::ScalarValue::new(None, Some(stmt_id)))
                    } else {
                        self.builder.ir_constant_int(0)
                    };
                    let stride_value = self
                        .builder
                        .ir_mul_i(&next_stride, &runtime_shape_values[axis + 1]);
                    runtime_strides[axis] =
                        crate::ops::dyn_ndarray::value_to_scalar_i64(&stride_value);
                }
                Value::DynamicNDArray(crate::types::DynamicNDArrayData {
                    envelope,
                    dtype: *dtype,
                    segment_id,
                    meta: crate::types::DynArrayMeta {
                        logical_shape: max_shape,
                        logical_offset: 0,
                        logical_strides: strides,
                        runtime_length,
                        runtime_rank,
                        runtime_shape,
                        runtime_strides,
                        runtime_offset: crate::types::ScalarValue::new(Some(0), None),
                    },
                    value_id: ValueId::next(),
                })
            }
            _ => {
                let path = InputPath::new(param_name, segments);
                self.builder.ir_read_integer(path, is_public)
            }
        }
    }

    pub(crate) fn register_global_datatypes(&mut self) {
        // Register Float, Integer, and Complex as class values
        let float_class = Value::Class(ZinniaType::Float);
        let int_class = Value::Class(ZinniaType::Integer);
        let complex_class = Value::Class(ZinniaType::Complex);
        for name in &["Float", "float"] {
            self.ctx.set(name, float_class.clone());
        }
        for name in &[
            "Integer", "int", "Int", "integer", "Boolean", "bool", "Bool", "boolean",
        ] {
            self.ctx.set(name, int_class.clone());
        }
        for name in &["Complex", "complex"] {
            self.ctx.set(name, complex_class.clone());
        }
    }

    pub(crate) fn parse_dt_descriptor(&self, dt_json: &serde_json::Value) -> ZinniaType {
        // Direct ZinniaType serde format: "Integer", {"NDArray": {...}}, etc.
        serde_json::from_value::<ZinniaType>(dt_json.clone()).unwrap_or_else(|e| {
            panic!(
                "invalid type annotation descriptor; symbolic shapes must be resolved before IR generation: {} ({})",
                dt_json, e
            )
        })
    }
}
