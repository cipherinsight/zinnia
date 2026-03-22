//! DynamicNDArray method implementations on IRGenerator.
//!
//! Extends `IRGenerator` with all bounded-dynamic array operations.
//! These mirror the Python `zinnia/op_def/dynamic_ndarray/` operators.

use std::collections::HashMap;

use crate::ir_gen::IRGenerator;
use crate::types::{
    CompositeData, DynArrayMeta, DynamicNDArrayData, NumberType, ScalarValue, Value, ZinniaType,
};

// ═══════════════════════════════════════════════════════════════════════════
// Aggregation kind enum
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Copy)]
pub enum DynAggKind {
    Sum,
    Prod,
    Max,
    Min,
    All,
    Any,
    Argmax,
    Argmin,
}

// ═══════════════════════════════════════════════════════════════════════════
// Phase 1: Utility functions (pure computation, no IR emission)
// ═══════════════════════════════════════════════════════════════════════════

/// Compute row-major strides from shape.
pub fn dyn_row_major_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

/// Decode a linear index into multi-dimensional coordinates.
pub fn dyn_decode_coords(linear: usize, shape: &[usize], strides: &[usize]) -> Vec<usize> {
    shape
        .iter()
        .zip(strides.iter())
        .map(|(&dim, &stride)| (linear / stride) % dim)
        .collect()
}

/// Encode multi-dimensional coordinates into a linear index.
pub fn dyn_encode_coords(coords: &[usize], strides: &[usize]) -> usize {
    coords.iter().zip(strides.iter()).map(|(&c, &s)| c * s).sum()
}

/// Product of shape dimensions.
pub fn dyn_num_elements(shape: &[usize]) -> usize {
    shape.iter().product()
}

// ═══════════════════════════════════════════════════════════════════════════
// Helper: convert between Value and ScalarValue<i64>
// ═══════════════════════════════════════════════════════════════════════════

/// Extract ScalarValue<i64> from a Value::Integer (or coerce Boolean).
pub fn value_to_scalar_i64(val: &Value) -> ScalarValue<i64> {
    match val {
        Value::Integer(sv) => sv.clone(),
        Value::Boolean(sv) => ScalarValue::new(
            sv.static_val.map(|b| if b { 1 } else { 0 }),
            sv.ptr,
        ),
        Value::Float(sv) => ScalarValue::new(
            sv.static_val.map(|f| f as i64),
            sv.ptr,
        ),
        _ => ScalarValue::new(None, None),
    }
}

/// Convert a stored element (ScalarValue<i64>) back to a Value.
pub fn scalar_i64_to_value(elem: &ScalarValue<i64>, dtype: NumberType) -> Value {
    match dtype {
        NumberType::Integer => Value::Integer(elem.clone()),
        NumberType::Float => {
            Value::Float(ScalarValue::new(
                elem.static_val.map(|v| v as f64),
                elem.ptr,
            ))
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// DynamicNDArray method implementations on IRGenerator
// ═══════════════════════════════════════════════════════════════════════════

impl IRGenerator {
    // ── Dispatch router ────────────────────────────────────────────────

    /// Main dispatch for DynamicNDArray method calls.
    pub fn dispatch_dyn_ndarray_method(
        &mut self,
        val: Value,
        method: &str,
        args: &[Value],
        kwargs: &HashMap<String, Value>,
    ) -> Value {
        // Extract data — we need ownership for some ops
        let data = match &val {
            Value::DynamicNDArray(d) => d.clone(),
            _ => panic!("dispatch_dyn_ndarray_method called on non-DynamicNDArray"),
        };

        match method {
            // Phase 2: pure metadata
            "ndim" => self.dyn_ndim(&data),
            "dtype" => self.dyn_dtype(&data),
            "shape" => self.dyn_shape(&data),
            "size" => self.dyn_size(&data),

            // Phase 2: simple value ops
            "astype" => self.dyn_astype(&data, args),
            "flatten" => self.dyn_flatten_to_list(&data),
            "flat" => self.dyn_flat(&data),
            "tolist" => self.dyn_tolist(&data),
            "T" => self.dyn_transpose(&data, &[]),
            "transpose" => {
                let axes_args = if let Some(axes_val) = kwargs.get("axes") {
                    vec![axes_val.clone()]
                } else {
                    args.to_vec()
                };
                self.dyn_transpose(&data, &axes_args)
            }
            "moveaxis" => self.dyn_moveaxis(&data, args),

            // Phase 4: aggregation ops
            "sum" => {
                let axis = kwargs.get("axis").or_else(|| args.first());
                self.dyn_aggregate(&data, axis, DynAggKind::Sum)
            }
            "prod" => {
                let axis = kwargs.get("axis").or_else(|| args.first());
                self.dyn_aggregate(&data, axis, DynAggKind::Prod)
            }
            "max" => {
                let axis = kwargs.get("axis").or_else(|| args.first());
                self.dyn_aggregate(&data, axis, DynAggKind::Max)
            }
            "min" => {
                let axis = kwargs.get("axis").or_else(|| args.first());
                self.dyn_aggregate(&data, axis, DynAggKind::Min)
            }
            "all" => {
                let axis = kwargs.get("axis").or_else(|| args.first());
                self.dyn_aggregate(&data, axis, DynAggKind::All)
            }
            "any" => {
                let axis = kwargs.get("axis").or_else(|| args.first());
                self.dyn_aggregate(&data, axis, DynAggKind::Any)
            }
            "argmax" => {
                let axis = kwargs.get("axis").or_else(|| args.first());
                self.dyn_aggregate(&data, axis, DynAggKind::Argmax)
            }
            "argmin" => {
                let axis = kwargs.get("axis").or_else(|| args.first());
                self.dyn_aggregate(&data, axis, DynAggKind::Argmin)
            }

            // Phase 5: memory-heavy ops
            "filter" => self.dyn_filter(&data, args),
            "repeat" => self.dyn_repeat(&data, args, kwargs),

            _ => panic!(
                "DynamicNDArray.{} not yet implemented in Rust IR generator",
                method
            ),
        }
    }

    // ── Phase 1: utility helpers (on IRGenerator for builder access) ──

    /// Create a default value for the given dtype.
    fn dyn_default_value(&mut self, dtype: NumberType) -> Value {
        match dtype {
            NumberType::Integer => self.builder.ir_constant_int(0),
            NumberType::Float => self.builder.ir_constant_float(0.0),
        }
    }

    /// Flatten a DynamicNDArray's elements respecting view metadata (offset, strides, shape).
    fn dyn_flatten_values(&self, data: &DynamicNDArrayData) -> Vec<ScalarValue<i64>> {
        let shape = &data.meta.logical_shape;
        let strides = &data.meta.logical_strides;
        let offset = data.meta.logical_offset;
        let numel = dyn_num_elements(shape);

        if numel == 0 {
            return vec![];
        }

        let row_strides = dyn_row_major_strides(shape);
        let mut result = Vec::with_capacity(numel);
        for i in 0..numel {
            let coords = dyn_decode_coords(i, shape, &row_strides);
            let src_idx = offset + dyn_encode_coords(&coords, strides);
            if src_idx < data.elements.len() {
                result.push(data.elements[src_idx].clone());
            } else {
                result.push(ScalarValue::new(Some(0), None));
            }
        }
        result
    }

    /// Convert a DynamicNDArray's flat elements to Vec<Value>.
    fn dyn_elements_to_values(&self, data: &DynamicNDArrayData) -> Vec<Value> {
        let flat = self.dyn_flatten_values(data);
        flat.iter()
            .map(|elem| scalar_i64_to_value(elem, data.dtype))
            .collect()
    }

    // ── Phase 2: pure metadata ops ────────────────────────────────────

    fn dyn_ndim(&mut self, data: &DynamicNDArrayData) -> Value {
        self.builder
            .ir_constant_int(data.meta.logical_shape.len() as i64)
    }

    fn dyn_dtype(&self, data: &DynamicNDArrayData) -> Value {
        match data.dtype {
            NumberType::Integer => Value::Class(ZinniaType::Integer),
            NumberType::Float => Value::Class(ZinniaType::Float),
        }
    }

    fn dyn_shape(&mut self, data: &DynamicNDArrayData) -> Value {
        let shape = &data.meta.logical_shape;
        if shape.len() == 1 {
            // 1D: use runtime_length if available (dynamic), else constant
            let len_val = if let Some(v) = data.meta.runtime_length.static_val {
                self.builder.ir_constant_int(v)
            } else if let Some(ptr) = data.meta.runtime_length.ptr {
                Value::Integer(ScalarValue::new(None, Some(ptr)))
            } else {
                self.builder.ir_constant_int(shape[0] as i64)
            };
            let types = vec![ZinniaType::Integer];
            Value::Tuple(CompositeData {
                elements_type: types,
                values: vec![len_val],
            })
        } else {
            let vals: Vec<Value> = shape
                .iter()
                .map(|&s| Value::Integer(ScalarValue::new(Some(s as i64), None)))
                .collect();
            let types = vec![ZinniaType::Integer; vals.len()];
            Value::Tuple(CompositeData {
                elements_type: types,
                values: vals,
            })
        }
    }

    fn dyn_size(&mut self, data: &DynamicNDArrayData) -> Value {
        let shape = &data.meta.logical_shape;
        if shape.len() == 1 {
            // 1D: return runtime_length if dynamic
            if let Some(v) = data.meta.runtime_length.static_val {
                self.builder.ir_constant_int(v)
            } else if let Some(ptr) = data.meta.runtime_length.ptr {
                Value::Integer(ScalarValue::new(None, Some(ptr)))
            } else {
                self.builder.ir_constant_int(shape[0] as i64)
            }
        } else {
            let total: usize = shape.iter().product();
            self.builder.ir_constant_int(total as i64)
        }
    }

    // ── Phase 2: simple value ops ─────────────────────────────────────

    fn dyn_astype(&mut self, data: &DynamicNDArrayData, args: &[Value]) -> Value {
        let target_float = matches!(args.first(), Some(Value::Class(ZinniaType::Float)));
        let new_dtype = if target_float {
            NumberType::Float
        } else {
            NumberType::Integer
        };

        // Cast each element
        let flat = self.dyn_flatten_values(data);
        let new_elements: Vec<ScalarValue<i64>> = flat
            .iter()
            .map(|elem| {
                let val = scalar_i64_to_value(elem, data.dtype);
                let cast_val = if target_float {
                    self.builder.ir_float_cast(&val)
                } else {
                    self.builder.ir_int_cast(&val)
                };
                value_to_scalar_i64(&cast_val)
            })
            .collect();

        Value::DynamicNDArray(DynamicNDArrayData {
            max_length: data.max_length,
            max_rank: data.max_rank,
            dtype: new_dtype,
            elements: new_elements,
            meta: data.meta.clone(),
        })
    }

    fn dyn_flatten_to_list(&self, data: &DynamicNDArrayData) -> Value {
        let values = self.dyn_elements_to_values(data);
        let types = values.iter().map(|v| v.zinnia_type()).collect();
        Value::List(CompositeData {
            elements_type: types,
            values,
        })
    }

    fn dyn_flat(&self, data: &DynamicNDArrayData) -> Value {
        let flat = self.dyn_flatten_values(data);
        Value::DynamicNDArray(DynamicNDArrayData {
            max_length: data.max_length,
            max_rank: 1,
            dtype: data.dtype,
            elements: flat,
            meta: DynArrayMeta {
                logical_shape: vec![data.max_length],
                logical_offset: 0,
                logical_strides: vec![1],
                runtime_length: data.meta.runtime_length.clone(),
                runtime_rank: ScalarValue::new(Some(1), None),
                runtime_shape: vec![data.meta.runtime_length.clone()],
                runtime_strides: vec![ScalarValue::new(Some(1), None)],
                runtime_offset: ScalarValue::new(Some(0), None),
            },
        })
    }

    fn dyn_tolist(&self, data: &DynamicNDArrayData) -> Value {
        let flat_values = self.dyn_elements_to_values(data);
        let shape = &data.meta.logical_shape;
        self.build_nested_list(&flat_values, shape)
    }

    /// Build nested Value::List from flat values and shape.
    fn build_nested_list(&self, flat: &[Value], shape: &[usize]) -> Value {
        if shape.len() <= 1 {
            let len = shape.first().copied().unwrap_or(flat.len());
            let vals: Vec<Value> = flat.iter().take(len).cloned().collect();
            let types = vals.iter().map(|v| v.zinnia_type()).collect();
            return Value::List(CompositeData {
                elements_type: types,
                values: vals,
            });
        }
        let inner_size: usize = shape[1..].iter().product();
        let mut rows = Vec::new();
        for chunk in flat.chunks(inner_size) {
            rows.push(self.build_nested_list(chunk, &shape[1..]));
        }
        let row_types = rows.iter().map(|v| v.zinnia_type()).collect();
        Value::List(CompositeData {
            elements_type: row_types,
            values: rows,
        })
    }

    // ── Phase 2: constructors (zeros, ones, eye) ──────────────────────

    /// DynamicNDArray.zeros(shape, dtype=...)
    pub fn dyn_zeros(&mut self, args: &[Value], kwargs: &HashMap<String, Value>) -> Value {
        self.dyn_fill(args, kwargs, 0)
    }

    /// DynamicNDArray.ones(shape, dtype=...)
    pub fn dyn_ones(&mut self, args: &[Value], kwargs: &HashMap<String, Value>) -> Value {
        self.dyn_fill(args, kwargs, 1)
    }

    fn dyn_fill(
        &mut self,
        args: &[Value],
        kwargs: &HashMap<String, Value>,
        fill_value: i64,
    ) -> Value {
        let shape = self.parse_shape_arg(args.first().expect("zeros/ones: requires shape arg"));
        let dtype = self.parse_dtype_kwarg(kwargs);
        let max_length: usize = shape.iter().product();
        let max_rank = shape.len();

        let fill_sv = match dtype {
            NumberType::Integer => {
                let v = self.builder.ir_constant_int(fill_value);
                value_to_scalar_i64(&v)
            }
            NumberType::Float => {
                let v = self.builder.ir_constant_float(fill_value as f64);
                value_to_scalar_i64(&v)
            }
        };
        let elements = vec![fill_sv; max_length];

        let strides = dyn_row_major_strides(&shape);
        Value::DynamicNDArray(DynamicNDArrayData {
            max_length,
            max_rank,
            dtype,
            elements,
            meta: DynArrayMeta {
                logical_shape: shape.clone(),
                logical_offset: 0,
                logical_strides: strides,
                runtime_length: ScalarValue::new(Some(max_length as i64), None),
                runtime_rank: ScalarValue::new(Some(max_rank as i64), None),
                runtime_shape: shape
                    .iter()
                    .map(|&s| ScalarValue::new(Some(s as i64), None))
                    .collect(),
                runtime_strides: dyn_row_major_strides(&shape)
                    .iter()
                    .map(|&s| ScalarValue::new(Some(s as i64), None))
                    .collect(),
                runtime_offset: ScalarValue::new(Some(0), None),
            },
        })
    }

    /// DynamicNDArray.eye(N, M=None, dtype=...)
    pub fn dyn_eye(&mut self, args: &[Value], kwargs: &HashMap<String, Value>) -> Value {
        let n = args
            .first()
            .and_then(|v| v.int_val())
            .expect("eye: N must be constant int") as usize;
        let m = args
            .get(1)
            .or_else(|| kwargs.get("M"))
            .and_then(|v| v.int_val())
            .unwrap_or(n as i64) as usize;
        let dtype = self.parse_dtype_kwarg(kwargs);

        let max_length = n * m;
        let shape = vec![n, m];
        let strides = dyn_row_major_strides(&shape);

        let zero = match dtype {
            NumberType::Integer => {
                let v = self.builder.ir_constant_int(0);
                value_to_scalar_i64(&v)
            }
            NumberType::Float => {
                let v = self.builder.ir_constant_float(0.0);
                value_to_scalar_i64(&v)
            }
        };
        let one = match dtype {
            NumberType::Integer => {
                let v = self.builder.ir_constant_int(1);
                value_to_scalar_i64(&v)
            }
            NumberType::Float => {
                let v = self.builder.ir_constant_float(1.0);
                value_to_scalar_i64(&v)
            }
        };

        let mut elements = Vec::with_capacity(max_length);
        for i in 0..n {
            for j in 0..m {
                elements.push(if i == j { one.clone() } else { zero.clone() });
            }
        }

        Value::DynamicNDArray(DynamicNDArrayData {
            max_length,
            max_rank: 2,
            dtype,
            elements,
            meta: DynArrayMeta {
                logical_shape: shape.clone(),
                logical_offset: 0,
                logical_strides: strides,
                runtime_length: ScalarValue::new(Some(max_length as i64), None),
                runtime_rank: ScalarValue::new(Some(2), None),
                runtime_shape: shape
                    .iter()
                    .map(|&s| ScalarValue::new(Some(s as i64), None))
                    .collect(),
                runtime_strides: dyn_row_major_strides(&shape)
                    .iter()
                    .map(|&s| ScalarValue::new(Some(s as i64), None))
                    .collect(),
                runtime_offset: ScalarValue::new(Some(0), None),
            },
        })
    }

    fn parse_shape_arg(&self, val: &Value) -> Vec<usize> {
        match val {
            Value::Tuple(data) | Value::List(data) => data
                .values
                .iter()
                .map(|v| v.int_val().expect("shape element must be constant int") as usize)
                .collect(),
            Value::Integer(_) => vec![val.int_val().unwrap() as usize],
            _ => panic!("shape must be tuple, list, or int"),
        }
    }

    fn parse_dtype_kwarg(&self, kwargs: &HashMap<String, Value>) -> NumberType {
        if let Some(Value::Class(ZinniaType::Integer)) = kwargs.get("dtype") {
            NumberType::Integer
        } else if let Some(Value::Class(ZinniaType::Float)) = kwargs.get("dtype") {
            NumberType::Float
        } else {
            NumberType::Float // default to float like Python
        }
    }

    // ── Phase 3: view transforms ──────────────────────────────────────

    fn dyn_transpose(&mut self, data: &DynamicNDArrayData, args: &[Value]) -> Value {
        let shape = &data.meta.logical_shape;
        let strides = &data.meta.logical_strides;
        let ndim = shape.len();

        if ndim <= 1 {
            return Value::DynamicNDArray(data.clone());
        }

        // Determine axis permutation
        let perm: Vec<usize> =
            if args.is_empty() || matches!(args.first(), Some(Value::None)) {
                // Default: reverse all axes
                (0..ndim).rev().collect()
            } else if let Some(Value::Tuple(perm_data)) | Some(Value::List(perm_data)) =
                args.first()
            {
                perm_data
                    .values
                    .iter()
                    .map(|v| {
                        let a = v.int_val().expect("transpose: axes must be constant ints");
                        let resolved = if a < 0 { ndim as i64 + a } else { a };
                        resolved as usize
                    })
                    .collect()
            } else {
                // Multiple int args as axes
                args.iter()
                    .map(|v| {
                        let a = v.int_val().expect("transpose: axes must be constant ints");
                        let resolved = if a < 0 { ndim as i64 + a } else { a };
                        resolved as usize
                    })
                    .collect()
            };

        assert_eq!(perm.len(), ndim, "transpose: permutation length must match rank");

        // Permute shape and strides
        let new_shape: Vec<usize> = perm.iter().map(|&p| shape[p]).collect();
        let new_strides: Vec<usize> = perm.iter().map(|&p| strides[p]).collect();
        let new_runtime_shape: Vec<ScalarValue<i64>> = perm
            .iter()
            .map(|&p| {
                if p < data.meta.runtime_shape.len() {
                    data.meta.runtime_shape[p].clone()
                } else {
                    ScalarValue::new(Some(shape[p] as i64), None)
                }
            })
            .collect();
        let new_runtime_strides: Vec<ScalarValue<i64>> = perm
            .iter()
            .map(|&p| {
                if p < data.meta.runtime_strides.len() {
                    data.meta.runtime_strides[p].clone()
                } else {
                    ScalarValue::new(Some(strides[p] as i64), None)
                }
            })
            .collect();

        Value::DynamicNDArray(DynamicNDArrayData {
            max_length: data.max_length,
            max_rank: data.max_rank,
            dtype: data.dtype,
            elements: data.elements.clone(), // same underlying storage
            meta: DynArrayMeta {
                logical_shape: new_shape,
                logical_offset: data.meta.logical_offset,
                logical_strides: new_strides,
                runtime_length: data.meta.runtime_length.clone(),
                runtime_rank: data.meta.runtime_rank.clone(),
                runtime_shape: new_runtime_shape,
                runtime_strides: new_runtime_strides,
                runtime_offset: data.meta.runtime_offset.clone(),
            },
        })
    }

    fn dyn_moveaxis(&mut self, data: &DynamicNDArrayData, args: &[Value]) -> Value {
        let ndim = data.meta.logical_shape.len();
        assert!(args.len() >= 2, "moveaxis: requires source and destination");

        let src = {
            let s = args[0]
                .int_val()
                .expect("moveaxis: source must be constant int");
            if s < 0 { (ndim as i64 + s) as usize } else { s as usize }
        };
        let dst = {
            let d = args[1]
                .int_val()
                .expect("moveaxis: destination must be constant int");
            if d < 0 { (ndim as i64 + d) as usize } else { d as usize }
        };
        assert!(src < ndim && dst < ndim, "moveaxis: axis out of bounds");

        // Build permutation: remove src, insert at dst
        let mut order: Vec<usize> = (0..ndim).filter(|&i| i != src).collect();
        order.insert(dst, src);

        let axes_val: Vec<Value> = order
            .iter()
            .map(|&a| Value::Integer(ScalarValue::new(Some(a as i64), None)))
            .collect();
        let axes_tuple = Value::Tuple(CompositeData {
            elements_type: vec![ZinniaType::Integer; order.len()],
            values: axes_val,
        });
        self.dyn_transpose(data, &[axes_tuple])
    }

    // ── Phase 4: aggregation ops ──────────────────────────────────────

    fn dyn_aggregate(
        &mut self,
        data: &DynamicNDArrayData,
        axis: Option<&Value>,
        agg: DynAggKind,
    ) -> Value {
        let axis_val = axis.and_then(|v| {
            if matches!(v, Value::None) {
                None
            } else {
                v.int_val()
            }
        });

        match axis_val {
            None => self.dyn_aggregate_all(data, agg),
            Some(ax) => self.dyn_aggregate_axis(data, ax, agg),
        }
    }

    /// Full reduction (axis=None): reduce all elements to a scalar.
    fn dyn_aggregate_all(&mut self, data: &DynamicNDArrayData, agg: DynAggKind) -> Value {
        let flat = self.dyn_flatten_values(data);
        let numel = dyn_num_elements(&data.meta.logical_shape);
        if numel == 0 {
            return self.dyn_agg_identity(agg, data.dtype);
        }

        let use_float = data.dtype == NumberType::Float
            && !matches!(agg, DynAggKind::All | DynAggKind::Any);

        let first_val = scalar_i64_to_value(&flat[0], data.dtype);
        let mut acc = first_val.clone();
        let mut acc_idx = self.builder.ir_constant_int(0);

        for i in 1..numel.min(flat.len()) {
            let elem = scalar_i64_to_value(&flat[i], data.dtype);
            let idx_val = self.builder.ir_constant_int(i as i64);
            let (new_acc, new_idx) =
                self.dyn_agg_step(&acc, &acc_idx, &elem, &idx_val, agg, use_float);
            acc = new_acc;
            acc_idx = new_idx;
        }

        // For argmax/argmin, return the index
        match agg {
            DynAggKind::Argmax | DynAggKind::Argmin => acc_idx,
            _ => acc,
        }
    }

    /// Axis reduction: reduce along a specific axis.
    fn dyn_aggregate_axis(
        &mut self,
        data: &DynamicNDArrayData,
        axis: i64,
        agg: DynAggKind,
    ) -> Value {
        let shape = &data.meta.logical_shape;
        let ndim = shape.len();
        let ax = if axis < 0 {
            (ndim as i64 + axis) as usize
        } else {
            axis as usize
        };
        assert!(ax < ndim, "aggregate axis out of bounds");

        let flat = self.dyn_flatten_values(data);
        let strides = dyn_row_major_strides(shape);
        let use_float = data.dtype == NumberType::Float
            && !matches!(agg, DynAggKind::All | DynAggKind::Any);

        // Output shape: remove the reduced axis
        let out_shape: Vec<usize> = shape
            .iter()
            .enumerate()
            .filter(|&(i, _)| i != ax)
            .map(|(_, &s)| s)
            .collect();
        let out_numel: usize = if out_shape.is_empty() {
            1
        } else {
            out_shape.iter().product()
        };
        let out_strides = dyn_row_major_strides(&out_shape);
        let axis_dim = shape[ax];

        let mut out_elements = Vec::with_capacity(out_numel);

        for out_idx in 0..out_numel {
            // Decode output coordinates
            let out_coords = if out_shape.is_empty() {
                vec![]
            } else {
                dyn_decode_coords(out_idx, &out_shape, &out_strides)
            };

            // Build input coordinates: insert axis position
            let mut in_coords = out_coords.clone();
            in_coords.insert(ax, 0);

            // Initialize accumulator with first element along axis
            let first_src_idx = dyn_encode_coords(&in_coords, &strides);
            let first_elem = if first_src_idx < flat.len() {
                scalar_i64_to_value(&flat[first_src_idx], data.dtype)
            } else {
                self.dyn_default_value(data.dtype)
            };

            let mut acc = first_elem;
            let mut acc_idx = self.builder.ir_constant_int(0);

            // Iterate along reduction axis
            for k in 1..axis_dim {
                in_coords[ax] = k;
                let src_idx = dyn_encode_coords(&in_coords, &strides);
                let elem = if src_idx < flat.len() {
                    scalar_i64_to_value(&flat[src_idx], data.dtype)
                } else {
                    self.dyn_default_value(data.dtype)
                };
                let k_val = self.builder.ir_constant_int(k as i64);
                let (new_acc, new_idx) =
                    self.dyn_agg_step(&acc, &acc_idx, &elem, &k_val, agg, use_float);
                acc = new_acc;
                acc_idx = new_idx;
            }

            let result = match agg {
                DynAggKind::Argmax | DynAggKind::Argmin => acc_idx,
                _ => acc,
            };
            out_elements.push(value_to_scalar_i64(&result));
        }

        // Determine output dtype
        let out_dtype = match agg {
            DynAggKind::All | DynAggKind::Any | DynAggKind::Argmax | DynAggKind::Argmin => {
                NumberType::Integer
            }
            _ => data.dtype,
        };

        if out_shape.is_empty() {
            // Scalar result
            return scalar_i64_to_value(&out_elements[0], out_dtype);
        }

        let out_strides_meta = dyn_row_major_strides(&out_shape);
        Value::DynamicNDArray(DynamicNDArrayData {
            max_length: out_numel,
            max_rank: out_shape.len(),
            dtype: out_dtype,
            elements: out_elements,
            meta: DynArrayMeta {
                logical_shape: out_shape.clone(),
                logical_offset: 0,
                logical_strides: out_strides_meta.clone(),
                runtime_length: ScalarValue::new(Some(out_numel as i64), None),
                runtime_rank: ScalarValue::new(Some(out_shape.len() as i64), None),
                runtime_shape: out_shape
                    .iter()
                    .map(|&s| ScalarValue::new(Some(s as i64), None))
                    .collect(),
                runtime_strides: out_strides_meta
                    .iter()
                    .map(|&s| ScalarValue::new(Some(s as i64), None))
                    .collect(),
                runtime_offset: ScalarValue::new(Some(0), None),
            },
        })
    }

    /// One step of the accumulator for a given aggregation kind.
    fn dyn_agg_step(
        &mut self,
        acc: &Value,
        acc_idx: &Value,
        elem: &Value,
        elem_idx: &Value,
        agg: DynAggKind,
        use_float: bool,
    ) -> (Value, Value) {
        match agg {
            DynAggKind::Sum => {
                let new_acc = if use_float {
                    self.builder.ir_add_f(acc, elem)
                } else {
                    self.builder.ir_add_i(acc, elem)
                };
                (new_acc, acc_idx.clone())
            }
            DynAggKind::Prod => {
                let new_acc = if use_float {
                    self.builder.ir_mul_f(acc, elem)
                } else {
                    self.builder.ir_mul_i(acc, elem)
                };
                (new_acc, acc_idx.clone())
            }
            DynAggKind::Max => {
                let cond = if use_float {
                    self.builder.ir_greater_than_f(elem, acc)
                } else {
                    self.builder.ir_greater_than_i(elem, acc)
                };
                let new_acc = if use_float {
                    self.builder.ir_select_f(&cond, elem, acc)
                } else {
                    self.builder.ir_select_i(&cond, elem, acc)
                };
                let new_idx = self.builder.ir_select_i(&cond, elem_idx, acc_idx);
                (new_acc, new_idx)
            }
            DynAggKind::Min => {
                let cond = if use_float {
                    self.builder.ir_less_than_f(elem, acc)
                } else {
                    self.builder.ir_less_than_i(elem, acc)
                };
                let new_acc = if use_float {
                    self.builder.ir_select_f(&cond, elem, acc)
                } else {
                    self.builder.ir_select_i(&cond, elem, acc)
                };
                let new_idx = self.builder.ir_select_i(&cond, elem_idx, acc_idx);
                (new_acc, new_idx)
            }
            DynAggKind::All => {
                let b = self.builder.ir_bool_cast(elem);
                let new_acc = self.builder.ir_logical_and(acc, &b);
                (new_acc, acc_idx.clone())
            }
            DynAggKind::Any => {
                let b = self.builder.ir_bool_cast(elem);
                let new_acc = self.builder.ir_logical_or(acc, &b);
                (new_acc, acc_idx.clone())
            }
            DynAggKind::Argmax => {
                let cond = if use_float {
                    self.builder.ir_greater_than_f(elem, acc)
                } else {
                    self.builder.ir_greater_than_i(elem, acc)
                };
                let new_acc = if use_float {
                    self.builder.ir_select_f(&cond, elem, acc)
                } else {
                    self.builder.ir_select_i(&cond, elem, acc)
                };
                let new_idx = self.builder.ir_select_i(&cond, elem_idx, acc_idx);
                (new_acc, new_idx)
            }
            DynAggKind::Argmin => {
                let cond = if use_float {
                    self.builder.ir_less_than_f(elem, acc)
                } else {
                    self.builder.ir_less_than_i(elem, acc)
                };
                let new_acc = if use_float {
                    self.builder.ir_select_f(&cond, elem, acc)
                } else {
                    self.builder.ir_select_i(&cond, elem, acc)
                };
                let new_idx = self.builder.ir_select_i(&cond, elem_idx, acc_idx);
                (new_acc, new_idx)
            }
        }
    }

    /// Identity value for aggregation init (used when array is empty).
    fn dyn_agg_identity(&mut self, agg: DynAggKind, dtype: NumberType) -> Value {
        match agg {
            DynAggKind::Sum => self.dyn_default_value(dtype),
            DynAggKind::Prod => match dtype {
                NumberType::Integer => self.builder.ir_constant_int(1),
                NumberType::Float => self.builder.ir_constant_float(1.0),
            },
            DynAggKind::All => self.builder.ir_constant_bool(true),
            DynAggKind::Any => self.builder.ir_constant_bool(false),
            DynAggKind::Max | DynAggKind::Min => self.dyn_default_value(dtype),
            DynAggKind::Argmax | DynAggKind::Argmin => self.builder.ir_constant_int(0),
        }
    }

    // ── Phase 5: memory-heavy ops ─────────────────────────────────────

    /// DynamicNDArray.filter(mask)
    fn dyn_filter(&mut self, data: &DynamicNDArrayData, args: &[Value]) -> Value {
        let mask = args
            .first()
            .expect("filter: requires a mask argument");
        let elements = self.dyn_elements_to_values(data);

        // Get mask elements
        let mask_elements: Vec<Value> = match mask {
            Value::DynamicNDArray(md) => self.dyn_elements_to_values(md),
            Value::List(cd) | Value::Tuple(cd) => cd.values.clone(),
            _ => panic!("filter: mask must be array-like"),
        };

        let max_len = data.max_length;

        // Build output via compaction with write pointer
        let mut write_ptr = self.builder.ir_constant_int(0);
        let mut out_values: Vec<Value> = (0..max_len)
            .map(|_| self.dyn_default_value(data.dtype))
            .collect();

        for i in 0..max_len.min(elements.len()) {
            // Get mask value (or default false)
            let mask_val = if i < mask_elements.len() {
                mask_elements[i].clone()
            } else {
                self.builder.ir_constant_int(0)
            };
            // Check if mask is truthy
            let keep = self.builder.ir_bool_cast(&mask_val);

            // Conditionally place element at write_ptr position
            for j in 0..max_len {
                let j_const = self.builder.ir_constant_int(j as i64);
                let is_target = self.builder.ir_equal_i(&write_ptr, &j_const);
                let should_write = self.builder.ir_logical_and(&keep, &is_target);
                out_values[j] = if data.dtype == NumberType::Float {
                    self.builder
                        .ir_select_f(&should_write, &elements[i], &out_values[j])
                } else {
                    self.builder
                        .ir_select_i(&should_write, &elements[i], &out_values[j])
                };
            }

            // Increment write_ptr if keep
            let one = self.builder.ir_constant_int(1);
            let zero = self.builder.ir_constant_int(0);
            let inc = self.builder.ir_select_i(&keep, &one, &zero);
            write_ptr = self.builder.ir_add_i(&write_ptr, &inc);
        }

        let new_elements: Vec<ScalarValue<i64>> =
            out_values.iter().map(|v| value_to_scalar_i64(v)).collect();

        Value::DynamicNDArray(DynamicNDArrayData {
            max_length: max_len,
            max_rank: 1,
            dtype: data.dtype,
            elements: new_elements,
            meta: DynArrayMeta {
                logical_shape: vec![max_len],
                logical_offset: 0,
                logical_strides: vec![1],
                runtime_length: value_to_scalar_i64(&write_ptr),
                runtime_rank: ScalarValue::new(Some(1), None),
                runtime_shape: vec![value_to_scalar_i64(&write_ptr)],
                runtime_strides: vec![ScalarValue::new(Some(1), None)],
                runtime_offset: ScalarValue::new(Some(0), None),
            },
        })
    }

    /// DynamicNDArray.repeat(repeats, axis=...)
    fn dyn_repeat(
        &mut self,
        data: &DynamicNDArrayData,
        args: &[Value],
        kwargs: &HashMap<String, Value>,
    ) -> Value {
        let repeats = args
            .first()
            .and_then(|v| v.int_val())
            .expect("repeat: repeats must be a constant integer") as usize;
        let axis = kwargs
            .get("axis")
            .or_else(|| args.get(1))
            .and_then(|v| v.int_val());

        let flat = self.dyn_flatten_values(data);
        let numel = dyn_num_elements(&data.meta.logical_shape);

        if let Some(_ax) = axis {
            // For axis-specific repeat, flatten first, repeat, return 1D
            // (full axis support would need coordinate transforms)
            let mut new_elements = Vec::new();
            for elem in flat.iter().take(numel) {
                for _ in 0..repeats {
                    new_elements.push(elem.clone());
                }
            }
            let new_len = new_elements.len();
            Value::DynamicNDArray(DynamicNDArrayData {
                max_length: new_len,
                max_rank: 1,
                dtype: data.dtype,
                elements: new_elements,
                meta: DynArrayMeta {
                    logical_shape: vec![new_len],
                    logical_offset: 0,
                    logical_strides: vec![1],
                    runtime_length: ScalarValue::new(Some(new_len as i64), None),
                    runtime_rank: ScalarValue::new(Some(1), None),
                    runtime_shape: vec![ScalarValue::new(Some(new_len as i64), None)],
                    runtime_strides: vec![ScalarValue::new(Some(1), None)],
                    runtime_offset: ScalarValue::new(Some(0), None),
                },
            })
        } else {
            // No axis: flatten then repeat each element
            let mut new_elements = Vec::new();
            for elem in flat.iter().take(numel) {
                for _ in 0..repeats {
                    new_elements.push(elem.clone());
                }
            }
            let new_len = new_elements.len();
            Value::DynamicNDArray(DynamicNDArrayData {
                max_length: new_len,
                max_rank: 1,
                dtype: data.dtype,
                elements: new_elements,
                meta: DynArrayMeta {
                    logical_shape: vec![new_len],
                    logical_offset: 0,
                    logical_strides: vec![1],
                    runtime_length: ScalarValue::new(Some(new_len as i64), None),
                    runtime_rank: ScalarValue::new(Some(1), None),
                    runtime_shape: vec![ScalarValue::new(Some(new_len as i64), None)],
                    runtime_strides: vec![ScalarValue::new(Some(1), None)],
                    runtime_offset: ScalarValue::new(Some(0), None),
                },
            })
        }
    }

    /// DynamicNDArray.concatenate(arrays, axis=0)
    pub fn dyn_concatenate(
        &mut self,
        args: &[Value],
        kwargs: &HashMap<String, Value>,
    ) -> Value {
        let arrays_val = args
            .first()
            .expect("concatenate: requires arrays argument");
        let arrays: Vec<&DynamicNDArrayData> = match arrays_val {
            Value::List(cd) | Value::Tuple(cd) => cd
                .values
                .iter()
                .map(|v| match v {
                    Value::DynamicNDArray(d) => d,
                    _ => panic!("concatenate: all elements must be DynamicNDArray"),
                })
                .collect(),
            _ => panic!("concatenate: first arg must be list/tuple of arrays"),
        };
        if arrays.is_empty() {
            return Value::None;
        }

        let axis = kwargs
            .get("axis")
            .or_else(|| args.get(1))
            .and_then(|v| v.int_val())
            .unwrap_or(0);
        let ndim = arrays[0].meta.logical_shape.len();
        let ax = if axis < 0 {
            (ndim as i64 + axis) as usize
        } else {
            axis as usize
        };
        assert!(ax < ndim, "concatenate: axis out of bounds");

        // Compute output shape
        let base_shape = &arrays[0].meta.logical_shape;
        let mut out_shape = base_shape.clone();
        let concat_dim: usize = arrays.iter().map(|a| a.meta.logical_shape[ax]).sum();
        out_shape[ax] = concat_dim;
        let out_numel: usize = out_shape.iter().product();
        let out_strides = dyn_row_major_strides(&out_shape);

        // Build output elements by coordinate mapping
        let mut out_elements = Vec::with_capacity(out_numel);
        // Compute axis offsets for each source
        let mut axis_offsets = Vec::new();
        let mut offset = 0usize;
        for arr in &arrays {
            axis_offsets.push(offset);
            offset += arr.meta.logical_shape[ax];
        }

        let dtype = arrays[0].dtype;

        for i in 0..out_numel {
            let out_coords = dyn_decode_coords(i, &out_shape, &out_strides);

            // Find which source array this coordinate belongs to
            let ax_coord = out_coords[ax];
            let mut src_idx = 0;
            for (si, &off) in axis_offsets.iter().enumerate() {
                let next = if si + 1 < axis_offsets.len() {
                    axis_offsets[si + 1]
                } else {
                    concat_dim
                };
                if ax_coord >= off && ax_coord < next {
                    src_idx = si;
                    break;
                }
            }

            // Adjust coordinate for source
            let mut src_coords = out_coords.clone();
            src_coords[ax] -= axis_offsets[src_idx];

            // Read from source
            let src = arrays[src_idx];
            let src_strides = dyn_row_major_strides(&src.meta.logical_shape);
            let src_flat = dyn_encode_coords(&src_coords, &src_strides);
            let src_linear = src.meta.logical_offset + src_flat;

            let elem = if src_linear < src.elements.len() {
                src.elements[src_linear].clone()
            } else {
                ScalarValue::new(Some(0), None)
            };
            out_elements.push(elem);
        }

        Value::DynamicNDArray(DynamicNDArrayData {
            max_length: out_numel,
            max_rank: ndim,
            dtype,
            elements: out_elements,
            meta: DynArrayMeta {
                logical_shape: out_shape.clone(),
                logical_offset: 0,
                logical_strides: out_strides.clone(),
                runtime_length: ScalarValue::new(Some(out_numel as i64), None),
                runtime_rank: ScalarValue::new(Some(ndim as i64), None),
                runtime_shape: out_shape
                    .iter()
                    .map(|&s| ScalarValue::new(Some(s as i64), None))
                    .collect(),
                runtime_strides: out_strides
                    .iter()
                    .map(|&s| ScalarValue::new(Some(s as i64), None))
                    .collect(),
                runtime_offset: ScalarValue::new(Some(0), None),
            },
        })
    }

    /// DynamicNDArray.stack(arrays, axis=0)
    pub fn dyn_stack(
        &mut self,
        args: &[Value],
        kwargs: &HashMap<String, Value>,
    ) -> Value {
        let arrays_val = args
            .first()
            .expect("stack: requires arrays argument");
        let arrays: Vec<&DynamicNDArrayData> = match arrays_val {
            Value::List(cd) | Value::Tuple(cd) => cd
                .values
                .iter()
                .map(|v| match v {
                    Value::DynamicNDArray(d) => d,
                    _ => panic!("stack: all elements must be DynamicNDArray"),
                })
                .collect(),
            _ => panic!("stack: first arg must be list/tuple of arrays"),
        };
        if arrays.is_empty() {
            return Value::None;
        }

        let axis = kwargs
            .get("axis")
            .or_else(|| args.get(1))
            .and_then(|v| v.int_val())
            .unwrap_or(0);
        let base_shape = &arrays[0].meta.logical_shape;
        let ndim = base_shape.len();
        let ax = if axis < 0 {
            (ndim as i64 + 1 + axis) as usize
        } else {
            axis as usize
        };
        assert!(ax <= ndim, "stack: axis out of bounds");

        // Output shape: insert num_arrays at axis position
        let num_arrays = arrays.len();
        let mut out_shape = base_shape.clone();
        out_shape.insert(ax, num_arrays);
        let out_numel: usize = out_shape.iter().product();
        let out_strides = dyn_row_major_strides(&out_shape);
        let out_ndim = out_shape.len();

        let dtype = arrays[0].dtype;
        let mut out_elements = Vec::with_capacity(out_numel);

        for i in 0..out_numel {
            let out_coords = dyn_decode_coords(i, &out_shape, &out_strides);
            // The axis coordinate selects which source array
            let src_idx = out_coords[ax];
            // Remove axis coordinate to get source coordinates
            let mut src_coords: Vec<usize> = out_coords.clone();
            src_coords.remove(ax);

            let src = arrays[src_idx];
            let src_strides = dyn_row_major_strides(&src.meta.logical_shape);
            let src_linear = src.meta.logical_offset + dyn_encode_coords(&src_coords, &src_strides);

            let elem = if src_linear < src.elements.len() {
                src.elements[src_linear].clone()
            } else {
                ScalarValue::new(Some(0), None)
            };
            out_elements.push(elem);
        }

        Value::DynamicNDArray(DynamicNDArrayData {
            max_length: out_numel,
            max_rank: out_ndim,
            dtype,
            elements: out_elements,
            meta: DynArrayMeta {
                logical_shape: out_shape.clone(),
                logical_offset: 0,
                logical_strides: out_strides.clone(),
                runtime_length: ScalarValue::new(Some(out_numel as i64), None),
                runtime_rank: ScalarValue::new(Some(out_ndim as i64), None),
                runtime_shape: out_shape
                    .iter()
                    .map(|&s| ScalarValue::new(Some(s as i64), None))
                    .collect(),
                runtime_strides: out_strides
                    .iter()
                    .map(|&s| ScalarValue::new(Some(s as i64), None))
                    .collect(),
                runtime_offset: ScalarValue::new(Some(0), None),
            },
        })
    }
}
