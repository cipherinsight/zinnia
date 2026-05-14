use std::collections::HashMap;

use crate::builder::IRBuilder;
use crate::optim::resolver::{require_provable_static_int, SiteKind};
use crate::types::{ValueId, 
    DynArrayMeta, DynamicNDArrayData, NumberType, ScalarValue, Value, ZinniaType,
};

use super::{dyn_row_major_strides, value_to_scalar_i64};

pub fn dyn_fill(
    b: &mut IRBuilder,
    args: &[Value],
    kwargs: &HashMap<String, Value>,
    fill_value: i64,
) -> Value {
    let shape = parse_shape_arg(b, args.first().expect("zeros/ones: requires shape arg"));
    let dtype = parse_dtype_kwarg(kwargs);
    let max_length: usize = shape.iter().product();
    let max_rank = shape.len();

    let fill_sv = match dtype {
        NumberType::Integer => {
            let v = b.ir_constant_int(fill_value);
            value_to_scalar_i64(&v)
        }
        NumberType::Float => {
            let v = b.ir_constant_float(fill_value as f64);
            value_to_scalar_i64(&v)
        }
        NumberType::Complex => panic!(
            "DynamicNDArray of Complex is not yet supported (compiler.complex-ndarray-ops scope)"
        ),
    };
    let values = vec![fill_sv; max_length];
    let segment_id = crate::helpers::segment::alloc_and_write(b, &values, dtype);

    let strides = dyn_row_major_strides(&shape);
    let _ = max_rank;
    let envelope = crate::types::Envelope::from_static_shape(&mut b.dim_table, &shape);
    let result = DynamicNDArrayData {
        envelope,
        dtype,
        segment_id,
        meta: DynArrayMeta {
            logical_shape: shape.clone(),
            logical_offset: 0,
            logical_strides: strides,
            runtime_length: ScalarValue::new(Some(max_length as i64), None),
            runtime_rank: ScalarValue::new(Some(shape.len() as i64), None),
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
        value_id: ValueId::next(),
    };
    Value::DynamicNDArray(result)
}

/// DynamicNDArray.eye(N, M=None, dtype=...)
pub fn dyn_eye(b: &mut IRBuilder, args: &[Value], kwargs: &HashMap<String, Value>) -> Value {
    let n = args
        .first()
        .and_then(|v| v.int_val())
        .expect("eye: N must be constant int") as usize;
    let m = args
        .get(1)
        .or_else(|| kwargs.get("M"))
        .and_then(|v| v.int_val())
        .unwrap_or(n as i64) as usize;
    let dtype = parse_dtype_kwarg(kwargs);

    let max_length = n * m;
    let shape = vec![n, m];
    let strides = dyn_row_major_strides(&shape);

    let zero = match dtype {
        NumberType::Integer => {
            let v = b.ir_constant_int(0);
            value_to_scalar_i64(&v)
        }
        NumberType::Float => {
            let v = b.ir_constant_float(0.0);
            value_to_scalar_i64(&v)
        }
        NumberType::Complex => panic!(
            "DynamicNDArray of Complex is not yet supported (compiler.complex-ndarray-ops scope)"
        ),
    };
    let one = match dtype {
        NumberType::Integer => {
            let v = b.ir_constant_int(1);
            value_to_scalar_i64(&v)
        }
        NumberType::Float => {
            let v = b.ir_constant_float(1.0);
            value_to_scalar_i64(&v)
        }
        NumberType::Complex => panic!(
            "DynamicNDArray of Complex is not yet supported (compiler.complex-ndarray-ops scope)"
        ),
    };

    let mut values = Vec::with_capacity(max_length);
    for i in 0..n {
        for j in 0..m {
            values.push(if i == j { one.clone() } else { zero.clone() });
        }
    }
    let segment_id = crate::helpers::segment::alloc_and_write(b, &values, dtype);

    let envelope = crate::types::Envelope::from_static_shape(&mut b.dim_table, &shape);
    let result = DynamicNDArrayData {
        envelope,
        dtype,
        segment_id,
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
        value_id: ValueId::next(),
    };
    Value::DynamicNDArray(result)
}

/// Build a 1-D `DynamicNDArray` from a fully-materialized buffer of
/// scalar values with a runtime-active length. `values.len()` becomes
/// the static envelope's `max_length`; `active_len` becomes the runtime
/// length.
///
/// This is the primitive used by both [`dyn_fill_with_active`] (fill
/// semantics: all-same fill value) and the bounded-admission paths in
/// `np.arange` / `np.tile` (arbitrary values). The contract `Output ==
/// active` fired by `dyn_fill_with_active` is specific to fill semantics
/// (where the runtime length is the only thing the buffer's contents
/// depend on); callers that fit a different relational fact must fire
/// their own contract on the returned value.
pub fn dyn_from_values_with_active(
    b: &mut IRBuilder,
    values: Vec<ScalarValue<i64>>,
    active_len: Value,
    dtype: NumberType,
) -> Value {
    let max_length = values.len();
    let segment_id = crate::helpers::segment::alloc_and_write(b, &values, dtype);

    // Static envelope: 1-D with the proven max_length.
    let logical_shape: Vec<usize> = vec![max_length];
    let logical_strides = dyn_row_major_strides(&logical_shape);
    let envelope = crate::types::Envelope::from_static_shape(&mut b.dim_table, &logical_shape);

    // Runtime active size: the user's `k`. Extract its ScalarValue.
    let active_sv = value_to_scalar_i64(&active_len);

    let result = DynamicNDArrayData {
        envelope,
        dtype,
        segment_id,
        meta: DynArrayMeta {
            logical_shape,
            logical_offset: 0,
            logical_strides: logical_strides.clone(),
            runtime_length: active_sv.clone(),
            runtime_rank: ScalarValue::new(Some(1), None),
            runtime_shape: vec![active_sv.clone()],
            // 1-D row-major strides at runtime: stride 0 is 1, period.
            runtime_strides: vec![ScalarValue::new(Some(1), None)],
            runtime_offset: ScalarValue::new(Some(0), None),
        },
        value_id: ValueId::next(),
    };

    Value::DynamicNDArray(result)
}

/// Multi-dim sibling of [`dyn_from_values_with_active`]. Builds an N-D
/// `DynamicNDArray` whose envelope's per-axis maxima come from `max_shape`
/// and whose runtime shape comes from `runtime_shape` (one SSA scalar per
/// axis, must match `max_shape.len()`). `runtime_length` is the proven SSA
/// scalar product of `runtime_shape`.
///
/// **Layout convention.** Both `logical_strides` and `runtime_strides` use
/// `row_major(max_shape)`. This is sound when the buffer content is *position-
/// independent inside the active region* — uniform fill (np_fill multi-dim)
/// and the identity matrix (where slot `i * N_max + j` carries the same
/// value `[i == j]` for any `N <= N_max`) both satisfy this. Constructors
/// whose buffer contents depend on the runtime shape must NOT use this
/// primitive.
pub fn dyn_from_values_with_active_nd(
    b: &mut IRBuilder,
    values: Vec<ScalarValue<i64>>,
    max_shape: Vec<usize>,
    runtime_shape: Vec<ScalarValue<i64>>,
    runtime_length: Value,
    dtype: NumberType,
) -> Value {
    let max_length: usize = max_shape.iter().product();
    assert_eq!(
        values.len(),
        max_length,
        "dyn_from_values_with_active_nd: values length {} != product(max_shape) {}",
        values.len(),
        max_length,
    );
    assert_eq!(
        runtime_shape.len(),
        max_shape.len(),
        "dyn_from_values_with_active_nd: runtime_shape rank {} != max_shape rank {}",
        runtime_shape.len(),
        max_shape.len(),
    );

    let segment_id = crate::helpers::segment::alloc_and_write(b, &values, dtype);
    let logical_strides = dyn_row_major_strides(&max_shape);
    let envelope = crate::types::Envelope::from_static_shape(&mut b.dim_table, &max_shape);
    let runtime_length_sv = value_to_scalar_i64(&runtime_length);

    let rank = max_shape.len();
    let result = DynamicNDArrayData {
        envelope,
        dtype,
        segment_id,
        meta: DynArrayMeta {
            logical_shape: max_shape.clone(),
            logical_offset: 0,
            logical_strides: logical_strides.clone(),
            runtime_length: runtime_length_sv,
            runtime_rank: ScalarValue::new(Some(rank as i64), None),
            runtime_shape,
            runtime_strides: logical_strides
                .iter()
                .map(|&s| ScalarValue::new(Some(s as i64), None))
                .collect(),
            runtime_offset: ScalarValue::new(Some(0), None),
        },
        value_id: ValueId::next(),
    };

    Value::DynamicNDArray(result)
}

/// Compact-buffer multi-D constructor. Sibling of
/// [`dyn_from_values_with_active_nd`] for the imbalanced-bound case where
/// `total_bound < product(max_shape)` (e.g., `np.zeros((m, n))` with
/// `m_max = n_max = 100` and `@requires(m * n <= 100)`).
///
/// Allocates a buffer of `total_bound` slots (vs `product(max_shape)`).
/// `runtime_strides[k]` is set to the SSA-`Value` chain
/// `prod(runtime_shape[k+1:])` so subscript-read against
/// [`StrideMode::SymbolicRuntime`] gets the right per-axis address
/// arithmetic. `runtime_length = prod(runtime_shape)` is the final SSA
/// product.
///
/// **Strict mode** (`ZINNIA_BOUNDED_AXIS_STRICT=1`): emits an
/// `assert prod(runtime_shape) <= total_bound` via
/// `lower_precondition_to_ir` so the witness side flags users whose
/// `@requires` facts are wrong. Default (lenient): trusts the user.
///
/// **Soundness invariant:** in lenient mode the user's `@requires` facts
/// establish `prod(runtime_shape) <= total_bound`. Reads beyond
/// `total_bound` are clamped by the segment-init value (typically 0);
/// strict mode upgrades that to a witness-time assertion.
pub fn dyn_from_values_with_active_compact(
    b: &mut IRBuilder,
    fill_value: ScalarValue<i64>,
    max_shape: Vec<usize>,
    runtime_shape: Vec<ScalarValue<i64>>,
    total_bound: usize,
    dtype: NumberType,
) -> Value {
    assert!(
        total_bound < max_shape.iter().product::<usize>(),
        "dyn_from_values_with_active_compact: total_bound {} should be < product(max_shape) {}; \
         use dyn_from_values_with_active_nd for the equal case",
        total_bound,
        max_shape.iter().product::<usize>(),
    );
    assert_eq!(
        runtime_shape.len(),
        max_shape.len(),
        "dyn_from_values_with_active_compact: runtime_shape rank {} != max_shape rank {}",
        runtime_shape.len(),
        max_shape.len(),
    );

    // Allocate the compact buffer at `total_bound` slots only.
    let values = vec![fill_value; total_bound];
    let segment_id = crate::helpers::segment::alloc_and_write(b, &values, dtype);

    // Build runtime_strides as the SSA-Value chain.
    // For row-major, axis k's stride is prod(runtime_shape[k+1:]).
    let rank = max_shape.len();
    let mut shape_values: Vec<Value> = runtime_shape
        .iter()
        .map(|sv| {
            if let Some(v) = sv.static_val {
                b.ir_constant_int(v)
            } else if let Some(ptr) = sv.stmt_id {
                Value::Integer(ScalarValue::new(None, Some(ptr)))
            } else {
                b.ir_constant_int(0)
            }
        })
        .collect();
    let mut runtime_strides: Vec<ScalarValue<i64>> = vec![ScalarValue::new(Some(1), None); rank];
    if rank > 0 {
        // Stride of last axis = 1 (already set).
        // Stride of axis k = stride[k+1] * runtime_shape[k+1].
        for k in (0..rank.saturating_sub(1)).rev() {
            let prev_stride_v = if let Some(v) = runtime_strides[k + 1].static_val {
                b.ir_constant_int(v)
            } else if let Some(ptr) = runtime_strides[k + 1].stmt_id {
                Value::Integer(ScalarValue::new(None, Some(ptr)))
            } else {
                b.ir_constant_int(0)
            };
            let new_stride = b.ir_mul_i(&prev_stride_v, &shape_values[k + 1]);
            runtime_strides[k] = value_to_scalar_i64(&new_stride);
        }
    }

    // Build runtime_length as prod(runtime_shape) via ir_mul_i chain.
    let runtime_length = if shape_values.is_empty() {
        b.ir_constant_int(1)
    } else {
        let mut acc = shape_values.remove(0);
        for v in &shape_values {
            acc = b.ir_mul_i(&acc, v);
        }
        acc
    };

    // Strict mode: assert prod(runtime_shape) <= total_bound at witness time.
    if crate::helpers::array_ops::bounded_axis_strict() {
        let bound_val = b.ir_constant_int(total_bound as i64);
        let le = b.ir_less_than_or_equal_i(&runtime_length, &bound_val);
        b.ir_assert(&le);
    }

    let runtime_length_sv = value_to_scalar_i64(&runtime_length);
    let logical_strides = dyn_row_major_strides(&max_shape);
    let dims: Vec<crate::types::Dim> = max_shape
        .iter()
        .map(|&s| crate::types::Dim::new_dynamic(&mut b.dim_table, 0, s))
        .collect();
    let envelope = crate::types::Envelope::new_with_bound(dims, total_bound);

    Value::DynamicNDArray(DynamicNDArrayData {
        envelope,
        dtype,
        segment_id,
        meta: DynArrayMeta {
            logical_shape: max_shape,
            logical_offset: 0,
            logical_strides,
            runtime_length: runtime_length_sv,
            runtime_rank: ScalarValue::new(Some(rank as i64), None),
            runtime_shape,
            runtime_strides,
            runtime_offset: ScalarValue::new(Some(0), None),
        },
        value_id: ValueId::next(),
    })
}

/// Build a 1-D `DynamicNDArray` whose envelope (`logical_shape`) is the
/// static `max_length` but whose runtime active size is `active_len`
/// (a `Value::Integer` whose `int_val` may be `None` — i.e., a symbolic
/// scalar resolved at prove time).
///
/// Used by `np_fill`'s bounded-fallback path: when `np.zeros(k, ...)` has
/// `k` proven `<= max_length` (via the structural-predicate machinery)
/// but `k` itself is symbolic, we build a dyn-ndarray with `max_length`
/// of slack and the runtime length set to `k`. Downstream subscript-read
/// / subscript-write ops handle the dyn-ndarray representation already.
pub fn dyn_fill_with_active(
    b: &mut IRBuilder,
    max_length: usize,
    active_len: Value,
    fill_value: i64,
    dtype: NumberType,
) -> Value {
    let fill_sv = match dtype {
        NumberType::Integer => {
            let v = b.ir_constant_int(fill_value);
            value_to_scalar_i64(&v)
        }
        NumberType::Float => {
            let v = b.ir_constant_float(fill_value as f64);
            value_to_scalar_i64(&v)
        }
        NumberType::Complex => panic!(
            "DynamicNDArray of Complex is not yet supported (compiler.complex-ndarray-ops scope)"
        ),
    };
    let values = vec![fill_sv; max_length];
    let result = dyn_from_values_with_active(b, values, active_len.clone(), dtype);

    // Op contract for the dyn-ndarray allocator
    // (compiler.fact-propagation-framework prototype + compiler.contract-fact-binding):
    // look up the template in the contract registry, bind its `Output`
    // formal to the runtime-length SSA ptr, and insert the instantiated
    // facts into the FactStack.
    //
    // The result's "stmt_id for length-reasoning purposes" is `active_sv.stmt_id`
    // (= the ptr of the input `active_len`'s scalar). For `zeros(k)`
    // specifically that aliases the user's `k`, which is intentional —
    // any `@requires(k >= 0)` fact about k is shared with len(out) via
    // ptr identity. The contract on top contributes `len(out) >= 0` as a
    // structural guarantee even when the user supplied no such bound.
    {
        // Multi-formal: bind the `active` formal to the input active-length
        // ValueId so the equality clause `Output == Formal("active")`
        // instantiates to a usable relational fact. When the constructor
        // aliases the input directly (the typical case for `zeros(k, ...)`),
        // Output and active resolve to the same ValueId and the equality
        // becomes a harmless `v == v` tautology.
        let active_sv = value_to_scalar_i64(&active_len);
        let mut formals = std::collections::HashMap::new();
        formals.insert("active".to_string(), active_sv.value_id);
        b.fire_contract(
            "dyn_fill_with_active",
            active_sv.value_id,
            &formals,
        );
    }

    // Group 4a (compiler.op-fact-group-4a-fill-constructors-forall-eq-const):
    // fill values 0 / 1 deposit `forall_eq_const(out, k)` on the result so
    // downstream consumers (Group 3d sum-on-constant, future where/select
    // arm-copy) can specialize. Fired right next to the length contract
    // because the content fact is the second half of the same constructor
    // semantics. Other fill values are handled by Group 4b's multi-formal
    // `full` contract.
    if let Some(name) = match fill_value {
        0 => Some("zeros_content"),
        1 => Some("ones_content"),
        _ => None,
    } {
        if let Some(vid) = result.value_id() {
            b.fire_contract(name, vid, &std::collections::HashMap::new());
        }
    }

    result
}

pub fn parse_shape_arg(b: &mut IRBuilder, val: &Value) -> Vec<usize> {
    match val {
        Value::Tuple(data) | Value::List(data) => data
            .values
            .iter()
            .enumerate()
            .map(|(i, v)| {
                let n: i64 = require_provable_static_int(b, v, SiteKind::ShapeAxis(i));
                n as usize
            })
            .collect(),
        Value::Integer(_) => {
            let n: i64 = require_provable_static_int(b, val, SiteKind::ShapeAxis(0));
            vec![n as usize]
        }
        _ => panic!("shape must be tuple, list, or int"),
    }
}

pub fn parse_dtype_kwarg(kwargs: &HashMap<String, Value>) -> NumberType {
    if let Some(Value::Class(ZinniaType::Integer)) = kwargs.get("dtype") {
        NumberType::Integer
    } else if let Some(Value::Class(ZinniaType::Float)) = kwargs.get("dtype") {
        NumberType::Float
    } else {
        NumberType::Float // default to float like Python
    }
}

// ---------------------------------------------------------------------------
// Tests for the fact-propagation prototype (compiler.fact-propagation-framework)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optim::predicates::formula::{CmpOp, ContractTerm, ContractVar};

    #[test]
    fn dyn_fill_with_active_fires_nonneg_length_fact() {
        let mut b = IRBuilder::new();
        let k = b.ir_constant_int(8);
        let k_vid = k.value_id().expect("ir_constant_int returns a value_id'd Value");

        let _out = dyn_fill_with_active(
            &mut b,
            /* max_length */ 16,
            /* active_len */ k.clone(),
            /* fill_value */ 0,
            NumberType::Integer,
        );

        // Expected fact: runtime_length(out) >= 0, anchored at k's
        // value_id (the active_sv aliases k's scalar; the contract
        // anchors on the active-len value_id).
        let expected = ContractTerm::Cmp {
            op: CmpOp::Ge,
            lhs: Box::new(ContractTerm::Var(ContractVar::Value(k_vid))),
            rhs: Box::new(ContractTerm::LitInt(0)),
        };

        let facts_at_k = b
            .facts
            .per_value
            .get(&k_vid)
            .expect("expected at least one fact anchored at the runtime-length value_id");
        assert!(
            facts_at_k.iter().any(|f| *f == expected),
            "fact set at k_vid={k_vid} did not contain the expected `>= 0` postcondition; got {facts_at_k:?}"
        );
    }
}
