//! Shared function/method spellings with identical ndarray semantics.
use crate::builder::IRBuilder;
use crate::types::Value;
use std::collections::HashMap;

pub(super) fn copy_array(b: &mut IRBuilder, input: &Value) -> Value {
    if let Value::DynamicNDArray(data) = input {
        let mut copied = crate::ops::dyn_ndarray::force_contiguous(b, data);
        let cells = crate::helpers::segment::read_all_typed(
            b,
            copied.segment_id,
            copied.max_length(),
            copied.dtype,
        );
        copied.segment_id =
            crate::helpers::segment::alloc_and_write_values(b, &cells, copied.dtype);
        copied.value_id = crate::types::ValueId::next();
        return Value::DynamicNDArray(copied);
    }
    let values = crate::helpers::static_array::base::deep_to_value_list(b, input);
    crate::helpers::static_array::base::to_static_array(b, &values)
        .expect("np.copy: expected a rectangular numeric array")
}

pub(super) fn clip(b: &mut IRBuilder, args: &[Value], kwargs: &HashMap<String, Value>) -> Value {
    for key in kwargs.keys() {
        assert!(
            ["a_min", "a_max", "min", "max", "out"].contains(&key.as_str()),
            "np.clip: keyword `{key}` is not supported"
        );
    }
    assert!(args.len() <= 4, "np.clip: too many positional arguments");
    let input = args.first().expect("np.clip: requires an array");
    let bound = |index, old: &str, new: &str| {
        let choices = [args.get(index), kwargs.get(old), kwargs.get(new)];
        assert!(
            choices.iter().filter(|v| v.is_some()).count() <= 1,
            "np.clip: multiple values for `{old}`"
        );
        choices
            .into_iter()
            .flatten()
            .next()
            .cloned()
            .unwrap_or(Value::None)
    };
    let low = crate::helpers::static_array::base::deep_to_value_list(b, &bound(1, "a_min", "min"));
    let high = crate::helpers::static_array::base::deep_to_value_list(b, &bound(2, "a_max", "max"));
    // np_where's legacy broadcast path consumes nested composites.
    let mut out = crate::helpers::static_array::base::deep_to_value_list(b, input);
    if !matches!(low, Value::None) {
        out = crate::ops::np_like::vectorize_minmax(
            b,
            &out,
            &low,
            IRBuilder::ir_greater_than_i,
            IRBuilder::ir_greater_than_f,
            true,
        );
    }
    if !matches!(high, Value::None) {
        out = crate::ops::np_like::vectorize_minmax(
            b,
            &out,
            &high,
            IRBuilder::ir_less_than_i,
            IRBuilder::ir_less_than_f,
            false,
        );
    }
    let out = if matches!(input, Value::StaticArray { .. }) {
        crate::helpers::static_array::base::to_static_array(b, &out)
            .expect("np.clip: rectangular numeric output required")
    } else {
        out
    };
    assert!(
        !(args.get(3).is_some() && kwargs.contains_key("out")),
        "np.clip: multiple values for out"
    );
    if let Some(target) = args.get(3).or_else(|| kwargs.get("out")).filter(|v| !matches!(v, Value::None)) {
        write_output(b, &out, target, None)
    } else { out }
}

/// NumPy out= writes through views and aliases, including masked ufunc writes.
pub(super) fn write_output(b: &mut IRBuilder, result: &Value, target: &Value, mask: Option<&Value>) -> Value {
    let target = match target {
        Value::Tuple(d) if d.values.len() == 1 => &d.values[0],
        other => other,
    };
    if matches!(target, Value::None) { return result.clone(); }
    let Value::StaticArray { shape, strides, offset, segment_id, dtype, .. } = target else {
        panic!("out must be a statically shaped ndarray");
    };
    let result = if let Some(mask) = mask {
        let values = [mask, result, target].iter().map(|v| crate::helpers::static_array::base::deep_to_value_list(b, v)).collect::<Vec<_>>();
        crate::ops::static_ndarray_ops::np_where(b, &values)
    } else { result.clone() };
    assert_eq!(*shape, crate::helpers::composite::get_composite_shape(&result), "output shape mismatch");
    let cells = crate::helpers::composite::flatten_composite_with_builder(b, &result);
    assert!(!cells.iter().any(|v| matches!(v, Value::Float(_))) || *dtype == crate::types::NumberType::Float,
            "cannot cast real results to an integer out array");
    crate::helpers::static_array::base::materialize_static_array_if_lazy(b, target);
    if b.static_array_product_provenance.contains_key(segment_id) {
        crate::helpers::static_array::base::force_static_array_product_provenance(b, *segment_id);
    }
    let logical_strides = crate::helpers::shape_arith::row_major_strides(shape);
    for (i, cell) in cells.iter().enumerate() {
        let coordinates = crate::helpers::shape_arith::decode_coords(i, shape, &logical_strides);
        let addr = offset + coordinates.iter().zip(strides).map(|(i,s)| i*s).sum::<usize>();
        let addr = b.ir_constant_int(addr as i64);
        let value = if *dtype == crate::types::NumberType::Float { crate::helpers::value_ops::ensure_float(b, cell) } else { cell.clone() };
        crate::helpers::segment::write_typed(b, *segment_id, &addr, &value, *dtype);
    }
    b.static_array_payload.remove(segment_id);
    target.clone()
}

/// `np.putmask(a, mask, values)`: for every flat position `i` where
/// `mask.flat[i]` holds, set `a.flat[i] = values.flat[i % len(values)]`.
/// NumPy repeats `values` in flat order; it does not broadcast it. `mask`
/// must have as many elements as `a`.
pub(super) fn putmask(b: &mut IRBuilder, target: &Value, mask: &Value, values: &Value) -> Value {
    let Value::StaticArray { shape, dtype, .. } = target else {
        panic!("np.putmask: the first argument must be a statically shaped ndarray");
    };
    let old = crate::helpers::composite::flatten_composite_with_builder(b, target);
    let mask_cells = crate::helpers::composite::flatten_composite_with_builder(b, mask);
    if mask_cells.len() != old.len() {
        panic!(
            "np.putmask: mask and data must be the same size ({} vs {})",
            mask_cells.len(),
            old.len()
        );
    }
    let value_cells = crate::helpers::composite::flatten_composite_with_builder(b, values);
    if value_cells.is_empty() {
        // NumPy leaves `a` unchanged when `values` is empty.
        return target.clone();
    }
    let cells: Vec<Value> = old
        .iter()
        .zip(&mask_cells)
        .enumerate()
        .map(|(i, (old_cell, mask_cell))| {
            let new_cell = &value_cells[i % value_cells.len()];
            let new_cell = match dtype {
                crate::types::NumberType::Float => crate::helpers::value_ops::ensure_float(b, new_cell),
                _ if matches!(new_cell, Value::Float(_)) => panic!(
                    "np.putmask: putting float values into an integer array (a truncating \
                     cast) is not supported"
                ),
                _ => new_cell.clone(),
            };
            let take = crate::helpers::value_ops::to_scalar_bool(b, mask_cell);
            crate::helpers::value_ops::select_value(b, &take, &new_cell, old_cell)
        })
        .collect();
    let result = crate::helpers::static_array::base::build_static_array_from_flat(
        b,
        cells,
        shape.clone(),
        *dtype,
    );
    write_output(b, &result, target, None)
}

/// Reduce bounded dimensions without discarding keepdims or tuple-axis
/// semantics. Retained dimensions keep their existing runtime metadata.
fn bounded_reduction(b: &mut IRBuilder, op: &str, input: &Value, axis: Option<&Value>, keepdims: bool) -> Value {
    let Value::DynamicNDArray(data) = input else {
        return crate::helpers::array_ops::reduce(b, op, input, axis);
    };
    let rank = data.envelope.rank();
    let all_axes = axis.is_none() || matches!(axis, Some(Value::None));
    let mut axes = if all_axes { (0..rank).collect::<Vec<_>>() } else {
        let axis = axis.unwrap();
        let entries = match axis {
            Value::List(d) | Value::Tuple(d) => d.values.clone(),
            _ => vec![axis.clone()],
        };
        entries.iter().map(|entry| {
            let raw = entry.int_val().expect("bounded reduction axes must be statically known integers");
            let normalized = if raw < 0 { raw + rank as i64 } else { raw };
            assert!(normalized >= 0 && normalized < rank as i64, "reduction axis out of bounds");
            normalized as usize
        }).collect::<Vec<_>>()
    };
    axes.sort_unstable();
    assert!(axes.windows(2).all(|pair| pair[0] != pair[1]), "duplicate reduction axis");
    let mut result = if all_axes {
        crate::helpers::array_ops::reduce(b, op, input, None)
    } else {
        let mut value = if axes.is_empty() && matches!(op, "any" | "all") {
            // A zero-axis Boolean reduction still converts each element
            // to truth, whereas numeric reductions preserve each element.
            let zero = b.ir_constant_int(0);
            crate::helpers::value_ops::apply_binary_op(b, "ne", input, &zero)
        } else { input.clone() };
        // Descending order keeps each original axis number valid after
        // removing dimensions to its right.
        for &axis in axes.iter().rev() {
            let index = b.ir_constant_int(axis as i64);
            value = crate::helpers::array_ops::reduce(b, op, &value, Some(&index));
        }
        value
    };
    if keepdims && !axes.is_empty() {
        if matches!(result, Value::DynamicNDArray(_)) {
            for &axis in &axes {
                let index = b.ir_constant_int(axis as i64);
                let Value::DynamicNDArray(data) = &result else { unreachable!() };
                result = crate::ops::dyn_ndarray::reshape::dyn_expand_dims(b, data, &[index]);
            }
        } else {
            // Reducing every dimension produces a scalar. Keep one unit
            // dimension per input axis while retaining its numeric dtype.
            let dtype = match result {
                Value::Float(_) => crate::types::NumberType::Float,
                Value::Complex { .. } => crate::types::NumberType::Complex,
                _ => crate::types::NumberType::Integer,
            };
            if dtype == crate::types::NumberType::Complex {
                let (real, imag) = crate::helpers::value_ops::unpack_value_to_complex_parts(b, &result);
                result = crate::helpers::static_array::base::build_static_array_from_flat_complex(b, vec![real], vec![imag], vec![1; rank]);
            } else {
                result = crate::helpers::static_array::base::build_static_array_from_flat(b, vec![result], vec![1; rank], dtype);
            }
        }
    }
    result
}

pub(super) fn reduction(b: &mut IRBuilder, op: &str, args: &[Value], kwargs: &HashMap<String, Value>) -> Value {
    let positional = if matches!(op, "sum" | "prod") { vec!["axis","dtype","out","keepdims","initial","where"] }
        else { vec!["axis","out","keepdims","initial","where"] };
    assert!(args.len() <= positional.len() + 1, "too many positional reduction arguments");
    let mut kw = kwargs.clone();
    for (key,value) in positional.iter().zip(args.iter().skip(1)) {
        assert!(!kw.contains_key(*key), "multiple values for reduction argument {key}");
        kw.insert((*key).to_string(), value.clone());
    }
    for key in kw.keys() { assert!(["axis","dtype","out","keepdims","initial","where"].contains(&key.as_str()), "unknown reduction keyword {key}"); }
    let original = args.first().expect("reduction requires an array");
    let materialized = if matches!(original,Value::List(_)|Value::Tuple(_)) { crate::helpers::static_array::base::deep_to_value_list(b,original) } else { original.clone() };
    let mut input = crate::helpers::static_array::base::to_static_array(b, &materialized).unwrap_or(materialized);
    if let Some(dtype) = kw.get("dtype").filter(|v| !matches!(v, Value::None)) {
        input = astype(b, &input, dtype);
    }
    let initial = kw.get("initial").filter(|v| !matches!(v, Value::None));
    if let Some(mask) = kw.get("where") {
        let neutral = match op {
            "sum" | "any" => b.ir_constant_int(0),
            "prod" | "all" => b.ir_constant_int(1),
            "min" | "max" => initial.cloned().expect("min/max with where requires initial"),
            _ => unreachable!(),
        };
        let values = [mask, &input, &neutral].iter().map(|v| crate::helpers::static_array::base::deep_to_value_list(b,v)).collect::<Vec<_>>();
        let masked = crate::ops::static_ndarray_ops::np_where(b,&values);
        input = crate::helpers::static_array::base::to_static_array(b,&masked).unwrap_or(masked);
    }
    let keepdims = super::kwarg_static_truthy(&kw,"keepdims",false);
    let mut result = if let Some(value) = crate::helpers::static_array::reductions::try_apply_reduce(b,op,&input,kw.get("axis"),keepdims) { value }
        else { bounded_reduction(b,op,&input,kw.get("axis"),keepdims) };
    if let Some(initial) = initial {
        result = match op {
            "sum" => crate::helpers::value_ops::apply_binary_op(b,"add",&result,initial),
            "prod" => crate::helpers::value_ops::apply_binary_op(b,"mul",&result,initial),
            "min" | "max" => {
                let list = crate::helpers::static_array::base::deep_to_value_list(b,&result);
                let value = if op=="min" { crate::ops::np_like::vectorize_minmax(b,&list,initial,IRBuilder::ir_less_than_i,IRBuilder::ir_less_than_f,false) }
                    else { crate::ops::np_like::vectorize_minmax(b,&list,initial,IRBuilder::ir_greater_than_i,IRBuilder::ir_greater_than_f,true) };
                if matches!(result,Value::StaticArray{..}) { crate::helpers::static_array::base::to_static_array(b,&value).unwrap_or(value) } else { value }
            }
            _ => panic!("initial is only supported for numeric reductions"),
        };
    }
    if let Some(target) = kw.get("out") { write_output(b,&result,target,None) } else { result }
}

/// Preserve ndarray arithmetic after a numeric dtype conversion.
pub(super) fn astype(b: &mut IRBuilder, input: &Value, dtype: &Value) -> Value {
    use crate::types::{NumberType, ZinniaType};
    #[derive(Clone, Copy, PartialEq)]
    enum Target {
        Int,
        Float,
        Bool,
    }
    let target = match dtype {
        Value::Class(ZinniaType::Float) => Target::Float,
        Value::Class(ZinniaType::Integer) => Target::Int,
        Value::Class(ZinniaType::Boolean) => Target::Bool,
        Value::String(s) if ["float", "float64", "f8"].contains(&s.val.as_str()) => Target::Float,
        Value::String(s) if ["int", "int64", "i8"].contains(&s.val.as_str()) => Target::Int,
        Value::String(s) if ["bool", "bool_", "?"].contains(&s.val.as_str()) => Target::Bool,
        _ => panic!("dtype conversion requires int64, float64 or bool"),
    };
    let to_float = target == Target::Float;
    // `astype(bool)` is each element's truth value (NumPy: `x != 0`).
    // `astype(int)` of a bool array is an integer array (its cells stop
    // being bools, so it indexes as positions).
    let convert = |b: &mut IRBuilder, value: &Value| {
        match value {
            Value::Boolean(_) if target == Target::Bool => value.clone(),
            _ if target == Target::Bool => b.ir_bool_cast(value),
            Value::Integer(_) | Value::Boolean(_) if to_float => b.ir_float_cast(value),
            Value::Float(_) | Value::Boolean(_) if !to_float => b.ir_int_cast(value),
            _ => value.clone(),
        }
    };
    if let Value::DynamicNDArray(data) = input {
        let mut result = crate::ops::dyn_ndarray::force_contiguous(b, data);
        let cells = crate::helpers::segment::read_all_typed(b, result.segment_id, result.max_length(), result.dtype);
        let cells = cells.iter().map(|cell| convert(b, cell)).collect::<Vec<_>>();
        result.dtype = if to_float { NumberType::Float } else { NumberType::Integer };
        result.segment_id = crate::helpers::segment::alloc_and_write_values(b, &cells, result.dtype);
        result.value_id = crate::types::ValueId::next();
        return Value::DynamicNDArray(result);
    }
    let shape = crate::helpers::composite::get_composite_shape(input);
    let cells = crate::helpers::composite::flatten_composite_with_builder(b, input);
    let cells = cells.iter().map(|cell| convert(b, cell)).collect();
    crate::helpers::static_array::base::build_static_array_from_flat(b, cells, shape,
        if to_float { NumberType::Float } else { NumberType::Integer })
}

/// Median of a statically shaped array, with ordinary NumPy axis semantics.
pub(super) fn median(b:&mut IRBuilder,args:&[Value],kwargs:&HashMap<String,Value>)->Value {
    use crate::types::{SliceIndex,NumberType};
    let mut input=args[0].clone();
    let axis_arg=kwargs.get("axis").or_else(||args.get(1));
    if let Value::DynamicNDArray(data) = &input {
        if axis_arg.is_some_and(|a| !matches!(a, Value::None)) {
            panic!("np.median of a runtime-shaped array: only axis=None is supported");
        }
        return crate::ops::static_ndarray_ops::runtime_shaped_median(b, data);
    }
    let axis = if axis_arg.is_none() || matches!(axis_arg,Some(Value::None)) {
        input=crate::helpers::static_array::shape::try_apply_flatten(b,&input).expect("median requires numeric array");0
    } else {
        let rank=crate::helpers::composite::get_composite_shape(&input).len() as i64;
        let axis=axis_arg.unwrap().int_val().expect("median axis must be static");
        let normalized=if axis<0 {axis+rank}else{axis};
        assert!(normalized>=0&&normalized<rank,"median axis out of bounds");normalized as usize
    };
    let ax=b.ir_constant_int(axis as i64);
    let sorted=crate::helpers::static_array::sorting::try_apply_sort(b,&input,Some(&ax),false).expect("median requires numeric array");
    let shape=crate::helpers::composite::get_composite_shape(&sorted);
    let n=shape[axis];assert!(n>0,"median requires nonempty reduction axis");
    let mut indices=vec![SliceIndex::Range(None,None,None);shape.len()];
    indices[axis]=SliceIndex::Single(b.ir_constant_int((n/2) as i64));
    let upper=crate::helpers::static_array::read::static_array_subscript(b,&sorted,&indices);
    let result=if n%2==1 {upper} else {
        indices[axis]=SliceIndex::Single(b.ir_constant_int((n/2-1) as i64));
        let lower=crate::helpers::static_array::read::static_array_subscript(b,&sorted,&indices);
        let sum=crate::helpers::value_ops::apply_binary_op(b,"add",&lower,&upper);
        let two=b.ir_constant_float(2.0);
        crate::helpers::value_ops::apply_binary_op(b,"div",&sum,&two)
    };
    let mut out_shape=shape.clone();out_shape.remove(axis);
    if super::kwarg_static_truthy(kwargs,"keepdims",false) {out_shape.insert(axis,1);}
    let cells=crate::helpers::composite::flatten_composite_with_builder(b,&result).iter().map(|cell|crate::helpers::value_ops::ensure_float(b,cell)).collect();
    crate::helpers::static_array::base::build_static_array_from_flat(b,cells,out_shape,NumberType::Float)
}
