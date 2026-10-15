//! `np.*` functions bound to their NumPy signatures here and computed by
//! other array operations: `np.ptp` is `max - min`, `np.count_nonzero` a
//! sum of `!= 0`, `np.average` a mean or a weighted sum over a sum,
//! `np.trace` the sum of `np.diagonal`, `np.ravel` a flattening view. The
//! cell-level functions (`np.kron`, `np.cross`, `np.polyval`, ...) are in
//! `ops::static_ndarray_ops`; this module only binds their arguments.

use std::collections::HashMap;

use crate::builder::IRBuilder;
use crate::helpers::static_args::{static_bool_arg, static_int_arg};
use crate::ops::static_ndarray_ops as ops;
use crate::types::{CompositeData, NumberType, Value, ValueId};

use super::array_compat::reduction;

/// The functions of this module. `trapz`, NumPy's name of `trapezoid` until
/// NumPy 2.4, stays accepted: see "Removed NumPy Names" in
/// docs-site/content/programming-guide/numpy-api.mdx.
pub(super) const FUNCTIONS: &[&str] = &[
    "average", "count_nonzero", "cross", "diagonal", "ediff1d", "kron", "polyval", "ptp", "ravel",
    "trace", "trapezoid", "trapz", "tri", "vander", "vdot",
];

/// `np.<name>(*args, **kwargs)` for a name of [`FUNCTIONS`].
pub(super) fn call(b: &mut IRBuilder, name: &str, args: &[Value], kwargs: &HashMap<String, Value>) -> Value {
    let op = format!("np.{name}");
    let bind = |parameters: &[&'static str]| Arguments::bind(&op, parameters, args, kwargs);
    match name {
        "average" => {
            let a = bind(&["a", "axis", "weights", "returned", "*", "keepdims"]);
            average(b, &a)
        }
        "count_nonzero" => {
            let a = bind(&["a", "axis", "*", "keepdims"]);
            let input = array_argument(b, a.required("a"));
            let zero = b.ir_constant_int(0);
            let nonzero = crate::helpers::value_ops::apply_binary_op(b, "ne", &input, &zero);
            reduction(b, "sum", &[nonzero], &a.only(&["axis", "keepdims"]))
        }
        "cross" => {
            let a = bind(&["a", "b", "axisa", "axisb", "axisc", "axis"]);
            for (name, default) in [("axisa", -1), ("axisb", -1), ("axisc", -1)] {
                if static_int_arg(a.get(name), default, &format!("{op}: `{name}`")) != -1 {
                    panic!("{op}: only the last axis holds the vectors (`{name}=-1`)");
                }
            }
            if a.get("axis").is_some() {
                panic!("{op}: only the last axis holds the vectors (`axis=None`)");
            }
            ops::np_cross(b, a.required("a"), a.required("b"))
        }
        "diagonal" => {
            let a = bind(&["a", "offset", "axis1", "axis2"]);
            let offset = static_int_arg(a.get("offset"), 0, &format!("{op}: `offset`"));
            let axis1 = static_int_arg(a.get("axis1"), 0, &format!("{op}: `axis1`"));
            let axis2 = static_int_arg(a.get("axis2"), 1, &format!("{op}: `axis2`"));
            ops::np_diagonal(b, a.required("a"), offset, axis1, axis2)
        }
        "ediff1d" => {
            let a = bind(&["ary", "to_end", "to_begin"]);
            ops::np_ediff1d(b, a.required("ary"), a.get("to_end"), a.get("to_begin"))
        }
        "kron" => {
            let a = bind(&["a", "b"]);
            ops::np_kron(b, a.required("a"), a.required("b"))
        }
        "polyval" => {
            let a = bind(&["p", "x"]);
            ops::np_polyval(b, a.required("p"), a.required("x"))
        }
        "ptp" => {
            let a = bind(&["a", "axis", "out", "keepdims"]);
            let input = array_argument(b, a.required("a"));
            if crate::helpers::array_ops::is_bool_array(b, &input) || b.is_bool_cell(&input) {
                panic!("{op}: boolean arrays are not supported (NumPy's subtract rejects them)");
            }
            let reduce = a.only(&["axis", "keepdims"]);
            let max = reduction(b, "max", &[input.clone()], &reduce);
            let min = reduction(b, "min", &[input], &reduce);
            let range = crate::helpers::value_ops::apply_binary_op(b, "sub", &max, &min);
            match a.get("out") {
                Some(out) => super::write_into(b, &range, out),
                None => range,
            }
        }
        "ravel" => {
            let a = bind(&["a", "order"]);
            if a.get("order").is_some_and(|order| order.string_val() != Some("C")) {
                panic!("{op}: only order='C' is supported");
            }
            ravel(b, a.required("a"))
        }
        "trace" => {
            let a = bind(&["a", "offset", "axis1", "axis2", "dtype", "out"]);
            let offset = static_int_arg(a.get("offset"), 0, &format!("{op}: `offset`"));
            let axis1 = static_int_arg(a.get("axis1"), 0, &format!("{op}: `axis1`"));
            let axis2 = static_int_arg(a.get("axis2"), 1, &format!("{op}: `axis2`"));
            let diagonal = ops::np_diagonal(b, a.required("a"), offset, axis1, axis2);
            let mut reduce = a.only(&["dtype", "out"]);
            reduce.insert("axis".to_string(), b.ir_constant_int(-1));
            reduction(b, "sum", &[diagonal], &reduce)
        }
        "trapezoid" | "trapz" => {
            let a = bind(&["y", "x", "dx", "axis"]);
            let axis = static_int_arg(a.get("axis"), -1, &format!("{op}: `axis`"));
            let default_dx = b.ir_constant_float(1.0);
            let spacing = match (a.get("x"), a.get("dx")) {
                (Some(x), _) => ops::TrapezoidSpacing::Points(x),
                (None, Some(dx @ (Value::Integer(_) | Value::Float(_)))) => ops::TrapezoidSpacing::Constant(dx),
                (None, Some(_)) => panic!("{op}: `dx` must be a number"),
                (None, None) => ops::TrapezoidSpacing::Constant(&default_dx),
            };
            ops::np_trapezoid(b, a.required("y"), spacing, axis)
        }
        "tri" => {
            let a = bind(&["N", "M", "k", "dtype", "*", "like"]);
            let n = dimension(a.required("N"), &op, "N");
            let m = a.get("M").map(|m| dimension(m, &op, "M"));
            let k = static_int_arg(a.get("k"), 0, &format!("{op}: `k`"));
            let dtype = a.get("dtype").map(|d| crate::helpers::dtype::Dtype::required(d, &op));
            if a.get("like").is_some() {
                panic!("{op}: `like` is not supported");
            }
            ops::np_tri(b, n, m, k, dtype)
        }
        "vander" => {
            let a = bind(&["x", "N", "increasing"]);
            let n = a.get("N").map(|n| dimension(n, &op, "N"));
            let increasing = static_bool_arg(a.get("increasing"), false, &format!("{op}: `increasing`"));
            ops::np_vander(b, a.required("x"), n, increasing)
        }
        "vdot" => {
            let a = bind(&["a", "b"]);
            ops::np_vdot(b, a.required("a"), a.required("b"))
        }
        _ => unreachable!("np.{name} is not a function of this module"),
    }
}

/// `np.average(a, axis=None, weights=None, returned=False, keepdims=False)`.
/// Without weights it is the mean. With weights it is NumPy's
/// `sum(a * w, axis) / sum(w, axis)` in floats; a zero weight sum has no
/// proof (NumPy raises `ZeroDivisionError`). Weights have `a`'s shape, or
/// are 1-D along a given integer `axis`. `returned=True` also gives the
/// sum of the weights (the number of elements averaged, without weights),
/// with the average's shape.
fn average(b: &mut IRBuilder, a: &Arguments) -> Value {
    let input = array_argument(b, a.required("a"));
    let reduce = a.only(&["axis", "keepdims"]);
    let returned = static_bool_arg(a.get("returned"), false, "np.average: `returned`");
    let (avg, scale) = match a.get("weights") {
        None => {
            let avg = mean(b, &input, &reduce);
            let scale = returned.then(|| element_count_like(b, &input, &avg));
            (avg, scale)
        }
        Some(weights) => {
            let input = as_float_array(b, &input);
            let weights = broadcast_weights(b, &input, weights, a.get("axis"));
            let scale = reduction(b, "sum", &[weights.clone()], &reduce);
            let weighted = crate::helpers::value_ops::apply_binary_op(b, "mul", &input, &weights);
            let total = reduction(b, "sum", &[weighted], &reduce);
            let avg = crate::helpers::value_ops::apply_binary_op(b, "div", &total, &scale);
            (avg, Some(scale))
        }
    };
    match scale.filter(|_| returned) {
        Some(scale) => {
            let values = vec![avg, scale];
            Value::Tuple(CompositeData {
                elements_type: values.iter().map(|v| v.zinnia_type()).collect(),
                values,
                value_id: ValueId::next(),
            })
        }
        None => avg,
    }
}

/// The mean of `input` (`axis=`, `keepdims=` in `reduce`): the static
/// reduction of the `np.mean` arm of `try_numpy_func`, and for an array
/// with a runtime length the bounded mean of `x.mean()`.
fn mean(b: &mut IRBuilder, input: &Value, reduce: &HashMap<String, Value>) -> Value {
    let keepdims = super::kwarg_static_truthy(reduce, "keepdims", false);
    if let Value::DynamicNDArray(data) = input {
        if keepdims {
            panic!("np.average(keepdims=True) is not supported for an array with a runtime shape");
        }
        return crate::ops::dyn_ndarray::aggregation::dyn_mean(b, data, reduce.get("axis"));
    }
    if let Some(out) = crate::helpers::static_array::reductions::try_apply_reduce(
        b,
        "mean",
        input,
        reduce.get("axis"),
        keepdims,
    ) {
        return out;
    }
    let legacy = crate::helpers::static_array::base::deep_to_value_list(b, input);
    ops::np_mean(b, &[legacy], reduce)
}

/// The number of elements of `input` averaged into each element of `avg`,
/// as a float of `avg`'s shape (`np.average(..., returned=True)` without
/// weights).
fn element_count_like(b: &mut IRBuilder, input: &Value, avg: &Value) -> Value {
    let (Value::StaticArray { shape: input_shape, .. }, avg_shape) =
        (input, crate::helpers::composite::get_composite_shape(avg))
    else {
        panic!("np.average: returned=True needs an array with a compile-time shape");
    };
    let total: usize = input_shape.iter().product();
    let per_value: usize = avg_shape.iter().product();
    let count = b.ir_constant_float((total / per_value.max(1)) as f64);
    if !matches!(avg, Value::StaticArray { .. }) {
        return count;
    }
    crate::helpers::static_array::base::build_static_array_from_flat(
        b,
        vec![count; per_value],
        avg_shape,
        NumberType::Float,
    )
}

/// `weights` with `input`'s shape: as given when it has that shape, else a
/// 1-D array along the integer `axis`, repeated over the other axes.
fn broadcast_weights(b: &mut IRBuilder, input: &Value, weights: &Value, axis: Option<&Value>) -> Value {
    let weights = array_argument(b, weights);
    let weights = as_float_array(b, &weights);
    let (Value::StaticArray { shape, .. }, Value::StaticArray { shape: weight_shape, .. }) = (input, &weights)
    else {
        panic!("np.average: weights need arrays with a compile-time shape");
    };
    if weight_shape == shape {
        return weights;
    }
    let Some(axis) = axis else {
        panic!("np.average: axis must be specified when shapes of a and weights differ");
    };
    let axis = axis
        .int_val()
        .unwrap_or_else(|| panic!("np.average: with 1-D weights, axis must be a compile-time integer"));
    let axis = crate::helpers::shape_arith::resolve_axis(axis, shape.len(), "np.average");
    if weight_shape.as_slice() != [shape[axis]] {
        panic!("np.average: shape of weights must be consistent with shape of a along specified axis");
    }
    let cells = crate::helpers::static_array::elementwise::payload_cells(b, &weights);
    let strides = crate::helpers::shape_arith::row_major_strides(shape);
    let total: usize = shape.iter().product();
    let repeated = (0..total).map(|i| cells[i / strides[axis] % shape[axis]].clone()).collect();
    crate::helpers::static_array::base::build_static_array_from_flat(b, repeated, shape.clone(), NumberType::Float)
}

/// `np.ravel(a)`: `a` flattened, a view of a contiguous array (as
/// `a.ravel()`).
fn ravel(b: &mut IRBuilder, a: &Value) -> Value {
    match a {
        Value::DynamicNDArray(_) => {
            let minus_one = b.ir_constant_int(-1);
            crate::helpers::array_ops::reshape(b, a, &[minus_one])
        }
        Value::Integer(_) | Value::Float(_) | Value::Boolean(_) => {
            let dtype = if matches!(a, Value::Float(_)) { NumberType::Float } else { NumberType::Integer };
            crate::helpers::static_array::base::build_static_array_from_flat(b, vec![a.clone()], vec![1], dtype)
        }
        _ => {
            let array = array_argument(b, a);
            crate::helpers::static_array::shape::try_apply_flatten(b, &array, false)
                .unwrap_or_else(|| panic!("np.ravel: expected an array"))
        }
    }
}

/// The compile-time array length `value` of parameter `name`.
fn dimension(value: &Value, op: &str, name: &str) -> usize {
    let n = static_int_arg(Some(value), 0, &format!("{op}: `{name}`"));
    usize::try_from(n).unwrap_or_else(|_| panic!("{op}: negative dimensions are not allowed"))
}

/// An array argument as NumPy reads it (`asarray`): a nested numeric list
/// becomes an ndarray; arrays and scalars are themselves.
fn array_argument(b: &mut IRBuilder, value: &Value) -> Value {
    match value {
        Value::List(_) | Value::Tuple(_) => crate::helpers::static_array::base::to_static_array(b, value)
            .unwrap_or_else(|| panic!("expected a rectangular numeric array")),
        other => other.clone(),
    }
}

/// `value` converted to a float array (`astype(float)`).
fn as_float_array(b: &mut IRBuilder, value: &Value) -> Value {
    crate::helpers::dtype::convert(b, value, crate::helpers::dtype::Dtype::Float)
}

/// A call's arguments bound to the parameter names of a NumPy signature.
struct Arguments {
    values: HashMap<String, Value>,
    op: String,
}

impl Arguments {
    /// Bind `args` and `kwargs` to `parameters`, NumPy's parameter names in
    /// order, where a `"*"` starts the keyword-only ones. An extra
    /// positional argument, an unknown keyword or a parameter given twice
    /// is rejected.
    fn bind(op: &str, parameters: &[&str], args: &[Value], kwargs: &HashMap<String, Value>) -> Self {
        let positional = parameters.iter().position(|p| *p == "*").unwrap_or(parameters.len());
        if args.len() > positional {
            panic!("{op}: takes at most {positional} positional arguments, got {}", args.len());
        }
        let mut values: HashMap<String, Value> = parameters
            .iter()
            .zip(args)
            .map(|(name, value)| (name.to_string(), value.clone()))
            .collect();
        for (key, value) in kwargs {
            if key == "*" || !parameters.contains(&key.as_str()) {
                panic!("{op}: unexpected keyword argument `{key}`");
            }
            if values.insert(key.clone(), value.clone()).is_some() {
                panic!("{op}: got multiple values for argument `{key}`");
            }
        }
        Arguments { values, op: op.to_string() }
    }

    /// The argument `name`, unless absent or `None`.
    fn get(&self, name: &str) -> Option<&Value> {
        self.values.get(name).filter(|v| !matches!(v, Value::None))
    }

    fn required(&self, name: &str) -> &Value {
        self.get(name)
            .unwrap_or_else(|| panic!("{}: missing required argument `{name}`", self.op))
    }

    /// The given arguments among `names`, as keywords.
    fn only(&self, names: &[&str]) -> HashMap<String, Value> {
        names
            .iter()
            .filter_map(|name| self.get(name).map(|v| (name.to_string(), v.clone())))
            .collect()
    }
}
