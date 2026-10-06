//! Small static-data statistical helpers.

use std::collections::HashMap;

use crate::builder::IRBuilder;
use crate::helpers::static_array::elementwise::payload_cells;
use crate::types::{CompositeData, NumberType, Value, ZinniaType};

fn static_number(cell: &Value, op: &str) -> f64 {
    if let Some(v) = cell.float_val() {
        v
    } else if let Some(v) = cell.int_val() {
        v as f64
    } else {
        panic!("{op}: first slice supports compile-time-data histogram only; runtime/private values are not supported");
    }
}

fn static_1d_numeric_values(b: &mut IRBuilder, input: &Value, op: &str) -> Vec<f64> {
    match input {
        Value::StaticArray { dtype, shape, .. } => {
            if shape.len() != 1 {
                panic!("{op}: first slice supports 1-D static numeric input only");
            }
            if matches!(dtype, NumberType::Complex) {
                panic!("{op}: Complex input is deferred for this first slice");
            }
            payload_cells(b, input)
                .iter()
                .map(|cell| static_number(cell, op))
                .collect()
        }
        Value::List(_) | Value::Tuple(_) => {
            let shape = crate::helpers::composite::get_composite_shape(input);
            if shape.len() != 1 {
                panic!("{op}: first slice supports 1-D static numeric input only");
            }
            crate::helpers::composite::flatten_composite_with_builder(b, input)
                .iter()
                .map(|cell| static_number(cell, op))
                .collect()
        }
        Value::DynamicNDArray(_) | Value::NDArray(_) => {
            panic!("{op}: first slice supports compile-time-data histogram only; runtime/private values are not supported");
        }
        _ => panic!("{op}: expected 1-D static numeric input"),
    }
}

fn static_f64_arg(value: &Value, name: &str) -> f64 {
    value
        .float_val()
        .or_else(|| value.int_val().map(|v| v as f64))
        .unwrap_or_else(|| panic!("histogram: {name} must be static in this first slice"))
}

fn parse_bins(args: &[Value], kwargs: &HashMap<String, Value>) -> usize {
    let bins = kwargs
        .get("bins")
        .or_else(|| args.get(1))
        .unwrap_or_else(|| panic!("histogram: bins must be provided as a static positive integer"));
    let Some(n) = bins.int_val() else {
        panic!("histogram: bins must be a static positive integer");
    };
    if n <= 0 {
        panic!("histogram: bins must be a static positive integer");
    }
    n as usize
}

fn parse_range(args: &[Value], kwargs: &HashMap<String, Value>) -> (f64, f64) {
    let range = kwargs
        .get("range")
        .or_else(|| args.get(2))
        .unwrap_or_else(|| panic!("histogram: range=(lo, hi) must be static in this first slice"));
    let values = match range {
        Value::Tuple(data) | Value::List(data) if data.values.len() == 2 => &data.values,
        _ => panic!("histogram: range must be a static (lo, hi) pair"),
    };
    let lo = static_f64_arg(&values[0], "range lower bound");
    let hi = static_f64_arg(&values[1], "range upper bound");
    if !lo.is_finite() || !hi.is_finite() || hi <= lo {
        panic!("histogram: range must be finite with hi > lo");
    }
    (lo, hi)
}

/// `np.histogram(a, bins=..., range=(lo, hi))` first slice.
///
/// This intentionally supports compile-time-data histograms only. It does not
/// lower a data-dependent/private histogram circuit.
pub fn np_histogram(b: &mut IRBuilder, args: &[Value], kwargs: &HashMap<String, Value>) -> Value {
    let input = args
        .first()
        .unwrap_or_else(|| panic!("histogram: expected input array"));
    let data = static_1d_numeric_values(b, input, "histogram");
    let bins = parse_bins(args, kwargs);
    let (lo, hi) = parse_range(args, kwargs);
    let width = (hi - lo) / bins as f64;

    let mut counts = vec![0i64; bins];
    for x in data {
        if x < lo || x > hi {
            continue;
        }
        let idx = if x == hi {
            bins - 1
        } else {
            ((x - lo) / width).floor() as usize
        };
        if idx < bins {
            counts[idx] += 1;
        }
    }

    let count_cells: Vec<Value> = counts
        .into_iter()
        .map(|count| b.ir_constant_int(count))
        .collect();
    let count_array = crate::helpers::static_array::base::build_static_array_from_flat(
        b,
        count_cells,
        vec![bins],
        NumberType::Integer,
    );

    let edge_cells: Vec<Value> = (0..=bins)
        .map(|i| {
            let edge = if i == bins { hi } else { lo + width * i as f64 };
            b.ir_constant_float(edge)
        })
        .collect();
    let edge_array = crate::helpers::static_array::base::build_static_array_from_flat(
        b,
        edge_cells,
        vec![bins + 1],
        NumberType::Float,
    );

    let values = vec![count_array, edge_array];
    let types: Vec<ZinniaType> = values.iter().map(|v| v.zinnia_type()).collect();
    Value::Tuple(CompositeData::new(types, values))
}

fn reject_present_kw(kwargs: &HashMap<String, Value>, name: &str) {
    if kwargs.contains_key(name) {
        panic!("cov: {name} is deferred for the first fixed-shape StaticArray slice");
    }
}

fn validate_cov_kwargs(kwargs: &HashMap<String, Value>) {
    for name in ["y", "bias", "ddof", "fweights", "aweights", "dtype"] {
        reject_present_kw(kwargs, name);
    }
    for name in kwargs.keys() {
        if name != "rowvar"
            && !matches!(
                name.as_str(),
                "y" | "bias" | "ddof" | "fweights" | "aweights" | "dtype"
            )
        {
            panic!("cov: unsupported keyword argument `{name}`");
        }
    }
}

fn cov_input_matrix(b: &mut IRBuilder, input: &Value) -> (Vec<Value>, usize, usize) {
    let (dtype, shape) = match input {
        Value::StaticArray { dtype, shape, .. } => (*dtype, shape.clone()),
        Value::DynamicNDArray(_) | Value::NDArray(_) => {
            panic!("cov: first slice supports fixed-shape StaticArray input only")
        }
        _ => panic!("cov: expected fixed-shape StaticArray input"),
    };
    if matches!(dtype, NumberType::Complex) {
        panic!("cov: Complex input is deferred");
    }
    let cells = payload_cells(b, input);
    match shape.as_slice() {
        [n] => (cells, 1, *n),
        [rows, cols] => (cells, *rows, *cols),
        _ => panic!("cov: first slice supports rank-1 or rank-2 StaticArray input only"),
    }
}

fn parse_cov_rowvar(kwargs: &HashMap<String, Value>) -> bool {
    match kwargs.get("rowvar") {
        None => true,
        Some(v) => v
            .bool_val()
            .unwrap_or_else(|| panic!("cov: rowvar must be a static boolean")),
    }
}

fn cov_cell_at(
    cells: &[Value],
    rows: usize,
    cols: usize,
    rowvar: bool,
    var_idx: usize,
    obs_idx: usize,
) -> Value {
    let idx = if rowvar {
        var_idx * cols + obs_idx
    } else {
        obs_idx * cols + var_idx
    };
    let _ = rows;
    cells[idx].clone()
}

fn mean_of_values(b: &mut IRBuilder, values: &[Value]) -> Value {
    let sum = values.iter().fold(b.ir_constant_float(0.0), |acc, v| {
        let vf = match v {
            Value::Float(_) => v.clone(),
            _ => b.ir_float_cast(v),
        };
        b.ir_add_f(&acc, &vf)
    });
    let denom = b.ir_constant_float(values.len() as f64);
    b.ir_div_f(&sum, &denom)
}

/// `np.cov(m, rowvar=True)` first StaticArray slice.
///
/// Supports one fixed-shape numeric StaticArray input. `rowvar=True`
/// treats each row as a variable and columns as observations; `rowvar=False`
/// treats each column as a variable. We use NumPy's default unbiased
/// normalization by `N - 1` and intentionally reject weights/bias/ddof/y.
pub fn np_cov(b: &mut IRBuilder, args: &[Value], kwargs: &HashMap<String, Value>) -> Value {
    if args.len() > 1 {
        panic!("cov: optional y input is deferred for the first StaticArray slice");
    }
    validate_cov_kwargs(kwargs);
    let input = args
        .first()
        .unwrap_or_else(|| panic!("cov: expected input array"));
    let (cells, rows, cols) = cov_input_matrix(b, input);
    let rowvar = parse_cov_rowvar(kwargs);
    let (var_count, obs_count) = if rowvar { (rows, cols) } else { (cols, rows) };
    if obs_count <= 1 {
        panic!(
            "cov: invalid observation count {}; default normalization requires N - 1 > 0",
            obs_count
        );
    }

    let mut means = Vec::with_capacity(var_count);
    for i in 0..var_count {
        let vals: Vec<Value> = (0..obs_count)
            .map(|obs| cov_cell_at(&cells, rows, cols, rowvar, i, obs))
            .collect();
        means.push(mean_of_values(b, &vals));
    }

    let denom = b.ir_constant_float((obs_count - 1) as f64);
    let mut out = Vec::with_capacity(var_count * var_count);
    for i in 0..var_count {
        for j in 0..var_count {
            let mut products = Vec::with_capacity(obs_count);
            for obs in 0..obs_count {
                let xi = cov_cell_at(&cells, rows, cols, rowvar, i, obs);
                let xj = cov_cell_at(&cells, rows, cols, rowvar, j, obs);
                let xi_f = match xi {
                    Value::Float(_) => xi,
                    _ => b.ir_float_cast(&xi),
                };
                let xj_f = match xj {
                    Value::Float(_) => xj,
                    _ => b.ir_float_cast(&xj),
                };
                let di = b.ir_sub_f(&xi_f, &means[i]);
                let dj = b.ir_sub_f(&xj_f, &means[j]);
                products.push((di, dj));
            }
            let acc = b.ir_dot_f(&products);
            out.push(b.ir_div_f(&acc, &denom));
        }
    }

    let out_shape = if var_count == 1 {
        vec![]
    } else {
        vec![var_count, var_count]
    };
    crate::helpers::static_array::base::build_static_array_from_flat(
        b,
        out,
        out_shape,
        NumberType::Float,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn list_of(values: Vec<Value>) -> Value {
        let types = values.iter().map(|v| v.zinnia_type()).collect();
        Value::List(CompositeData::new(types, values))
    }

    #[test]
    fn np_histogram_static_counts_and_edges() {
        let mut b = IRBuilder::new();
        let input = list_of(vec![
            b.ir_constant_int(0),
            b.ir_constant_int(1),
            b.ir_constant_int(2),
            b.ir_constant_int(3),
            b.ir_constant_int(4),
        ]);
        let bins = b.ir_constant_int(2);
        let lo = b.ir_constant_float(0.0);
        let hi = b.ir_constant_float(4.0);
        let range = Value::Tuple(CompositeData::new(
            vec![lo.zinnia_type(), hi.zinnia_type()],
            vec![lo, hi],
        ));
        let out = np_histogram(&mut b, &[input, bins, range], &HashMap::new());
        let Value::Tuple(tuple) = out else {
            panic!("expected tuple");
        };
        assert_eq!(tuple.values.len(), 2);

        let counts = payload_cells(&mut b, &tuple.values[0]);
        assert_eq!(counts[0].int_val(), Some(2));
        assert_eq!(counts[1].int_val(), Some(3));

        let edges = payload_cells(&mut b, &tuple.values[1]);
        assert_eq!(edges[0].float_val(), Some(0.0));
        assert_eq!(edges[1].float_val(), Some(2.0));
        assert_eq!(edges[2].float_val(), Some(4.0));
    }
}

/// `np.percentile(a, q)` of a 1-D static array with NumPy's default
/// ("linear") method: on the sorted values `s` of the `n` elements, with
/// `r = q / 100 * (n - 1)` and `lo = floor(r)`, the result is
/// `s[lo] + (s[lo + 1] - s[lo]) * (r - lo)` (just `s[lo]` when `r` is
/// integral). `q` is a compile-time number or a runtime integer; NumPy
/// rejects `q` outside `[0, 100]`, so such a `q` is a compile error or has
/// no proof.
///
/// Only the sorted values of ranks `lo` and `lo + 1` are used, as sort
/// outputs at those ranks (`IR::Sorted`), which the sort lowering pins on
/// their own when that is cheaper than sorting. A runtime `q` is lowered
/// arithmetically: `r * 100 = q * (n - 1)` is an integer, so
/// `lo = (q * (n - 1)) // 100` (a pinned floor division by the constant
/// 100), `frac = q * (n - 1) - 100 * lo` in `[0, 100)`, the upper rank is
/// `lo + [lo < n - 1]` (`frac` is 0 at `lo = n - 1`), and the result is
/// `s[lo] + (s[hi] - s[lo]) * frac / 100`: an exact product with the
/// integer `frac` and one division by the constant 100, a single rounding.
pub fn np_percentile(b: &mut IRBuilder, args: &[Value], kwargs: &HashMap<String, Value>) -> Value {
    if let Some(key) = kwargs.keys().find(|k| k.as_str() != "q") {
        panic!("np.percentile: keyword argument `{key}` is not supported");
    }
    assert!(args.len() >= 2 || kwargs.contains_key("q"), "np.percentile: requires a and q");
    let arr = &args[0];
    let q = kwargs.get("q").or_else(|| args.get(1)).expect("np.percentile: requires q");
    if let Value::DynamicNDArray(data) = arr {
        return runtime_shaped_percentile(b, data, q);
    }
    let n = match arr {
        Value::StaticArray { shape, dtype, .. } if shape.len() == 1 && *dtype != NumberType::Complex => {
            shape[0]
        }
        _ => panic!("np.percentile: supports a 1-D real static array"),
    };
    if n == 0 {
        panic!("np.percentile: empty arrays are not supported");
    }
    let elements: std::sync::Arc<[Value]> =
        payload_cells(b, arr).iter().map(|c| b.ensure_ptr(c)).collect();
    if let Some(q) = q.float_val().or_else(|| q.int_val().map(|v| v as f64)) {
        return percentile_at_constant(b, &elements, q);
    }
    if !matches!(q, Value::Integer(_)) {
        panic!("np.percentile: a runtime q must be an integer");
    }
    percentile_at_runtime(b, &elements, q)
}

/// The sorted value of `rank` (constant or runtime), as a float.
fn sorted_value(b: &mut IRBuilder, elements: &std::sync::Arc<[Value]>, rank: &Value) -> Value {
    let value = b.ir_sorted_at_runtime_rank(elements, rank, false);
    crate::helpers::value_ops::ensure_float(b, &value)
}

fn percentile_at_constant(b: &mut IRBuilder, elements: &std::sync::Arc<[Value]>, q: f64) -> Value {
    if !(0.0..=100.0).contains(&q) {
        panic!("np.percentile: q must be in [0, 100], got {q}");
    }
    let r = q / 100.0 * (elements.len() - 1) as f64;
    let lo = r.floor() as usize;
    let weight = r - lo as f64;
    let lo_rank = b.ir_constant_int(lo as i64);
    let low = sorted_value(b, elements, &lo_rank);
    if weight == 0.0 {
        return low;
    }
    let hi_rank = b.ir_constant_int(lo as i64 + 1);
    let high = sorted_value(b, elements, &hi_rank);
    let diff = b.ir_sub_f(&high, &low);
    let w = b.ir_constant_float(weight);
    let step = b.ir_mul_f(&diff, &w);
    b.ir_add_f(&low, &step)
}

fn percentile_at_runtime(b: &mut IRBuilder, elements: &std::sync::Arc<[Value]>, q: &Value) -> Value {
    let n_minus_1 = b.ir_constant_int(elements.len() as i64 - 1);
    interpolated_percentile(b, q, &n_minus_1, |b, rank| sorted_value(b, elements, rank))
}

/// NumPy's linear percentile at an integer `q` (checked to `[0, 100]`) of
/// `n` values whose order statistics `value_at_rank` gives:
/// `r * 100 = q * (n - 1)`, `lo = (q * (n - 1)) // 100` (a floor division
/// by the constant 100), `frac = q * (n - 1) - 100 * lo` in `[0, 100)`, the
/// upper rank `lo + [lo < n - 1]` (`frac` is 0 at `lo = n - 1`), and the
/// result `s[lo] + (s[hi] - s[lo]) * frac / 100`: an exact product with the
/// integer `frac` and one division by the constant 100.
fn interpolated_percentile(
    b: &mut IRBuilder,
    q: &Value,
    n_minus_1: &Value,
    mut value_at_rank: impl FnMut(&mut IRBuilder, &Value) -> Value,
) -> Value {
    // The index the computation uses (in range also where the code does not
    // run, see `discharge_index_in_range`).
    let q = &crate::optim::resolver::discharge_index_in_range(b, q, 0, 101, "np.percentile");
    let hundred = b.ir_constant_int(100);
    let scaled = b.ir_mul_i(q, n_minus_1);
    let lo = b.ir_floor_div_i(&scaled, &hundred);
    let lo_times_100 = b.ir_mul_i(&lo, &hundred);
    let frac = b.ir_sub_i(&scaled, &lo_times_100);
    let below_last = b.ir_less_than_i(&lo, n_minus_1);
    let step_up = b.ir_int_cast(&below_last);
    let hi = b.ir_add_i(&lo, &step_up);

    let low = value_at_rank(b, &lo);
    let high = value_at_rank(b, &hi);
    let diff = b.ir_sub_f(&high, &low);
    let frac_f = b.ir_float_cast(&frac);
    let scaled_diff = b.ir_mul_f(&diff, &frac_f);
    let hundred_f = b.ir_constant_float(100.0);
    let step = b.ir_div_f(&scaled_diff, &hundred_f);
    b.ir_add_f(&low, &step)
}

/// The elements of a runtime-shaped array for order statistics, which
/// ignore positions: every buffer slot, its activity flag (whether it holds
/// an element) and the element count.
struct RuntimeShapedElements {
    values: Vec<Value>,
    active: Vec<Value>,
    count: Value,
    dtype: NumberType,
}

fn runtime_shaped_elements(b: &mut IRBuilder, data: &crate::types::DynamicNDArrayData, op: &str) -> RuntimeShapedElements {
    if data.dtype == NumberType::Complex {
        panic!("{op}: complex arrays are not ordered");
    }
    let data = crate::ops::dyn_ndarray::force_contiguous(b, data);
    let values = crate::helpers::segment::read_all_typed(b, data.segment_id, data.max_length(), data.dtype);
    let active = crate::ops::dyn_ndarray::aggregation::active_slots(b, &data)
        .unwrap_or_else(|| (0..values.len()).map(|_| b.ir_constant_bool(true)).collect());
    let count = crate::ops::dyn_ndarray::scalar_i64_to_value(&data.meta.runtime_length, NumberType::Integer);
    let count = b.ensure_ptr(&count);
    RuntimeShapedElements { values, active, count, dtype: data.dtype }
}

impl RuntimeShapedElements {
    /// The value of `rank` among the elements, as a float (see
    /// `lower::sort::masked_order_statistic`).
    fn value_at_rank(&self, b: &mut IRBuilder, rank: &Value) -> Value {
        let value = crate::lower::sort::masked_order_statistic(
            b,
            &self.values,
            &self.active,
            self.dtype,
            rank,
        );
        let value = crate::ops::dyn_ndarray::scalar_i64_to_value(
            &crate::ops::dyn_ndarray::value_to_scalar_i64(&value),
            self.dtype,
        );
        crate::helpers::value_ops::ensure_float(b, &value)
    }
}

/// `np.median` of a runtime-shaped array (all its elements): the mean of
/// the values of ranks `(L - 1) // 2` and `L // 2` of its `L` elements (the
/// same rank when `L` is odd), each an order statistic pinned by counting.
/// An empty array has no proof (NumPy returns nan with a warning).
pub fn runtime_shaped_median(b: &mut IRBuilder, data: &crate::types::DynamicNDArrayData) -> Value {
    let elements = runtime_shaped_elements(b, data, "np.median");
    let one = b.ir_constant_int(1);
    let two = b.ir_constant_int(2);
    let last = b.ir_sub_i(&elements.count, &one);
    let lower_rank = b.ir_floor_div_i(&last, &two);
    let upper_rank = b.ir_floor_div_i(&elements.count, &two);
    let lower = elements.value_at_rank(b, &lower_rank);
    let upper = elements.value_at_rank(b, &upper_rank);
    let sum = b.ir_add_f(&lower, &upper);
    let two_f = b.ir_constant_float(2.0);
    b.ir_div_f(&sum, &two_f)
}

/// `np.percentile(a, q)` of a runtime-shaped array at an integer `q`
/// (constant or runtime), NumPy's linear method over its `L` elements.
fn runtime_shaped_percentile(b: &mut IRBuilder, data: &crate::types::DynamicNDArrayData, q: &Value) -> Value {
    let integral = q.int_val().is_some() || matches!(q, Value::Integer(_))
        || q.float_val().is_some_and(|v| v.fract() == 0.0);
    if !integral {
        panic!("np.percentile of a runtime-shaped array: q must be an integer");
    }
    let q = match q.float_val() {
        Some(v) if q.int_val().is_none() => b.ir_constant_int(v as i64),
        _ => q.clone(),
    };
    let elements = runtime_shaped_elements(b, data, "np.percentile");
    let one = b.ir_constant_int(1);
    let n_minus_1 = b.ir_sub_i(&elements.count, &one);
    interpolated_percentile(b, &q, &n_minus_1, |b, rank| elements.value_at_rank(b, rank))
}

/// `np.corrcoef(x, y)` of two 1-D static arrays of the same length
/// `n >= 2`: `[[1, r], [r, 1]]` with
/// `r = sum(dx * dy) / sqrt(sum(dx^2) * sum(dy^2))`, `dx = x - mean(x)`.
/// Each of the three sums of products is one `ProductSumF` (rescaled
/// once, `IRBuilder::ir_dot_f`).
pub fn np_corrcoef(b: &mut IRBuilder, args: &[Value]) -> Value {
    assert!(args.len() >= 2, "np.corrcoef: requires x and y");
    let length_of = |v: &Value| match v {
        Value::StaticArray { shape, .. } if shape.len() == 1 => shape[0],
        _ => panic!("np.corrcoef: supports two 1-D static arrays"),
    };
    let n = length_of(&args[0]);
    if length_of(&args[1]) != n {
        panic!("np.corrcoef: x and y must have the same length");
    }
    if n <= 1 {
        panic!("np.corrcoef: at least two observations are required");
    }
    let deviations = |b: &mut IRBuilder, arr: &Value| -> Vec<Value> {
        let cells: Vec<Value> = payload_cells(b, arr)
            .iter()
            .map(|c| crate::helpers::value_ops::ensure_float(b, c))
            .collect();
        let sum = b.ir_sum_f(&cells);
        let count = b.ir_constant_float(n as f64);
        let mean = b.ir_div_f(&sum, &count);
        cells.iter().map(|c| b.ir_sub_f(c, &mean)).collect()
    };
    let dx = deviations(b, &args[0]);
    let dy = deviations(b, &args[1]);
    let pairs = |u: &[Value], v: &[Value]| -> Vec<(Value, Value)> {
        u.iter().cloned().zip(v.iter().cloned()).collect()
    };
    let cov = b.ir_dot_f(&pairs(&dx, &dy));
    let var_x = b.ir_dot_f(&pairs(&dx, &dx));
    let var_y = b.ir_dot_f(&pairs(&dy, &dy));
    let denom_sq = b.ir_mul_f(&var_x, &var_y);
    let denom = b.ir_sqrt_f(&denom_sq);
    let corr = b.ir_div_f(&cov, &denom);
    let one = b.ir_constant_float(1.0);
    crate::helpers::static_array::base::build_static_array_from_flat(
        b,
        vec![one.clone(), corr.clone(), corr, one],
        vec![2, 2],
        NumberType::Float,
    )
}
