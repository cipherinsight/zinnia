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
            let mut acc = b.ir_constant_float(0.0);
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
                let prod = b.ir_mul_f(&di, &dj);
                acc = b.ir_add_f(&acc, &prod);
            }
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
