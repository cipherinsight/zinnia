//! Index-producing NumPy helpers for static numeric arrays.

use crate::builder::IRBuilder;
use crate::ops::dyn_ndarray::value_to_scalar_i64;
use crate::types::{
    CompositeData, Dim, DynArrayMeta, DynamicNDArrayData, NumberType, ScalarValue, SliceIndex,
    Value, ValueId, ZinniaType,
};

use crate::helpers::static_array::elementwise::payload_cells;

struct StaticInput {
    shape: Vec<usize>,
    cells: Vec<Value>,
}

fn static_numeric_input(b: &mut IRBuilder, val: &Value, op: &str) -> StaticInput {
    match val {
        Value::StaticArray {
            dtype,
            shape,
            segment_id: _,
            ..
        } => {
            if matches!(dtype, NumberType::Complex) {
                panic!("{op}: Complex static arrays are not supported in this first slice");
            }
            StaticInput {
                shape: shape.clone(),
                cells: payload_cells(b, val),
            }
        }
        Value::List(_) | Value::Tuple(_) => {
            let shape = crate::helpers::composite::get_composite_shape(val);
            let cells = crate::helpers::composite::flatten_composite_with_builder(b, val);
            if cells
                .iter()
                .any(|c| !matches!(c, Value::Integer(_) | Value::Float(_) | Value::Boolean(_)))
            {
                panic!("{op}: expected a numeric StaticArray/List/Tuple input");
            }
            StaticInput { shape, cells }
        }
        Value::DynamicNDArray(_) => {
            panic!("{op}: DynamicNDArray input is deferred; use static numeric arrays for now")
        }
        _ => panic!("{op}: expected a static numeric array input"),
    }
}

fn nonzero_indicator(b: &mut IRBuilder, cell: &Value) -> Value {
    match cell {
        Value::Boolean(_) => cell.clone(),
        Value::Integer(_) => {
            let zero = b.ir_constant_int(0);
            b.ir_not_equal_i(cell, &zero)
        }
        Value::Float(_) => {
            let zero = b.ir_constant_float(0.0);
            b.ir_not_equal_f(cell, &zero)
        }
        _ => panic!("argwhere/nonzero: expected numeric scalar cell"),
    }
}

fn static_row_major_coords(flat_idx: usize, shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        return Vec::new();
    }
    let mut rem = flat_idx;
    let mut coords = vec![0; shape.len()];
    for axis in (0..shape.len()).rev() {
        let dim = shape[axis];
        coords[axis] = if dim == 0 { 0 } else { rem % dim };
        if dim != 0 {
            rem /= dim;
        }
    }
    coords
}

fn strides_for_shape(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    let mut acc = 1usize;
    for (idx, dim) in shape.iter().enumerate().rev() {
        strides[idx] = acc;
        acc = acc.saturating_mul(*dim);
    }
    strides
}

fn zero_for_dtype(b: &mut IRBuilder, dtype: NumberType) -> Value {
    match dtype {
        NumberType::Integer => b.ir_constant_int(0),
        NumberType::Float => b.ir_constant_float(0.0),
        NumberType::Complex => crate::helpers::value_ops::pack_complex_value(
            b.ir_constant_float(0.0),
            b.ir_constant_float(0.0),
        ),
    }
}

fn cast_cell_to_dtype(b: &mut IRBuilder, cell: &Value, dtype: NumberType) -> Value {
    match dtype {
        NumberType::Integer => match cell {
            Value::Integer(_) | Value::Boolean(_) => cell.clone(),
            _ => b.ir_int_cast(cell),
        },
        NumberType::Float => match cell {
            Value::Float(_) => cell.clone(),
            _ => b.ir_float_cast(cell),
        },
        NumberType::Complex => match cell {
            Value::Complex { .. } => cell.clone(),
            _ => crate::helpers::value_ops::pack_complex_value(
                b.ir_float_cast(cell),
                b.ir_constant_float(0.0),
            ),
        },
    }
}

fn static_array_shape_dtype(arr: &Value, op: &str) -> (NumberType, Vec<usize>) {
    match arr {
        Value::StaticArray { dtype, shape, .. } => (*dtype, shape.clone()),
        _ => panic!("{op}: only StaticArray inputs are supported"),
    }
}

fn sum_cells_for_dtype(b: &mut IRBuilder, cells: &[Value], dtype: NumberType) -> Value {
    if cells.is_empty() {
        return zero_for_dtype(b, dtype);
    }
    let mut acc = cast_cell_to_dtype(b, &cells[0], dtype);
    for cell in cells.iter().skip(1) {
        let rhs = cast_cell_to_dtype(b, cell, dtype);
        acc = if dtype == NumberType::Float {
            b.ir_add_f(&acc, &rhs)
        } else {
            b.ir_add_i(&acc, &rhs)
        };
    }
    acc
}

pub fn np_bincount(
    b: &mut IRBuilder,
    args: &[Value],
    kwargs: &std::collections::HashMap<String, Value>,
) -> Value {
    assert!(!args.is_empty(), "np.bincount: requires x");
    if kwargs
        .keys()
        .any(|key| key != "weights" && key != "minlength")
    {
        panic!("np.bincount: only weights and minlength are supported in this bounded slice");
    }
    let x = &args[0];
    let weights_arg = kwargs.get("weights").or_else(|| args.get(1));
    let minlength_arg = kwargs.get("minlength").or_else(|| args.get(2));

    let (x_dtype, x_shape) = static_array_shape_dtype(x, "np.bincount");
    if x_dtype != NumberType::Integer {
        panic!("np.bincount: x must be an integer StaticArray");
    }
    if x_shape.len() != 1 {
        panic!("np.bincount: this bounded slice supports 1-D x only");
    }
    let x_cells = payload_cells(b, x);
    let (weights, out_dtype) = match weights_arg {
        Some(w) if !matches!(w, Value::None) => {
            let (wdtype, wshape) = static_array_shape_dtype(w, "np.bincount");
            if wshape != x_shape {
                panic!("np.bincount: weights must have the same shape as x");
            }
            if wdtype == NumberType::Complex {
                panic!("np.bincount: Complex weights are not supported");
            }
            (payload_cells(b, w), NumberType::Float)
        }
        _ => (
            vec![b.ir_constant_int(1); x_cells.len()],
            NumberType::Integer,
        ),
    };

    // Soundness check (compiler.np-bincount-no-silent-default-bound):
    // bincount's output length is `max(x) + 1` — runtime-determined by
    // the *values* inside `x`. The previous behaviour silently capped
    // the output at DEFAULT_BINCOUNT_BOUND=256 and injected a witness
    // assert that every x entry is in [0, 255]. Sound but a hidden
    // contract — a user passing `np.bincount([1000, 2000, 3000])` gets
    // a 256-element output with proof-time failure rather than a
    // compile-time error explaining what's wrong.
    //
    // Reject loudly: each runtime x entry must either carry a
    // compile-time-derivable upper bound (literal int, or
    // `ask_bounds` returns a tight range, or @requires on x's element
    // values), otherwise we can't size the output and the user needs
    // to add an annotation.
    let mut max_bound: i64 = 0; // tightest provable upper bound across cells
    for idx in &x_cells {
        if !matches!(idx, Value::Integer(_) | Value::Boolean(_)) {
            panic!("np.bincount: x entries must be integer scalars");
        }
        if let Some(n) = idx.int_val() {
            assert!(n >= 0, "np.bincount: x entries must be nonnegative");
            max_bound = max_bound.max(n);
        } else if let Some((min, max)) = b.ask_bounds(idx) {
            if min < 0 {
                panic!(
                    "np.bincount: x entry can be negative ({min}). Add a precondition \
                     (e.g. `@requires(lambda x: forall(x, lambda v: v >= 0))`)."
                );
            }
            max_bound = max_bound.max(max);
        } else {
            panic!(
                "np.bincount: x entry value is not statically bounded — its runtime \
                 value could be anything, and the output length is `max(x)+1`. The \
                 previous compiler revision silently capped the output at 256 with a \
                 proof-time witness assert; that has been removed. To compile this \
                 program, either pass a literal int array, or annotate the input with \
                 `@requires(lambda x: forall(x, lambda v: v <= K))` so the compiler \
                 can derive a tight, sound upper bound."
            )
        }
    }

    let zero = b.ir_constant_int(0);
    let mut max_idx = b.ir_constant_int(0);
    for idx in &x_cells {
        let gt = b.ir_greater_than_i(idx, &max_idx);
        max_idx = b.ir_select_i(&gt, idx, &max_idx);
    }
    let one = b.ir_constant_int(1);
    let mut runtime_len = b.ir_add_i(&max_idx, &one);
    let mut output_cap = (max_bound as usize).saturating_add(1);
    if let Some(minlength) = minlength_arg {
        if !matches!(minlength, Value::None) {
            if let Some(n) = minlength.int_val() {
                assert!(n >= 0, "np.bincount: minlength must be nonnegative");
                output_cap = output_cap.max(n as usize);
            } else if let Some((mn, mx)) = b.ask_bounds(minlength) {
                if mn < 0 {
                    panic!(
                        "np.bincount: minlength can be negative ({mn}). Add a \
                         precondition with a nonneg lower bound."
                    );
                }
                output_cap = output_cap.max(mx as usize);
            } else {
                panic!(
                    "np.bincount: minlength is not statically bounded. Add a \
                     precondition (e.g. `@requires(lambda minlength: minlength <= K)`) \
                     so the compiler can derive an upper bound on the output."
                );
            }
            let gt_min = b.ir_greater_than_i(minlength, &runtime_len);
            runtime_len = b.ir_select_i(&gt_min, minlength, &runtime_len);
        }
    }
    let _ = zero; // silence unused-warning in the trimmed path

    let mut out = Vec::with_capacity(output_cap);
    for bin in 0..output_cap {
        let bin_val = b.ir_constant_int(bin as i64);
        let mut terms = Vec::with_capacity(x_cells.len());
        for (idx, weight) in x_cells.iter().zip(weights.iter()) {
            let is_bin = b.ir_equal_i(idx, &bin_val);
            if out_dtype == NumberType::Float {
                let zero_f = b.ir_constant_float(0.0);
                let w = cast_cell_to_dtype(b, weight, NumberType::Float);
                terms.push(b.ir_select_f(&is_bin, &w, &zero_f));
            } else {
                let zero_i = b.ir_constant_int(0);
                let w = cast_cell_to_dtype(b, weight, NumberType::Integer);
                terms.push(b.ir_select_i(&is_bin, &w, &zero_i));
            }
        }
        out.push(sum_cells_for_dtype(b, &terms, out_dtype));
    }

    crate::ops::dyn_ndarray::constructors::dyn_from_values_with_active_values(
        b,
        out,
        runtime_len,
        out_dtype,
    )
}

/// The coordinates of the non-zero cells of a static array, packed to the
/// front in row-major order: the shared core of `np.argwhere`,
/// `np.nonzero` and `np.flatnonzero`.
struct NonzeroCoordinates {
    /// `rows * rank` cells, row-major: row `j` holds the coordinates of the
    /// `j`-th non-zero cell, rows past the count hold zeros.
    packed: Vec<Value>,
    /// The number of non-zero cells.
    count: Value,
    /// Per cell: its non-zero indicator, the running count after it, and
    /// its coordinate constants (for the compaction provenance record).
    keeps: Vec<Value>,
    prefix: Vec<Value>,
    coordinates: Vec<Vec<Value>>,
}

/// Packs the coordinates of the non-zero cells with the boolean-mask
/// compaction network (`compact_by_mask`), one row of `rank` coordinates
/// per cell. (It used to write each cell's coordinates to the runtime
/// address "count so far", which every backend lowers as a one-hot write
/// over all slots per cell.)
fn nonzero_coordinates(b: &mut IRBuilder, cells: &[Value], shape: &[usize]) -> NonzeroCoordinates {
    let rank = shape.len();
    let keeps: Vec<Value> = cells.iter().map(|cell| nonzero_indicator(b, cell)).collect();
    let coordinates: Vec<Vec<Value>> = (0..cells.len())
        .map(|flat_idx| {
            static_row_major_coords(flat_idx, shape)
                .into_iter()
                .map(|c| b.ir_constant_int(c as i64))
                .collect()
        })
        .collect();
    let flat_coordinates: Vec<Value> = coordinates.iter().flatten().cloned().collect();
    let zero = b.ir_constant_int(0);
    let compaction = crate::ops::dyn_ndarray::memory_ops::compact_by_mask(
        b,
        &flat_coordinates,
        &keeps,
        &zero,
        rank,
    );
    let mut count = b.ir_constant_int(0);
    let mut prefix = Vec::with_capacity(cells.len());
    for keep in &keeps {
        count = b.ir_add_i(&count, keep);
        prefix.push(count.clone());
    }
    NonzeroCoordinates { packed: compaction.acc, count, keeps, prefix, coordinates }
}

impl NonzeroCoordinates {
    /// Records that lane `axis` of `segment_id` holds the kept coordinates
    /// along `axis` (`provenance::CompactionInfo`), for the bounding-box
    /// fast paths of `min`/`max`. Only axis 0's coordinates are
    /// non-decreasing in row-major order.
    fn record(&self, b: &mut IRBuilder, segment_id: u32, lane: usize, axis: usize) {
        let vids = |vs: &mut dyn Iterator<Item = &Value>| -> Option<Vec<crate::types::ValueId>> {
            vs.map(|v| v.value_id()).collect()
        };
        let (Some(keep), Some(source), Some(prefix), Some(count)) = (
            vids(&mut self.keeps.iter()),
            vids(&mut self.coordinates.iter().map(|c| &c[axis])),
            vids(&mut self.prefix.iter()),
            self.count.value_id(),
        ) else {
            return;
        };
        b.provenance.set_compaction(
            segment_id,
            lane,
            crate::provenance::CompactionInfo { keep, source, prefix, count, sorted: axis == 0 },
        );
    }
}

pub fn np_argwhere(b: &mut IRBuilder, args: &[Value]) -> Value {
    let input = args
        .first()
        .unwrap_or_else(|| panic!("argwhere: expected one argument"));
    let StaticInput { shape, cells } = static_numeric_input(b, input, "argwhere");
    let rank = shape.len();
    if !(rank == 1 || rank == 2) {
        panic!("argwhere: first slice supports rank-1 and rank-2 static arrays only");
    }

    let row_bound = cells.len();
    let cell_bound = row_bound * rank;
    let rank_const = b.ir_constant_int(rank as i64);
    let found = nonzero_coordinates(b, &cells, &shape);
    let segment_id =
        crate::helpers::segment::alloc_and_write_values(b, &found.packed, NumberType::Integer);
    for axis in 0..rank {
        found.record(b, segment_id, axis, axis);
    }
    let true_count = found.count.clone();
    let runtime_rows = value_to_scalar_i64(&true_count);
    let runtime_rank_width = ScalarValue::new(Some(rank as i64), None);
    let runtime_len_val = b.ir_mul_i(&true_count, &rank_const);
    let runtime_length = value_to_scalar_i64(&runtime_len_val);
    let logical_shape = vec![row_bound, rank];
    let envelope = crate::types::Envelope::new_with_bound(
        vec![
            Dim::new_dynamic(&mut b.dim_table, 0, row_bound),
            Dim::new_static(&mut b.dim_table, rank),
        ],
        cell_bound,
    );

    Value::DynamicNDArray(DynamicNDArrayData {
        envelope,
        dtype: NumberType::Integer,
        segment_id,
        meta: DynArrayMeta {
            logical_shape: logical_shape.clone(),
            logical_offset: 0,
            logical_strides: strides_for_shape(&logical_shape),
            runtime_length,
            runtime_rank: ScalarValue::new(Some(2), None),
            runtime_shape: vec![runtime_rows, runtime_rank_width.clone()],
            runtime_strides: vec![runtime_rank_width, ScalarValue::new(Some(1), None)],
            runtime_offset: ScalarValue::new(Some(0), None),
        },
        value_id: ValueId::next(),
    })
}

pub fn np_nonzero(b: &mut IRBuilder, args: &[Value]) -> Value {
    let input = args
        .first()
        .unwrap_or_else(|| panic!("nonzero: expected one argument"));
    let StaticInput { shape, cells } = static_numeric_input(b, input, "nonzero");
    let rank = shape.len();
    if !(rank == 1 || rank == 2) {
        panic!("nonzero: first slice supports rank-1 and rank-2 static arrays only");
    }

    let row_bound = cells.len();
    let found = nonzero_coordinates(b, &cells, &shape);
    let mut segment_ids = Vec::with_capacity(rank);
    for axis in 0..rank {
        let lane: Vec<Value> = found.packed.iter().skip(axis).step_by(rank).cloned().collect();
        let seg = crate::helpers::segment::alloc_and_write_values(b, &lane, NumberType::Integer);
        // Every cell of this segment is a coordinate along `axis` or 0.
        // Publish that as a segment-wide bound so downstream readers
        // (e.g. `flatnonzero().min() / .max()` for the bbox-crop family
        // 0507/0509/0510) discharge their slice bounds from facts instead
        // of emitting u128 bound asserts.
        b.segment_default_bounds.insert(seg, (0, shape[axis] as i64 - 1));
        found.record(b, seg, 0, axis);
        segment_ids.push(seg);
    }
    let true_count = found.count.clone();
    let keep_cells = found.keeps.clone();

    let mut outputs = Vec::with_capacity(rank);
    for segment_id in segment_ids {
        let runtime_length = value_to_scalar_i64(&true_count);
        let envelope = crate::types::Envelope::new_with_bound(
            vec![Dim::new_dynamic(&mut b.dim_table, 0, row_bound)],
            row_bound,
        );
        outputs.push(Value::DynamicNDArray(DynamicNDArrayData {
            envelope,
            dtype: NumberType::Integer,
            segment_id,
            meta: DynArrayMeta {
                logical_shape: vec![row_bound],
                logical_offset: 0,
                logical_strides: vec![1],
                runtime_length: runtime_length.clone(),
                runtime_rank: ScalarValue::new(Some(1), None),
                runtime_shape: vec![runtime_length],
                runtime_strides: vec![ScalarValue::new(Some(1), None)],
                runtime_offset: ScalarValue::new(Some(0), None),
            },
            value_id: ValueId::next(),
        }));
    }
    let types = vec![
        ZinniaType::DynamicNDArray {
            dtype: NumberType::Integer,
            max_length: row_bound,
            max_rank: 1,
            max_shape: Some(vec![row_bound]),
        };
        outputs.len()
    ];
    let index_tuple = Value::Tuple(CompositeData::new(types, outputs));
    let mask = crate::helpers::static_array::base::build_static_array_from_flat(
        b,
        keep_cells,
        shape,
        NumberType::Integer,
    );
    if let Some(vid) = index_tuple.value_id() {
        b.nonzero_index_masks.insert(vid, mask);
    }
    index_tuple
}

/// `np.unique(arr, return_inverse=...)` returning a bounded-dynamic
/// output: the distinct values of `arr` (flattened) in **ascending
/// order**, as NumPy returns them.
///
/// **Construction (O(n^2) comparisons, no sorting network needed).**
/// With `n` input cells `x[0..n)`:
///
/// - `is_first[i] = AND_{j<i} (x[j] != x[i])` picks one representative
///   (the first occurrence) of every distinct value;
/// - `rank[i] = sum_j is_first[j] * (x[j] < x[i])` is the number of
///   distinct values strictly below `x[i]`, i.e. the position of `x[i]`
///   in the sorted unique array;
/// - `unique[k]` is the representative whose rank is `k`; exactly one
///   representative has rank `k` for every `k < count`, where
///   `count = sum_i is_first[i]` is the runtime length;
/// - `inverse[i] = rank[i]`, so `unique[inverse] == x` as in NumPy.
///
/// Every output cell is a function of the inputs computed by in-circuit
/// comparisons and selects, so the result is fully constrained.
///
/// **Bounded-dynamic output.** Worst case every input cell is unique, so
/// the output envelope is `[Dim::dynamic(0, n)]` with `total_bound = n`.
/// Slots `k >= count` hold 0 and sit past `runtime_length`.
///
/// **`return_inverse`.** Returns a `Tuple(unique_dyn, inverse_static)`
/// where `inverse_static` is a `Value::StaticArray` of shape `(n,)`.
///
/// Unsupported kwargs (`axis`, `return_index`, `return_counts`) panic
/// with a clear message rather than silently producing the wrong
/// shape.
pub fn np_unique(
    b: &mut IRBuilder,
    args: &[Value],
    kwargs: &std::collections::HashMap<String, Value>,
) -> Value {
    use crate::types::{CompositeData, ZinniaType};

    let return_inverse = kwargs
        .get("return_inverse")
        .and_then(|v| v.bool_val())
        .unwrap_or(false);
    for opt in ["return_index", "return_counts"] {
        if kwargs.get(opt).and_then(|v| v.bool_val()).unwrap_or(false) {
            panic!("np.unique: kwarg `{opt}` is not yet supported");
        }
    }
    if kwargs
        .get("axis")
        .map(|v| !matches!(v, Value::None))
        .unwrap_or(false)
    {
        panic!("np.unique: kwarg `axis` is not yet supported (input is implicitly flattened)");
    }

    // Resolve input. Accept any rank-N StaticArray (flattened) or a
    // tuple/list of scalars; DynamicNDArray inputs are rejected by
    // `static_numeric_input`.
    let input = args
        .first()
        .unwrap_or_else(|| panic!("np.unique: requires an input array"));
    let StaticInput { shape: _, cells } = static_numeric_input(b, input, "np.unique");
    let n = cells.len();
    if n == 0 {
        panic!("np.unique: empty input is not supported");
    }
    // A StaticArray has one declared dtype and `static_numeric_input`
    // rejected non-numeric composites, so the first cell decides.
    let dtype = match &cells[0] {
        Value::Float(_) => NumberType::Float,
        Value::Integer(_) | Value::Boolean(_) => NumberType::Integer,
        _ => panic!("np.unique: input cells must be Integer / Float / Boolean"),
    };
    let cells: Vec<Value> = cells.iter().map(|c| cast_to_dtype(b, c, dtype)).collect();

    let ranking = rank_distinct_values(b, &cells, dtype);
    let sorted_unique: Vec<Value> = (0..n)
        .map(|k| select_representative_with_rank(b, &cells, &ranking, k, dtype))
        .collect();
    let segment_id = crate::helpers::segment::alloc_and_write_values(b, &sorted_unique, dtype);

    let envelope =
        crate::types::Envelope::new_with_bound(vec![Dim::new_dynamic(&mut b.dim_table, 0, n)], n);
    let runtime_length = value_to_scalar_i64(&ranking.count);
    let unique_dyn = Value::DynamicNDArray(DynamicNDArrayData {
        envelope,
        dtype,
        segment_id,
        meta: DynArrayMeta {
            logical_shape: vec![n],
            logical_offset: 0,
            logical_strides: vec![1],
            runtime_length: runtime_length.clone(),
            runtime_rank: ScalarValue::new(Some(1), None),
            runtime_shape: vec![runtime_length],
            runtime_strides: vec![ScalarValue::new(Some(1), None)],
            runtime_offset: ScalarValue::new(Some(0), None),
        },
        value_id: ValueId::next(),
    });

    if !return_inverse {
        return unique_dyn;
    }

    let inverse_arr = crate::helpers::static_array::base::build_static_array_from_flat(
        b,
        ranking.rank,
        vec![n],
        NumberType::Integer,
    );
    let types = vec![
        ZinniaType::DynamicNDArray {
            dtype,
            max_length: n,
            max_rank: 1,
            max_shape: Some(vec![n]),
        },
        ZinniaType::NDArray {
            dtype: NumberType::Integer,
            shape: vec![n],
        },
    ];
    Value::Tuple(CompositeData::new(types, vec![unique_dyn, inverse_arr]))
}

/// Per-cell ranking of the distinct values of `cells` (see `np_unique`).
struct DistinctRanking {
    /// `is_first[i]`: `cells[i]` is the first occurrence of its value.
    is_first: Vec<Value>,
    /// `rank[i]`: number of distinct values strictly below `cells[i]`.
    rank: Vec<Value>,
    /// Number of distinct values.
    count: Value,
}

fn rank_distinct_values(b: &mut IRBuilder, cells: &[Value], dtype: NumberType) -> DistinctRanking {
    let n = cells.len();
    let (eq, lt): (fn(&mut IRBuilder, &Value, &Value) -> Value, fn(&mut IRBuilder, &Value, &Value) -> Value) =
        match dtype {
            NumberType::Float => (IRBuilder::ir_equal_f, IRBuilder::ir_less_than_f),
            NumberType::Integer => (IRBuilder::ir_equal_i, IRBuilder::ir_less_than_i),
            NumberType::Complex => unreachable!("np.unique: Complex inputs not accepted"),
        };
    // One comparison per unordered pair: `x[j] > x[i]` is derived as
    // `!(x[j] < x[i]) && !(x[j] == x[i])`.
    let mut equal = vec![vec![None; n]; n];
    let mut less = vec![vec![None; n]; n];
    for i in 0..n {
        for j in 0..i {
            let e = eq(b, &cells[j], &cells[i]);
            let l = lt(b, &cells[j], &cells[i]);
            let not_e = b.ir_logical_not(&e);
            let not_l = b.ir_logical_not(&l);
            let g = b.ir_logical_and(&not_e, &not_l);
            equal[i][j] = Some(e.clone());
            equal[j][i] = Some(e);
            less[j][i] = Some(l); // x[j] < x[i]
            less[i][j] = Some(g); // x[i] < x[j]
        }
    }
    let mut is_first: Vec<Value> = Vec::with_capacity(n);
    for i in 0..n {
        let mut first = b.ir_constant_bool(true);
        for j in 0..i {
            let e = equal[i][j].as_ref().expect("pair compared above");
            let ne = b.ir_logical_not(e);
            first = b.ir_logical_and(&first, &ne);
        }
        is_first.push(first);
    }
    let mut rank: Vec<Value> = Vec::with_capacity(n);
    for i in 0..n {
        let mut r = b.ir_constant_int(0);
        for j in (0..n).filter(|&j| j != i) {
            let below = less[j][i].as_ref().expect("pair compared above");
            let counted = b.ir_logical_and(&is_first[j], below);
            let counted = b.ir_int_cast(&counted);
            r = b.ir_add_i(&r, &counted);
        }
        rank.push(r);
    }
    let mut count = b.ir_constant_int(0);
    for first in &is_first {
        let one_if_first = b.ir_int_cast(first);
        count = b.ir_add_i(&count, &one_if_first);
    }
    DistinctRanking {
        is_first,
        rank,
        count,
    }
}

/// The representative cell whose rank is `k`, or 0 when `k >= count`.
fn select_representative_with_rank(
    b: &mut IRBuilder,
    cells: &[Value],
    ranking: &DistinctRanking,
    k: usize,
    dtype: NumberType,
) -> Value {
    let k_v = b.ir_constant_int(k as i64);
    let mut acc = match dtype {
        NumberType::Float => b.ir_constant_float(0.0),
        _ => b.ir_constant_int(0),
    };
    for (i, cell) in cells.iter().enumerate() {
        let rank_is_k = b.ir_equal_i(&ranking.rank[i], &k_v);
        let picked = b.ir_logical_and(&ranking.is_first[i], &rank_is_k);
        acc = match dtype {
            NumberType::Float => b.ir_select_f(&picked, cell, &acc),
            _ => b.ir_select_i(&picked, cell, &acc),
        };
    }
    acc
}

/// Cast a numeric Value to the requested NumberType, preserving an
/// already-matching wire to avoid spurious IR. Mirrors the helper used
/// in `helpers::static_array::mask` but kept local to avoid pulling
/// that module's narrow visibility outward.
fn cast_to_dtype(b: &mut IRBuilder, v: &Value, dtype: NumberType) -> Value {
    match dtype {
        NumberType::Float => {
            if matches!(v, Value::Float(_)) {
                v.clone()
            } else {
                b.ir_float_cast(v)
            }
        }
        NumberType::Integer => {
            if matches!(v, Value::Integer(_)) {
                v.clone()
            } else {
                b.ir_int_cast(v)
            }
        }
        NumberType::Complex => unreachable!("np.unique: Complex inputs rejected upstream"),
    }
}

fn static_shape_from_value(b: &mut IRBuilder, shape_value: &Value, op: &str) -> Vec<usize> {
    match shape_value {
        Value::Tuple(d) | Value::List(d) => d
            .values
            .iter()
            .map(|v| {
                let n = v
                    .int_val()
                    .unwrap_or_else(|| panic!("{op}: shape dimensions must be static integers"));
                assert!(n >= 0, "{op}: shape dimensions must be nonnegative");
                n as usize
            })
            .collect(),
        Value::StaticArray { dtype, shape, .. } => {
            if *dtype != NumberType::Integer || shape.len() != 1 {
                panic!("{op}: dims array must be a 1-D integer StaticArray");
            }
            payload_cells(b, shape_value)
                .iter()
                .map(|v| {
                    let n = v.int_val().unwrap_or_else(|| {
                        panic!("{op}: dims array values must be static integers")
                    });
                    assert!(n >= 0, "{op}: shape dimensions must be nonnegative");
                    n as usize
                })
                .collect()
        }
        _ => panic!("{op}: shape must be a static tuple/list of integers"),
    }
}

fn order_from_args(
    args: &[Value],
    kwargs: &std::collections::HashMap<String, Value>,
    op: &str,
) -> String {
    let order = kwargs
        .get("order")
        .or_else(|| args.get(2))
        .and_then(|v| v.string_val())
        .unwrap_or("C");
    if order != "C" && order != "F" {
        panic!("{op}: only order='C' and order='F' are supported");
    }
    order.to_string()
}

pub fn np_unravel_index(
    b: &mut IRBuilder,
    args: &[Value],
    kwargs: &std::collections::HashMap<String, Value>,
) -> Value {
    assert!(
        args.len() >= 2,
        "np.unravel_index: requires indices and shape"
    );
    let shape = static_shape_from_value(b, &args[1], "np.unravel_index");
    if shape.is_empty() {
        panic!("np.unravel_index: scalar shape is not supported");
    }
    let order = order_from_args(args, kwargs, "np.unravel_index");
    let idx = args[0].clone();
    if !matches!(idx, Value::Integer(_)) {
        panic!("np.unravel_index: only scalar integer indices are supported");
    }
    let total: usize = shape.iter().product();
    if let Some(i) = idx.int_val() {
        if i < 0 || i >= total as i64 {
            panic!(
                "np.unravel_index: index {} is out of bounds for shape {:?}",
                i, shape
            );
        }
    } else {
        crate::optim::resolver::discharge_index_in_range(
            b,
            &idx,
            0,
            total as i64,
            "np.unravel_index",
        );
    }

    let mut coords = vec![b.ir_constant_int(0); shape.len()];
    let mut rem = idx;
    let axes: Vec<usize> = if order == "C" {
        (0..shape.len()).rev().collect()
    } else {
        (0..shape.len()).collect()
    };
    for axis in axes {
        let dim = shape[axis] as i64;
        let dim_val = b.ir_constant_int(dim);
        coords[axis] = b.ir_mod_i(&rem, &dim_val);
        rem = b.ir_floor_div_i(&rem, &dim_val);
    }
    let types = vec![ZinniaType::Integer; coords.len()];
    Value::Tuple(CompositeData::new(types, coords))
}

fn static_delete_indices(b: &mut IRBuilder, idx: &Value, axis_len: usize) -> Vec<usize> {
    let raw: Vec<i64> = match idx {
        Value::Integer(_) => vec![idx
            .int_val()
            .expect("np.delete: obj scalar must be a static integer")],
        Value::List(d) | Value::Tuple(d) => d
            .values
            .iter()
            .map(|v| {
                v.int_val()
                    .expect("np.delete: obj array values must be static integers")
            })
            .collect(),
        Value::StaticArray { dtype, shape, .. } => {
            if *dtype != NumberType::Integer || shape.len() != 1 {
                panic!("np.delete: obj array must be a 1-D integer StaticArray");
            }
            payload_cells(b, idx)
                .iter()
                .map(|v| {
                    v.int_val()
                        .expect("np.delete: obj array values must be static integers")
                })
                .collect()
        }
        _ => panic!("np.delete: only static integer obj values are supported"),
    };
    let mut out = Vec::with_capacity(raw.len());
    for r in raw {
        let n = axis_len as i64;
        let normalized = if r < 0 { n + r } else { r };
        if normalized < 0 || normalized >= n {
            panic!(
                "np.delete: index {} is out of bounds for axis with size {}",
                r, axis_len
            );
        }
        let u = normalized as usize;
        if !out.contains(&u) {
            out.push(u);
        }
    }
    out.sort_unstable();
    out
}

pub fn np_delete(
    b: &mut IRBuilder,
    args: &[Value],
    kwargs: &std::collections::HashMap<String, Value>,
) -> Value {
    assert!(args.len() >= 2, "np.delete: requires arr and obj");
    let arr = &args[0];
    let obj = &args[1];
    let axis_arg = kwargs.get("axis").or_else(|| args.get(2));
    let (dtype, shape) = match arr {
        Value::StaticArray { dtype, shape, .. } => (*dtype, shape.clone()),
        _ => panic!("np.delete: only StaticArray inputs are supported"),
    };
    if dtype == NumberType::Complex {
        panic!("np.delete: Complex StaticArray delete is not yet implemented");
    }
    if shape.is_empty() {
        panic!("np.delete: scalar arrays are not supported");
    }
    let axis = match axis_arg {
        Some(Value::None) | None if shape.len() == 1 => 0,
        Some(Value::None) | None => {
            panic!("np.delete with axis=None flattening is not yet implemented")
        }
        Some(v) => {
            let raw = v
                .int_val()
                .expect("np.delete: axis must be a static integer or None");
            crate::helpers::shape_arith::resolve_axis(raw, shape.len(), "np.delete")
        }
    };
    let deleted = static_delete_indices(b, obj, shape[axis]);
    let mut out_shape = shape.clone();
    out_shape[axis] = out_shape[axis].saturating_sub(deleted.len());
    let src_strides = crate::helpers::shape_arith::row_major_strides(&shape);
    let out_strides = crate::helpers::shape_arith::row_major_strides(&out_shape);
    let out_total: usize = out_shape.iter().product();
    let cells = payload_cells(b, arr);
    let mut out = Vec::with_capacity(out_total);
    for out_idx in 0..out_total {
        let mut coords =
            crate::helpers::shape_arith::decode_coords(out_idx, &out_shape, &out_strides);
        let mut src_axis = coords[axis];
        for del in &deleted {
            if src_axis >= *del {
                src_axis += 1;
            }
        }
        coords[axis] = src_axis;
        let src_idx = crate::helpers::shape_arith::encode_coords(&coords, &src_strides);
        out.push(cells[src_idx].clone());
    }
    let result =
        crate::helpers::static_array::base::build_static_array_from_flat(b, out, out_shape, dtype);
    if let (Some(in_vid), Some(out_vid)) = (arr.value_id(), result.value_id()) {
        crate::optim::resolver::relay_forall_eq_const_from_input(b, in_vid, out_vid);
    }
    result
}

pub fn np_take_along_axis(
    b: &mut IRBuilder,
    args: &[Value],
    kwargs: &std::collections::HashMap<String, Value>,
) -> Value {
    assert!(
        args.len() >= 2,
        "np.take_along_axis: requires arr and indices"
    );
    let arr = &args[0];
    let indices = &args[1];
    let axis_arg = kwargs
        .get("axis")
        .or_else(|| args.get(2))
        .unwrap_or_else(|| panic!("np.take_along_axis: axis is required"));
    let (dtype, arr_shape) = match arr {
        Value::StaticArray { dtype, shape, .. } => (*dtype, shape.clone()),
        _ => panic!("np.take_along_axis: only StaticArray inputs are supported"),
    };
    if dtype == NumberType::Complex {
        panic!("np.take_along_axis: Complex StaticArray input is not yet implemented");
    }
    let (idx_dtype, idx_shape) = match indices {
        Value::StaticArray { dtype, shape, .. } => (*dtype, shape.clone()),
        _ => panic!("np.take_along_axis: indices must be an integer StaticArray"),
    };
    if idx_dtype != NumberType::Integer {
        panic!("np.take_along_axis: indices must have integer dtype");
    }
    if arr_shape.len() != idx_shape.len() {
        panic!("np.take_along_axis: arr and indices must have the same rank");
    }
    let axis = {
        let raw = axis_arg
            .int_val()
            .expect("np.take_along_axis: axis must be a static integer");
        crate::helpers::shape_arith::resolve_axis(raw, arr_shape.len(), "np.take_along_axis")
    };
    for ax in 0..arr_shape.len() {
        if ax != axis && arr_shape[ax] != idx_shape[ax] {
            panic!("np.take_along_axis: indices shape must match arr shape except along axis");
        }
    }

    let idx_cells = payload_cells(b, indices);
    let out_shape = idx_shape.clone();
    let out_strides = crate::helpers::shape_arith::row_major_strides(&out_shape);
    let arr_strides = crate::helpers::shape_arith::row_major_strides(&arr_shape);
    let out_total: usize = out_shape.iter().product();

    // Small-axis fast path: when the indexed axis has size ≤ SMALL_AXIS,
    // a select chain is cheaper than the zkram path. Per output cell:
    //   zkram path = 1 ir_read_memory (~2 rows) + bounds-check (~10 rows)
    //   select chain (N candidates) = (N-1) selects + 1 bounds-check (~N+10 rows)
    // The zkram path also amortises a per-array materialise (O(total)
    // ir_write_memory). For small axis size + many output cells, the chain
    // wins by avoiding the materialise AND the per-access bounds-check chain.
    //
    // For axis-size 2 (the most common case: pick one of two channels),
    // we can use ir_bool_cast(idx) directly as the select bit. For larger
    // axis sizes we build a select chain over equality comparisons.
    const SMALL_AXIS: usize = 16;
    let axis_size = arr_shape[axis];

    if axis_size > 0 && axis_size <= SMALL_AXIS {
        let arr_payload = crate::helpers::static_array::elementwise::payload_cells(b, arr);
        let mut out = Vec::with_capacity(out_total);
        for out_idx in 0..out_total {
            let coords =
                crate::helpers::shape_arith::decode_coords(out_idx, &out_shape, &out_strides);
            let take_index = idx_cells[out_idx].clone();
            if let Some(i) = take_index.int_val() {
                if i < 0 || (i as usize) >= axis_size {
                    panic!(
                        "np.take_along_axis: index {} is out of bounds for axis {} with size {}",
                        i, axis, axis_size
                    );
                }
                // Static index: pick the candidate directly, no IR ops.
                let mut src_flat: usize = 0;
                for ax in 0..arr_shape.len() {
                    let c = if ax == axis { i as usize } else { coords[ax] };
                    src_flat += c * arr_strides[ax];
                }
                out.push(arr_payload[src_flat].clone());
                continue;
            }
            // Pre-load the N candidates for this output cell's "row".
            let mut candidates: Vec<Value> = Vec::with_capacity(axis_size);
            for k in 0..axis_size {
                let mut src_flat: usize = 0;
                for ax in 0..arr_shape.len() {
                    let c = if ax == axis { k } else { coords[ax] };
                    src_flat += c * arr_strides[ax];
                }
                candidates.push(arr_payload[src_flat].clone());
            }
            // Build the select + lookup-free bounds soundness check.
            //
            // Note: we deliberately do NOT call `discharge_index_in_range`
            // here. That helper emits LessThan + GreaterThanOrEqual, both
            // of which trigger range-check lookups. For the small-axis
            // path we already produce equality witnesses (eq_k) as part
            // of the select chain; constraining `sum_k(eq_k) == 1` proves
            // `idx ∈ {0..N-1}` soundly with no lookups, just arithmetic.
            let picked = if axis_size == 1 {
                candidates[0].clone()
            } else if axis_size == 2 {
                // Constrain idx ∈ {0, 1} while selecting in the same
                // backend gate where supported. idx doubles as the select
                // condition.
                if dtype == NumberType::Float {
                    b.ir_select_bit_f(&take_index, &candidates[1], &candidates[0])
                } else {
                    b.ir_select_bit_i(&take_index, &candidates[1], &candidates[0])
                }
            } else if axis_size <= 8 {
                // Static small-axis bit routing. The backend constrains
                // idx < axis_size and exposes low index bits, so we can
                // route without per-candidate equality masks.
                let bits = b.ir_assert_u128_lt_bits(&take_index, axis_size);
                let select = |b: &mut IRBuilder, cond: &Value, t: &Value, f: &Value| {
                    if dtype == NumberType::Float {
                        b.ir_select_f(cond, t, f)
                    } else {
                        b.ir_select_i(cond, t, f)
                    }
                };
                match axis_size {
                    3 => {
                        let t0 = select(b, &bits[0], &candidates[1], &candidates[0]);
                        select(b, &bits[1], &candidates[2], &t0)
                    }
                    4 => {
                        let t0 = select(b, &bits[0], &candidates[1], &candidates[0]);
                        let t1 = select(b, &bits[0], &candidates[3], &candidates[2]);
                        select(b, &bits[1], &t1, &t0)
                    }
                    5 => {
                        let t0 = select(b, &bits[0], &candidates[1], &candidates[0]);
                        let t1 = select(b, &bits[0], &candidates[3], &candidates[2]);
                        let t01 = select(b, &bits[1], &t1, &t0);
                        select(b, &bits[2], &candidates[4], &t01)
                    }
                    6 => {
                        let t0 = select(b, &bits[0], &candidates[1], &candidates[0]);
                        let t1 = select(b, &bits[0], &candidates[3], &candidates[2]);
                        let t01 = select(b, &bits[1], &t1, &t0);
                        let t2 = select(b, &bits[0], &candidates[5], &candidates[4]);
                        select(b, &bits[2], &t2, &t01)
                    }
                    7 => {
                        let t0 = select(b, &bits[0], &candidates[1], &candidates[0]);
                        let t1 = select(b, &bits[0], &candidates[3], &candidates[2]);
                        let t01 = select(b, &bits[1], &t1, &t0);
                        let t2 = select(b, &bits[0], &candidates[5], &candidates[4]);
                        let t23 = select(b, &bits[1], &candidates[6], &t2);
                        select(b, &bits[2], &t23, &t01)
                    }
                    8 => {
                        let t0 = select(b, &bits[0], &candidates[1], &candidates[0]);
                        let t1 = select(b, &bits[0], &candidates[3], &candidates[2]);
                        let t2 = select(b, &bits[0], &candidates[5], &candidates[4]);
                        let t3 = select(b, &bits[0], &candidates[7], &candidates[6]);
                        let t01 = select(b, &bits[1], &t1, &t0);
                        let t23 = select(b, &bits[1], &t3, &t2);
                        select(b, &bits[2], &t23, &t01)
                    }
                    _ => unreachable!("axis_size > 2 and <= 8 handled here"),
                }
            } else {
                // General small-N: select chain. Default to candidates[0];
                // for k in 1..N, replace if idx == k. We also compute
                // `eq_0 = is_equal(idx, 0)` and assert
                // `sum_k(eq_k) == 1` — this proves idx ∈ {0..N-1} without
                // any range-check lookup.
                let zero_const = b.ir_constant_int(0);
                let eq_0 = b.ir_equal_i(&take_index, &zero_const);
                let mut acc = candidates[0].clone();
                let mut eq_sum = eq_0;
                for k in 1..axis_size {
                    let k_const = b.ir_constant_int(k as i64);
                    let eq_k = b.ir_equal_i(&take_index, &k_const);
                    acc = if dtype == NumberType::Float {
                        b.ir_select_f(&eq_k, &candidates[k], &acc)
                    } else {
                        b.ir_select_i(&eq_k, &candidates[k], &acc)
                    };
                    eq_sum = b.ir_add_i(&eq_sum, &eq_k);
                }
                let one_const = b.ir_constant_int(1);
                let is_in_range = b.ir_equal_i(&eq_sum, &one_const);
                b.ir_assert(&is_in_range);
                acc
            };
            out.push(picked);
        }
        return crate::helpers::static_array::base::build_static_array_from_flat(
            b, out, out_shape, dtype,
        );
    }

    let mut out = Vec::with_capacity(out_total);
    for out_idx in 0..out_total {
        let coords = crate::helpers::shape_arith::decode_coords(out_idx, &out_shape, &out_strides);
        let take_index = idx_cells[out_idx].clone();
        let take_index = if let Some(i) = take_index.int_val() {
            let dim = arr_shape[axis] as i64;
            if i < 0 || i >= dim {
                panic!(
                    "np.take_along_axis: index {} is out of bounds for axis {} with size {}",
                    i, axis, dim
                );
            }
            take_index
        } else {
            // The returned index stays in range where the code does not run.
            crate::optim::resolver::discharge_index_in_range(
                b,
                &take_index,
                0,
                arr_shape[axis] as i64,
                "np.take_along_axis",
            )
        };
        let mut src_indices = Vec::with_capacity(arr_shape.len());
        for ax in 0..arr_shape.len() {
            if ax == axis {
                src_indices.push(SliceIndex::Single(take_index.clone()));
            } else {
                src_indices.push(SliceIndex::Single(b.ir_constant_int(coords[ax] as i64)));
            }
        }
        let _src_linear_hint = crate::helpers::shape_arith::encode_coords(&coords, &arr_strides);
        out.push(
            crate::helpers::static_array::read::static_array_subscript_with_preproved_index_bounds(
                b,
                arr,
                &src_indices,
            ),
        );
    }
    crate::helpers::static_array::base::build_static_array_from_flat(b, out, out_shape, dtype)
}

pub fn np_insert(
    b: &mut IRBuilder,
    args: &[Value],
    kwargs: &std::collections::HashMap<String, Value>,
) -> Value {
    assert!(args.len() >= 3, "np.insert: requires arr, obj, and values");
    let arr = &args[0];
    let obj = &args[1];
    let values = &args[2];
    let axis_arg = kwargs.get("axis").or_else(|| args.get(3));
    let (dtype, shape) = static_array_shape_dtype(arr, "np.insert");
    if dtype == NumberType::Complex {
        panic!("np.insert: Complex StaticArray insert is not yet implemented");
    }
    if shape.is_empty() {
        panic!("np.insert: scalar arrays are not supported");
    }
    let axis = match axis_arg {
        Some(Value::None) | None if shape.len() == 1 => 0,
        Some(Value::None) | None => {
            panic!("np.insert with axis=None flattening is not yet implemented")
        }
        Some(v) => {
            let raw = v
                .int_val()
                .expect("np.insert: axis must be a static integer or None");
            crate::helpers::shape_arith::resolve_axis(raw, shape.len(), "np.insert")
        }
    };

    if !matches!(obj, Value::Integer(_) | Value::Boolean(_)) {
        panic!("np.insert: this bounded slice supports one scalar obj");
    }
    if let Some(raw_pos) = obj.int_val() {
        if raw_pos < 0 || raw_pos > shape[axis] as i64 {
            panic!(
                "np.insert: obj {} is out of bounds for this bounded slice; expected 0 <= obj <= {}",
                raw_pos, shape[axis]
            );
        }
    } else {
        crate::optim::resolver::discharge_index_in_range(
            b,
            obj,
            0,
            shape[axis] as i64 + 1,
            "np.insert",
        );
    }

    let insert_count = 1usize;
    let mut out_shape = shape.clone();
    out_shape[axis] += insert_count;
    let out_total: usize = out_shape.iter().product();
    let out_strides = crate::helpers::shape_arith::row_major_strides(&out_shape);
    let src_strides = crate::helpers::shape_arith::row_major_strides(&shape);
    let arr_cells = payload_cells(b, arr);
    let value_cells: Vec<Value> = match values {
        Value::StaticArray { .. } => payload_cells(b, values),
        Value::List(_) | Value::Tuple(_) => {
            crate::helpers::composite::flatten_composite_with_builder(b, values)
        }
        _ => vec![values.clone()],
    };
    let expected_value_cells: usize = shape
        .iter()
        .enumerate()
        .filter_map(|(i, dim)| if i == axis { None } else { Some(*dim) })
        .product();
    if !(value_cells.len() == 1 || value_cells.len() == expected_value_cells) {
        panic!(
            "np.insert: values must be scalar or match one inserted axis slice; got {} cells, expected 1 or {}",
            value_cells.len(),
            expected_value_cells
        );
    }
    let value_shape: Vec<usize> = shape
        .iter()
        .enumerate()
        .filter_map(|(i, dim)| if i == axis { None } else { Some(*dim) })
        .collect();
    let value_strides = crate::helpers::shape_arith::row_major_strides(&value_shape);

    let select_typed = |b: &mut IRBuilder, cond: &Value, tv: &Value, fv: &Value| -> Value {
        if dtype == NumberType::Float {
            b.ir_select_f(cond, tv, fv)
        } else {
            b.ir_select_i(cond, tv, fv)
        }
    };
    let select_many_typed =
        |b: &mut IRBuilder, cond: &Value, tv: &[Value], fv: &[Value]| -> Vec<Value> {
            if dtype == NumberType::Float {
                b.ir_select_many_f(cond, tv, fv)
            } else {
                b.ir_select_many_i(cond, tv, fv)
            }
        };

    if obj.int_val().is_none() {
        let n_axis = shape[axis];
        let lane_total = expected_value_cells;
        let mut out_slots: Vec<Option<Value>> = vec![None; out_total];
        // Output slot `i` takes the source element `i` before the insertion
        // point, the inserted value at it and the source element `i - 1`
        // after it. `[obj <= i]` is the running OR of the equality tests
        // `[obj == k]`, k <= i: `obj` is bounds-checked to `0..=n_axis`
        // above, so exactly one of those tests holds, and the prefix OR is
        // set from `k = obj` on. That replaces a comparison per slot with
        // one gate.
        let mut reached: Option<Value> = None;
        for i in 0..=n_axis {
            let i_const = b.ir_constant_int(i as i64);
            let is_eq = b.ir_equal_i(&i_const, obj);
            let at_or_after = match &reached {
                Some(prev) => b.ir_logical_or(prev, &is_eq),
                None => is_eq.clone(),
            };
            reached = Some(at_or_after.clone());
            let mut inserted_values = Vec::with_capacity(lane_total);
            let mut pre_values = Vec::with_capacity(lane_total);
            let mut post_values = Vec::with_capacity(lane_total);
            let mut out_indices = Vec::with_capacity(lane_total);
            for lane_idx in 0..lane_total {
                let lane_coords = if value_shape.is_empty() {
                    Vec::new()
                } else {
                    crate::helpers::shape_arith::decode_coords(
                        lane_idx,
                        &value_shape,
                        &value_strides,
                    )
                };
                let mut coords = Vec::with_capacity(shape.len());
                let mut lane_pos = 0usize;
                for ax in 0..shape.len() {
                    if ax == axis {
                        coords.push(i);
                    } else {
                        coords.push(lane_coords[lane_pos]);
                        lane_pos += 1;
                    }
                }
                out_indices.push(crate::helpers::shape_arith::encode_coords(
                    &coords,
                    &out_strides,
                ));
                let value_idx = if value_cells.len() == 1 { 0 } else { lane_idx };
                inserted_values.push(cast_cell_to_dtype(b, &value_cells[value_idx], dtype));

                let pre_val = if i < n_axis {
                    let mut src_coords = coords.clone();
                    src_coords[axis] = i;
                    let src_idx =
                        crate::helpers::shape_arith::encode_coords(&src_coords, &src_strides);
                    arr_cells[src_idx].clone()
                } else {
                    let mut src_coords = coords.clone();
                    src_coords[axis] = 0;
                    let src_idx =
                        crate::helpers::shape_arith::encode_coords(&src_coords, &src_strides);
                    arr_cells[src_idx].clone()
                };
                pre_values.push(pre_val);

                let post_val = if i >= 1 {
                    let mut src_coords = coords.clone();
                    src_coords[axis] = i - 1;
                    let src_idx =
                        crate::helpers::shape_arith::encode_coords(&src_coords, &src_strides);
                    arr_cells[src_idx].clone()
                } else {
                    let mut src_coords = coords.clone();
                    src_coords[axis] = 0;
                    let src_idx =
                        crate::helpers::shape_arith::encode_coords(&src_coords, &src_strides);
                    arr_cells[src_idx].clone()
                };
                post_values.push(post_val);
            }
            let mid_values = select_many_typed(b, &is_eq, &inserted_values, &post_values);
            let chosen_values = select_many_typed(b, &at_or_after, &mid_values, &pre_values);
            for (out_idx, value) in out_indices.into_iter().zip(chosen_values) {
                out_slots[out_idx] = Some(value);
            }
        }
        let out = out_slots
            .into_iter()
            .map(|slot| slot.expect("np.insert: internal output slot not filled"))
            .collect();
        return crate::helpers::static_array::base::build_static_array_from_flat(
            b, out, out_shape, dtype,
        );
    }

    let mut out = Vec::with_capacity(out_total);
    for out_idx in 0..out_total {
        let coords = crate::helpers::shape_arith::decode_coords(out_idx, &out_shape, &out_strides);
        let inserted_value = {
            let value_idx = if value_cells.len() == 1 {
                0
            } else {
                let value_coords: Vec<usize> = coords
                    .iter()
                    .enumerate()
                    .filter_map(|(i, c)| if i == axis { None } else { Some(*c) })
                    .collect();
                crate::helpers::shape_arith::encode_coords(&value_coords, &value_strides)
            };
            cast_cell_to_dtype(b, &value_cells[value_idx], dtype)
        };
        if let Some(raw_pos) = obj.int_val() {
            let pos = raw_pos as usize;
            if coords[axis] == pos {
                out.push(inserted_value);
            } else {
                let mut src_coords = coords.clone();
                if coords[axis] > pos {
                    src_coords[axis] -= insert_count;
                }
                let src_idx = crate::helpers::shape_arith::encode_coords(&src_coords, &src_strides);
                out.push(arr_cells[src_idx].clone());
            }
        } else {
            // Runtime `pos`: at each output position i along `axis` the source
            // is fully determined by two runtime predicates `(i < pos)` and
            // `(i == pos)`. The mux to N candidates each via N-1 selects was
            // O(N) per output cell — replace with the 2-cell static-shift
            // pattern: pre_val = arr[i] (used when i<pos), post_val =
            // arr[i-1] (used when i>pos), and the inserted value in the
            // middle. Bounds-only-when-selected: the i=0 branch's post_val
            // is never picked (is_lt or is_eq must hold); the
            // i=shape[axis] branch's pre_val is never picked. We default
            // out-of-range coords to a valid in-bounds clone so the
            // backend gets a wire to read but the constraint never
            // observes a malformed value.
            let i = coords[axis];
            let n_axis = shape[axis];
            // pre_val: arr at coords (used when i < pos, valid for i < n_axis)
            let pre_val = if i < n_axis {
                let src_idx = crate::helpers::shape_arith::encode_coords(&coords, &src_strides);
                arr_cells[src_idx].clone()
            } else {
                let mut src_coords = coords.clone();
                src_coords[axis] = 0; // dummy; never selected at i == n_axis
                let src_idx = crate::helpers::shape_arith::encode_coords(&src_coords, &src_strides);
                arr_cells[src_idx].clone()
            };
            // post_val: arr at coords with axis-1 (used when i > pos, valid for i >= 1)
            let post_val = if i >= 1 {
                let mut src_coords = coords.clone();
                src_coords[axis] = i - 1;
                let src_idx = crate::helpers::shape_arith::encode_coords(&src_coords, &src_strides);
                arr_cells[src_idx].clone()
            } else {
                let mut src_coords = coords.clone();
                src_coords[axis] = 0; // dummy; never selected at i == 0
                let src_idx = crate::helpers::shape_arith::encode_coords(&src_coords, &src_strides);
                arr_cells[src_idx].clone()
            };
            // Runtime predicates — independent of `j`, so the IRBuilder
            // value-numbering naturally dedupes them across all j for the
            // same i.
            let i_const = b.ir_constant_int(i as i64);
            let is_lt = b.ir_less_than_i(&i_const, obj);
            let is_eq = b.ir_equal_i(&i_const, obj);
            // Two selects per output cell: middle picks between inserted and
            // post; outer picks between pre and middle.
            let mid = select_typed(b, &is_eq, &inserted_value, &post_val);
            let chosen = select_typed(b, &is_lt, &pre_val, &mid);
            out.push(chosen);
        }
    }
    crate::helpers::static_array::base::build_static_array_from_flat(b, out, out_shape, dtype)
}

/// Extract `dims` as a `Vec<Value>` (rank is statically known; each value
/// may be a compile-time integer or a runtime IR integer wire).
///
/// Accepts:
/// - `Value::Tuple(_) | Value::List(_)` of integer cells.
/// - 1-D integer `Value::StaticArray`.
///
/// Statically-known nonneg dims are still validated at compile time;
/// dynamic dim wires are returned as-is and bounds are enforced at the
/// per-index discharge site below.
fn dims_cells_from_value(b: &mut IRBuilder, dims_value: &Value, op: &str) -> Vec<Value> {
    let cells = match dims_value {
        Value::Tuple(d) | Value::List(d) => d.values.clone(),
        Value::StaticArray { dtype, shape, .. } => {
            if *dtype != NumberType::Integer || shape.len() != 1 {
                panic!("{op}: dims array must be a 1-D integer StaticArray");
            }
            payload_cells(b, dims_value)
        }
        _ => panic!("{op}: dims must be a tuple/list/1-D array of integers"),
    };
    for c in &cells {
        if !matches!(c, Value::Integer(_) | Value::Boolean(_)) {
            panic!("{op}: dims entries must be integers");
        }
        if let Some(n) = c.int_val() {
            assert!(n >= 0, "{op}: dims must be nonnegative (got {n})");
        }
    }
    cells
}

pub fn np_ravel_multi_index(
    b: &mut IRBuilder,
    args: &[Value],
    kwargs: &std::collections::HashMap<String, Value>,
) -> Value {
    let multi_index = args
        .first()
        .unwrap_or_else(|| panic!("np.ravel_multi_index: requires multi_index"));
    let dims_arg = kwargs
        .get("dims")
        .or_else(|| args.get(1))
        .unwrap_or_else(|| panic!("np.ravel_multi_index: requires dims"));
    let dims_cells = dims_cells_from_value(b, dims_arg, "np.ravel_multi_index");
    let rank = dims_cells.len();
    let order = order_from_args(args, kwargs, "np.ravel_multi_index");

    // ── Discover multi_index layout ──────────────────────────────────────
    //
    // Two shapes accepted:
    //   (a) "scalar-vector": rank scalars → produce a scalar Integer.
    //       Sources: 1-D StaticArray of shape (rank,), or List/Tuple of
    //       rank cells.
    //   (b) "vectorized 2-D": StaticArray of shape (rank, N) → produce a
    //       1-D StaticArray of shape (N,). Each output cell j is the flat
    //       index of column j.
    let (idx_cells, vec_len) = match multi_index {
        Value::StaticArray { dtype, shape, .. } => {
            if *dtype != NumberType::Integer {
                panic!("np.ravel_multi_index: multi_index must have integer dtype");
            }
            match shape.as_slice() {
                [r] => {
                    if *r != rank {
                        panic!("np.ravel_multi_index: multi_index shape must be (len(dims),)");
                    }
                    (payload_cells(b, multi_index), None)
                }
                [r, n] => {
                    if *r != rank {
                        panic!(
                            "np.ravel_multi_index: multi_index axis-0 length {r} must equal len(dims)={rank}"
                        );
                    }
                    (payload_cells(b, multi_index), Some(*n))
                }
                _ => {
                    panic!("np.ravel_multi_index: multi_index must be 1-D (rank,) or 2-D (rank, N)")
                }
            }
        }
        Value::List(d) | Value::Tuple(d) => {
            if d.values.len() != rank {
                panic!("np.ravel_multi_index: multi_index length must equal len(dims)");
            }
            (d.values.clone(), None)
        }
        _ => panic!("np.ravel_multi_index: multi_index must be a static integer array/list/tuple"),
    };

    // ── Precompute strides (static if every dim is static) ───────────────
    //
    // C order: stride[k] = prod(dims[k+1..]); F order: stride[0]=1,
    // stride[k] = stride[k-1] * dims[k-1].
    let static_dims: Option<Vec<usize>> = dims_cells
        .iter()
        .map(|d| {
            d.int_val()
                .and_then(|n| if n >= 0 { Some(n as usize) } else { None })
        })
        .collect();

    enum Strides {
        Static(Vec<usize>),
        Dynamic(Vec<Value>),
    }
    let strides = if let Some(sd) = static_dims.as_ref() {
        let s = if order == "C" {
            crate::helpers::shape_arith::row_major_strides(sd)
        } else {
            let mut s = vec![1usize; rank];
            for i in 1..rank {
                s[i] = s[i - 1] * sd[i - 1];
            }
            s
        };
        Strides::Static(s)
    } else {
        let mut s: Vec<Value> = vec![b.ir_constant_int(1); rank];
        if order == "C" {
            for k in (0..rank.saturating_sub(1)).rev() {
                s[k] = b.ir_mul_i(&s[k + 1], &dims_cells[k + 1]);
            }
        } else {
            for k in 1..rank {
                s[k] = b.ir_mul_i(&s[k - 1], &dims_cells[k - 1]);
            }
        }
        Strides::Dynamic(s)
    };

    // ── Per-output-cell flat-index emission ─────────────────────────────
    //
    // For each output cell j (or the single scalar case j=0), accumulate
    //     acc[j] = sum_{axis=0..rank} index_cells[axis * N + j] * stride[axis]
    // and discharge `0 <= index_cells[axis, j] < dims[axis]` per axis.
    let n_out = vec_len.unwrap_or(1);
    let mut out_cells: Vec<Value> = Vec::with_capacity(n_out);
    for j in 0..n_out {
        let mut acc = b.ir_constant_int(0);
        for axis in 0..rank {
            let idx = if vec_len.is_some() {
                &idx_cells[axis * n_out + j]
            } else {
                &idx_cells[axis]
            };
            if !matches!(idx, Value::Integer(_) | Value::Boolean(_)) {
                panic!("np.ravel_multi_index: multi_index entries must be integers");
            }
            match (idx.int_val(), static_dims.as_ref().map(|sd| sd[axis])) {
                (Some(i), Some(dim)) => {
                    if i < 0 || i >= dim as i64 {
                        panic!(
                            "np.ravel_multi_index: index {i} is out of bounds for axis {axis} with size {dim}"
                        );
                    }
                }
                (_, Some(dim)) => {
                    crate::optim::resolver::discharge_index_in_range(
                        b,
                        idx,
                        0,
                        dim as i64,
                        "np.ravel_multi_index",
                    );
                }
                (_, None) => {
                    crate::optim::resolver::discharge_index_below_value(
                        b,
                        idx,
                        0,
                        &dims_cells[axis],
                        "np.ravel_multi_index",
                    );
                }
            }
            let stride_val = match &strides {
                Strides::Static(s) => b.ir_constant_int(s[axis] as i64),
                Strides::Dynamic(s) => s[axis].clone(),
            };
            let term = b.ir_mul_i(idx, &stride_val);
            acc = b.ir_add_i(&acc, &term);
        }
        out_cells.push(acc);
    }

    match vec_len {
        None => out_cells.into_iter().next().unwrap(),
        Some(n) => crate::helpers::static_array::base::build_static_array_from_flat(
            b,
            out_cells,
            vec![n],
            NumberType::Integer,
        ),
    }
}

fn static_pad_width_from_value(
    b: &mut IRBuilder,
    pad_width: &Value,
    rank: usize,
) -> Vec<(usize, usize)> {
    fn static_nonnegative_usize(v: &Value, op: &str) -> usize {
        let n = v
            .int_val()
            .unwrap_or_else(|| panic!("{op}: pad_width values must be static"));
        assert!(n >= 0, "{op}: pad widths must be nonnegative");
        n as usize
    }
    if let Value::List(d) | Value::Tuple(d) = pad_width {
        if d.values.len() == rank
            && d.values
                .iter()
                .all(|v| matches!(v, Value::List(_) | Value::Tuple(_)))
        {
            let mut out = Vec::with_capacity(rank);
            for pair in &d.values {
                let pair_data = match pair {
                    Value::List(data) | Value::Tuple(data) => data,
                    _ => unreachable!(),
                };
                if pair_data.values.len() != 2 {
                    panic!("np.pad: each per-axis pad_width entry must have length 2");
                }
                out.push((
                    static_nonnegative_usize(&pair_data.values[0], "np.pad"),
                    static_nonnegative_usize(&pair_data.values[1], "np.pad"),
                ));
            }
            return out;
        }
    }
    let cells = match pad_width {
        Value::Integer(_) => {
            let n = static_nonnegative_usize(pad_width, "np.pad");
            return vec![(n, n); rank];
        }
        Value::List(d) | Value::Tuple(d) => d.values.clone(),
        Value::StaticArray { dtype, shape, .. } => {
            if *dtype != NumberType::Integer {
                panic!("np.pad: pad_width array must have integer dtype");
            }
            let vals = payload_cells(b, pad_width);
            if shape.as_slice() == [rank, 2] {
                let mut out = Vec::with_capacity(rank);
                for axis in 0..rank {
                    let before = vals[axis * 2]
                        .int_val()
                        .expect("np.pad: pad_width values must be static");
                    let after = vals[axis * 2 + 1]
                        .int_val()
                        .expect("np.pad: pad_width values must be static");
                    assert!(
                        before >= 0 && after >= 0,
                        "np.pad: pad widths must be nonnegative"
                    );
                    out.push((before as usize, after as usize));
                }
                return out;
            }
            vals
        }
        _ => panic!("np.pad: pad_width must be static"),
    };
    if rank == 1 && cells.len() == 2 && cells.iter().all(|v| v.int_val().is_some()) {
        let before = cells[0].int_val().unwrap();
        let after = cells[1].int_val().unwrap();
        assert!(
            before >= 0 && after >= 0,
            "np.pad: pad widths must be nonnegative"
        );
        return vec![(before as usize, after as usize)];
    }
    if cells.len() == rank {
        let mut out = Vec::with_capacity(rank);
        for v in cells {
            let n = v
                .int_val()
                .expect("np.pad: pad_width values must be static");
            assert!(n >= 0, "np.pad: pad widths must be nonnegative");
            out.push((n as usize, n as usize));
        }
        return out;
    }
    if cells.len() == rank * 2 {
        let mut out = Vec::with_capacity(rank);
        for axis in 0..rank {
            let before = cells[axis * 2]
                .int_val()
                .expect("np.pad: pad_width values must be static");
            let after = cells[axis * 2 + 1]
                .int_val()
                .expect("np.pad: pad_width values must be static");
            assert!(
                before >= 0 && after >= 0,
                "np.pad: pad widths must be nonnegative"
            );
            out.push((before as usize, after as usize));
        }
        return out;
    }
    panic!("np.pad: unsupported pad_width shape for static lowering");
}

fn runtime_pad_width_pairs(pad_width: &Value, rank: usize) -> Option<Vec<(Value, Value)>> {
    let elems = match pad_width {
        Value::List(d) | Value::Tuple(d) => d.values.clone(),
        _ => return None,
    };
    if rank == 1 && elems.len() == 2 && elems.iter().all(|v| v.int_val().is_some()) {
        return Some(vec![(elems[0].clone(), elems[1].clone())]);
    }
    if elems.len() == rank
        && elems
            .iter()
            .all(|v| matches!(v, Value::List(_) | Value::Tuple(_)))
    {
        let mut out = Vec::with_capacity(rank);
        for pair in elems {
            let pair_data = match pair {
                Value::List(d) | Value::Tuple(d) => d,
                _ => unreachable!(),
            };
            if pair_data.values.len() != 2 {
                return None;
            }
            out.push((pair_data.values[0].clone(), pair_data.values[1].clone()));
        }
        return Some(out);
    }
    if elems.len() == rank * 2 {
        let mut out = Vec::with_capacity(rank);
        for axis in 0..rank {
            out.push((elems[axis * 2].clone(), elems[axis * 2 + 1].clone()));
        }
        return Some(out);
    }
    None
}

fn checked_runtime_pad_width_pairs(
    b: &mut IRBuilder,
    pad_width: &Value,
    rank: usize,
) -> Option<Vec<(Value, Value)>> {
    let pairs = runtime_pad_width_pairs(pad_width, rank)?;
    let zero = b.ir_constant_int(0);
    for (before, after) in &pairs {
        for width in [before, after] {
            if !matches!(width, Value::Integer(_) | Value::Boolean(_)) {
                panic!("np.pad: runtime pad_width entries must be integer scalars");
            }
            if let Some(n) = width.int_val() {
                assert!(n >= 0, "np.pad: pad widths must be nonnegative");
            } else {
                let ok = b.ir_greater_than_or_equal_i(width, &zero);
                b.ir_assert(&ok);
            }
        }
    }
    Some(pairs)
}

fn pad_width_upper_bound(b: &mut IRBuilder, width: &Value) -> usize {
    if let Some(n) = width.int_val() {
        assert!(n >= 0, "np.pad: pad widths must be nonnegative");
        return n as usize;
    }
    if let Some((min, max)) = b.ask_bounds(width) {
        if min >= 0 && max >= 0 && max <= 1_000_000 {
            return max as usize;
        }
    }
    if let Some((min, max)) = pad_width_expr_bounds(b, width) {
        if min >= 0 && max >= 0 && max <= 1_000_000 {
            return max as usize;
        }
    }
    // Final fallback: the prove-based chokepoint. `ask_bounds` uses the
    // SMT-Optimize path which doesn't always pick up scalar precondi-
    // tions on derived values; `resolve_int_or_bounded` runs the
    // outward-doubling prove probe over the SMT-Solver path which does
    // discharge precondition-derived bounds. This is the same path
    // `np_arange` uses to admit `@requires`-bounded scalars.
    use crate::optim::resolver::{resolve_int_or_bounded, BoundedInt, SiteKind};
    let (lo, hi) = match resolve_int_or_bounded(b, width, SiteKind::RangeStop, None) {
        BoundedInt::Static(n) => (n, n),
        BoundedInt::Bounded { min, max } => (min, max),
        BoundedInt::Neither => (i64::MIN, i64::MAX),
    };
    if lo >= 0 && hi >= 0 && hi <= 1_000_000 {
        return hi as usize;
    }
    // The previous behaviour silently capped the pad width at 128 and
    // injected a witness assert `width <= 128` — sound but a hidden
    // contract that the user never asked for. We now reject loudly:
    // every pad width must either be a literal, be provably bounded
    // by visible facts (e.g. via `@requires`), or be derivable from
    // a bounded arithmetic expression. See
    // `compiler.np-pad-no-silent-default-bound`.
    panic!(
        "np.pad: pad width is not statically bounded — its value at runtime \
         could be anything. The previous compiler revision silently capped it \
         at 128 with a proof-time witness assert; that has been removed. To \
         compile this program, either pass a literal pad width, or annotate \
         the kernel inputs that drive the pad width with `@requires(lambda \
         x: x <= K)` so the compiler can derive a tight, sound upper bound."
    )
}

fn value_for_stmt_id(b: &IRBuilder, stmt_id: crate::types::StmtId) -> Value {
    let stmt = &b.stmts[stmt_id as usize];
    match &stmt.ir {
        crate::ir_defs::IR::ConstantInt { value } => {
            Value::Integer(ScalarValue::new(Some(*value), Some(stmt_id)))
        }
        _ => Value::Integer(ScalarValue::new(None, Some(stmt_id))),
    }
}

fn pad_width_expr_bounds(b: &mut IRBuilder, value: &Value) -> Option<(i64, i64)> {
    if let Some(n) = value.int_val() {
        return Some((n, n));
    }
    let stmt_id = value.stmt_id()?;
    let stmt = b.stmts.get(stmt_id as usize)?.clone();
    match stmt.ir {
        crate::ir_defs::IR::ConstantInt { value } => Some((value, value)),
        crate::ir_defs::IR::AddI => {
            let lhs = value_for_stmt_id(b, stmt.arguments[0]);
            let rhs = value_for_stmt_id(b, stmt.arguments[1]);
            let (lmin, lmax) = b
                .ask_bounds(&lhs)
                .or_else(|| pad_width_expr_bounds(b, &lhs))?;
            let (rmin, rmax) = b
                .ask_bounds(&rhs)
                .or_else(|| pad_width_expr_bounds(b, &rhs))?;
            Some((lmin.saturating_add(rmin), lmax.saturating_add(rmax)))
        }
        crate::ir_defs::IR::SubI => {
            let lhs = value_for_stmt_id(b, stmt.arguments[0]);
            let rhs = value_for_stmt_id(b, stmt.arguments[1]);
            let (lmin, lmax) = b
                .ask_bounds(&lhs)
                .or_else(|| pad_width_expr_bounds(b, &lhs))?;
            let (rmin, rmax) = b
                .ask_bounds(&rhs)
                .or_else(|| pad_width_expr_bounds(b, &rhs))?;
            Some((lmin.saturating_sub(rmax), lmax.saturating_sub(rmin)))
        }
        crate::ir_defs::IR::FloorDivI | crate::ir_defs::IR::DivI => {
            let lhs = value_for_stmt_id(b, stmt.arguments[0]);
            let rhs = value_for_stmt_id(b, stmt.arguments[1]);
            let (lmin, lmax) = b
                .ask_bounds(&lhs)
                .or_else(|| pad_width_expr_bounds(b, &lhs))?;
            let (rmin, rmax) = b
                .ask_bounds(&rhs)
                .or_else(|| pad_width_expr_bounds(b, &rhs))?;
            if rmin <= 0 || rmax <= 0 {
                return None;
            }
            Some((lmin.div_euclid(rmax), lmax.div_euclid(rmin)))
        }
        crate::ir_defs::IR::ModI => {
            let rhs = value_for_stmt_id(b, stmt.arguments[1]);
            let (rmin, rmax) = b
                .ask_bounds(&rhs)
                .or_else(|| pad_width_expr_bounds(b, &rhs))?;
            if rmin <= 0 || rmax <= 0 {
                return None;
            }
            Some((0, rmax - 1))
        }
        _ => None,
    }
}

fn scalar_from_usize(b: &mut IRBuilder, n: usize) -> ScalarValue<i64> {
    value_to_scalar_i64(&b.ir_constant_int(n as i64))
}

fn value_from_scalar_i64(b: &mut IRBuilder, sv: &ScalarValue<i64>) -> Value {
    if let Some(n) = sv.static_val {
        b.ir_constant_int(n)
    } else if let Some(stmt_id) = sv.stmt_id {
        Value::Integer(ScalarValue::new(None, Some(stmt_id)))
    } else {
        b.ir_constant_int(0)
    }
}

fn runtime_strides_from_shape(
    b: &mut IRBuilder,
    runtime_shape_values: &[Value],
) -> Vec<ScalarValue<i64>> {
    let rank = runtime_shape_values.len();
    let mut strides = vec![ScalarValue::new(Some(1), None); rank];
    if rank == 0 {
        return strides;
    }
    for axis in (0..rank.saturating_sub(1)).rev() {
        let next_stride = value_from_scalar_i64(b, &strides[axis + 1]);
        let stride = b.ir_mul_i(&next_stride, &runtime_shape_values[axis + 1]);
        strides[axis] = value_to_scalar_i64(&stride);
    }
    strides
}

fn np_pad_runtime_static_input(
    b: &mut IRBuilder,
    arr: &Value,
    dtype: NumberType,
    shape: &[usize],
    runtime_pairs: Vec<(Value, Value)>,
    constant_value: &Value,
) -> Value {
    let pad_bounds: Vec<(usize, usize)> = runtime_pairs
        .iter()
        .map(|(before, after)| {
            (
                pad_width_upper_bound(b, before),
                pad_width_upper_bound(b, after),
            )
        })
        .collect();
    let max_shape: Vec<usize> = shape
        .iter()
        .zip(pad_bounds.iter())
        .map(|(dim, (before, after))| dim + before + after)
        .collect();
    let max_total: usize = max_shape.iter().product();
    let max_strides = crate::helpers::shape_arith::row_major_strides(&max_shape);
    let src_strides = crate::helpers::shape_arith::row_major_strides(shape);
    let zero = b.ir_constant_int(0);

    let runtime_shape_values: Vec<Value> = shape
        .iter()
        .zip(runtime_pairs.iter())
        .map(|(dim, (before, after))| {
            let dim_v = b.ir_constant_int(*dim as i64);
            let with_before = b.ir_add_i(&dim_v, before);
            b.ir_add_i(&with_before, after)
        })
        .collect();
    let runtime_shape: Vec<ScalarValue<i64>> = runtime_shape_values
        .iter()
        .map(value_to_scalar_i64)
        .collect();
    let runtime_length_value = if runtime_shape_values.is_empty() {
        b.ir_constant_int(1)
    } else {
        let mut acc = runtime_shape_values[0].clone();
        for axis_len in runtime_shape_values.iter().skip(1) {
            acc = b.ir_mul_i(&acc, axis_len);
        }
        acc
    };
    let runtime_strides = runtime_strides_from_shape(b, &runtime_shape_values);

    let mut out = Vec::with_capacity(max_total);
    for out_idx in 0..max_total {
        let out_coords =
            crate::helpers::shape_arith::decode_coords(out_idx, &max_shape, &max_strides);
        let mut in_runtime_extent = b.ir_constant_bool(true);
        let mut inside_src = b.ir_constant_bool(true);
        let mut src_idx = b.ir_constant_int(0);

        for (axis, coord) in out_coords.iter().enumerate() {
            let coord_v = b.ir_constant_int(*coord as i64);
            let runtime_dim = &runtime_shape_values[axis];
            let before = &runtime_pairs[axis].0;
            let coord_before_runtime_end = b.ir_less_than_i(&coord_v, runtime_dim);
            in_runtime_extent = b.ir_logical_and(&in_runtime_extent, &coord_before_runtime_end);

            let coord_ge_before = b.ir_greater_than_or_equal_i(&coord_v, before);
            let src_coord_v = b.ir_sub_i(&coord_v, before);
            let dim_v = b.ir_constant_int(shape[axis] as i64);
            let src_coord_lt_dim = b.ir_less_than_i(&src_coord_v, &dim_v);
            inside_src = b.ir_logical_and(&inside_src, &coord_ge_before);
            inside_src = b.ir_logical_and(&inside_src, &src_coord_lt_dim);

            if shape[axis] > 0 {
                let max_src_coord = b.ir_constant_int(shape[axis] as i64 - 1);
                let below_zero = b.ir_less_than_i(&src_coord_v, &zero);
                let above_hi = b.ir_greater_than_i(&src_coord_v, &max_src_coord);
                let clamped_hi = b.ir_select_i(&above_hi, &max_src_coord, &src_coord_v);
                let clamped = b.ir_select_i(&below_zero, &zero, &clamped_hi);
                let stride = b.ir_constant_int(src_strides[axis] as i64);
                let term = b.ir_mul_i(&clamped, &stride);
                src_idx = b.ir_add_i(&src_idx, &term);
            }
        }

        let src_cell =
            crate::helpers::static_array::read::static_array_subscript_with_preproved_index_bounds(
                b,
                arr,
                &[SliceIndex::Single(src_idx)],
            );
        let active_src = b.ir_logical_and(&in_runtime_extent, &inside_src);
        let not_inside_src = b.ir_logical_not(&inside_src);
        let active_pad = b.ir_logical_and(&in_runtime_extent, &not_inside_src);
        let value = if dtype == NumberType::Float {
            let with_pad = b.ir_select_f(&active_pad, constant_value, &constant_value);
            b.ir_select_f(&active_src, &src_cell, &with_pad)
        } else {
            let with_pad = b.ir_select_i(&active_pad, constant_value, &constant_value);
            b.ir_select_i(&active_src, &src_cell, &with_pad)
        };
        let final_value = if dtype == NumberType::Float {
            let zero_f = b.ir_constant_float(0.0);
            b.ir_select_f(&in_runtime_extent, &value, &zero_f)
        } else {
            b.ir_select_i(&in_runtime_extent, &value, &zero)
        };
        out.push(final_value);
    }

    let segment_id = crate::helpers::segment::alloc_and_write_values(b, &out, dtype);
    let dims: Vec<Dim> = max_shape
        .iter()
        .map(|&n| Dim::new_dynamic(&mut b.dim_table, 0, n))
        .collect();
    let envelope = crate::types::Envelope::new_with_bound(dims, max_total);
    Value::DynamicNDArray(DynamicNDArrayData {
        envelope,
        dtype,
        segment_id,
        meta: DynArrayMeta {
            logical_shape: max_shape.clone(),
            logical_offset: 0,
            logical_strides: max_strides.clone(),
            runtime_length: value_to_scalar_i64(&runtime_length_value),
            runtime_rank: ScalarValue::new(Some(max_shape.len() as i64), None),
            runtime_shape,
            runtime_strides,
            runtime_offset: scalar_from_usize(b, 0),
        },
        value_id: ValueId::next(),
    })
}

pub fn np_pad(
    b: &mut IRBuilder,
    args: &[Value],
    kwargs: &std::collections::HashMap<String, Value>,
) -> Value {
    assert!(args.len() >= 2, "np.pad: requires array and pad_width");
    let arr = &args[0];
    let (dtype, shape) = static_array_shape_dtype(arr, "np.pad");
    if dtype == NumberType::Complex {
        panic!("np.pad: Complex StaticArray pad is not yet implemented");
    }
    let mode = kwargs
        .get("mode")
        .or_else(|| args.get(2))
        .and_then(|v| v.string_val())
        .unwrap_or("constant");
    if mode != "constant" {
        panic!("np.pad: only mode='constant' is supported");
    }
    let constant_value = kwargs
        .get("constant_values")
        .or_else(|| args.get(3))
        .map(|v| cast_cell_to_dtype(b, v, dtype))
        .unwrap_or_else(|| zero_for_dtype(b, dtype));

    if let Some(runtime_pairs) = checked_runtime_pad_width_pairs(b, &args[1], shape.len()) {
        if runtime_pairs
            .iter()
            .all(|(before, after)| before.int_val().is_some() && after.int_val().is_some())
        {
            let pad_width = runtime_pairs
                .iter()
                .map(|(before, after)| {
                    (
                        before.int_val().unwrap() as usize,
                        after.int_val().unwrap() as usize,
                    )
                })
                .collect::<Vec<_>>();
            let out_shape: Vec<usize> = shape
                .iter()
                .zip(pad_width.iter())
                .map(|(dim, (before, after))| dim + before + after)
                .collect();
            let out_total: usize = out_shape.iter().product();
            let out_strides = crate::helpers::shape_arith::row_major_strides(&out_shape);
            let src_strides = crate::helpers::shape_arith::row_major_strides(&shape);
            let arr_cells = payload_cells(b, arr);
            let mut out = Vec::with_capacity(out_total);
            for out_idx in 0..out_total {
                let out_coords =
                    crate::helpers::shape_arith::decode_coords(out_idx, &out_shape, &out_strides);
                let mut src_coords = Vec::with_capacity(shape.len());
                let mut inside = true;
                for (axis, coord) in out_coords.iter().enumerate() {
                    let before = pad_width[axis].0;
                    if *coord < before || *coord >= before + shape[axis] {
                        inside = false;
                        break;
                    }
                    src_coords.push(*coord - before);
                }
                if inside {
                    let src_idx =
                        crate::helpers::shape_arith::encode_coords(&src_coords, &src_strides);
                    out.push(arr_cells[src_idx].clone());
                } else {
                    out.push(constant_value.clone());
                }
            }
            return crate::helpers::static_array::base::build_static_array_from_flat(
                b, out, out_shape, dtype,
            );
        } else {
            return np_pad_runtime_static_input(
                b,
                arr,
                dtype,
                &shape,
                runtime_pairs,
                &constant_value,
            );
        }
    }

    let pad_width = static_pad_width_from_value(b, &args[1], shape.len());
    let out_shape: Vec<usize> = shape
        .iter()
        .zip(pad_width.iter())
        .map(|(dim, (before, after))| dim + before + after)
        .collect();
    let out_total: usize = out_shape.iter().product();
    let out_strides = crate::helpers::shape_arith::row_major_strides(&out_shape);
    let src_strides = crate::helpers::shape_arith::row_major_strides(&shape);
    let arr_cells = payload_cells(b, arr);
    let mut out = Vec::with_capacity(out_total);
    for out_idx in 0..out_total {
        let out_coords =
            crate::helpers::shape_arith::decode_coords(out_idx, &out_shape, &out_strides);
        let mut src_coords = Vec::with_capacity(shape.len());
        let mut inside = true;
        for (axis, coord) in out_coords.iter().enumerate() {
            let before = pad_width[axis].0;
            if *coord < before || *coord >= before + shape[axis] {
                inside = false;
                break;
            }
            src_coords.push(*coord - before);
        }
        if inside {
            let src_idx = crate::helpers::shape_arith::encode_coords(&src_coords, &src_strides);
            out.push(arr_cells[src_idx].clone());
        } else {
            out.push(constant_value.clone());
        }
    }
    crate::helpers::static_array::base::build_static_array_from_flat(b, out, out_shape, dtype)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn list_of(values: Vec<Value>) -> Value {
        let types = values.iter().map(|v| v.zinnia_type()).collect();
        Value::List(CompositeData::new(types, values))
    }

    fn make_static_array(b: &mut IRBuilder, values: &[i64], shape: Vec<usize>) -> Value {
        let leaves: Vec<Value> = values.iter().map(|n| b.ir_constant_int(*n)).collect();
        crate::helpers::static_array::base::build_static_array_from_flat(
            b,
            leaves,
            shape,
            NumberType::Integer,
        )
    }

    #[test]
    fn np_argwhere_static_1d_has_coordinate_cell_runtime_length() {
        let mut b = IRBuilder::new();
        let arr = make_static_array(&mut b, &[0, 5, 0, 7], vec![4]);
        let out = np_argwhere(&mut b, &[arr]);
        let Value::DynamicNDArray(data) = out else {
            panic!("expected DynamicNDArray");
        };
        assert_eq!(data.max_length(), 4);
        assert_eq!(data.meta.logical_shape, vec![4, 1]);
        assert_eq!(data.meta.runtime_shape[0].static_val, Some(2));
        assert_eq!(data.meta.runtime_shape[1].static_val, Some(1));
        assert_eq!(data.meta.runtime_length.static_val, Some(2));
    }

    #[test]
    fn np_argwhere_static_2d_runtime_length_is_true_count_times_rank() {
        let mut b = IRBuilder::new();
        let arr = make_static_array(&mut b, &[0, 2, 3, 0, 0, 4], vec![2, 3]);
        let out = np_argwhere(&mut b, &[arr]);
        let Value::DynamicNDArray(data) = out else {
            panic!("expected DynamicNDArray");
        };
        assert_eq!(data.max_length(), 12);
        assert_eq!(data.meta.logical_shape, vec![6, 2]);
        assert_eq!(data.meta.runtime_shape[0].static_val, Some(3));
        assert_eq!(data.meta.runtime_shape[1].static_val, Some(2));
        assert_eq!(data.meta.runtime_length.static_val, Some(6));
    }

    #[test]
    fn np_nonzero_static_2d_returns_per_axis_dynamic_arrays() {
        let mut b = IRBuilder::new();
        let arr = make_static_array(&mut b, &[0, 2, 3, 0, 0, 4], vec![2, 3]);
        let out = np_nonzero(&mut b, &[arr]);
        let Value::Tuple(tuple) = out else {
            panic!("expected tuple");
        };
        assert_eq!(tuple.values.len(), 2);
        for axis_out in tuple.values {
            let Value::DynamicNDArray(data) = axis_out else {
                panic!("expected DynamicNDArray axis output");
            };
            assert_eq!(data.max_length(), 6);
            assert_eq!(data.meta.logical_shape, vec![6]);
            assert_eq!(data.meta.runtime_length.static_val, Some(3));
        }
    }

    #[test]
    fn np_argwhere_accepts_nested_list_numeric_input() {
        let mut b = IRBuilder::new();
        let row0 = list_of(vec![b.ir_constant_int(1), b.ir_constant_int(0)]);
        let row1 = list_of(vec![b.ir_constant_int(0), b.ir_constant_int(4)]);
        let arr = list_of(vec![row0, row1]);
        let out = np_argwhere(&mut b, &[arr]);
        let Value::DynamicNDArray(data) = out else {
            panic!("expected DynamicNDArray");
        };
        assert_eq!(data.meta.logical_shape, vec![4, 2]);
        assert_eq!(data.meta.runtime_length.static_val, Some(4));
    }
}
