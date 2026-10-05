//! Statically shaped einsum: contractions, diagonals, and ellipsis broadcasting.
//! Output and contraction loops depend only on operand shapes and subscripts.
use std::collections::{BTreeMap, BTreeSet};

use crate::builder::IRBuilder;
use crate::types::{NumberType, Value};

fn letters(text: &str) -> Vec<usize> {
    text.chars()
        .map(|c| {
            assert!(
                c.is_ascii_alphabetic(),
                "np.einsum: invalid subscript `{c}`"
            );
            c as usize
        })
        .collect()
}

fn expand(text: &str, rank: usize, ellipsis_rank: usize) -> Vec<usize> {
    if let Some((before, after)) = text.split_once("...") {
        let mut left = letters(before);
        let right = letters(after);
        assert!(
            rank >= left.len() + right.len(),
            "np.einsum: subscript has too many axes"
        );
        let count = rank - left.len() - right.len();
        left.extend((ellipsis_rank - count..ellipsis_rank).map(|i| 128 + i));
        left.extend(right);
        left
    } else {
        let labels = letters(text);
        assert_eq!(
            labels.len(),
            rank,
            "np.einsum: operand rank does not match subscript"
        );
        labels
    }
}

pub fn einsum(b: &mut IRBuilder, args: &[Value]) -> Value {
    assert!(
        args.len() >= 2,
        "np.einsum: requires subscripts and operands"
    );
    let subscript = args[0]
        .string_val()
        .expect("np.einsum: subscripts must be a static string");
    let compact: String = subscript.chars().filter(|c| !c.is_whitespace()).collect();
    let (inputs, explicit_output) = match compact.split_once("->") {
        Some((a, z)) => (a, Some(z)),
        None => (compact.as_str(), None),
    };
    let specs: Vec<_> = inputs.split(',').collect();
    let operands = &args[1..];
    assert_eq!(
        specs.len(),
        operands.len(),
        "np.einsum: operand count mismatch"
    );
    let mut dtype = NumberType::Integer;
    let mut shapes = Vec::new();
    let mut payloads = Vec::new();
    for operand in operands {
        let owned = match operand {
            Value::StaticArray { .. } => operand.clone(),
            _ => crate::helpers::static_array::base::to_static_array(b, operand)
                .expect("np.einsum: requires statically shaped numeric arrays"),
        };
        let Value::StaticArray {
            dtype: dt,
            ref shape,
            ..
        } = owned
        else {
            unreachable!()
        };
        assert!(
            matches!(dt, NumberType::Integer | NumberType::Float),
            "np.einsum: only integer and real operands are supported"
        );
        if dt == NumberType::Float {
            dtype = dt;
        }
        shapes.push(shape.clone());
        payloads.push(crate::helpers::static_array::elementwise::payload_cells(
            b, &owned,
        ));
    }
    let ellipsis_rank = specs
        .iter()
        .zip(&shapes)
        .map(|(s, shape)| {
            if s.contains("...") {
                let named = s.chars().filter(|c| c.is_ascii_alphabetic()).count();
                shape
                    .len()
                    .checked_sub(named)
                    .expect("np.einsum: too many named axes")
            } else {
                0
            }
        })
        .max()
        .unwrap_or(0);
    let labels: Vec<Vec<usize>> = specs
        .iter()
        .zip(&shapes)
        .map(|(s, shape)| expand(s, shape.len(), ellipsis_rank))
        .collect();
    let mut dims = BTreeMap::new();
    let mut counts = BTreeMap::new();
    for (axes, shape) in labels.iter().zip(&shapes) {
        let mut within = BTreeMap::new();
        for (&label, &dim) in axes.iter().zip(shape) {
            if let Some(previous) = within.insert(label, dim) {
                assert_eq!(
                    previous, dim,
                    "np.einsum: repeated axes must have equal dimensions"
                );
            }
            *counts.entry(label).or_insert(0usize) += 1;
            let old = *dims.get(&label).unwrap_or(&1);
            assert!(
                old == dim || old == 1 || dim == 1,
                "np.einsum: dimensions cannot broadcast"
            );
            dims.insert(label, if old == 1 { dim } else { old });
        }
    }
    let output_labels = if let Some(text) = explicit_output {
        let named = text.chars().filter(|c| c.is_ascii_alphabetic()).count();
        expand(
            text,
            named
                + if text.contains("...") {
                    ellipsis_rank
                } else {
                    0
                },
            ellipsis_rank,
        )
    } else {
        (0..ellipsis_rank)
            .map(|i| 128 + i)
            .chain(
                counts
                    .iter()
                    .filter(|(label, count)| **label < 128 && **count == 1)
                    .map(|(label, _)| *label),
            )
            .collect()
    };
    let output_set: BTreeSet<_> = output_labels.iter().copied().collect();
    assert_eq!(
        output_set.len(),
        output_labels.len(),
        "np.einsum: repeated output axis"
    );
    assert!(
        output_labels.iter().all(|axis| dims.contains_key(axis)),
        "np.einsum: unknown output axis"
    );
    let summed: Vec<_> = dims
        .keys()
        .filter(|axis| !output_set.contains(axis))
        .copied()
        .collect();
    let output_shape: Vec<_> = output_labels.iter().map(|axis| dims[axis]).collect();
    if let [operand] = operands {
        if summed.is_empty() {
            if let Some(view) = relabel_view(operand, &labels[0], &output_labels, &output_shape) {
                return view;
            }
        }
    }
    let summed_shape: Vec<_> = summed.iter().map(|axis| dims[axis]).collect();
    let strides: Vec<_> = shapes
        .iter()
        .map(|shape| crate::helpers::shape_arith::row_major_strides(shape))
        .collect();
    let out_strides = crate::helpers::shape_arith::row_major_strides(&output_shape);
    let sum_strides = crate::helpers::shape_arith::row_major_strides(&summed_shape);
    let mut out = Vec::new();
    for out_index in 0..output_shape.iter().product::<usize>() {
        let coords =
            crate::helpers::shape_arith::decode_coords(out_index, &output_shape, &out_strides);
        let mut index: BTreeMap<usize, usize> = output_labels.iter().copied().zip(coords).collect();
        let mut total = if dtype == NumberType::Float {
            b.ir_constant_float(0.0)
        } else {
            b.ir_constant_int(0)
        };
        for sum_index in 0..summed_shape.iter().product::<usize>() {
            let coords =
                crate::helpers::shape_arith::decode_coords(sum_index, &summed_shape, &sum_strides);
            index.extend(summed.iter().copied().zip(coords));
            let mut product = if dtype == NumberType::Float {
                b.ir_constant_float(1.0)
            } else {
                b.ir_constant_int(1)
            };
            for i in 0..operands.len() {
                let offset: usize = labels[i]
                    .iter()
                    .enumerate()
                    .map(|(axis, label)| {
                        if shapes[i][axis] == 1 {
                            0
                        } else {
                            index[label] * strides[i][axis]
                        }
                    })
                    .sum();
                product = crate::helpers::value_ops::apply_binary_op(
                    b,
                    "mul",
                    &product,
                    &payloads[i][offset],
                );
            }
            total = crate::helpers::value_ops::apply_binary_op(b, "add", &total, &product);
        }
        out.push(total);
    }
    if output_shape.is_empty() {
        out.remove(0)
    } else {
        crate::helpers::static_array::base::build_static_array_from_flat(
            b,
            out,
            output_shape,
            dtype,
        )
    }
}

/// A single-operand einsum that sums nothing (a diagonal and/or an axis
/// permutation, e.g. `'ii->i'`, `'ij->ji'`) is, in NumPy, a writeable VIEW
/// of its operand: writing through the result writes the operand (case
/// 0478 writes a saved diagonal back through `np.einsum('ii->i', a)`).
/// The view has the operand's segment and offset; along each output label
/// it steps by the sum of the operand strides of the axes with that label.
/// Returns `None` when the operand is not a segment-backed array (a list
/// operand is converted to a fresh array, as NumPy does).
fn relabel_view(
    operand: &Value,
    input_labels: &[usize],
    output_labels: &[usize],
    output_shape: &[usize],
) -> Option<Value> {
    let Value::StaticArray {
        dtype,
        segment_id,
        strides,
        offset,
        imag_segment_id,
        ..
    } = operand
    else {
        return None;
    };
    let view_strides = output_labels
        .iter()
        .map(|label| {
            input_labels
                .iter()
                .zip(strides)
                .filter(|(l, _)| *l == label)
                .map(|(_, stride)| stride)
                .sum()
        })
        .collect();
    Some(Value::StaticArray {
        dtype: *dtype,
        shape: output_shape.to_vec(),
        segment_id: *segment_id,
        strides: view_strides,
        offset: *offset,
        imag_segment_id: *imag_segment_id,
        value_id: crate::types::ValueId::next(),
    })
}
