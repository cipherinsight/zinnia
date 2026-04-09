//! Static-ndarray materialization for broadcasting.
//!
//! Static ndarrays in Zinnia are represented as nested `Value::List`/`Value::Tuple`
//! whose shape is fully known at IR-generation time. Broadcasting therefore
//! reduces to a pure compile-time tile/repeat: we expand both operands to a
//! common target shape and then dispatch the existing element-wise op.
//!
//! Per project decision: static broadcasting only **materializes** — we do not
//! play stride tricks. The nested-list representation makes the materialized
//! form trivial to construct via `composite::build_nested_value`.
//!
//! The pure shape arithmetic (`broadcast_shapes`, etc.) lives in
//! [`super::shape_arith`]. This module re-exports `broadcast_shapes` for
//! backwards compatibility with existing call sites.

use super::composite;
use crate::types::Value;

pub use super::shape_arith::broadcast_shapes;

/// Materialize a static composite value into the given `target_shape` by
/// tiling axes of size 1 (and prepending unit axes when the source has lower
/// rank). Assumes `target_shape` is broadcast-compatible with the source's
/// shape (callers should validate via `broadcast_shapes`).
///
/// Scalars (non-composite leaves) are tiled to fill the entire target.
pub fn materialize_to_shape(val: &Value, target_shape: &[usize]) -> Value {
    let total: usize = target_shape.iter().product();

    // Scalar fast path: just repeat the value.
    let src_shape = composite::get_composite_shape(val);
    if src_shape.is_empty() {
        if target_shape.is_empty() {
            return val.clone();
        }
        let flat: Vec<Value> = (0..total).map(|_| val.clone()).collect();
        let types = flat.iter().map(|v| v.zinnia_type()).collect();
        return composite::build_nested_value(flat, types, target_shape);
    }

    if target_shape.is_empty() {
        // Target is a scalar; only valid if source is also effectively a scalar.
        return val.clone();
    }

    // Left-pad source shape with 1s to match target rank.
    let rank = target_shape.len();
    let mut padded_src = vec![1usize; rank.saturating_sub(src_shape.len())];
    padded_src.extend_from_slice(&src_shape);

    // Strides for the (left-padded) source. A padded leading "1" axis has
    // stride 0 since the source has no real elements there. Non-leading
    // axes-of-size-1 also map to stride 0 (broadcast tiling). All other
    // axes use the natural row-major stride over the *original* source.
    let mut src_strides = vec![0i64; rank];
    {
        let pad = rank - src_shape.len();
        // Build natural strides over the unpadded source first.
        let mut natural = vec![1i64; src_shape.len()];
        for i in (0..src_shape.len().saturating_sub(1)).rev() {
            natural[i] = natural[i + 1] * src_shape[i + 1] as i64;
        }
        for d in 0..rank {
            if d < pad {
                src_strides[d] = 0; // synthesised leading axis
            } else {
                let sd = d - pad;
                src_strides[d] = if src_shape[sd] == 1 { 0 } else { natural[sd] };
            }
        }
    }

    // Output strides (row-major).
    let mut out_strides = vec![1usize; rank];
    for i in (0..rank.saturating_sub(1)).rev() {
        out_strides[i] = out_strides[i + 1] * target_shape[i + 1];
    }

    let flat_src = composite::flatten_composite(val);
    let mut out_flat: Vec<Value> = Vec::with_capacity(total);
    for out_idx in 0..total {
        // Decompose out_idx into multi-index over target_shape.
        let mut remainder = out_idx;
        let mut src_flat_idx: i64 = 0;
        for d in 0..rank {
            let coord = remainder / out_strides[d];
            remainder %= out_strides[d];
            src_flat_idx += coord as i64 * src_strides[d];
        }
        out_flat.push(flat_src[src_flat_idx as usize].clone());
    }

    let types = out_flat.iter().map(|v| v.zinnia_type()).collect();
    composite::build_nested_value(out_flat, types, target_shape)
}
