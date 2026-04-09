//! Pure shape arithmetic shared by static and dynamic ndarray surfaces.
//!
//! Everything in this module operates on `&[usize]` shape vectors and returns
//! plain Rust values — no `Value`, no `IRBuilder`. The point is to keep the
//! shape-level reasoning in one place so the static and dynamic ndarray
//! operator implementations can both build on top of it.

// ────────────────────────────────────────────────────────────────────────
// Strides and linearization
// ────────────────────────────────────────────────────────────────────────

/// Row-major (C-order) strides for a shape. For shape `[a, b, c]` returns
/// `[b*c, c, 1]`. The product of dim and stride at each axis gives the
/// total element count, and `linear_idx = sum(coord[i] * strides[i])`.
pub fn row_major_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

/// Decode a row-major linear index into multi-dimensional coordinates.
/// `shape` and `strides` must have the same length.
pub fn decode_coords(linear: usize, shape: &[usize], strides: &[usize]) -> Vec<usize> {
    shape
        .iter()
        .zip(strides.iter())
        .map(|(&dim, &stride)| (linear / stride) % dim)
        .collect()
}

/// Encode multi-dimensional coordinates back to a linear index.
pub fn encode_coords(coords: &[usize], strides: &[usize]) -> usize {
    coords.iter().zip(strides.iter()).map(|(&c, &s)| c * s).sum()
}

/// Total element count for a shape (product of dims). Empty shape (a 0-D
/// scalar) returns 1.
pub fn num_elements(shape: &[usize]) -> usize {
    shape.iter().product()
}

// ────────────────────────────────────────────────────────────────────────
// Broadcasting (value-bound only — no symbolic relations here)
// ────────────────────────────────────────────────────────────────────────

/// Compute the NumPy broadcast of two shapes. Returns the broadcast shape
/// on success, or `None` if the shapes are not broadcast-compatible.
/// Scalars are represented as the empty shape `[]`.
///
/// Rules (NumPy):
/// - Right-align the two shape vectors (pad the shorter on the left with 1s).
/// - For each axis, the dims must be equal, or one of them must be 1.
/// - The broadcast dim is the max of the two.
pub fn broadcast_shapes(a: &[usize], b: &[usize]) -> Option<Vec<usize>> {
    let rank = a.len().max(b.len());
    let mut out = Vec::with_capacity(rank);
    for i in 0..rank {
        // Right-aligned: index from the tail.
        let da = if i < a.len() { a[a.len() - 1 - i] } else { 1 };
        let db = if i < b.len() { b[b.len() - 1 - i] } else { 1 };
        let dim = if da == db {
            da
        } else if da == 1 {
            db
        } else if db == 1 {
            da
        } else {
            return None;
        };
        out.push(dim);
    }
    out.reverse();
    Some(out)
}

// ────────────────────────────────────────────────────────────────────────
// Axis resolution
// ────────────────────────────────────────────────────────────────────────

/// Resolve a possibly-negative axis against a given rank. Negative axes are
/// counted from the back (`-1` means last axis). Panics with a descriptive
/// message if the axis is out of bounds. The `op` argument is the operator
/// name used in the error.
pub fn resolve_axis(axis: i64, ndim: usize, op: &str) -> usize {
    let resolved = if axis < 0 { ndim as i64 + axis } else { axis };
    if resolved < 0 || resolved >= ndim as i64 {
        panic!(
            "{}: axis {} is out of bounds for array of rank {}",
            op, axis, ndim
        );
    }
    resolved as usize
}

// ────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strides_row_major() {
        assert_eq!(row_major_strides(&[2, 3, 4]), vec![12, 4, 1]);
        assert_eq!(row_major_strides(&[5]), vec![1]);
        assert_eq!(row_major_strides(&[]), Vec::<usize>::new());
    }

    #[test]
    fn coord_round_trip() {
        let shape = [2, 3, 4];
        let strides = row_major_strides(&shape);
        for linear in 0..num_elements(&shape) {
            let coords = decode_coords(linear, &shape, &strides);
            assert_eq!(encode_coords(&coords, &strides), linear);
        }
    }

    #[test]
    fn num_elements_basic() {
        assert_eq!(num_elements(&[2, 3, 4]), 24);
        assert_eq!(num_elements(&[]), 1);
        assert_eq!(num_elements(&[0, 5]), 0);
    }

    #[test]
    fn broadcast_same_shape() {
        assert_eq!(broadcast_shapes(&[3, 4], &[3, 4]), Some(vec![3, 4]));
    }

    #[test]
    fn broadcast_outer_product_classic() {
        // (3, 1) and (1, 4) -> (3, 4)
        assert_eq!(broadcast_shapes(&[3, 1], &[1, 4]), Some(vec![3, 4]));
    }

    #[test]
    fn broadcast_lower_rank_left_pads() {
        // (3,) broadcasts against (2, 3) -> (2, 3)
        assert_eq!(broadcast_shapes(&[3], &[2, 3]), Some(vec![2, 3]));
    }

    #[test]
    fn broadcast_scalar_against_anything() {
        assert_eq!(broadcast_shapes(&[], &[2, 3]), Some(vec![2, 3]));
        assert_eq!(broadcast_shapes(&[2, 3], &[]), Some(vec![2, 3]));
        assert_eq!(broadcast_shapes(&[], &[]), Some(vec![]));
    }

    #[test]
    fn broadcast_three_dim() {
        // (2, 1, 3) and (1, 4, 1) -> (2, 4, 3)
        assert_eq!(broadcast_shapes(&[2, 1, 3], &[1, 4, 1]), Some(vec![2, 4, 3]));
    }

    #[test]
    fn broadcast_incompatible_returns_none() {
        assert_eq!(broadcast_shapes(&[2, 3], &[2, 4]), None);
        assert_eq!(broadcast_shapes(&[3], &[4]), None);
    }

    #[test]
    fn resolve_axis_positive() {
        assert_eq!(resolve_axis(0, 3, "test"), 0);
        assert_eq!(resolve_axis(2, 3, "test"), 2);
    }

    #[test]
    fn resolve_axis_negative() {
        assert_eq!(resolve_axis(-1, 3, "test"), 2);
        assert_eq!(resolve_axis(-3, 3, "test"), 0);
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn resolve_axis_too_large() {
        resolve_axis(5, 3, "test");
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn resolve_axis_too_negative() {
        resolve_axis(-5, 3, "test");
    }
}
