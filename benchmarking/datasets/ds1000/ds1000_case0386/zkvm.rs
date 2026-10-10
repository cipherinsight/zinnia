//! DS-1000 case 0386: overlapping 2x2 windows, reshaped to (-1, 2, 2).
use crate::prelude::*;

const ROWS: usize = 100;
const COLS: usize = 200;

pub type Output = Vec<Q32>;

pub fn main(input: &mut impl Input) -> Output {
    let a: Vec<Q32> = input.read_vec(ROWS * COLS);
    let mut result = Vec::with_capacity((ROWS - 1) * (COLS - 1) * 4);
    // sliding_window_view has axes [window row, window column, row, column].
    // The reshape preserves that order, including overlapping entries.
    for i in 0..ROWS - 1 {
        for j in 0..COLS - 1 {
            let top = i * COLS + j;
            result.extend_from_slice(&a[top..top + 2]);
            result.extend_from_slice(&a[top + COLS..top + COLS + 2]);
        }
    }
    result
}
