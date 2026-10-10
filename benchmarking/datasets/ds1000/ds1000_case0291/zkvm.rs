//! DS-1000 case 0291: return the fixed shape of `a`.
use crate::prelude::*;

pub type Output = (i64, i64, i64, i64);

pub fn main(input: &mut impl Input) -> Output {
    let _a: [Q32; 2 * 5 * 5 * 2] = input.read_array();
    (2, 5, 5, 2)
}
