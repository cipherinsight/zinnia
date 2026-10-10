//! DS-1000 case 0349: return the fixed column count of `a`.
use crate::prelude::*;

pub type Output = i64;

pub fn main(input: &mut impl Input) -> Output {
    let _a: [[Q32; 6]; 8] = input.read_matrix();
    6
}
