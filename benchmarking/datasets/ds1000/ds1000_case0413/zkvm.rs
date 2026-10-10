//! DS-1000 case 0413: keep complex entries with nonzero imaginary part.
use crate::prelude::*;

pub type Output = Vec<C32>;

pub fn main(input: &mut impl Input) -> Output {
    let x: Vec<C32> = input.read_vec(10);
    x.into_iter().filter(|v| v.im != Q32::ZERO).collect()
}
