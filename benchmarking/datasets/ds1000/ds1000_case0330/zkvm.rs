//! DS-1000 case 0330: elementwise sixth power.
use crate::prelude::*;

pub type Output = Vec<Q32>;

pub fn main(input: &mut impl Input) -> Output {
    let a: Vec<Q32> = input.read_vec(8 * 6);
    a.into_iter().map(|v| v.powi(6)).collect()
}
