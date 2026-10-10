//! DS-1000 case 0373: sort the grades, then return ECDF probabilities.
use crate::prelude::*;

const N: usize = 50;

pub type Output = Vec<Q32>;

pub fn main(input: &mut impl Input) -> Output {
    let mut xs: [Q32; N] = input.read_array();
    xs.sort_unstable();
    // Keep the kernel's sort even though only the sorted array's length is used.
    let xs = core::hint::black_box(&xs);
    let n = xs.len() as i64;
    (1..=n)
        .map(|rank| Q32::from_int(rank) / Q32::from_int(n))
        .collect()
}
