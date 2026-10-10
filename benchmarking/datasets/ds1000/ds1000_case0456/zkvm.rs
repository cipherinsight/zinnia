//! DS-1000 case 0456: pairwise Euclidean distances.
use crate::prelude::*;

pub type Output = Vec<Q32>;

pub fn main(input: &mut impl Input) -> Output {
    let a: [[Q32; 3]; 8] = input.read_matrix();
    let mut out = Vec::with_capacity(64);
    for i in 0..8 {
        for j in 0..8 {
            let mut squared = Q32::ZERO;
            for k in 0..3 {
                let delta = a[j][k] - a[i][k];
                squared += delta * delta;
            }
            out.push(squared.sqrt());
        }
    }
    out
}
