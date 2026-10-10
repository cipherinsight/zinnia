//! DS-1000 case 0477: np.diag of a four-element float vector.
use crate::prelude::*;

pub type Output = Vec<Q32>;

pub fn main(input: &mut impl Input) -> Output {
    let i = input.read_array::<Q32, 4>();
    let mut result = vec![Q32::ZERO; 16];
    for k in 0..4 {
        result[k * 4 + k] = i[k];
    }
    result
}
