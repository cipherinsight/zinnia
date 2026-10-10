//! DS-1000 case 0451: np.zeros((20, 10, 10, 2)).
use crate::prelude::*;

pub type Output = Vec<Q32>;

pub fn main(_input: &mut impl Input) -> Output {
    vec![Q32::ZERO; 20 * 10 * 10 * 2]
}
