//! Calibration program (not a DS-1000 case): the private input becomes the
//! public output, so its proof measures the zkVM's fixed costs.
//!
//! ```python
//! def kernel(x: Integer):
//!     return x
//! ```
use crate::prelude::*;

pub type Output = i64;

pub fn main(input: &mut impl Input) -> Output {
    input.read()
}
