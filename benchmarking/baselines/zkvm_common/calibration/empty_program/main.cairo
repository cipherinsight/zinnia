//! Calibration program (not a DS-1000 case): the private input becomes the
//! public output, so its proof measures the fixed costs of Scarb and Stwo.
//!
//!     def kernel(x: Integer):
//!         return x
// output: i64

#[executable]
fn main(x: i64) -> i64 {
    x
}
