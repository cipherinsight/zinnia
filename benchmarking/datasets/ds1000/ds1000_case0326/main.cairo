//! DS-1000 case 0326: arcsine in degrees.
//!
//!     def kernel(value: Float):
//!         result = np.degrees(np.arcsin(value))
//!         return result
// output: q32
use crate::q32::{Q32, Q32Trait, RAD_TO_DEG};

#[executable]
fn main(value: Q32) -> Q32 {
    value.asin() * RAD_TO_DEG
}
