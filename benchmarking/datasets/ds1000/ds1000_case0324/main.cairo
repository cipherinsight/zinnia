//! DS-1000 case 0324: cosine of an angle given in degrees.
//!
//!     def kernel(degree: Integer):
//!         result = np.cos(np.deg2rad(degree))
//!         return result
// output: q32
use crate::q32::{DEG_TO_RAD, Q32, Q32Trait};

#[executable]
fn main(degree: i64) -> Q32 {
    (Q32Trait::from_int(degree) * DEG_TO_RAD).cos()
}
