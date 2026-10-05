//! DS-1000 case 0325: 1 if `number` read as radians has a larger sine than
//! `number` read as degrees, else 0.
//!
//!     def kernel(number: Integer):
//!         deg = np.sin(np.deg2rad(number))
//!         rad = np.sin(number)
//!         result = int(rad > deg)
//!         return result
// output: i64
use crate::q32::{DEG_TO_RAD, Q32Trait};

#[executable]
fn main(number: i64) -> i64 {
    let number = Q32Trait::from_int(number);
    let deg = (number * DEG_TO_RAD).sin();
    let rad = number.sin();
    if rad > deg {
        1
    } else {
        0
    }
}
