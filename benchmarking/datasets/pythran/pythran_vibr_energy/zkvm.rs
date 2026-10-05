//! Pythran vibr_energy: `exp(-harmonic * i - anharmonic * i^2)`,
//! element-wise.
//!
//! ```python
//! def calculate_vibr_energy(harmonic: Float[Array, '1000000'], anharmonic: Float[Array, '1000000'], i: Float[Array, '1000000']):
//!     return numpy.exp(-harmonic * i - anharmonic * i ** 2)
//! ```
use crate::prelude::*;

pub mod calculate_vibr_energy {
    use super::*;

    pub type Output = Array1<Q32>;

    pub fn main(args: &mut Args<impl Input>) -> Output {
        let harmonic: Array1<Q32> = args.array1();
        let anharmonic: Array1<Q32> = args.array1();
        let i: Array1<Q32> = args.array1();
        assert!(
            harmonic.len() == i.len() && anharmonic.len() == i.len(),
            "operands could not be broadcast together"
        );
        Zip::from(&harmonic)
            .and(&anharmonic)
            .and(&i)
            .map_collect(|&h, &a, &i| (-h * i - a * (i * i)).exp())
    }
}
