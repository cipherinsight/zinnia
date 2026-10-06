//! Pythran ln: the degree-9 Taylor polynomial of `log(X)` around 1, written
//! into `Y`, which is returned.
//!
//! ```python
//! def ln(X: Float[Array, '100'], Y: Float[Array, '100']):
//!     Y[:] = X - 1 - (X - 1) ** 2 / 2 + (X - 1) ** 3 / 3 - (X - 1) ** 4 / 4 + (X - 1) ** 5 / 5 - (X - 1) ** 6 / 6 + (X - 1) ** 7 / 7 - (X - 1) ** 8 / 8 + (X - 1) ** 9 / 9
//!     return Y
//! ```
use crate::prelude::*;

pub mod ln {
    use super::*;

    pub type Output = Array1<Q32>;

    #[allow(non_snake_case)]
    pub fn main(args: &mut Args<impl Input>) -> Output {
        let X: Array1<Q32> = args.array1();
        let mut Y: Array1<Q32> = args.array1();
        let series = X.mapv(|x| {
            // `(X - 1) ** n / n` with alternating signs; `** n` multiplies
            // by `X - 1` repeatedly, so each power extends the previous one.
            let d = x - Q32::ONE;
            let (mut power, mut sum) = (d, d);
            for n in 2..=9i64 {
                power = power * d;
                let term = power / n;
                sum = if n % 2 == 0 { sum - term } else { sum + term };
            }
            sum
        });
        // `Y[:] = ...` broadcasts like NumPy.
        Y.assign(&series);
        Y
    }
}
