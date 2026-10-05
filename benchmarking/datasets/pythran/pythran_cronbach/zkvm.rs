//! Pythran cronbach: Cronbach's alpha of an items x subjects score matrix.
//!
//! ```python
//! def cronbach(itemscores: Float[Array, '800 800']):
//!     itemvars = itemscores.var(1, None, None, 1)
//!     tscores = itemscores.sum(0)
//!     nitems = len(itemscores)
//!     return nitems / (nitems - 1) * (1 - itemvars.sum() / tscores.var(None, None, None, 1))
//! ```
use crate::prelude::*;

/// `np.var(values, ddof=1)`: `sum((v - mean) ** 2) / (n - 1)`, each square
/// rescaled.
fn sample_variance(values: ArrayView1<Q32>) -> Q32 {
    let n = values.len() as i64;
    let mean = values.sum() / n;
    values.iter().map(|&v| (v - mean) * (v - mean)).sum::<Q32>() / (n - 1)
}

pub mod cronbach {
    use super::*;

    pub type Output = Q32;

    pub fn main(args: &mut Args<impl Input>) -> Output {
        let itemscores: Array2<Q32> = args.array2();
        let itemvars_sum: Q32 = itemscores.rows().into_iter().map(sample_variance).sum();
        let tscores = itemscores.sum_axis(Axis(0));
        let nitems = itemscores.nrows() as i64;
        // `nitems / (nitems - 1)` divides two Python ints (ZeroDivisionError
        // for one item).
        assert!(nitems != 1, "division by zero");
        Q32::from_int(nitems) / (nitems - 1)
            * (Q32::ONE - itemvars_sum / sample_variance(tscores.view()))
    }
}
