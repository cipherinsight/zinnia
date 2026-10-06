//! Pythran monte_carlo: Monte Carlo integration of `sin` over
//! `[xmin, xmax]`, with the samples taken from `rand`.
//!
//! ```python
//! @requires(lambda xmin, xmax, numSteps, rand, randsize: numSteps != 0)
//! @requires(lambda xmin, xmax, numSteps, rand, randsize: randsize != 0)
//! def montecarlo_integration(xmin: Float, xmax: Float, numSteps: Integer, rand: Float[Array, '100'], randsize: Integer):
//!     def f(x):
//!         return math.sin(x)
//!     ymin = f(xmin)
//!     ymax = ymin
//!     for i in range(numSteps):
//!         x = xmin + (xmax - xmin) * float(i) / numSteps
//!         y = f(x)
//!         if y < ymin:
//!             ymin = y
//!         if y > ymax:
//!             ymax = y
//!     rectArea = (xmax - xmin) * (ymax - ymin)
//!     numPoints = numSteps
//!     ctr = 0
//!     for j in range(numPoints):
//!         x = xmin + (xmax - xmin) * rand[j % randsize]
//!         y = ymin + (ymax - ymin) * rand[j % randsize]
//!         if math.fabs(y) <= math.fabs(f(x)):
//!             if f(x) > 0 and y > 0 and (y <= f(x)):
//!                 ctr += 1
//!             if f(x) < 0 and y < 0 and (y >= f(x)):
//!                 ctr -= 1
//!     fnArea = rectArea * float(ctr) / numPoints
//!     return fnArea
//! ```
use crate::prelude::*;

pub mod montecarlo_integration {
    use super::*;

    pub type Output = Q32;

    pub fn main(args: &mut Args<impl Input>) -> Output {
        let xmin: Q32 = args.scalar();
        let xmax: Q32 = args.scalar();
        let num_steps: i64 = args.scalar();
        let rand: Array1<Q32> = args.array1();
        let randsize: i64 = args.scalar();
        assert!(num_steps != 0, "requires numSteps != 0");
        assert!(randsize != 0, "requires randsize != 0");

        // The range of sin over the grid: `(xmax - xmin) * float(i)` is exact,
        // then `/ numSteps`.
        let width = xmax - xmin;
        let mut ymin = xmin.sin();
        let mut ymax = ymin;
        for i in 0..num_steps {
            let y = (xmin + width * i / num_steps).sin();
            ymin = ymin.min(y);
            ymax = ymax.max(y);
        }
        let rect_area = width * (ymax - ymin);

        // Signed count of the samples between the axis and the curve.
        let mut ctr: i64 = 0;
        for j in 0..num_steps {
            let r = rand[np_index(floor_mod(j, randsize), rand.len())];
            let fx = (xmin + width * r).sin();
            let y = ymin + (ymax - ymin) * r;
            if y.abs() <= fx.abs() {
                if fx > Q32::ZERO && y > Q32::ZERO && y <= fx {
                    ctr += 1;
                }
                if fx < Q32::ZERO && y < Q32::ZERO && y >= fx {
                    ctr -= 1;
                }
            }
        }
        rect_area * ctr / num_steps
    }
}
