//! Pythran mandel: escape-time colouring of a `size x size` grid of the
//! Mandelbrot set centred at `(x_center, y_center)`.
//!
//! ```python
//! @requires(lambda size, x_center, y_center, max_iteration: 0 <= size <= 1024)
//! @requires(lambda size, x_center, y_center, max_iteration: size != 0)
//! def mandel(size: Integer, x_center: Float, y_center: Float, max_iteration: Integer):
//!     out = [[0 for i in range(size)] for j in range(size)]
//!     for i in range(size):
//!         for j in range(size):
//!             x, y = (x_center + 4.0 * float(i - size / 2) / size, y_center + 4.0 * float(j - size / 2) / size)
//!             a, b = (0.0, 0.0)
//!             iteration = 0
//!             while a ** 2 + b ** 2 <= 4.0 and iteration < max_iteration:
//!                 a, b = (a ** 2 - b ** 2 + x, 2 * a * b + y)
//!                 iteration += 1
//!             if iteration == max_iteration:
//!                 color_value = 255
//!             else:
//!                 color_value = iteration * 10 % 255
//!             out[i][j] = color_value
//!     return out
//! ```
use crate::prelude::*;

pub mod mandel {
    use super::*;

    pub type Output = Array2<i64>;

    pub fn main(args: &mut Args<impl Input>) -> Output {
        let size: i64 = args.scalar();
        let x_center: Q32 = args.scalar();
        let y_center: Q32 = args.scalar();
        let max_iteration: i64 = args.scalar();
        assert!((0..=1024).contains(&size), "requires 0 <= size <= 1024");
        assert!(size != 0, "requires size != 0");

        // `4.0 * float(k - size / 2) / size`: `size / 2` is a float division.
        let offset = |k: usize| (Q32::from_int(k as i64) - Q32::from_int(size) / 2) * 4 / size;
        let four = Q32::from_int(4);
        let n = size as usize;
        Array2::from_shape_fn((n, n), |(i, j)| {
            let (x, y) = (x_center + offset(i), y_center + offset(j));
            let (mut a, mut b) = (Q32::ZERO, Q32::ZERO);
            let mut iteration = 0;
            while a * a + b * b <= four && iteration < max_iteration {
                // `a ** 2` is `1 * a * a`, i.e. `a * a`; `2 * a * b` is `(2 * a) * b`.
                (a, b) = (a * a - b * b + x, a * 2 * b + y);
                iteration += 1;
            }
            if iteration == max_iteration {
                255
            } else {
                iteration * 10 % 255
            }
        })
    }
}
