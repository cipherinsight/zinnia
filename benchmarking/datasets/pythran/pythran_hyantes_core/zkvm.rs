//! Pythran hyantes_core: on a `range_x x range_y` grid of points on the
//! sphere, sum the potentials `k[2] / (1 + d)` of the towns `k` of `t` within
//! great-circle distance `d < range_` (in km).
//!
//! ```python
//! def run(xmin: Float, ymin: Float, xmax: Float, ymax: Float, step: Float, range_: Float, range_x: Integer, range_y: Integer, t: Float[Array, '80 3']):
//!     pt = [[0] * range_y for _ in range(range_x)]
//!     for i in range(range_x):
//!         for j in range(range_y):
//!             s = 0
//!             for k in t:
//!                 tmp = 6368.0 * math.acos(math.cos(xmin + step * i) * math.cos(k[0]) * math.cos(ymin + step * j - k[1]) + math.sin(xmin + step * i) * math.sin(k[0]))
//!                 if tmp < range_:
//!                     s += k[2] / (1 + tmp)
//!             pt[i][j] = s
//!     return pt
//! ```
use crate::prelude::*;

/// `6368.0`, the Earth radius in km.
const EARTH_RADIUS: Q32 = Q32::from_int(6368);

/// `math.acos` on `[-1, 1]`: `pi/2 - asin(x)`.
fn acos(x: Q32) -> Q32 {
    Q32::FRAC_PI_2 - x.asin()
}

pub mod run {
    use super::*;

    // unused parameters: xmax, ymax
    pub type Output = Array2<Q32>;

    pub fn main(args: &mut Args<impl Input>) -> Output {
        let xmin: Q32 = args.scalar();
        let ymin: Q32 = args.scalar();
        let step: Q32 = args.scalar();
        let range: Q32 = args.scalar();
        let range_x: i64 = args.scalar();
        let range_y: i64 = args.scalar();
        let t: Array2<Q32> = args.array2();
        assert!(step != Q32::ZERO, "requires step != 0.0");
        assert!(range != Q32::ZERO, "requires range_ != 0.0");

        // `range` and `[0] * n` of a negative count are empty.
        let extent = |n: i64| usize::try_from(n.max(0)).expect("grid exceeds the address space");
        let (rows, cols) = (extent(range_x), extent(range_y));
        // Per town: cos and sin of its latitude, its longitude, its weight.
        let towns: Vec<(Q32, Q32, Q32, Q32)> =
            t.rows().into_iter().map(|k| (k[0].cos(), k[0].sin(), k[1], k[2])).collect();
        let mut pt = Array2::<Q32>::zeros((rows, cols));
        for i in 0..rows {
            let x = xmin + step * i as i64;
            let (cos_x, sin_x) = (x.cos(), x.sin());
            for j in 0..cols {
                let y = ymin + step * j as i64;
                let mut s = Q32::ZERO;
                for &(cos_k0, sin_k0, k1, k2) in &towns {
                    let tmp = EARTH_RADIUS * acos(cos_x * cos_k0 * (y - k1).cos() + sin_x * sin_k0);
                    if tmp < range {
                        s += k2 / (Q32::ONE + tmp);
                    }
                }
                pt[(i, j)] = s;
            }
        }
        pt
    }
}
