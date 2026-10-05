//! Pythran sobelfilter: Sobel edge detection on a flattened `rows x cols`
//! integer image; border pixels are 255.
//!
//! ```python
//! def sobelFilter(original_image: Integer[Array, '10000'], cols: Integer, rows: Integer):
//!     edge_image = list(range(len(original_image)))
//!     for i in range(rows):
//!         edge_image[i * cols] = 255
//!         edge_image[(i + 1) * cols - 1] = 255
//!     for i in range(1, cols - 1):
//!         edge_image[i] = 255
//!         edge_image[i + (rows - 1) * cols] = 255
//!     for iy in range(1, rows - 1):
//!         for ix in range(1, cols - 1):
//!             sum_x = 0
//!             sum_y = 0
//!             sum = 0
//!             sum_x += original_image[ix - 1 + (iy - 1) * cols] * -1
//!             sum_x += original_image[ix + (iy - 1) * cols] * -2
//!             sum_x += original_image[ix + 1 + (iy - 1) * cols] * -1
//!             sum_x += original_image[ix - 1 + (iy + 1) * cols] * 1
//!             sum_x += original_image[ix + (iy + 1) * cols] * 2
//!             sum_x += original_image[ix + 1 + (iy + 1) * cols] * 1
//!             sum_x = min(255, max(0, sum_x))
//!             sum_y += original_image[ix - 1 + (iy - 1) * cols] * 1
//!             sum_y += original_image[ix + 1 + (iy - 1) * cols] * -1
//!             sum_y += original_image[ix - 1 + iy * cols] * 2
//!             sum_y += original_image[ix + 1 + iy * cols] * -2
//!             sum_y += original_image[ix - 1 + (iy + 1) * cols] * 1
//!             sum_y += original_image[ix + 1 + (iy + 1) * cols] * -1
//!             sum_y = min(255, max(0, sum_y))
//!             sum = abs(sum_x) + abs(sum_y)
//!             edge_image[ix + iy * cols] = 255 - (255 & sum)
//!     return edge_image
//! ```
use crate::prelude::*;

#[allow(non_snake_case)]
pub mod sobelFilter {
    use super::*;

    pub type Output = Array1<i64>;

    pub fn main(args: &mut Args<impl Input>) -> Output {
        let image: Array1<i64> = args.array1();
        let cols: i64 = args.scalar();
        let rows: i64 = args.scalar();
        let n = image.len();
        // Python list/array indices: negative ones wrap, others must be in range.
        let at = |k: i64| np_index(k, n);
        let pixel = |x: i64, y: i64| image[at(x + y * cols)];

        let mut edges: Array1<i64> = (0..n as i64).collect();
        for i in 0..rows {
            edges[at(i * cols)] = 255;
            edges[at((i + 1) * cols - 1)] = 255;
        }
        for i in 1..cols - 1 {
            edges[at(i)] = 255;
            edges[at(i + (rows - 1) * cols)] = 255;
        }
        for iy in 1..rows - 1 {
            for ix in 1..cols - 1 {
                let sum_x = -pixel(ix - 1, iy - 1) - 2 * pixel(ix, iy - 1) - pixel(ix + 1, iy - 1)
                    + pixel(ix - 1, iy + 1)
                    + 2 * pixel(ix, iy + 1)
                    + pixel(ix + 1, iy + 1);
                let sum_x = sum_x.clamp(0, 255);
                let sum_y = pixel(ix - 1, iy - 1) - pixel(ix + 1, iy - 1) + 2 * pixel(ix - 1, iy)
                    - 2 * pixel(ix + 1, iy)
                    + pixel(ix - 1, iy + 1)
                    - pixel(ix + 1, iy + 1);
                let sum_y = sum_y.clamp(0, 255);
                let sum = sum_x.abs() + sum_y.abs();
                edges[at(ix + iy * cols)] = 255 - (255 & sum);
            }
        }
        edges
    }
}
