//! Pythran lttb_pt: Largest-Triangle-Three-Buckets downsampling of a 2-D
//! series to `n_out` points.
//!
//! ```python
//! def _areas_of_triangles(a, bs, c):
//!     bs_minus_a = bs - a
//!     a_minus_bs = a - bs
//!     return 0.5 * abs((a[0] - c[0]) * bs_minus_a[:, 1] - a_minus_bs[:, 0] * (c[1] - a[1]))
//!
//! @requires(lambda data, n_out: 0 <= n_out <= 1024)
//! def downsample(data: Float[Array, '100 2'], n_out: Integer):
//!     if n_out > data.shape[0]:
//!         raise ValueError('n_out must be <= number of rows in data')
//!     if n_out == data.shape[0]:
//!         return data
//!     if n_out < 3:
//!         raise ValueError('Can only downsample to a minimum of 3 points')
//!     n_bins = n_out - 2
//!     data_bins = np.array_split(data[1:len(data) - 1], n_bins)
//!     out = np.zeros((n_out, 2))
//!     out[0] = data[0]
//!     out[len(out) - 1] = data[len(data) - 1]
//!     for i in range(len(data_bins)):
//!         this_bin = data_bins[i]
//!         if i < n_bins - 1:
//!             next_bin = data_bins[i + 1]
//!         else:
//!             next_bin = data[len(data) - 1:]
//!         a = out[i]
//!         bs = this_bin
//!         c = next_bin.mean(axis=0)
//!         areas = _areas_of_triangles(a, bs, c)
//!         out[i + 1] = bs[np.argmax(areas)]
//!     return out
//! ```
use crate::prelude::*;

/// The row ranges of `np.array_split` of `len` rows into `sections` parts:
/// the first `len % sections` parts have one row more.
fn array_split(len: usize, sections: usize) -> Vec<core::ops::Range<usize>> {
    let (size, extra) = (len / sections, len % sections);
    let mut start = 0;
    (0..sections)
        .map(|k| {
            let end = start + size + usize::from(k < extra);
            let range = start..end;
            start = end;
            range
        })
        .collect()
}

/// `next_bin.mean(axis=0)`.
fn column_means(rows: ArrayView2<Q32>) -> [Q32; 2] {
    let n = rows.nrows() as i64;
    core::array::from_fn(|axis| rows.column(axis).sum() / n)
}

pub mod downsample {
    use super::*;

    pub type Output = Array2<Q32>;

    pub fn main(args: &mut Args<impl Input>) -> Output {
        let data: Array2<Q32> = args.array2();
        let n_out: i64 = args.scalar();
        assert!((0..=1024).contains(&n_out), "requires 0 <= n_out <= 1024");
        let rows = data.nrows();
        let n_out = n_out as usize;
        assert!(n_out <= rows, "n_out must be <= number of rows in data");
        if n_out == rows {
            return data;
        }
        assert!(n_out >= 3, "Can only downsample to a minimum of 3 points");
        let n_bins = n_out - 2;
        // Bin ranges as rows of `data` (the bins split `data[1:-1]`).
        let bins: Vec<_> = array_split(rows - 2, n_bins)
            .into_iter()
            .map(|r| r.start + 1..r.end + 1)
            .collect();
        let mut out = Array2::<Q32>::zeros((n_out, 2));
        out.row_mut(0).assign(&data.row(0));
        out.row_mut(n_out - 1).assign(&data.row(rows - 1));
        for (i, bin) in bins.iter().enumerate() {
            let next_bin = if i < n_bins - 1 {
                bins[i + 1].clone()
            } else {
                rows - 1..rows
            };
            let (a0, a1) = (out[(i, 0)], out[(i, 1)]);
            let [c0, c1] = column_means(data.slice(s![next_bin, ..]));
            // `_areas_of_triangles(a, bs, c)` for each row of the bin.
            let areas: Vec<Q32> = bin
                .clone()
                .map(|r| {
                    Q32::HALF * ((a0 - c0) * (data[(r, 1)] - a1) - (a0 - data[(r, 0)]) * (c1 - a1)).abs()
                })
                .collect();
            let chosen = bin.start + np::argmax(&areas);
            out.row_mut(i + 1).assign(&data.row(chosen));
        }
        out
    }
}
