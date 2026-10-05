//! NPBench scattering_self_energies: electron-phonon scattering self-energies
//! `Sigma[k, E, a] += (G[k, E - w, neigh_idx[a, b]] @ dH[a, b, i]) @ (dH[a, b, j] * D[q, w, a, b, i, j])`,
//! in place.
//!
//! ```python
//! def scattering_self_energies(neigh_idx: Integer[Array, '6 2'], dH: Complex[Array, '6 2 2 3 3'], G: Complex[Array, '2 4 6 3 3'], D: Complex[Array, '2 2 6 2 2 2'], Sigma: Complex[Array, '2 4 6 3 3']):
//!     for k in range(G.shape[0]):
//!         for E in range(G.shape[1]):
//!             for q in range(D.shape[0]):
//!                 for w in range(D.shape[1]):
//!                     for i in range(D.shape[-2]):
//!                         for j in range(D.shape[-1]):
//!                             for a in range(neigh_idx.shape[0]):
//!                                 for b in range(neigh_idx.shape[1]):
//!                                     if E - w >= 0:
//!                                         dHG = G[k, E - w, neigh_idx[a, b]] @ dH[a, b, i]
//!                                         dHD = dH[a, b, j] * D[q, w, a, b, i, j]
//!                                         Sigma[k, E, a] += dHG @ dHD
//! ```
use crate::prelude::*;
use ndarray::{Array5, Array6, Ix5, Ix6};

/// `a @ b` for complex matrices: each part of each entry accumulates its raw
/// products and rescales once, like `np::matmul`.
fn complex_matmul(a: ArrayView2<C32>, b: ArrayView2<C32>) -> Array2<C32> {
    assert_eq!(a.ncols(), b.nrows(), "shapes not aligned");
    Array2::from_shape_fn((a.nrows(), b.ncols()), |(r, c)| {
        let pairs = || a.row(r).into_iter().zip(b.column(c));
        C32::new(
            dot(pairs().flat_map(|(x, y)| [(x.re, y.re), (-x.im, y.im)])),
            dot(pairs().flat_map(|(x, y)| [(x.re, y.im), (x.im, y.re)])),
        )
    })
}

pub mod scattering_self_energies {
    use super::*;

    // outputs: mutated Sigma
    pub type Output = Array5<C32>;

    #[allow(non_snake_case)]
    pub fn main(args: &mut Args<impl Input>) -> Output {
        let neigh_idx: Array2<i64> = args.array2();
        let dH: Array5<C32> = args.array::<C32, Ix5>();
        let G: Array5<C32> = args.array::<C32, Ix5>();
        let D: Array6<C32> = args.array::<C32, Ix6>();
        let mut Sigma: Array5<C32> = args.array::<C32, Ix5>();
        let (nk, ne) = (G.dim().0, G.dim().1);
        let (nq, nw, .., ni, nj) = D.dim();
        let (na, nb) = neigh_idx.dim();
        for k in 0..nk {
            for E in 0..ne {
                for q in 0..nq {
                    // `if E - w >= 0`: only `w <= E` contributes.
                    for w in 0..nw.min(E + 1) {
                        for i in 0..ni {
                            for j in 0..nj {
                                for a in 0..na {
                                    for b in 0..nb {
                                        let atom = np_index(neigh_idx[(a, b)], G.dim().2);
                                        let dHG = complex_matmul(
                                            G.slice(s![k, E - w, atom, .., ..]),
                                            dH.slice(s![a, b, i, .., ..]),
                                        );
                                        let coupling = D[(q, w, a, b, i, j)];
                                        let dHD =
                                            dH.slice(s![a, b, j, .., ..]).mapv(|v| v * coupling);
                                        let term = complex_matmul(dHG.view(), dHD.view());
                                        Sigma
                                            .slice_mut(s![k, E, a, .., ..])
                                            .zip_mut_with(&term, |s, &t| *s += t);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        Sigma
    }
}
