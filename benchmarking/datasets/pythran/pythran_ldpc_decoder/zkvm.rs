//! Pythran ldpc_decoder: belief-propagation decoding of an LDPC code in the
//! log domain. `Bits[i]` lists the variable nodes of check node `i`,
//! `Nodes[j]` the check nodes of variable node `j`.
//!
//! ```python
//! def phi0(x):
//!     x = abs(x)
//!     if x < 9.08e-05:
//!         return 10
//!     else:
//!         return -log(tanh(x / 2))
//!
//! def G(Lq):
//!     X = sum((phi0(e) for e in Lq))
//!     s = np.prod(np.sign(Lq))
//!     return s * phi0(X)
//!
//! def BinaryProduct(X, Y):
//!     A = X.dot(Y)
//!     return A % 2
//!
//! def InCode(H, x):
//!     return (BinaryProduct(H, x) == 0).all()
//!
//! @requires(lambda H, Bits, Nodes, Lq, Lc, max_iter: 0 <= max_iter <= 1024)
//! def Decoding_logBP(H: Integer[Array, '8 8'], Bits: Integer[Array, '8 8'], Nodes: Integer[Array, '8 8'], Lq: Float[Array, '8 8'], Lc: Float[Array, '8'], max_iter: Integer=1):
//!     m, n = H.shape
//!     if not len(Lc) == n:
//!         raise ValueError('La taille de y doit correspondre au nombre de colonnes de H')
//!     if m >= n:
//!         raise ValueError('H doit avoir plus de colonnes que de lignes')
//!     Lr = np.zeros(shape=(m, n))
//!     count = 0
//!     Lq += Lc
//!     while True:
//!         count += 1
//!         for i in range(m):
//!             Ni = Bits[i]
//!             for j in Ni:
//!                 Nij = list(Ni)
//!                 Nij.remove(j)
//!                 Lr[i, j] = G(Lq[i][Nij])
//!         Lr = np.clip(Lr, -100, 100)
//!         for j in range(n):
//!             Mj = Nodes[j]
//!             for i in Mj:
//!                 Mji = list(Mj)
//!                 Mji.remove(i)
//!                 Lq[i, j] = Lc[j] + sum(Lr[Mji][:, j])
//!         extrinsic = np.empty(n)
//!         for j in range(n):
//!             Mj = Nodes[j]
//!             extrinsic[j] = sum(Lr[Mj][:, j])
//!         L_posteriori = extrinsic + Lc
//!         x = np.array(extrinsic <= 0).astype(int)
//!         product = InCode(H, x)
//!         if product or count >= max_iter:
//!             break
//!     return (np.array(L_posteriori <= 0).astype(int), Lq - Lc, extrinsic, product)
//! ```
use crate::prelude::*;

/// The kernel's `9.08e-05`, quantized: `round(9.08e-05 * 2^32)`.
const PHI0_CUTOFF: Q32 = Q32::from_bits(389_983);

/// `phi0(x) = -log(tanh(|x| / 2))`, and 10 near zero.
fn phi0(x: Q32) -> Q32 {
    let x = x.abs();
    if x < PHI0_CUTOFF {
        Q32::from_int(10)
    } else {
        -(x / 2).tanh().ln()
    }
}

/// `np.sign`.
fn sign(x: Q32) -> i64 {
    match x.cmp(&Q32::ZERO) {
        core::cmp::Ordering::Less => -1,
        core::cmp::Ordering::Equal => 0,
        core::cmp::Ordering::Greater => 1,
    }
}

/// `G(values)`: the check-node update over the incoming messages.
fn g(values: impl Iterator<Item = Q32> + Clone) -> Q32 {
    let x: Q32 = values.clone().map(phi0).sum();
    let s: i64 = values.map(sign).product();
    phi0(x) * s
}

/// The items of `list` except the first one equal to `item`
/// (`l = list(list); l.remove(item)`), which must occur.
fn without_first<'a>(list: ArrayView1<'a, i64>, item: i64) -> impl Iterator<Item = i64> + Clone + 'a {
    let skip = list.iter().position(|&v| v == item).expect("list.remove(x): x not in list");
    list.into_iter().enumerate().filter(move |&(k, _)| k != skip).map(|(_, &v)| v)
}

#[allow(non_snake_case)]
pub mod Decoding_logBP {
    use super::*;

    /// `(L_posteriori <= 0, Lq - Lc, extrinsic, product)`.
    pub type Output = (Array1<i64>, Array2<Q32>, Array1<Q32>, bool);

    pub fn main(args: &mut Args<impl Input>) -> Output {
        let h: Array2<i64> = args.array2();
        let bits: Array2<i64> = args.array2();
        let nodes: Array2<i64> = args.array2();
        let mut lq: Array2<Q32> = args.array2();
        let lc: Array1<Q32> = args.array1();
        let max_iter: i64 = args.scalar();
        assert!((0..=1024).contains(&max_iter), "requires 0 <= max_iter <= 1024");

        let (m, n) = h.dim();
        assert!(lc.len() == n, "La taille de y doit correspondre au nombre de colonnes de H");
        assert!(m < n, "H doit avoir plus de colonnes que de lignes");
        let mut lr = Array2::<Q32>::zeros((m, n));
        let mut count = 0;
        lq += &lc;
        let (extrinsic, product) = loop {
            count += 1;
            // Check-to-variable messages.
            for i in 0..m {
                let ni = bits.row(i);
                for &j in ni {
                    let others = without_first(ni, j).map(|k| lq[(i, np_index(k, lq.ncols()))]);
                    lr[(i, np_index(j, n))] = g(others);
                }
            }
            let (lo, hi) = (Q32::from_int(-100), Q32::from_int(100));
            lr.mapv_inplace(|v| v.clamp(lo, hi));
            // Variable-to-check messages.
            for j in 0..n {
                let mj = nodes.row(j);
                for &i in mj {
                    let others: Q32 = without_first(mj, i).map(|k| lr[(np_index(k, m), j)]).sum();
                    let row = np_index(i, lq.nrows());
                    lq[(row, j)] = lc[j] + others;
                }
            }
            let extrinsic: Array1<Q32> = (0..n)
                .map(|j| nodes.row(j).iter().map(|&k| lr[(np_index(k, m), j)]).sum())
                .collect();
            // InCode(H, x): every parity check of the hard decision holds.
            let x = extrinsic.mapv(|e| (e <= Q32::ZERO) as i64);
            let product = h.rows().into_iter().all(|row| row.dot(&x) % 2 == 0);
            if product || count >= max_iter {
                break (extrinsic, product);
            }
        };
        let l_posteriori = &extrinsic + &lc;
        (l_posteriori.mapv(|v| (v <= Q32::ZERO) as i64), &lq - &lc, extrinsic, product)
    }
}
