// DS-1000 case 0298, Noir baseline.
// Kernel: `vals, idx = np.unique(a, return_inverse=True); b = np.zeros((5,
// vals.size)); b[np.arange(5), idx] = 1` on Float[5]: the one-hot rows of each
// value's rank among the distinct values (NumPy's `unique` is sorted).
// Output: shape (5, number of distinct values) as dynamic2, the cells compact
// and row-major, zero-padded to 5 x 5.
// The rank of a_i is the number of distinct values below it, each distinct
// value counted at its first occurrence. Row i of b starts at cell
// i * n_unique, so cell t is 1.0 when i * n_unique + rank_i = t for some i
// (Field equalities over the cells row i can reach, t in [i, 5 i + 4]).
// @outputs: dynamic2 float[5x5]

use noir_hw::{SCALE, window};

global N: u32 = 5;

fn main(a: [Field; N]) -> pub [Field; 2 + N * N] {
    let mut w: [u64; N] = [0; N];
    for i in 0..N {
        w[i] = window(a[i]);
    }
    let mut first: [Field; N] = [0; N];
    let mut n_unique: Field = 0;
    for j in 0..N {
        let mut new = true;
        for i in 0..N {
            if i < j {
                new &= a[i] != a[j];
            }
        }
        first[j] = new as Field;
        n_unique += first[j];
    }
    let mut out: [Field; 2 + N * N] = [0; 2 + N * N];
    out[0] = N as Field;
    out[1] = n_unique;
    for i in 0..N {
        let mut rank: Field = 0;
        for j in 0..N {
            if j != i {
                rank += first[j] * ((w[j] < w[i]) as Field);
            }
        }
        let cell = (i as Field) * n_unique + rank;
        std::as_witness(cell); // one witness for the equalities below
        for t in 0..N * N {
            if (t >= i) & (t <= N * i + N - 1) {
                out[2 + t] += SCALE * ((cell == t as Field) as Field);
            }
        }
    }
    out
}
