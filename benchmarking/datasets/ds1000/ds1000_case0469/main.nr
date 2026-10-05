// DS-1000 case 0469 -- Noir baseline.
// np.real(s)/n + 1j*np.imag(s)/n with s = np.sum(a) on Float[3].
// For real inputs imag(s) = 0, so result = (mean, 0). We expose the real
// and imaginary components as two public Q32 Fields, matching the halo2
// baseline's two-public-output shape.

use noir_fpc::{qadd, qdiv, q_from_int};

fn main(a: [Field; 3]) -> pub [Field; 2] {
    let s = qadd(qadd(a[0], a[1]), a[2]);
    let n = q_from_int(3);
    let res_re = qdiv(s, n);
    let res_im: Field = 0;
    [res_re, res_im]
}
