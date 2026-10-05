// DS-1000 case 0470 -- Noir baseline. Identical to 0469: complex mean
// of real Float[3]; imag part is zero.

use noir_fpc::{qadd, qdiv, q_from_int};

fn main(a: [Field; 3]) -> pub [Field; 2] {
    let s = qadd(qadd(a[0], a[1]), a[2]);
    let n = q_from_int(3);
    let res_re = qdiv(s, n);
    let res_im: Field = 0;
    [res_re, res_im]
}
