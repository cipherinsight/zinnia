// DS-1000 case 0485 -- Noir baseline.
// np.tensordot(A, B, axes=((2),(0))) on Float[5,6,3]x[3,3] -> Float[5,6,3].
// out[i,j,k] = sum_p A[i,j,p] * B[p,k].

use noir_fpc::{qmul, qadd};

global I: u32 = 5;
global J: u32 = 6;
global P: u32 = 3;
global K: u32 = 3;

fn main(a: [Field; 90], b: [Field; 9]) -> pub [Field; 90] {
    let mut out: [Field; 90] = [0; 90];
    for i in 0..I {
        for j in 0..J {
            for k in 0..K {
                let mut s: Field = qmul(a[i * J * P + j * P], b[k]);
                for p in 1..P {
                    s = qadd(s, qmul(a[i * J * P + j * P + p], b[p * K + k]));
                }
                out[i * J * K + j * K + k] = s;
            }
        }
    }
    out
}
