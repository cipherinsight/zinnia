// DS-1000 case 0505 -- Noir baseline. Field convention.
// sum(a) - sum(take_along_axis(...)).

use noir_fpc::{qadd, qsub};

global D0: u32 = 13;
global D1: u32 = 13;
global D2: u32 = 5;
global TOTAL: u32 = 845;

fn main(a: [Field; 845], b: [Field; 169]) -> pub Field {
    let mut s_all: Field = a[0];
    for k in 1..TOTAL { s_all = qadd(s_all, a[k]); }
    let mut s_picked: Field = 0;
    for k in 0..(D0 * D1) {
        let i = k / D1;
        let j = k % D1;
        let target = (i * D1 * D2) as Field + (j * D2) as Field + b[k];
        let mut v: Field = 0;
        for m in 0..(D0 * D1 * D2) {
            if (m as Field) == target { v = a[m]; }
        }
        s_picked = qadd(s_picked, v);
    }
    qsub(s_all, s_picked)
}
