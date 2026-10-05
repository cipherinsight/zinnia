// DS-1000 case 0409 -- Noir baseline. z = x + y on Float[8,6].

use noir_fpc::qadd;

fn main(x: [Field; 48], y: [Field; 48]) -> pub [Field; 48] {
    let mut out: [Field; 48] = [0; 48];
    for k in 0..48 {
        out[k] = qadd(x[k], y[k]);
    }
    out
}
