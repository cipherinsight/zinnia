// DS-1000 case 0412 -- Noir baseline. x[x >= 0] on Float[10].
// Fixed-shape buffer: keep value if >= 0, else 0 (per halo2 baseline shape).

use noir_fpc::is_negative;

fn main(x: [Field; 10]) -> pub [Field; 10] {
    let mut out: [Field; 10] = [0; 10];
    for k in 0..10 {
        let neg = is_negative(x[k]);
        out[k] = if neg { 0 } else { x[k] };
    }
    out
}
