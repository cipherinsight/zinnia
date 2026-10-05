// DS-1000 case 0325 -- Noir baseline.
// int(sin(number) > sin(deg2rad(number))). Sample number=1.0 (float-cast on host).
// Note: qsin Taylor degrades for |x| >> 1; sample stays within accuracy.

use noir_fpc::{qmul, qsin, qgt};

fn main(number: Field) -> pub Field {
    let pi_over_180: Field = 74961321;
    let rad = qmul(number, pi_over_180);
    let sin_deg = qsin(rad);
    let sin_n = qsin(number);
    if qgt(sin_n, sin_deg) { 1 } else { 0 }
}
