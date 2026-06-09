// DS-1000 case 0324 -- Noir baseline. cos(deg2rad(degree)).
// degree is float-typed in the kernel (60.0 in fixture). Receive as Q-scaled.

use noir_fpc::{qmul, qcos};

fn main(degree: Field) -> pub Field {
    // pi/180 * Q = 74961321.
    let pi_over_180: Field = 74961321;
    let rad = qmul(degree, pi_over_180);
    qcos(rad)
}
