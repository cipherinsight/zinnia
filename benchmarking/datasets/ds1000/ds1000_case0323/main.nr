// DS-1000 case 0323 -- Noir baseline. result = sin(deg2rad(degree)).
// Q32 fixed-point per CONVENTIONS.md. Tolerance loosened to ~5e-3 because
// the 4-term Taylor sin (used by the qsin gadget) accumulates rounding.
//
// `degree` is an integer (typical sample input is degree=60, claimed
// result sin(pi/3) ~ 0.866). Provided in Prover.toml as a Field equal
// to the raw signed integer (not Q-scaled).

use noir_fpc::{qmul, qsin, Q};

fn main(degree: Field) -> pub Field {
    let degree_fp: Field = degree * Q;
    // pi/180 ~= 0.017453292519943295; Q-encoded as round(pi/180 * 2^32) = 74961321.
    let pi_over_180: Field = 74961321;
    let rad = qmul(degree_fp, pi_over_180);
    qsin(rad)
}
