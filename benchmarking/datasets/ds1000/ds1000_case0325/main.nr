// DS-1000 case 0325, Noir baseline.
// Kernel: `int(np.sin(number) > np.sin(np.deg2rad(number)))` for an Integer
// number (does `number` read as radians give a larger sine than as degrees).
// @outputs: int[1]

use noir_hw::{from_int, gt, sin};

global DEG_TO_RAD_Q: Field = 74961321; // round(pi / 180 * 2^32)

fn main(number: Field) -> pub Field {
    let deg = sin(number * DEG_TO_RAD_Q);
    let rad = sin(from_int(number));
    if gt(rad, deg) { 1 } else { 0 }
}
