// DS-1000 case 0326 -- Noir baseline. degrees(arcsin(value)) on Float.
// Uses the shared noir_fpc::qarcsin (the same lowering Zinnia auto-emits
// for arcsin_f); same approximation kernel for apples-to-apples gate
// count comparison. The earlier 5-term Taylor version diverges as |x|
// approaches 1, so wasn't equivalent across the input range.

use noir_fpc::{qmul, qarcsin};

fn main(value: Field) -> pub Field {
    let arcsin = qarcsin(value);
    // *180/pi; 180/pi * Q = 246083499208.
    let deg_per_rad: Field = 246083499208;
    qmul(arcsin, deg_per_rad)
}
