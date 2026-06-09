// DS-1000 case 0302 -- Noir baseline. A.reshape(nrow, -1) on Float[20].
// Field convention (CONVENTIONS.md sec 3).

use noir_fpc::q_floor_div_hint;

fn main(a: [Field; 20], nrow: Field) -> pub [Field; 20] {
    assert(nrow != 0, "0302: nrow must be > 0");
    let q = q_floor_div_hint(20 as Field, nrow);
    assert(q * nrow == 20, "0302: nrow must divide len(a)");
    a
}
