// DS-1000 case 0302 -- Noir baseline. A.reshape(nrow, -1) on Float[20].
// Field convention (CONVENTIONS.md sec 3).

use noir_fpc::q_floor_div_bounded;

fn main(a: [Field; 20], nrow: Field) -> pub [Field; 22] {
    assert(nrow != 0, "0302: nrow must be > 0");
    // Sound floor-div: bit-decomposition range checks (operands fit in 2^6 > 20)
    // pin q to the true floor and reject the `nrow = 20*q^{-1}` field-inverse
    // cheat that `q*nrow == 20` alone would admit. Matches Zinnia's
    // `FloorDivIPinned` lowering.
    let q = q_floor_div_bounded::<6>(20 as Field, nrow);
    assert(q * nrow == 20, "0302: nrow must divide len(a)");
    // Runtime shape metadata (rows, cols), mirroring Zinnia's dyn-array
    // output contract.
    let mut out: [Field; 22] = [0; 22];
    out[0] = nrow;
    out[1] = q;
    for k in 0..20 {
        out[k + 2] = a[k];
    }
    out
}
