// DS-1000 case 0301 -- Noir baseline. A.reshape(-1, ncol) on Float[20].
// Field convention (CONVENTIONS.md sec 3).

use noir_fpc::q_floor_div_hint;

fn main(a: [Field; 20], ncol: Field) -> pub [Field; 22] {
    assert(ncol != 0, "0301: ncol must be > 0");
    // Sound divisibility: q = floor(20/ncol); assert q*ncol == 20.
    // The hint alone is unconstrained; the multiplication-equality binds q
    // (and therefore the divisibility claim) into the constraint system.
    let q = q_floor_div_hint(20 as Field, ncol);
    assert(q * ncol == 20, "0301: ncol must divide len(a)");
    // Runtime shape metadata (rows, cols), mirroring Zinnia's dyn-array
    // output contract.
    let mut out: [Field; 22] = [0; 22];
    out[0] = q;
    out[1] = ncol;
    for k in 0..20 {
        out[k + 2] = a[k];
    }
    out
}
