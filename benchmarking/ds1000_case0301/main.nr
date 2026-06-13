// DS-1000 case 0301 -- Noir baseline. A.reshape(-1, ncol) on Float[20].
// Field convention (CONVENTIONS.md sec 3).

use noir_fpc::q_floor_div_bounded;

fn main(a: [Field; 20], ncol: Field) -> pub [Field; 22] {
    assert(ncol != 0, "0301: ncol must be > 0");
    // Sound divisibility: q = floor(20/ncol); assert q*ncol == 20.
    // NOTE: `q*ncol == 20` ALONE does not bind q to a small integer -- over the
    // field a malicious prover can pick any small q and set ncol = 20*q^{-1}
    // (a huge field element), satisfying the equality with a bogus shape.
    // `q_floor_div_bounded` range-checks q, the remainder, and ncol via bit
    // decomposition (all fit in 2^6 > 20), pinning q to the true floor and
    // rejecting the field-inverse substitution. Mirrors the Zinnia emitter's
    // `FloorDivIPinned` lowering, so the benchmark is sound-vs-sound.
    let q = q_floor_div_bounded::<6>(20 as Field, ncol);
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
