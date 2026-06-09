// DS-1000 case 0303 -- Noir baseline. Field convention (CONVENTIONS.md sec 3:
// "Noir: Field with u64/u128 view; order comparisons require an integer cast").
// col = (N // ncol) * ncol; B = A[:col].reshape(-1, ncol) on Float[23].

use noir_fpc::q_floor_div_hint;

global N: u32 = 23;

fn main(a: [Field; 23], ncol: Field) -> pub [Field; 23] {
    assert(ncol != 0, "0303: ncol must be > 0");
    // floor_div via hint + reconstruction, matching the Zinnia emitter's
    // pattern (noir_fpc::q_floor_div_hint + downstream q*ncol == _ check).
    let q = q_floor_div_hint(N as Field, ncol);
    let col = q * ncol;
    let mut out: [Field; 23] = [0; 23];
    for k in 0..N {
        // Per-cell `(k as Field) < col`: u32-safe operands fit u128 so
        // the bounded form is sound. Mirrors the Zinnia emitter's
        // slice-mask `lt_i` lowering.
        if ((k as u128) < (col as u128)) {
            out[k] = a[k];
        }
    }
    out
}
