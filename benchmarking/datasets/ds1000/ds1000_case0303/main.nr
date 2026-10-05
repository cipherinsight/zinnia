// DS-1000 case 0303 -- Noir baseline. Field convention (CONVENTIONS.md sec 3:
// "Noir: Field with u64/u128 view; order comparisons require an integer cast").
// col = (N // ncol) * ncol; B = A[:col].reshape(-1, ncol) on Float[23].

use noir_fpc::q_floor_div_bounded;

global N: u32 = 23;

fn main(a: [Field; 23], ncol: Field) -> pub [Field; 25] {
    assert(ncol != 0, "0303: ncol must be > 0");
    // Sound floor-div: q = floor(N/ncol). Here `col = q*ncol` truncates (no
    // divisibility assert), so a bare hint would leave q -- and thus the
    // reported nrow and the slice length -- entirely unconstrained.
    // `q_floor_div_bounded` pins q to the true Euclidean floor with `r < ncol`
    // via bit-decomposition range checks (operands fit in 2^6 > 23). Mirrors
    // Zinnia's `FloorDivIPinned` lowering.
    let q = q_floor_div_bounded::<6>(N as Field, ncol);
    let col = q * ncol;
    let mut out: [Field; 25] = [0; 25];
    // Runtime shape metadata (rows, cols), mirroring Zinnia's dyn-array
    // output contract.
    out[0] = q;
    out[1] = ncol;
    for k in 0..N {
        // Per-cell `(k as Field) < col`: u32-safe operands fit u128 so
        // the bounded form is sound. Mirrors the Zinnia emitter's
        // slice-mask `lt_i` lowering.
        if ((k as u128) < (col as u128)) {
            out[k + 2] = a[k];
        }
    }
    out
}
