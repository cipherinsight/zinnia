// DS-1000 case 0304 -- Noir baseline. Field convention (CONVENTIONS.md sec 3).
// col = (N // ncol) * ncol; B = A[len(A)-col:][::-1].reshape(-1, ncol)
// on Float[23].
// Mirrors Zinnia's lowering ALGORITHM: nonzero indicator + assert on ncol,
// hinted floor-div for col, remainder bound assert, clamped dyn-slice
// length, reversed PREFIX-MASKED gather via signed comparisons against
// the projected runtime length, and the reshape divisibility assert
// (nrow * ncol == len). Bound as [nrow, ncol, padded payload] (numpy
// dyn-shape contract).

use noir_fpc::{q_floor_div_hint, qlt_signed, to_signed};

global N: u32 = 23;

fn main(a: [Field; 23], ncol: Field) -> pub [Field; 25] {
    let nz: Field = if ncol == 0 { 0 } else { 1 };
    assert(nz == 1);
    let q0 = q_floor_div_hint(N as Field, ncol);
    let col = q0 * ncol;
    let rem = (N as Field) - col;
    assert((rem as u128) < (24 as u128));
    // Slice length, clamped non-negative (guarded select form).
    let lenf = (N as Field) - rem;
    let lpos: Field = if (lenf as u8) > 0 { 1 } else { 0 };
    let len = lpos * lenf;
    // Reversed prefix-masked gather: payload[k] = [k < len] * a[N-1-k].
    let (ls, lm) = to_signed(len);
    let mut payload: [Field; 23] = [0; 23];
    for k in 0..N {
        let ind: Field = if qlt_signed(false, k as u128, ls, lm) { 1 } else { 0 };
        payload[k] = ind * a[N - 1 - k];
    }
    // Reshape divisibility: nrow * ncol == len.
    let nrow = q_floor_div_hint(len, ncol);
    let ok: Field = if nrow * ncol == len { 1 } else { 0 };
    assert(ok == 1);
    let mut out: [Field; 25] = [0; 25];
    out[0] = nrow;
    out[1] = ncol;
    for k in 0..N {
        out[k + 2] = payload[k];
    }
    out
}
