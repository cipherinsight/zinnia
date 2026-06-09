// DS-1000 case 0304 -- Noir baseline. Field convention (CONVENTIONS.md sec 3).
// col = (N // ncol) * ncol; B = A[len(A)-col:][::-1].reshape(-1, ncol)
// on Float[23].

use noir_fpc::q_floor_div_hint;

global N: u32 = 23;

fn main(a: [Field; 23], ncol: Field) -> pub [Field; 23] {
    assert(ncol != 0, "0304: ncol must be > 0");
    let q = q_floor_div_hint(N as Field, ncol);
    let col = q * ncol;
    let mut out: [Field; 23] = [0; 23];
    for k in 0..N {
        if ((k as u128) < (col as u128)) {
            out[k] = a[N - 1 - k];
        }
    }
    out
}
