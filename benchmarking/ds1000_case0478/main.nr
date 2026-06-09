// DS-1000 case 0478 -- Noir baseline.
// einsum('ii->i', a) on Integer[4,4] = diagonal. a is then zeroed and re-set;
// result is the saved diagonal.

global N: u32 = 4;

fn main(a: [Field; 16]) -> pub [Field; 4] {
    let mut out: [Field; 4] = [0; 4];
    for i in 0..N {
        out[i] = a[i * N + i];
    }
    out
}
