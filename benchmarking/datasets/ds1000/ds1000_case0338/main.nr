// DS-1000 case 0338 -- Noir baseline.
// np.vstack((np.diag(a), np.diag(np.fliplr(a)))) on Float[8,8] -> (2, 8).

global N: u32 = 8;

fn main(a: [Field; 64]) -> pub [Field; 16] {
    let mut out: [Field; 16] = [0; 16];
    for i in 0..N {
        out[i] = a[i * N + i];
        out[N + i] = a[i * N + (N - 1 - i)];
    }
    out
}
