// DS-1000 case 0361 -- Noir baseline. np.delete(a, [0,2], axis=1) on Float[6,6].
// Output (6, 4): drop columns 0 and 2.

fn main(a: [Field; 36]) -> pub [Field; 24] {
    let mut out: [Field; 24] = [0; 24];
    let cols: [u32; 4] = [1, 3, 4, 5];
    for i in 0..6 {
        for j in 0..4 {
            out[i * 4 + j] = a[i * 6 + cols[j]];
        }
    }
    out
}
