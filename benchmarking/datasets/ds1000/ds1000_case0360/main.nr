// DS-1000 case 0360 -- Noir baseline. np.delete(a, 2, axis=0) on Float[4,4].
// Output (3, 4): drop row 2.

fn main(a: [Field; 16]) -> pub [Field; 12] {
    let mut out: [Field; 12] = [0; 12];
    let src_rows: [u32; 3] = [0, 1, 3];
    for i in 0..3 {
        let r = src_rows[i];
        for j in 0..4 {
            out[i * 4 + j] = a[r * 4 + j];
        }
    }
    out
}
