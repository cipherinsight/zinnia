// DS-1000 case 0359 -- Noir baseline. np.delete(a, 2, axis=1) on Float[3,3].
// Output (3, 2): drop column 2.

fn main(a: [Field; 9]) -> pub [Field; 6] {
    let mut out: [Field; 6] = [0; 6];
    for i in 0..3 {
        out[i * 2] = a[i * 3];
        out[i * 2 + 1] = a[i * 3 + 1];
    }
    out
}
