// DS-1000 case 0336 -- Noir baseline. result = np.diag(np.fliplr(a)) on Float[8,8].
// Q32 fixed-point. Pure static-index gather (no arithmetic), so quantization
// is just passthrough.

fn main(a: [Field; 64]) -> pub [Field; 8] {
    let mut out: [Field; 8] = [0; 8];
    for i in 0..8 {
        let j = 7 - i;
        out[i] = a[i * 8 + j];
    }
    out
}
