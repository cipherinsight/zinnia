// DS-1000 case 0472, Noir baseline.
// Kernel: `a[-1:, ...]` on Float[6, 2, 9, 3, 7, 4]: the last of the six leading
// slices (1512 values).
// @outputs: float[1512]

global SLICE: u32 = 1512;

fn main(a: [Field; 9072]) -> pub [Field; 1512] {
    let mut out = [0; 1512];
    for k in 0..SLICE {
        out[k] = noir_hw::checked(a[5 * SLICE + k]);
    }
    out
}
