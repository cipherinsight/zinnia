// DS-1000 case 0471, Noir baseline.
// Kernel: `Z[..., -1:]` on Float[6, 2, 9, 3, 7, 4]: the last entry of every
// length-4 innermost row, shape (6, 2, 9, 3, 7, 1).
// @outputs: float[2268]

fn main(Z: [Field; 9072]) -> pub [Field; 2268] {
    let mut out = [0; 2268];
    for k in 0..2268 {
        out[k] = Z[k * 4 + 3];
    }
    out
}
