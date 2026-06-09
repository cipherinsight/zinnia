// DS-1000 case 0472 -- Noir baseline. a[-1:, ...] on Float[6,2,9,3,7,4] -> (1,2,9,3,7,4).
// Take only the last layer (a[5, ...]). Output 1512 cells.

global TAIL: u32 = 1512;

fn main(a: [Field; 9072]) -> pub [Field; 1512] {
    let mut out: [Field; 1512] = [0; 1512];
    let base = 5 * TAIL;
    for k in 0..TAIL {
        out[k] = a[base + k];
    }
    out
}
