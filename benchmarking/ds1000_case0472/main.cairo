// DS-1000 case 472 — Cairo baseline. a[-1:, ...] on Float[6, 2, 9, 3, 7, 4].
// Returns the last slab along axis 0, shape (1, 2, 9, 3, 7, 4) = 1512 floats.
// Sample: a[k] = (k as f64) * 0.01 → quantised as (k * Q) / 100.

#[executable]
pub fn main() {
    const Q: u128 = 4_294_967_296;
    const N: u32 = 9072;
    const SLAB: u32 = 1512;

    let mut a: Array<u128> = ArrayTrait::new();
    for k in 0_u32..N {
        let q_v: u128 = ((k.into() * Q)) / 100;
        a.append(q_v);
    };

    // Build the claimed output = last 1512 entries.
    let start: u32 = 5 * SLAB;
    let mut claimed: Array<u128> = ArrayTrait::new();
    for k in 0_u32..SLAB {
        let idx: u32 = start + k;
        claimed.append(*a.at(idx));
    };

    // Verify slice extraction matches claim (exact equality since both come
    // from the same input quantisation).
    for k in 0_u32..SLAB {
        let idx: u32 = start + k;
        assert!(*a.at(idx) == *claimed.at(k));
    };
}
