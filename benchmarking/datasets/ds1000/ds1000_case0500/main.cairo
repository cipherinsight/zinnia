// DS-1000 case 500 — Cairo baseline.
// a.reshape(-1, 3) on Float[54] → Float[18, 3]. Layout-only.
// Inputs Q-scaled (×2^32); sample a[k] = k * 0.1.

#[executable]
pub fn main() {
    const Q: u128 = 4_294_967_296;
    const N: u32 = 54;

    let mut a: Array<u128> = ArrayTrait::new();
    for k in 0_u32..N {
        let raw: u128 = k.into();
        a.append((raw * Q) / 10);
    };

    let mut claimed: Array<u128> = ArrayTrait::new();
    for k in 0_u32..N {
        claimed.append(*a.at(k));
    };

    for i in 0_u32..18 {
        for j in 0_u32..3 {
            let idx: u32 = i * 3 + j;
            assert!(*a.at(idx) == *claimed.at(idx));
        };
    };
}
