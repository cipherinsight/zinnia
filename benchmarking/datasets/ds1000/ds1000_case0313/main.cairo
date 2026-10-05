// DS-1000 case 313 — a.argmax() over Float[8, 6] (flat index).
// Sample: a[3][2] = 99.0 → flat 20.
// Quantisation ×2^32 (Q).

#[executable]
pub fn main() {
    const Q: u128 = 4_294_967_296;
    let mut data: Array<u128> = ArrayTrait::new();
    for i in 0_u32..8 {
        for j in 0_u32..6 {
            let q_v: u128 = if i == 3 && j == 2 {
                99_u128 * Q
            } else {
                let num: u128 = ((i + 1) * 10 + (j + 1)).into();
                (num * Q) / 100
            };
            data.append(q_v);
        };
    };
    let expected: u128 = 20;

    let mut best_val: u128 = *data.at(0);
    let mut best_idx: u128 = 0;
    let mut k: u32 = 1;
    while k < 48 {
        let v: u128 = *data.at(k.try_into().unwrap());
        if v > best_val {
            best_val = v;
            best_idx = k.into();
        }
        k += 1;
    };

    assert!(best_idx == expected);
}
