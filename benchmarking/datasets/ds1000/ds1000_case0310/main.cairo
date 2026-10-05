// DS-1000 case 310 — a.argmin() over Float[8, 6].
// Use i128 because the min entry is negative.
// Quantisation ×2^32 (Q). Sample: a[5][3] = -99.0, deliberate min at flat 33.

#[executable]
pub fn main() {
    const Q: i128 = 4_294_967_296;
    let mut data: Array<i128> = ArrayTrait::new();
    for i in 0_u32..8 {
        for j in 0_u32..6 {
            let q_v: i128 = if i == 5 && j == 3 {
                -99_i128 * Q
            } else {
                let num: i128 = ((i + 1) * 10 + (j + 1)).into();
                (num * Q) / 100
            };
            data.append(q_v);
        };
    };
    let expected: i128 = 33;  // 5*6 + 3

    // Seed best_val with cells[0] (matches Zinnia IR's fold-from-first scheme).
    let mut best_val: i128 = *data.at(0);
    let mut best_idx: i128 = 0;
    let mut k: u32 = 1;
    while k < 48 {
        let v: i128 = *data.at(k.try_into().unwrap());
        if v < best_val {
            best_val = v;
            best_idx = k.into();
        }
        k += 1;
    };

    assert!(best_idx == expected);
}
