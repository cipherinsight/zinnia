// DS-1000 case 349 — a.shape[1] on Float[8, 6] → constant 6.
// Mirrors Zinnia IR: 48 witness reads (touched here) + constant 6.

#[executable]
pub fn main() {
    const Q: u128 = 4_294_967_296;

    let mut data: Array<u128> = ArrayTrait::new();
    for i in 0_u32..8 {
        for j in 0_u32..6 {
            let num: u128 = ((i + 1) * 10 + (j + 1)).into();
            data.append((num * Q) / 100);
        };
    };
    let claimed: u128 = 6;

    // Touch each cell to mirror Zinnia's ReadFloatIR (sum-with-zero pattern).
    let mut sink: u128 = 0;
    for k in 0_u32..48 {
        sink = sink + (*data.at(k)) * 0;
    };
    assert!(sink == 0);

    let result: u128 = 6;
    assert!(result == claimed);
}
