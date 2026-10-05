// DS-1000 case 413 — `x[x.imag != 0]` on a real Float[10].
// For purely real input the mask is identically false; result is empty.
// We mirror Zinnia's 10× ReadFloatIR (witness all 10 entries) with no output.

#[executable]
pub fn main() {
    const Q: i128 = 4_294_967_296;
    // Sample inputs in Q-scale (alternating sign, ±k/10).
    let raw: Array<i128> = array![
        Q / 10,
        -(2 * Q) / 10,
        (3 * Q) / 10,
        -(4 * Q) / 10,
        (5 * Q) / 10,
        -(6 * Q) / 10,
        (7 * Q) / 10,
        -(8 * Q) / 10,
        (9 * Q) / 10,
        -Q,
    ];
    let mut x: Array<i128> = ArrayTrait::new();
    let mut i: u32 = 0;
    while i < 10 {
        x.append(*raw.at(i));
        i += 1;
    };
    // Empty output — nothing to assert.
}
