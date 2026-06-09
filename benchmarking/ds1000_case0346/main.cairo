// DS-1000 case 346 — a[row-1, :] *= multiply_number; cumsum.
// Float[8, 6] with runtime row, multiply_number. Output: 6 floats.

#[executable]
pub fn main() {
    const Q: u128 = 4_294_967_296;

    // Row 2 (0-indexed) holds [1, 2, 3, 4, 5, 6]; others are filler 0.1..0.6.
    let mut data: Array<u128> = ArrayTrait::new();
    for i in 0_u32..8 {
        for j in 0_u32..6 {
            let q_v: u128 = if i == 2 {
                ((j + 1).into()) * Q
            } else {
                let num: u128 = (j + 1).into();
                (num * Q) / 10
            };
            data.append(q_v);
        };
    };
    let row: i128 = 3;
    let multiply_number: i128 = 2;
    let claimed_q: Array<u128> = array![
        2 * Q, 6 * Q, 12 * Q, 20 * Q, 30 * Q, 42 * Q,
    ];

    let row_idx: u32 = (row - 1).try_into().unwrap();
    let mn_q: u128 = multiply_number.try_into().unwrap();
    let mn_q: u128 = mn_q * Q;

    // row_vals[j] = a[row_idx][j] * mn
    let mut row_vals: Array<u128> = ArrayTrait::new();
    for j in 0_u32..6 {
        let cell: u128 = *data.at(row_idx * 6 + j);
        let scaled: u128 = (cell * mn_q) / Q;
        row_vals.append(scaled);
    };

    // cumsum
    let mut result: Array<u128> = ArrayTrait::new();
    let mut acc: u128 = *row_vals.at(0);
    result.append(acc);
    for k in 1_u32..6 {
        acc = acc + *row_vals.at(k);
        result.append(acc);
    };

    for k in 0_u32..6 {
        assert!(*result.at(k) == *claimed_q.at(k));
    };
}
