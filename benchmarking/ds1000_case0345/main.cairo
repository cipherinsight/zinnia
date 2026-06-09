// DS-1000 case 345 — a[:, col-1] *= multiply_number; cumsum.
// Float[8, 6] with runtime col, multiply_number. Output: 8 floats.
// Quantisation ×2^32 to match Zinnia.

#[executable]
pub fn main() {
    const Q: u128 = 4_294_967_296;

    // Build 8x6 matrix; col 2 is [1.0, 2.0, ... , 8.0], other cols filler.
    let mut data: Array<u128> = ArrayTrait::new();
    for i in 0_u32..8 {
        for j in 0_u32..6 {
            let q_v: u128 = if j == 2 {
                ((i + 1).into()) * Q
            } else {
                // 0.1 .. 0.6 scaled by 10 in fixed point
                let num: u128 = (j + 1).into();
                (num * Q) / 10
            };
            data.append(q_v);
        };
    };
    let col: i128 = 3;
    let multiply_number: i128 = 2;
    let claimed_q: Array<u128> = array![
        2 * Q, 6 * Q, 12 * Q, 20 * Q, 30 * Q, 42 * Q, 56 * Q, 72 * Q,
    ];

    let col_idx: u32 = (col - 1).try_into().unwrap();
    // multiply_number is an integer; lifted to fp as mn*Q.
    let mn_q: u128 = multiply_number.try_into().unwrap();
    let mn_q: u128 = mn_q * Q;

    // col_vals[i] = a[i][col_idx] * mn (q-scaled mul -> /Q rescale).
    let mut col_vals: Array<u128> = ArrayTrait::new();
    for i in 0_u32..8 {
        let cell: u128 = *data.at(i * 6 + col_idx);
        let scaled: u128 = (cell * mn_q) / Q;
        col_vals.append(scaled);
    };

    // cumsum
    let mut result: Array<u128> = ArrayTrait::new();
    let mut acc: u128 = *col_vals.at(0);
    result.append(acc);
    for k in 1_u32..8 {
        acc = acc + *col_vals.at(k);
        result.append(acc);
    };

    for k in 0_u32..8 {
        assert!(*result.at(k) == *claimed_q.at(k));
    };
}
