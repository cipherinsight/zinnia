// DS-1000 case 347 — a[row-1, :] /= divide_number; result = multiply.reduce.
// Float[8, 6] with runtime row, divide_number. Output: scalar float.

#[executable]
pub fn main() {
    const Q: u128 = 4_294_967_296;

    // Row 2 -> [2, 4, 2, 2, 1, 1]; others filler 0.5.
    let mut data: Array<u128> = ArrayTrait::new();
    for i in 0_u32..8 {
        for j in 0_u32..6 {
            let q_v: u128 = if i == 2 {
                let vals: Array<u128> = array![2, 4, 2, 2, 1, 1];
                (*vals.at(j)) * Q
            } else {
                Q / 2
            };
            data.append(q_v);
        };
    };
    let row: i128 = 3;
    let divide_number: i128 = 2;
    // Expected: 1 * 2 * 1 * 1 * 0.5 * 0.5 = 0.25
    let claimed_q: u128 = Q / 4;

    let row_idx: u32 = (row - 1).try_into().unwrap();
    let dn_q: u128 = divide_number.try_into().unwrap();
    let dn_q: u128 = dn_q * Q;

    // q-div: (a * Q) / b in same Q scale.
    let mut row_vals: Array<u128> = ArrayTrait::new();
    for j in 0_u32..6 {
        let cell: u128 = *data.at(row_idx * 6 + j);
        let divided: u128 = (cell * Q) / dn_q;
        row_vals.append(divided);
    };

    // product reduce
    let mut acc: u128 = *row_vals.at(0);
    for k in 1_u32..6 {
        acc = (acc * (*row_vals.at(k))) / Q;
    };

    // ±0.005 tolerance (no abs in Cairo).
    let tol_q: u128 = (5 * Q) / 1000;
    assert!(acc + tol_q >= claimed_q);
    assert!(acc <= claimed_q + tol_q);
}
