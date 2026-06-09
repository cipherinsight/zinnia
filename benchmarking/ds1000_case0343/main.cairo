// DS-1000 case 343 — Cairo baseline.
// Kernel: iterate `X.T.flat` and append — column-major flatten of Integer[5, 6].

#[executable]
pub fn main() {
    let mut x: Array<i128> = ArrayTrait::new();
    for i in 0_u32..5 {
        for j in 0_u32..6 {
            let v: i128 = ((i + 1) * 10 + (j + 1)).into();
            x.append(v);
        };
    };
    // Expected output: column-major order.
    let mut claimed: Array<i128> = ArrayTrait::new();
    for j in 0_u32..6 {
        for i in 0_u32..5 {
            claimed.append(*x.at(i * 6 + j));
        };
    };

    let mut result: Array<i128> = ArrayTrait::new();
    for j in 0_u32..6 {
        for i in 0_u32..5 {
            result.append(*x.at(i * 6 + j));
        };
    };

    for k in 0_u32..30 {
        assert!(*result.at(k) == *claimed.at(k));
    };
}
