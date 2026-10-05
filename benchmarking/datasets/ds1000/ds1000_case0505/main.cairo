// DS-1000 case 505 — Cairo baseline.
// kernel.py:
//     arr = np.take_along_axis(a, b[..., np.newaxis], axis=-1)[..., 0]
//     result = np.sum(a) - np.sum(arr)
//
// Quantisation ×2^32 to match Zinnia's `precision_bits = 32`. Pure gather +
// additive reductions, so Q-scaled equality is exact.
//
// Sample inputs:
//   a[i][j][k] = (i*65 + j*5 + k + 1) / 100
//   b[i][j]    = (i + j) % 5

#[executable]
pub fn main() {
    const Q: u128 = 4_294_967_296;

    let mut a: Array<u128> = ArrayTrait::new();
    for i in 0_u128..13 {
        for j in 0_u128..13 {
            for k in 0_u128..5 {
                let num: u128 = i * 65 + j * 5 + k + 1;
                a.append((num * Q) / 100);
            };
        };
    };

    let mut b: Array<u128> = ArrayTrait::new();
    for i in 0_u128..13 {
        for j in 0_u128..13 {
            b.append((i + j) % 5);
        };
    };

    // Compute claimed = sum(a) - sum(arr) in Q-scale by iterating once.
    let mut sum_a_q: u128 = 0;
    for i in 0_u128..13 {
        for j in 0_u128..13 {
            for k in 0_u128..5 {
                let num: u128 = i * 65 + j * 5 + k + 1;
                sum_a_q = sum_a_q + (num * Q) / 100;
            };
        };
    };
    let mut sum_arr_q: u128 = 0;
    for i in 0_u128..13 {
        for j in 0_u128..13 {
            let bij: u128 = (i + j) % 5;
            let num: u128 = i * 65 + j * 5 + bij + 1;
            sum_arr_q = sum_arr_q + (num * Q) / 100;
        };
    };
    let claimed_q: u128 = sum_a_q - sum_arr_q;

    // Recompute via the gather pipeline and compare.
    let mut arr: Array<u128> = ArrayTrait::new();
    for i in 0_u128..13 {
        for j in 0_u128..13 {
            let bij: u128 = *b.at((i * 13 + j).try_into().unwrap());
            let a_idx: u32 = (i * 65 + j * 5 + bij).try_into().unwrap();
            arr.append(*a.at(a_idx));
        };
    };

    let mut total_a_q: u128 = 0;
    for i in 0_u128..13 {
        for j in 0_u128..13 {
            for k in 0_u128..5 {
                let idx: u32 = (i * 65 + j * 5 + k).try_into().unwrap();
                total_a_q = total_a_q + *a.at(idx);
            };
        };
    };
    let mut total_arr_q: u128 = 0;
    for i in 0_u128..13 {
        for j in 0_u128..13 {
            let idx: u32 = (i * 13 + j).try_into().unwrap();
            total_arr_q = total_arr_q + *arr.at(idx);
        };
    };
    let result_q: u128 = total_a_q - total_arr_q;

    assert!(result_q == claimed_q);
}
