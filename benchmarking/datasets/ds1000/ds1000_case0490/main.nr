// DS-1000 case 0490, Noir baseline.
// Kernel: for each row a of arr: Float[5, 50] with t1 = n1[r], t2 = n2[r]
// (n1, n2: Integer[5]): temp = a.copy(); a[temp < t1] = 0;
// a[temp >= t2] = 30; a[(temp >= t1) & (temp < t2)] += 5.
// In assignment order a cell x ends as 30 if x >= t2 (the second write wins,
// also when t2 < t1), else 0 if x < t1, else x + 5: two comparisons per cell,
// against the row's thresholds converted to floats once per row (t * 2^32 must
// lie in the window: -2^31 <= t < 2^31).
// @outputs: float[250]

use noir_hw::{from_int, window, SCALE};

fn main(arr: [Field; 250], n1: [Field; 5], n2: [Field; 5]) -> pub [Field; 250] {
    let mut result: [Field; 250] = [0; 250];
    for r in 0..5 {
        let t1 = window(from_int(n1[r]));
        let t2 = window(from_int(n2[r]));
        for c in 0..50 {
            let x = arr[r * 50 + c];
            let w = window(x);
            result[r * 50 + c] = if w >= t2 {
                30 * SCALE
            } else if w < t1 {
                0
            } else {
                x + 5 * SCALE
            };
        }
    }
    result
}
