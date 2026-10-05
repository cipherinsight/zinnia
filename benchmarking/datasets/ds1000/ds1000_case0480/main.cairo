// DS-1000 case 480 — Cairo baseline.
// kernel.py: result = ((x==a) & (y==b)).argmax(); if x[result]!=a or y[result]!=b: result = -1.
// Integers, no Q scaling.

#[executable]
pub fn main() {
    let x: Array<i128> = array![10, 20, 30, 40, 50, 60, 70, 80];
    let y: Array<i128> = array![1, 2, 3, 4, 5, 6, 7, 8];
    let a: i128 = 50;
    let b: i128 = 5;
    let claimed: i128 = 4;

    let mut mask: Array<u128> = ArrayTrait::new();
    for k in 0_u32..8 {
        let xa: bool = *x.at(k) == a;
        let yb: bool = *y.at(k) == b;
        mask.append(if xa && yb { 1 } else { 0 });
    };

    let mut best_val: u128 = *mask.at(0);
    let mut best_idx: i128 = 0;
    for k in 1_u32..8 {
        let v = *mask.at(k);
        if v > best_val {
            best_val = v;
            best_idx = k.into();
        }
    };

    let best_u: u32 = best_idx.try_into().unwrap();
    let result: i128 = if *x.at(best_u) != a || *y.at(best_u) != b { -1 } else { best_idx };
    assert!(result == claimed);
}
