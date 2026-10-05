// DS-1000 case 473 — Cairo baseline.
// any(np.array_equal(c, x) for x in CNTS) — c is shape (4,1,2), each x is scalar.
// np.array_equal returns False for shape mismatch, so the answer is always False.

#[executable]
pub fn main() {
    let c: Array<i128> = array![1, 2, 3, 4, 5, 6, 7, 8];
    let cnts: Array<i128> = array![1, 5, 7];
    let claimed: u128 = 0; // False

    let mut result: bool = false;
    for x_idx in 0_u32..3 {
        let mut all_match = true;
        for k in 0_u32..8 {
            if *c.at(k) != *cnts.at(x_idx) {
                all_match = false;
            }
        };
        let shape_equal = false;
        result = result || (all_match && shape_equal);
    };

    let computed: u128 = if result { 1 } else { 0 };
    assert!(computed == claimed);
}
