// DS-1000 case 353 — rows of A not contained in B (via ravel_multi_index trick).
// A: Integer[2,3], B: Integer[7,3]. Output: variable rows of length 3.
// Integers; no fractional rescale.

#[executable]
pub fn main() {
    // A: 2x3 row-major. B: 7x3 row-major.
    let a: Array<u128> = array![
        1, 2, 3,
        4, 5, 6,
    ];
    let b: Array<u128> = array![
        0, 0, 0,
        1, 1, 1,
        1, 2, 3,
        2, 2, 2,
        3, 3, 3,
        7, 8, 9,
        9, 9, 9,
    ];
    // Expected output: 1 row [4, 5, 6].
    let claim_n: u32 = 1;
    let claimed: Array<u128> = array![4, 5, 6];

    // dims = elementwise max(A.max(0), B.max(0)) + 1
    let mut dims: Array<u128> = ArrayTrait::new();
    for k in 0_u32..3 {
        let mut m: u128 = *a.at(0 * 3 + k);
        for i in 1_u32..2 {
            let v: u128 = *a.at(i * 3 + k);
            if v > m { m = v; }
        };
        for i in 0_u32..7 {
            let v: u128 = *b.at(i * 3 + k);
            if v > m { m = v; }
        };
        dims.append(m + 1);
    };

    let d1: u128 = *dims.at(1);
    let d2: u128 = *dims.at(2);

    let mut a_idx: Array<u128> = ArrayTrait::new();
    for r in 0_u32..2 {
        let v: u128 = ((*a.at(r * 3 + 0)) * d1 + *a.at(r * 3 + 1)) * d2 + *a.at(r * 3 + 2);
        a_idx.append(v);
    };
    let mut b_idx: Array<u128> = ArrayTrait::new();
    for r in 0_u32..7 {
        let v: u128 = ((*b.at(r * 3 + 0)) * d1 + *b.at(r * 3 + 1)) * d2 + *b.at(r * 3 + 2);
        b_idx.append(v);
    };

    // in1d(a_idx, b_idx); ~in_b filter.
    let mut in_b: Array<bool> = ArrayTrait::new();
    for r in 0_u32..2 {
        let av: u128 = *a_idx.at(r);
        let mut found = false;
        for s in 0_u32..7 {
            if av == *b_idx.at(s) { found = true; }
        };
        in_b.append(found);
    };

    let mut out_rows: u32 = 0;
    let mut out: Array<u128> = ArrayTrait::new();
    for r in 0_u32..2 {
        if !*in_b.at(r) {
            out.append(*a.at(r * 3 + 0));
            out.append(*a.at(r * 3 + 1));
            out.append(*a.at(r * 3 + 2));
            out_rows = out_rows + 1;
        }
    };

    assert!(out_rows == claim_n);
    for k in 0_u32..(claim_n * 3) {
        assert!(*out.at(k) == *claimed.at(k));
    };
}
