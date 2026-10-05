// DS-1000 case 507 — Cairo baseline.
// kernel.py:
//     mask = im == 0
//     rows = np.flatnonzero((~mask).sum(axis=1))
//     cols = np.flatnonzero((~mask).sum(axis=0))
//     result = im[rows.min():rows.max()+1, cols.min():cols.max()+1]
//
// Output shape is data-dependent in NumPy. The sample input is designed so
// the bbox is a fixed 3×3 (rows 1..3, cols 2..4). Integer-only kernel — no
// quantisation needed. Indices and slice values use exact equality.

#[executable]
pub fn main() {
    // 5×6 input flattened row-major.
    let im: Array<u128> = array![
        0, 0, 0, 0, 0, 0,
        0, 0, 1, 2, 3, 0,
        0, 0, 4, 5, 6, 0,
        0, 0, 7, 8, 9, 0,
        0, 0, 0, 0, 0, 0,
    ];
    let claimed_rmin: u128 = 1;
    let claimed_rmax: u128 = 3;
    let claimed_cmin: u128 = 2;
    let claimed_cmax: u128 = 4;
    let claimed_result: Array<u128> = array![1, 2, 3, 4, 5, 6, 7, 8, 9];

    // mask[i][j] = (im[i][j] == 0)
    let mut mask: Array<u128> = ArrayTrait::new();
    for idx in 0_u32..30 {
        let v: u128 = *im.at(idx);
        let m: u128 = if v == 0 { 1 } else { 0 };
        mask.append(m);
    };

    // (~mask).sum(axis=1): per-row count of nonzero entries.
    let mut row_sum: Array<u128> = ArrayTrait::new();
    for i in 0_u32..5 {
        let mut s: u128 = 0;
        for j in 0_u32..6 {
            let m: u128 = *mask.at(i * 6 + j);
            if m == 0 { s = s + 1; };
        };
        row_sum.append(s);
    };
    // (~mask).sum(axis=0): per-col count of nonzero entries.
    let mut col_sum: Array<u128> = ArrayTrait::new();
    for j in 0_u32..6 {
        let mut s: u128 = 0;
        for i in 0_u32..5 {
            let m: u128 = *mask.at(i * 6 + j);
            if m == 0 { s = s + 1; };
        };
        col_sum.append(s);
    };

    // rmin/rmax: min/max of indices with non-zero row_sum.
    let mut rmin: u128 = 999;
    let mut rmax: u128 = 0;
    for i in 0_u32..5 {
        if *row_sum.at(i) != 0 {
            let ii: u128 = i.into();
            if ii < rmin { rmin = ii; };
            if ii > rmax { rmax = ii; };
        };
    };
    let mut cmin: u128 = 999;
    let mut cmax: u128 = 0;
    for j in 0_u32..6 {
        if *col_sum.at(j) != 0 {
            let jj: u128 = j.into();
            if jj < cmin { cmin = jj; };
            if jj > cmax { cmax = jj; };
        };
    };

    assert!(rmin == claimed_rmin);
    assert!(rmax == claimed_rmax);
    assert!(cmin == claimed_cmin);
    assert!(cmax == claimed_cmax);

    // Slice [rmin..=rmax, cmin..=cmax] and compare to claimed_result.
    let mut idx: u32 = 0;
    let rmin_u: u32 = claimed_rmin.try_into().unwrap();
    let rmax_u: u32 = claimed_rmax.try_into().unwrap();
    let cmin_u: u32 = claimed_cmin.try_into().unwrap();
    let cmax_u: u32 = claimed_cmax.try_into().unwrap();
    for i in rmin_u..(rmax_u + 1) {
        for j in cmin_u..(cmax_u + 1) {
            let v: u128 = *im.at(i * 6 + j);
            let e: u128 = *claimed_result.at(idx);
            assert!(v == e);
            idx = idx + 1;
        };
    };
}
