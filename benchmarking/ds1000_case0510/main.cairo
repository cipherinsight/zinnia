// DS-1000 case 510 — Cairo baseline.
// kernel.py:
//     mask = im == 0
//     rows = np.flatnonzero((~mask).sum(axis=1))
//     cols = np.flatnonzero((~mask).sum(axis=0))
//     result = im[rows.min():rows.max()+1, cols.min():cols.max()+1]
//
// im: Float[10,10]. Quantisation ×2^32. Sample input has a 4×4 block of
// nonzeros at rows 3..6, cols 2..5; the bbox shape is fixed at 4×4. Pure
// gather (no qmul), so Q-scaled equality is exact.

#[executable]
pub fn main() {
    const Q: u128 = 4_294_967_296;

    // 4×4 block of nonzeros: value (k+1)/10 in row-major order within the block.
    let mut im: Array<u128> = ArrayTrait::new();
    for i in 0_u32..10 {
        for j in 0_u32..10 {
            if i >= 3 && i <= 6 && j >= 2 && j <= 5 {
                let di: u32 = i - 3;
                let dj: u32 = j - 2;
                let k: u128 = (di * 4 + dj + 1).into();
                im.append((k * Q) / 10);
            } else {
                im.append(0);
            };
        };
    };

    let claimed_rmin: u128 = 3;
    let claimed_rmax: u128 = 6;
    let claimed_cmin: u128 = 2;
    let claimed_cmax: u128 = 5;
    let mut claimed_result: Array<u128> = ArrayTrait::new();
    for i in 3_u32..7 {
        for j in 2_u32..6 {
            let di: u32 = i - 3;
            let dj: u32 = j - 2;
            let k: u128 = (di * 4 + dj + 1).into();
            claimed_result.append((k * Q) / 10);
        };
    };

    // mask, row_sum (count of nonzero per row), col_sum (per col).
    let mut mask: Array<u128> = ArrayTrait::new();
    for idx in 0_u32..100 {
        let v: u128 = *im.at(idx);
        let m: u128 = if v == 0 { 1 } else { 0 };
        mask.append(m);
    };
    let mut row_sum: Array<u128> = ArrayTrait::new();
    for i in 0_u32..10 {
        let mut s: u128 = 0;
        for j in 0_u32..10 {
            if *mask.at(i * 10 + j) == 0 { s = s + 1; };
        };
        row_sum.append(s);
    };
    let mut col_sum: Array<u128> = ArrayTrait::new();
    for j in 0_u32..10 {
        let mut s: u128 = 0;
        for i in 0_u32..10 {
            if *mask.at(i * 10 + j) == 0 { s = s + 1; };
        };
        col_sum.append(s);
    };

    let mut rmin: u128 = 999;
    let mut rmax: u128 = 0;
    for i in 0_u32..10 {
        if *row_sum.at(i) != 0 {
            let ii: u128 = i.into();
            if ii < rmin { rmin = ii; };
            if ii > rmax { rmax = ii; };
        };
    };
    let mut cmin: u128 = 999;
    let mut cmax: u128 = 0;
    for j in 0_u32..10 {
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

    let rmin_u: u32 = claimed_rmin.try_into().unwrap();
    let rmax_u: u32 = claimed_rmax.try_into().unwrap();
    let cmin_u: u32 = claimed_cmin.try_into().unwrap();
    let cmax_u: u32 = claimed_cmax.try_into().unwrap();
    let mut idx: u32 = 0;
    for i in rmin_u..(rmax_u + 1) {
        for j in cmin_u..(cmax_u + 1) {
            let v: u128 = *im.at(i * 10 + j);
            let e: u128 = *claimed_result.at(idx);
            assert!(v == e);
            idx = idx + 1;
        };
    };
}
