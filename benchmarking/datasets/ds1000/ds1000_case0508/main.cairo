// DS-1000 case 508 — Cairo baseline.
// kernel.py:
//     B = np.argwhere(A)
//     (ystart, xstart), (ystop, xstop) = B.min(0), B.max(0) + 1
//     result = A[ystart:ystop, xstart:xstop]
//
// Output shape is data-dependent in NumPy. The sample input is designed so
// the bbox is a fixed 4×4 region: rows [2..5], cols [3..6]. Integer-only,
// no quantisation needed.

#[executable]
pub fn main() {
    // 10×10 input flattened row-major. Nonzero 4×4 block at rows 2..5, cols 3..6.
    let mut a: Array<u128> = ArrayTrait::new();
    let block: Array<u128> = array![
         1,  2,  3,  4,
         5,  6,  7,  8,
         9, 10, 11, 12,
        13, 14, 15, 16,
    ];
    for i in 0_u32..10 {
        for j in 0_u32..10 {
            if i >= 2 && i <= 5 && j >= 3 && j <= 6 {
                let di: u32 = i - 2;
                let dj: u32 = j - 3;
                a.append(*block.at(di * 4 + dj));
            } else {
                a.append(0);
            };
        };
    };

    let claimed_ystart: u128 = 2;
    let claimed_ystop: u128 = 6;
    let claimed_xstart: u128 = 3;
    let claimed_xstop: u128 = 7;
    let claimed_result: Array<u128> = array![
         1,  2,  3,  4,
         5,  6,  7,  8,
         9, 10, 11, 12,
        13, 14, 15, 16,
    ];

    // B = argwhere(A): traverse all (i, j); when nonzero, update min/max.
    let mut ystart: u128 = 999;
    let mut xstart: u128 = 999;
    let mut ystop_m1: u128 = 0;
    let mut xstop_m1: u128 = 0;
    for i in 0_u32..10 {
        for j in 0_u32..10 {
            let v: u128 = *a.at(i * 10 + j);
            if v != 0 {
                let yi: u128 = i.into();
                let xj: u128 = j.into();
                if yi < ystart { ystart = yi; };
                if xj < xstart { xstart = xj; };
                if yi > ystop_m1 { ystop_m1 = yi; };
                if xj > xstop_m1 { xstop_m1 = xj; };
            };
        };
    };
    let ystop: u128 = ystop_m1 + 1;
    let xstop: u128 = xstop_m1 + 1;

    assert!(ystart == claimed_ystart);
    assert!(ystop == claimed_ystop);
    assert!(xstart == claimed_xstart);
    assert!(xstop == claimed_xstop);

    let ystart_u: u32 = claimed_ystart.try_into().unwrap();
    let ystop_u: u32 = claimed_ystop.try_into().unwrap();
    let xstart_u: u32 = claimed_xstart.try_into().unwrap();
    let xstop_u: u32 = claimed_xstop.try_into().unwrap();
    let mut idx: u32 = 0;
    for i in ystart_u..ystop_u {
        for j in xstart_u..xstop_u {
            let v: u128 = *a.at(i * 10 + j);
            let e: u128 = *claimed_result.at(idx);
            assert!(v == e);
            idx = idx + 1;
        };
    };
}
