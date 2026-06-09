// DS-1000 case 0411 — fill_crop on Float[5,5] with runtime crop region.
// kernel.py: fill_crop(a, [low, low], np.zeros((H-L, H-L)))
// a[i][j] = i*5 + j + 1. low=2, high=5 → 3×3 = [[13,14,15],[18,19,20],[23,24,25]].
// Quantisation ×2^32 to match Zinnia's `precision_bits = 32` default.

#[executable]
pub fn main() {
    const Q: i128 = 4_294_967_296;
    let low: i64 = 2;
    let high: i64 = 5;
    let r: i64 = 5;
    let mut a: Array<i128> = ArrayTrait::new();
    let mut i: i64 = 0;
    while i < 5 {
        let mut j: i64 = 0;
        while j < 5 {
            a.append((i * 5 + j + 1).into() * Q);
            j += 1;
        };
        i += 1;
    };
    let h: u32 = (high - low).try_into().unwrap();
    let mut out: Array<i128> = ArrayTrait::new();
    for ii in 0_u32..h {
        for jj in 0_u32..h {
            let si: i64 = low + ii.into();
            let sj: i64 = low + jj.into();
            if si >= 0 && si < r && sj >= 0 && sj < r {
                let su: u32 = si.try_into().unwrap();
                let sju: u32 = sj.try_into().unwrap();
                let idx: u32 = su * 5 + sju;
                out.append(*a.at(idx.try_into().unwrap()));
            } else {
                out.append(0);
            };
        };
    };
    let exp_unq: Array<i128> = array![13, 14, 15, 18, 19, 20, 23, 24, 25];
    for k in 0_usize..9 {
        assert!(*out.at(k) == *exp_unq.at(k) * Q);
    };
}
