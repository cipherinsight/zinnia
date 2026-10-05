// DS-1000 case 0471 — Z[..., -1:] (last-element slice on final axis).
// Concrete dims N=2, M=3, K=4. Z[i][j][k] = i*100 + j*10 + k.
// Quantisation ×2^32 to match Zinnia's `precision_bits = 32` default.

#[executable]
pub fn main() {
    const Q: i128 = 4_294_967_296;
    let n: u32 = 2;
    let m: u32 = 3;
    let k: u32 = 4;
    let mut z: Array<i128> = ArrayTrait::new();
    let mut i: u32 = 0;
    while i < n {
        let mut j: u32 = 0;
        while j < m {
            let mut kk: u32 = 0;
            while kk < k {
                z.append((i * 100 + j * 10 + kk).into() * Q);
                kk += 1;
            };
            j += 1;
        };
        i += 1;
    };
    let mut out: Array<i128> = ArrayTrait::new();
    let mut ii: u32 = 0;
    while ii < n {
        let mut jj: u32 = 0;
        while jj < m {
            let idx: u32 = ii * (m * k) + jj * k + (k - 1);
            out.append(*z.at(idx.try_into().unwrap()));
            jj += 1;
        };
        ii += 1;
    };
    let expect_unq: Array<i128> = array![3, 13, 23, 103, 113, 123];
    for kk in 0_usize..6 { assert!(*out.at(kk) == *expect_unq.at(kk) * Q); };
}
