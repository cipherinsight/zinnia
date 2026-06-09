// DS-1000 case 0498 — 3D array with 2D pad on first two axes.
// N=2, M=3, K=2; target shape=(4, 5), padded with zeros.
// Driver: arr[i][j][k] = i*100 + j*10 + k (whole numbers).
// Quantisation ×2^32 to match Zinnia.

#[executable]
pub fn main() {
    const Q: u128 = 4_294_967_296;
    let n: u32 = 2;
    let m: u32 = 3;
    let k: u32 = 2;
    let new_n: u32 = 4;
    let new_m: u32 = 5;
    let mut arr: Array<u128> = ArrayTrait::new();
    let mut i: u32 = 0;
    while i < n {
        let mut j: u32 = 0;
        while j < m {
            let mut kk: u32 = 0;
            while kk < k {
                let raw: u128 = (i * 100 + j * 10 + kk).into();
                arr.append(raw * Q);
                kk += 1;
            };
            j += 1;
        };
        i += 1;
    };
    let mut out: Array<u128> = ArrayTrait::new();
    let mut ii: u32 = 0;
    while ii < new_n {
        let mut jj: u32 = 0;
        while jj < new_m {
            let mut kk: u32 = 0;
            while kk < k {
                if ii < n && jj < m {
                    let idx: u32 = ii * (m * k) + jj * k + kk;
                    out.append(*arr.at(idx.try_into().unwrap()));
                } else {
                    out.append(0);
                };
                kk += 1;
            };
            jj += 1;
        };
        ii += 1;
    };
    // arr[0][0][0] = 0 → 0 * Q = 0.
    assert!(*out.at(0) == 0);
    let pad_idx: u32 = 3 * (new_m * k) + 4 * k + 1;
    assert!(*out.at(pad_idx.try_into().unwrap()) == 0);
}
