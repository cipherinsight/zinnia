// DS-1000 case 389 — Float[6, 3, 4] reshape via (h/nrows, -1, nrows, ncols).swapaxes(1,2).reshape(h,w).
// Sample h=6, w=12. Quantisation ×2^32. Values are 0..72 (integer-valued floats).
// Index mapping (verified): out[out_i][out_j] = a[i0*3 + i2][i1][i3] where
//   out_idx = out_i*w + out_j; i0 = out_idx/36, i1 = (out_idx/12)%3, i2 = (out_idx/4)%3, i3 = out_idx%4.

#[executable]
pub fn main() {
    const Q: u128 = 4_294_967_296;
    let n: u32 = 6;
    let nrows: u32 = 3;
    let ncols: u32 = 4;
    let h: u32 = 6;
    let w: u32 = 12;

    // a[i][j][k] = i*12 + j*4 + k. Flat layout: a_flat[i*nrows*ncols + j*ncols + k].
    let mut a: Array<u128> = ArrayTrait::new();
    let mut i: u32 = 0;
    while i != n {
        let mut j: u32 = 0;
        while j != nrows {
            let mut k: u32 = 0;
            while k != ncols {
                let v: u128 = (i * 12 + j * 4 + k).into();
                a.append(v * Q);
                k = k + 1;
            };
            j = j + 1;
        };
        i = i + 1;
    };

    // Build expected via the same permutation.
    let mut expected: Array<u128> = ArrayTrait::new();
    let mut out_i: u32 = 0;
    while out_i != h {
        let mut out_j: u32 = 0;
        while out_j != w {
            let out_idx: u32 = out_i * w + out_j;
            let i0: u32 = out_idx / 36;
            let i1: u32 = (out_idx / 12) % 3;
            let i2: u32 = (out_idx / 4) % 3;
            let i3: u32 = out_idx % 4;
            let src_i: u32 = i0 * 3 + i2;
            let src_idx: u32 = src_i * 12 + i1 * 4 + i3;
            expected.append(*a.at(src_idx));
            out_j = out_j + 1;
        };
        out_i = out_i + 1;
    };

    // Recompute and verify.
    let mut idx: u32 = 0;
    let mut oi: u32 = 0;
    while oi != h {
        let mut oj: u32 = 0;
        while oj != w {
            let out_idx: u32 = oi * w + oj;
            let i0: u32 = out_idx / 36;
            let i1: u32 = (out_idx / 12) % 3;
            let i2: u32 = (out_idx / 4) % 3;
            let i3: u32 = out_idx % 4;
            let src_i: u32 = i0 * 3 + i2;
            let src_idx: u32 = src_i * 12 + i1 * 4 + i3;
            assert!(*a.at(src_idx) == *expected.at(idx));
            idx = idx + 1;
            oj = oj + 1;
        };
        oi = oi + 1;
    };
}
