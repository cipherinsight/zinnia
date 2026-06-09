// DS-1000 case 0497 — 2D pad with runtime shape and constant element.
// kernel.py: np.pad(a, ((0, shape[0]-42), (0, shape[1]-10)), 'constant', constant_values=element)
// Driver: a[i][j] = (i+1)*0.1 + (j+1)*0.01; shape=(45, 12); element=7.
// Quantisation ×2^32 to match Zinnia: a[i][j] = ((i+1)*10 + (j+1)) * Q / 100; pad = 7 * Q.

#[executable]
pub fn main() {
    const Q: u128 = 4_294_967_296;
    let r: u32 = 42;
    let c: u32 = 10;
    let new_rows: u32 = 45;
    let new_cols: u32 = 12;
    let pad_q: u128 = 7 * Q;
    let mut a: Array<u128> = ArrayTrait::new();
    let mut i: u32 = 0;
    while i < r {
        let mut j: u32 = 0;
        while j < c {
            let num: u128 = ((i + 1) * 10 + (j + 1)).into();
            a.append((num * Q) / 100);
            j += 1;
        };
        i += 1;
    };
    let mut out: Array<u128> = ArrayTrait::new();
    let mut ii: u32 = 0;
    while ii < new_rows {
        let mut jj: u32 = 0;
        while jj < new_cols {
            if ii < r && jj < c {
                let idx: u32 = ii * c + jj;
                out.append(*a.at(idx.try_into().unwrap()));
            } else {
                out.append(pad_q);
            };
            jj += 1;
        };
        ii += 1;
    };
    // Spot-check corners.
    assert!(*out.at(0) == (11_u128 * Q) / 100);  // a[0][0] = 0.11
    assert!(*out.at(9) == (20_u128 * Q) / 100);  // a[0][9] = 0.20
    let last_orig: u32 = 41 * new_cols + 9;       // a[41][9] = 4.20
    assert!(*out.at(last_orig.try_into().unwrap()) == (420_u128 * Q) / 100);
    let pad_idx: u32 = 41 * new_cols + 10;
    assert!(*out.at(pad_idx.try_into().unwrap()) == pad_q);
    let last_pad: u32 = 44 * new_cols + 11;
    assert!(*out.at(last_pad.try_into().unwrap()) == pad_q);
}
