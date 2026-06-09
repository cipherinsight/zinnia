// DS-1000 case 0495 — 2D zero pad to runtime shape on Float[41,13] → (45, 15).
// Driver: a[i][j] = (i+1)*0.1 + (j+1)*0.01. Quantisation ×2^32 to match Zinnia.

#[executable]
pub fn main() {
    const Q: u128 = 4_294_967_296;
    let r: u32 = 41;
    let c: u32 = 13;
    let new_rows: u32 = 45;
    let new_cols: u32 = 15;
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
                out.append(0);
            };
            jj += 1;
        };
        ii += 1;
    };
    assert!(*out.at(0) == (11_u128 * Q) / 100);
    let last_orig: u32 = 40 * new_cols + 12;
    assert!(*out.at(last_orig.try_into().unwrap()) == (((40 + 1) * 10 + (12 + 1)) * Q) / 100);
    let pad: u32 = 44 * new_cols + 14;
    assert!(*out.at(pad.try_into().unwrap()) == 0);
}
