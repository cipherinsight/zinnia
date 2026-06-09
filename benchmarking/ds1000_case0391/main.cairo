// DS-1000 case 391 — Cairo baseline.
// Kernel: result = a[:, low:high] on Float[20, 10].
// Sample: low=3, high=7; a[i][j] = (i+1)*10 + (j+1). Result shape (20, 4).
// Quantisation ×2^32.

#[executable]
pub fn main() {
    const Q: u128 = 4_294_967_296;
    let rows: u32 = 20;
    let cols: u32 = 10;
    let low: u32 = 3;
    let high: u32 = 7;
    let width: u32 = high - low;

    let mut a: Array<u128> = ArrayTrait::new();
    for i in 0_u32..rows {
        for j in 0_u32..cols {
            let v: u128 = ((i + 1) * 10 + (j + 1)).into();
            a.append(v * Q);
        };
    };

    let mut expected: Array<u128> = ArrayTrait::new();
    for i in 0_u32..rows {
        for k in 0_u32..width {
            let v: u128 = ((i + 1) * 10 + (low + k + 1)).into();
            expected.append(v * Q);
        };
    };

    for i in 0_u32..rows {
        for k in 0_u32..width {
            let aidx: u32 = i * cols + (low + k);
            let eidx: u32 = i * width + k;
            assert!(*a.at(aidx.try_into().unwrap()) == *expected.at(eidx.try_into().unwrap()));
        };
    };
}
