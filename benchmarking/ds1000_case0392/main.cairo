// DS-1000 case 392 — Cairo baseline.
// Kernel: result = a[low:high, :] on Float[20, 10].
// Sample: low=5, high=15; a[i][j] = (i+1)*10 + (j+1).
// Quantisation ×2^32.

#[executable]
pub fn main() {
    const Q: u128 = 4_294_967_296;
    let rows: u32 = 20;
    let cols: u32 = 10;
    let low: u32 = 5;
    let high: u32 = 15;
    let height: u32 = high - low;

    let mut a: Array<u128> = ArrayTrait::new();
    for i in 0_u32..rows {
        for j in 0_u32..cols {
            let v: u128 = ((i + 1) * 10 + (j + 1)).into();
            a.append(v * Q);
        };
    };

    let mut expected: Array<u128> = ArrayTrait::new();
    for i in 0_u32..height {
        for j in 0_u32..cols {
            let v: u128 = ((low + i + 1) * 10 + (j + 1)).into();
            expected.append(v * Q);
        };
    };

    for i in 0_u32..height {
        for j in 0_u32..cols {
            let aidx: u32 = (low + i) * cols + j;
            let eidx: u32 = i * cols + j;
            assert!(*a.at(aidx.try_into().unwrap()) == *expected.at(eidx.try_into().unwrap()));
        };
    };
}
