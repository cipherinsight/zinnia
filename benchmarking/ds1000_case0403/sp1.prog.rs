// DS-1000 case 403 — SP1 guest program.
// Kernel: result = np.ravel_multi_index(index, dims=dims, order='C').
// Signature: dims Integer[5], a Float[8,9,8,8,8], index Integer[5].
// 'C' order: result = ((((i0*d1 + i1)*d2 + i2)*d3 + i3)*d4 + i4).
// Sample: dims=[8,9,8,8,8], index=[2,3,4,5,6] → 11054.

#![no_main]
sp1_zkvm::entrypoint!(main);

pub fn main() {
    let mut dims: [u64; 5] = [0; 5];
    for k in 0..5 { dims[k] = sp1_zkvm::io::read::<u64>(); }
    let total: usize = 8 * 9 * 8 * 8 * 8;
    let mut s: f64 = 0.0;
    for _ in 0..total {
        let v: f64 = sp1_zkvm::io::read::<f64>();
        s += v * 0.0;
    }
    let mut index: [u64; 5] = [0; 5];
    for k in 0..5 { index[k] = sp1_zkvm::io::read::<u64>(); }
    let claimed: u64 = sp1_zkvm::io::read::<u64>();

    let mut result: u64 = 0;
    for k in 0..5 {
        result = result * dims[k] + index[k];
    }

    assert_eq!(result, claimed);
    assert!(s == 0.0);
}
