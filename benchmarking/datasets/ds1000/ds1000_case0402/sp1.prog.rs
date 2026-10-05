// DS-1000 case 402 — SP1 guest program.
// Kernel: result = np.ravel_multi_index(index, dims=dims, order='F').
// Signature: dims Integer[5], a Float[8,9,8,8,8], index Integer[5].
// `a` is unused by the math (np.ravel_multi_index only uses index + dims) but
// we read it through to mirror Zinnia's 5_184 ReadFloatIR ops.
// 'F' order: result = sum_k index[k] * prod_{j<k} dims[j].
// Sample: dims=[8,9,8,8,8], index=[2,3,4,5,6] → result = 30842.

#![no_main]
sp1_zkvm::entrypoint!(main);

pub fn main() {
    let mut dims: [u64; 5] = [0; 5];
    for k in 0..5 { dims[k] = sp1_zkvm::io::read::<u64>(); }
    // a is 8*9*8*8*8 = 36864 floats — read and discard to mirror signature.
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
    let mut stride: u64 = 1;
    for k in 0..5 {
        result += index[k] * stride;
        stride *= dims[k];
    }

    assert_eq!(result, claimed);
    assert!(s == 0.0);
}
