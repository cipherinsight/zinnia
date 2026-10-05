// DS-1000 case 398 — SP1 guest program.
// Kernel (effective):
//   B = np.empty(len(A))
//   for k in range(0, len(A)):
//       if k == 0:    B[k] = a * A[k]
//       else:         B[k] = a * A[k] + b * B[k-1]
// The original signature is `Float[Array, "N<=64 M<=64 K<=64"]` but the body
// only uses `len(A)` and indexes `A[k]` once; this is a 1-D recurrence in
// practice (no broadcast over M, K), so we materialise A as a single 1-D row.
// Sample: N=8, a=0.5, b=0.5, A = [4, 2, 1, 3, 2, 0, 1, 4]
//   B = [2, 2, 1.5, 2.25, 2.125, 1.0625, 1.03125, 2.515625]

#![no_main]
sp1_zkvm::entrypoint!(main);

pub fn main() {
    const N: usize = 8;
    let mut a_arr: [f64; N] = [0.0; N];
    for k in 0..N { a_arr[k] = sp1_zkvm::io::read::<f64>(); }
    let a: f64 = sp1_zkvm::io::read::<f64>();
    let b: f64 = sp1_zkvm::io::read::<f64>();
    let mut claimed: [f64; N] = [0.0; N];
    for k in 0..N { claimed[k] = sp1_zkvm::io::read::<f64>(); }

    let mut b_arr: [f64; N] = [0.0; N];
    for k in 0..N {
        if k == 0 {
            b_arr[k] = a * a_arr[k];
        } else {
            b_arr[k] = a * a_arr[k] + b * b_arr[k - 1];
        }
    }

    for k in 0..N {
        assert!((b_arr[k] - claimed[k]).abs() < 0.001);
    }
}
