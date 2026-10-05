// DS-1000 case 399 — SP1 guest program.
// Kernel (effective):
//   B = np.empty(len(A))
//   for k in range(0, len(A)):
//       if k == 0:    B[k] = a*A[k]
//       elif k == 1:  B[k] = a*A[k] + b*B[k-1]
//       else:         B[k] = a*A[k] + b*B[k-1] + c[k]*B[k-2]
// As in 0398, the M,K trailing dims are not used by the recurrence body, so
// we materialise A and c as 1-D rows of length N.
// Sample: N=8, a=b=0.5, c=[0.5; 8], A=[4,2,1,3,2,0,1,4] →
//   B = [2, 2, 2.5, 3.75, 4.125, 3.9375, 4.53125, 6.234375].

#![no_main]
sp1_zkvm::entrypoint!(main);

pub fn main() {
    const N: usize = 8;
    let mut a_arr: [f64; N] = [0.0; N];
    for k in 0..N { a_arr[k] = sp1_zkvm::io::read::<f64>(); }
    let a: f64 = sp1_zkvm::io::read::<f64>();
    let b: f64 = sp1_zkvm::io::read::<f64>();
    let mut c_arr: [f64; N] = [0.0; N];
    for k in 0..N { c_arr[k] = sp1_zkvm::io::read::<f64>(); }
    let mut claimed: [f64; N] = [0.0; N];
    for k in 0..N { claimed[k] = sp1_zkvm::io::read::<f64>(); }

    let mut b_arr: [f64; N] = [0.0; N];
    for k in 0..N {
        if k == 0 {
            b_arr[k] = a * a_arr[k];
        } else if k == 1 {
            b_arr[k] = a * a_arr[k] + b * b_arr[k - 1];
        } else {
            b_arr[k] = a * a_arr[k] + b * b_arr[k - 1] + c_arr[k] * b_arr[k - 2];
        }
    }

    for k in 0..N {
        assert!((b_arr[k] - claimed[k]).abs() < 0.001);
    }
}
