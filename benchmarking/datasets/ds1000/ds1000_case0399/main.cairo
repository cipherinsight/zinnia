// DS-1000 case 399 — Cairo baseline.
// Kernel (effective):
//   B[0] = a*A[0]; B[1] = a*A[1] + b*B[0];
//   B[k] = a*A[k] + b*B[k-1] + c[k]*B[k-2] for k>=2.
// Sample: N=8, a=b=0.5, c=[0.5; 8], A=[4,2,1,3,2,0,1,4] →
//   B = [2, 2, 2.5, 3.75, 4.125, 3.9375, 4.53125, 6.234375].
// Quantisation ×2^32; qmul = (a*b)/Q.

#[executable]
pub fn main() {
    const Q: u128 = 4_294_967_296;
    let n: u32 = 8;
    let a_q: u128 = Q / 2;
    let b_q: u128 = Q / 2;

    let mut a_arr: Array<u128> = ArrayTrait::new();
    a_arr.append(4 * Q);
    a_arr.append(2 * Q);
    a_arr.append(1 * Q);
    a_arr.append(3 * Q);
    a_arr.append(2 * Q);
    a_arr.append(0 * Q);
    a_arr.append(1 * Q);
    a_arr.append(4 * Q);

    let mut c_arr: Array<u128> = ArrayTrait::new();
    for _ in 0_u32..n { c_arr.append(Q / 2); };

    let mut expected: Array<u128> = ArrayTrait::new();
    expected.append(2 * Q);                                // 2.0
    expected.append(2 * Q);                                // 2.0
    expected.append(2 * Q + Q / 2);                        // 2.5
    expected.append(3 * Q + (3 * Q) / 4);                  // 3.75
    expected.append(4 * Q + Q / 8);                        // 4.125
    expected.append(3 * Q + (15 * Q) / 16);                // 3.9375
    expected.append(4 * Q + (17 * Q) / 32);                // 4.53125
    expected.append(6 * Q + (15 * Q) / 64);                // 6.234375

    let mut b_arr: Array<u128> = ArrayTrait::new();
    let mut prev1: u128 = 0;
    let mut prev2: u128 = 0;
    for k in 0_u32..n {
        let ak: u128 = *a_arr.at(k.try_into().unwrap());
        let ck: u128 = *c_arr.at(k.try_into().unwrap());
        let v: u128 = if k == 0 {
            (a_q * ak) / Q
        } else if k == 1 {
            (a_q * ak) / Q + (b_q * prev1) / Q
        } else {
            (a_q * ak) / Q + (b_q * prev1) / Q + (ck * prev2) / Q
        };
        b_arr.append(v);
        prev2 = prev1;
        prev1 = v;
    };

    for k in 0_u32..n {
        let got: u128 = *b_arr.at(k.try_into().unwrap());
        let want: u128 = *expected.at(k.try_into().unwrap());
        assert!(got == want);
    };
}
