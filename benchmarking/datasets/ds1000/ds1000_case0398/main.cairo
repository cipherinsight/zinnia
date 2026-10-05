// DS-1000 case 398 — Cairo baseline.
// Kernel (effective):
//   B = np.empty(N); B[0] = a*A[0]; B[k] = a*A[k] + b*B[k-1] for k>=1.
// Sample: N=8, a=b=0.5, A=[4,2,1,3,2,0,1,4] → B=[2, 2, 1.5, 2.25, 2.125,
// 1.0625, 1.03125, 2.515625]. Quantisation ×2^32; qmul = (a*b)/Q.

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

    let mut expected: Array<u128> = ArrayTrait::new();
    expected.append(2 * Q);                            // 2.0
    expected.append(2 * Q);                            // 2.0
    expected.append(1 * Q + Q / 2);                    // 1.5
    expected.append(2 * Q + Q / 4);                    // 2.25
    expected.append(2 * Q + Q / 8);                    // 2.125
    expected.append(1 * Q + Q / 16);                   // 1.0625
    expected.append(1 * Q + Q / 32);                   // 1.03125
    expected.append(2 * Q + Q / 2 + Q / 64);           // 2.515625

    let mut b_arr: Array<u128> = ArrayTrait::new();
    let mut prev: u128 = 0;
    for k in 0_u32..n {
        let ak: u128 = *a_arr.at(k.try_into().unwrap());
        let v: u128 = if k == 0 {
            (a_q * ak) / Q
        } else {
            (a_q * ak) / Q + (b_q * prev) / Q
        };
        b_arr.append(v);
        prev = v;
    };

    for k in 0_u32..n {
        let got: u128 = *b_arr.at(k.try_into().unwrap());
        let want: u128 = *expected.at(k.try_into().unwrap());
        assert!(got == want);
    };
}
