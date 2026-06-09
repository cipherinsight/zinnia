// DS-1000 case 485 — Cairo baseline.
// tensordot(A, B, axes=((2),(0))) for A: Float[5,6,3], B: Float[3,3].
// result[i,j,k] = sum_{p=0..3} A[i,j,p] * B[p,k]. Q-scaled (×2^32); qmul = (a*b)/Q.
// A[i,j,p] = (i*18 + j*3 + p) * 0.1; B[p,k] = (p*3 + k + 1) * 0.5.

#[executable]
pub fn main() {
    const Q: u128 = 4_294_967_296;
    let tol_q: u128 = (5 * Q) / 1000; // ±0.005

    // Build A in row-major (i*18 + j*3 + p).
    let mut a: Array<u128> = ArrayTrait::new();
    for i in 0_u32..5 {
        for j in 0_u32..6 {
            for p in 0_u32..3 {
                let raw: u128 = (i * 18 + j * 3 + p).into();
                let q_v: u128 = (raw * Q) / 10;
                a.append(q_v);
            };
        };
    };
    // Build B in row-major (p*3 + k).
    let mut b: Array<u128> = ArrayTrait::new();
    for p in 0_u32..3 {
        for k in 0_u32..3 {
            let raw: u128 = (p * 3 + k + 1).into();
            let q_v: u128 = (raw * Q) / 2;
            b.append(q_v);
        };
    };
    // Expected values in centi-units (×100); convert to Q-scale.
    let mut expected_q: Array<u128> = ArrayTrait::new();
    let exp100: Array<u128> = array![
        90, 105, 120, 270, 330, 390, 450, 555, 660, 630, 780, 930, 810, 1005, 1200, 990, 1230, 1470,
        1170, 1455, 1740, 1350, 1680, 2010, 1530, 1905, 2280, 1710, 2130, 2550, 1890, 2355, 2820, 2070, 2580, 3090,
        2250, 2805, 3360, 2430, 3030, 3630, 2610, 3255, 3900, 2790, 3480, 4170, 2970, 3705, 4440, 3150, 3930, 4710,
        3330, 4155, 4980, 3510, 4380, 5250, 3690, 4605, 5520, 3870, 4830, 5790, 4050, 5055, 6060, 4230, 5280, 6330,
        4410, 5505, 6600, 4590, 5730, 6870, 4770, 5955, 7140, 4950, 6180, 7410, 5130, 6405, 7680, 5310, 6630, 7950,
    ];
    // Build expected_q = exp100[i] * Q / 100.
    for k in 0_u32..90 {
        let v = *exp100.at(k);
        expected_q.append((v * Q) / 100);
    };

    for i in 0_u32..5 {
        for j in 0_u32..6 {
            for k in 0_u32..3 {
                let mut acc_q: u128 = 0;
                for p in 0_u32..3 {
                    let a_idx: u32 = i * 18 + j * 3 + p;
                    let b_idx: u32 = p * 3 + k;
                    let av = *a.at(a_idx);
                    let bv = *b.at(b_idx);
                    acc_q = acc_q + (av * bv) / Q;
                };
                let idx: u32 = i * 18 + j * 3 + k;
                let claim = *expected_q.at(idx);
                assert!(acc_q + tol_q >= claim);
                assert!(acc_q <= claim + tol_q);
            };
        };
    };

}
