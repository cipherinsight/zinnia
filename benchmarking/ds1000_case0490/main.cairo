// DS-1000 case 490 — Cairo baseline.
// In-place threshold replacement per-slab on a 5×4×4 float tensor.
// Inputs Q-scaled (×2^32). Thresholds are ints (no scale).

#[executable]
pub fn main() {
    const Q: u128 = 4_294_967_296;
    const N: u32 = 5;
    const M: u32 = 4;
    const K: u32 = 4;

    // Build arr[i][j][k] = ((i*16 + j*4 + k) % 30) * Q.
    let mut arr: Array<u128> = ArrayTrait::new();
    for i in 0_u32..N {
        for j in 0_u32..M {
            for k in 0_u32..K {
                let raw: u128 = ((i * 16 + j * 4 + k) % 30).into();
                arr.append(raw * Q);
            };
        };
    };
    let n1: Array<u128> = array![5, 10, 15, 0, 8];
    let n2: Array<u128> = array![20, 25, 22, 30, 18];

    let expected_int: Array<u128> = array![
        0, 0, 0, 0, 0, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
        21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 30, 30, 30, 30, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 20, 21, 22,
        23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 5, 6, 7, 8,
        0, 0, 0, 0, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 30, 30,
    ];
    let mut claimed: Array<u128> = ArrayTrait::new();
    for k in 0_u32..80 {
        claimed.append(*expected_int.at(k) * Q);
    };

    // We rebuild arr as new buffer per slab (no mutate in place).
    let mut out: Array<u128> = ArrayTrait::new();
    for i in 0_u32..N {
        let t1_q: u128 = *n1.at(i) * Q;
        let t2_q: u128 = *n2.at(i) * Q;
        // snapshot temp values for this slab.
        let mut temp: Array<u128> = ArrayTrait::new();
        for j in 0_u32..M {
            for k in 0_u32..K {
                let idx: u32 = i * M * K + j * K + k;
                temp.append(*arr.at(idx));
            };
        };
        // produce 16 output cells for this slab.
        for j in 0_u32..M {
            for k in 0_u32..K {
                let local_idx: u32 = j * K + k;
                let v = *temp.at(local_idx);
                let mut next: u128 = v;
                if v < t1_q {
                    next = 0;
                } else if v >= t2_q {
                    next = 30 * Q;
                } else {
                    // v in [t1_q, t2_q): add 5
                    next = v + 5 * Q;
                }
                out.append(next);
            };
        };
    };

    let total: u32 = N * M * K;
    for k in 0_u32..total {
        assert!(*out.at(k) == *claimed.at(k));
    };
}
