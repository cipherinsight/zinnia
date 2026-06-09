// DS-1000 case 443 — `C = A[np.in1d(A, B)]` on Integer[20] × Integer[3].
// A = 0..20, B = [3, 7, 13]. Expected: [3, 7, 13].

#[executable]
pub fn main() {
    let a: Array<i128> = array![0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19];
    let b: Array<i128> = array![3, 7, 13];
    let claimed: Array<i128> = array![3, 7, 13];

    let mut out: Array<i128> = ArrayTrait::new();
    for i in 0_u32..20 {
        let v: i128 = *a.at(i);
        let mut in_b: u128 = 0;
        for j in 0_u32..3 {
            let eq: u128 = if v == *b.at(j) { 1 } else { 0 };
            in_b = in_b | eq;
        };
        if in_b == 1 {
            out.append(v);
        }
    };

    assert!(out.len() == claimed.len());
    for k in 0_u32..out.len() {
        assert!(*out.at(k) == *claimed.at(k));
    };
}
