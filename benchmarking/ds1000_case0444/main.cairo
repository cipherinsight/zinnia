// DS-1000 case 444 — A[(A > B[0] & A < B[1]) | (A > B[1] & A < B[2])].
// A = 0..20, B = [3, 10, 17]. Expected: (3,10) ∪ (10,17).

#[executable]
pub fn main() {
    let a: Array<i128> = array![0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19];
    let b: Array<i128> = array![3, 10, 17];
    let claimed: Array<i128> = array![4,5,6,7,8,9,11,12,13,14,15,16];

    let b0: i128 = *b.at(0);
    let b1: i128 = *b.at(1);
    let b2: i128 = *b.at(2);

    let mut out: Array<i128> = ArrayTrait::new();
    for i in 0_u32..20 {
        let v: i128 = *a.at(i);
        let lo1: u128 = if v > b0 { 1 } else { 0 };
        let hi1: u128 = if v < b1 { 1 } else { 0 };
        let lo2: u128 = if v > b1 { 1 } else { 0 };
        let hi2: u128 = if v < b2 { 1 } else { 0 };
        let band1: u128 = lo1 & hi1;
        let band2: u128 = lo2 & hi2;
        let mask: u128 = band1 | band2;
        if mask == 1 {
            out.append(v);
        }
    };

    assert!(out.len() == claimed.len());
    for k in 0_u32..out.len() {
        assert!(*out.at(k) == *claimed.at(k));
    };
}
