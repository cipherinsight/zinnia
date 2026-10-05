// DS-1000 case 462 — keep a[i] iff (i == 0 or a[i] != a[i-1]) AND a[i] != 0.

#[executable]
pub fn main() {
    let a: Array<i128> = array![0, 0, 1, 1, 2, 3, 3, 0, 4, 4, 5, 0, 6, 7, 7, 8, 0, 9, 9, 0];
    let claimed: Array<i128> = array![1, 2, 3, 4, 5, 6, 7, 8, 9];

    let mut selection: Array<u128> = ArrayTrait::new();
    selection.append(1);
    for i in 1_u32..20 {
        let s: u128 = if *a.at(i) != *a.at(i - 1) { 1 } else { 0 };
        selection.append(s);
    };

    let mut selection2: Array<u128> = ArrayTrait::new();
    for i in 0_u32..20 {
        let nz: u128 = if *a.at(i) != 0 { 1 } else { 0 };
        selection2.append(*selection.at(i) & nz);
    };

    let mut out: Array<i128> = ArrayTrait::new();
    for i in 0_u32..20 {
        if *selection2.at(i) == 1 {
            out.append(*a.at(i));
        }
    };

    assert!(out.len() == claimed.len());
    for k in 0_u32..out.len() {
        assert!(*out.at(k) == *claimed.at(k));
    };
}
