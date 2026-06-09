// DS-1000 case 481 — Cairo baseline.
// idx_list = ((x == a) & (y == b)); result = idx_list.nonzero()[0]
// Dynamic-length output: padded buffer + actual length.

#[executable]
pub fn main() {
    let x: Array<i128> = array![1, 5, 5, 2, 5, 5, 3, 5, 4];
    let y: Array<i128> = array![2, 7, 7, 3, 8, 7, 4, 7, 5];
    let a: i128 = 5;
    let b: i128 = 7;
    let claimed_len: u32 = 4;
    let claimed: Array<u32> = array![1, 2, 5, 7];

    let mut idx_list: Array<bool> = ArrayTrait::new();
    for k in 0_u32..9 {
        idx_list.append(*x.at(k) == a && *y.at(k) == b);
    };

    let mut out: Array<u32> = ArrayTrait::new();
    for k in 0_u32..9 {
        if *idx_list.at(k) {
            out.append(k);
        }
    };

    assert!(out.len() == claimed_len);
    for k in 0_u32..claimed_len {
        assert!(*out.at(k) == *claimed.at(k));
    };
}
