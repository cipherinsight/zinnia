// DS-1000 case 463 — Cairo baseline.
// Kernel removes adjacent (pre-removal) duplicates and zeros from an Int[20,1] array.
// Inputs are integers — no Q scaling needed. The compact output length is dynamic
// so the claim is a padded buffer plus its actual length.

#[executable]
pub fn main() {
    let a: Array<i128> = array![
        0, 0, 1, 1, 1, 2, 2, 0, 1, 3, 3, 3, 0, 0, 5, 5, 4, 4, 0, 7,
    ];

    let claimed_len: u32 = 7;
    let claimed: Array<i128> = array![1, 2, 1, 3, 5, 4, 7];

    // selection[0] = true; selection[k] = (a[k] != a[k-1]) for k>=1.
    let mut selection: Array<bool> = ArrayTrait::new();
    selection.append(true);
    for k in 1_u32..20 {
        let cur = *a.at(k);
        let prev = *a.at(k - 1);
        selection.append(cur != prev);
    };

    // selection &= (a != 0).
    let mut selection2: Array<bool> = ArrayTrait::new();
    for k in 0_u32..20 {
        let s = *selection.at(k);
        let v = *a.at(k);
        selection2.append(s && v != 0);
    };

    // Gather.
    let mut out: Array<i128> = ArrayTrait::new();
    let mut out_len: u32 = 0;
    for k in 0_u32..20 {
        if *selection2.at(k) {
            out.append(*a.at(k));
            out_len += 1;
        }
    };

    assert!(out_len == claimed_len);
    for k in 0_u32..7 {
        assert!(*out.at(k) == *claimed.at(k));
    };
}
