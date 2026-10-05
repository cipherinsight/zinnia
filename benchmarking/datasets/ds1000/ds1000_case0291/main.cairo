// DS-1000 case 291 — `a.shape` on Integer[2, 5, 5, 2].
// kernel.py: result = a.shape  → (2, 5, 5, 2).
// 100 integer cells are witnessed but unused; only the 4-tuple shape is exposed.

#[executable]
pub fn main() {
    let mut a: Array<i64> = ArrayTrait::new();
    let mut i: i64 = 0;
    while i < 100 {
        a.append(i + 1);
        i += 1;
    };

    let d0: u64 = 2;
    let d1: u64 = 5;
    let d2: u64 = 5;
    let d3: u64 = 2;

    let claim_d0: u64 = 2;
    let claim_d1: u64 = 5;
    let claim_d2: u64 = 5;
    let claim_d3: u64 = 2;

    assert!(d0 == claim_d0);
    assert!(d1 == claim_d1);
    assert!(d2 == claim_d2);
    assert!(d3 == claim_d3);
    let _ = a.len();
}
