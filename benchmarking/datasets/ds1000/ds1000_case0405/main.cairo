// DS-1000 case 405 — weighted bincount.
// kernel.py: np.bincount(accmap, weights=a).
// a = [1..101], accmap = [i%5 for i in 0..100]. Output: [970, 990, 1010, 1030, 1050].

#[executable]
pub fn main() {
    let n: u32 = 100;
    let out_len: u32 = 5;
    let mut a: Array<i64> = ArrayTrait::new();
    let mut i: i64 = 1;
    while i <= 100 { a.append(i); i += 1; };
    let mut accmap: Array<i64> = ArrayTrait::new();
    let mut k: i64 = 0;
    while k < 100 { accmap.append(k % 5); k += 1; };
    let mut result: Array<i64> = ArrayTrait::new();
    let mut j: u32 = 0;
    while j < out_len { result.append(0); j += 1; };
    let mut new_result: Array<i64> = ArrayTrait::new();
    // Re-bucket: for each output bin k, sum a[i] where accmap[i] == k.
    let mut bin: i64 = 0;
    while bin < 5 {
        let mut s: i64 = 0;
        for ii in 0_usize..100 {
            if *accmap.at(ii) == bin { s += *a.at(ii); };
        };
        new_result.append(s);
        bin += 1;
    };
    assert!(*new_result.at(0) == 970);
    assert!(*new_result.at(1) == 990);
    assert!(*new_result.at(2) == 1010);
    assert!(*new_result.at(3) == 1030);
    assert!(*new_result.at(4) == 1050);
    let _ = result;
    let _ = n;
}
