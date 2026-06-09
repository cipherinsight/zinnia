// DS-1000 case 0407 — bincount with negative-shifted accmap.
// kernel.py: add = max(accmap); accmap[accmap<0] += add+1; bincount(accmap, weights=a).
// a = [1..101], accmap = [(i%5)-2 for i in 0..100]. add = 2. Expected: [1010, 2000, 2040].

#[executable]
pub fn main() {
    let n: u32 = 100;
    let out_len: u32 = 3;
    let mut a: Array<i64> = ArrayTrait::new();
    let mut i: i64 = 1;
    while i <= 100 { a.append(i); i += 1; };
    let mut accmap: Array<i64> = ArrayTrait::new();
    let mut k: i64 = 0;
    while k < 100 { accmap.append((k % 5) - 2); k += 1; };
    // Compute add = max(accmap)
    let mut add: i64 = *accmap.at(0);
    for kk in 1_usize..100 {
        let v: i64 = *accmap.at(kk);
        if v > add { add = v; };
    };
    // Shift negatives
    let mut acc2: Array<i64> = ArrayTrait::new();
    for kk in 0_usize..100 {
        let v: i64 = *accmap.at(kk);
        if v < 0 { acc2.append(v + add + 1); } else { acc2.append(v); };
    };
    // Bincount with weights
    let mut result: Array<i64> = ArrayTrait::new();
    let mut bin: i64 = 0;
    while bin < 3 {
        let mut s: i64 = 0;
        for ii in 0_usize..100 {
            if *acc2.at(ii) == bin { s += *a.at(ii); };
        };
        result.append(s);
        bin += 1;
    };
    assert!(*result.at(0) == 1010);
    assert!(*result.at(1) == 2000);
    assert!(*result.at(2) == 2040);
    let _ = (n, out_len);
}
