// DS-1000 case 0408 — group-by-max with negative shift.
// add = max(index); index[index<0] += add+1; then per-bucket max(a).
// a = [1..101], index = [(i%5)-2]. Expected: [98, 99, 100].

#[executable]
pub fn main() {
    let out_len: u32 = 3;
    let mut a: Array<i64> = ArrayTrait::new();
    let mut i: i64 = 1;
    while i <= 100 { a.append(i); i += 1; };
    let mut index: Array<i64> = ArrayTrait::new();
    let mut k: i64 = 0;
    while k < 100 { index.append((k % 5) - 2); k += 1; };
    let mut add: i64 = *index.at(0);
    for kk in 1_usize..100 { let v = *index.at(kk); if v > add { add = v; }; };
    let mut idx2: Array<i64> = ArrayTrait::new();
    for kk in 0_usize..100 {
        let v: i64 = *index.at(kk);
        if v < 0 { idx2.append(v + add + 1); } else { idx2.append(v); };
    };
    let mut result: Array<i64> = ArrayTrait::new();
    let mut bin: i64 = 0;
    while bin < 3 {
        let mut best: i64 = 0;
        let mut seen: bool = false;
        for ii in 0_usize..100 {
            if *idx2.at(ii) == bin {
                let v: i64 = *a.at(ii);
                if !seen || v > best { best = v; seen = true; };
            };
        };
        result.append(if seen { best } else { 0 });
        bin += 1;
    };
    assert!(*result.at(0) == 98);
    assert!(*result.at(1) == 99);
    assert!(*result.at(2) == 100);
    let _ = out_len;
}
