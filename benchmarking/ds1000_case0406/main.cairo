// DS-1000 case 0406 — group-by-max over index.
// a = [1..101], index = [i%5 for i in 0..100]. Output: [96, 97, 98, 99, 100].

#[executable]
pub fn main() {
    let out_len: u32 = 5;
    let mut a: Array<i64> = ArrayTrait::new();
    let mut i: i64 = 1;
    while i <= 100 { a.append(i); i += 1; };
    let mut index: Array<i64> = ArrayTrait::new();
    let mut k: i64 = 0;
    while k < 100 { index.append(k % 5); k += 1; };
    let mut result: Array<i64> = ArrayTrait::new();
    let mut bin: i64 = 0;
    while bin < 5 {
        let mut best: i64 = 0;
        let mut seen: bool = false;
        for ii in 0_usize..100 {
            if *index.at(ii) == bin {
                let v: i64 = *a.at(ii);
                if !seen || v > best { best = v; seen = true; };
            };
        };
        result.append(if seen { best } else { 0 });
        bin += 1;
    };
    assert!(*result.at(0) == 96);
    assert!(*result.at(1) == 97);
    assert!(*result.at(2) == 98);
    assert!(*result.at(3) == 99);
    assert!(*result.at(4) == 100);
    let _ = out_len;
}
