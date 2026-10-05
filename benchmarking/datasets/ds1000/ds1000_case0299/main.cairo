// DS-1000 case 0299 — one-hot on raveled 2D integers after subtract-min.
// a[i][j] = ((i*20+j) % 4) + 5. raveled len 200. cols=4.

#[executable]
pub fn main() {
    let n: u32 = 200;
    let cols: u32 = 4;
    let mut flat: Array<i64> = ArrayTrait::new();
    let mut k: i64 = 0;
    while k < 200 { flat.append((k % 4) + 5); k += 1; };
    let mut m: i64 = *flat.at(0);
    for kk in 1_usize..200 { let v = *flat.at(kk); if v < m { m = v; }; };
    let mut b: Array<u64> = ArrayTrait::new();
    for ii in 0_usize..200 {
        let pick: i64 = *flat.at(ii) - m;
        let mut c: i64 = 0;
        while c < 4 {
            if c == pick { b.append(1); } else { b.append(0); };
            c += 1;
        };
    };
    assert!(*b.at(0) == 1);
    assert!(*b.at(5) == 1);
    let _ = (n, cols);
}
