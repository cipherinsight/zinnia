// DS-1000 case 0297 — one-hot after subtracting min.
// a = [(i%4)+5 for i in 0..50]. min=5; temp values cycle 0..3; cols=4.

#[executable]
pub fn main() {
    let n: u32 = 50;
    let cols: u32 = 4;
    let mut a: Array<i64> = ArrayTrait::new();
    let mut i: i64 = 0;
    while i < 50 { a.append((i % 4) + 5); i += 1; };
    let mut m: i64 = *a.at(0);
    for k in 1_usize..50 { let v = *a.at(k); if v < m { m = v; }; };
    let mut temp: Array<i64> = ArrayTrait::new();
    for k in 0_usize..50 { temp.append(*a.at(k) - m); };
    let mut b: Array<u64> = ArrayTrait::new();
    for ii in 0_usize..50 {
        let pick: i64 = *temp.at(ii);
        let mut c: i64 = 0;
        while c < 4 {
            if c == pick { b.append(1); } else { b.append(0); };
            c += 1;
        };
    };
    assert!(*b.at(0) == 1);   // temp[0]=0 → col 0
    assert!(*b.at(5) == 1);   // temp[1]=1 → col 1
    assert!(*b.at(11) == 1);  // temp[2]=2 → col 2
    let _ = (n, cols);
}
