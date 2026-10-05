// DS-1000 case 0296 — one-hot encoding.
// kernel.py: b = np.zeros((a.size, a.max()+1)); b[np.arange(a.size), a] = 1.
// a = [i%4 for i in 0..50]. max=3 → 50×4 one-hot.

#[executable]
pub fn main() {
    let n: u32 = 50;
    let cols: u32 = 4;
    let mut a: Array<u64> = ArrayTrait::new();
    let mut i: u64 = 0;
    while i < 50 { a.append(i % 4); i += 1; };
    let mut b: Array<u64> = ArrayTrait::new();
    for ii in 0_usize..50 {
        let pick: u64 = *a.at(ii);
        let mut c: u64 = 0;
        while c < 4 {
            if c == pick { b.append(1); } else { b.append(0); };
            c += 1;
        };
    };
    // Spot-check a few cells.
    assert!(*b.at(0) == 1);  // a[0]=0, b[0][0]=1
    assert!(*b.at(1) == 0);  // b[0][1]=0
    assert!(*b.at(5) == 1);  // a[1]=1, b[1][1] = idx 5
    assert!(*b.at(8 * 4 + 0) == 1); // a[8]=0
    let _ = (n, cols);
}
