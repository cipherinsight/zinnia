// DS-1000 case 0354 — np.in1d set-difference between row tuples of A and B.
// A = [[1,2,0],[3,1,2]]; B has A[0] present so the kept row is A[1] = [3,1,2].

#[executable]
pub fn main() {
    let na: u32 = 2;
    let nb: u32 = 7;
    let k: u32 = 3;
    let mut a: Array<i64> = ArrayTrait::new();
    a.append(1); a.append(2); a.append(0);
    a.append(3); a.append(1); a.append(2);
    let mut b: Array<i64> = ArrayTrait::new();
    b.append(1); b.append(2); b.append(0);
    b.append(0); b.append(0); b.append(0);
    b.append(5); b.append(5); b.append(5);
    b.append(4); b.append(4); b.append(4);
    b.append(9); b.append(9); b.append(9);
    b.append(8); b.append(8); b.append(8);
    b.append(7); b.append(7); b.append(7);
    // dims = max(A.max(0), B.max(0)) + 1. max per col across A∪B.
    let mut dims: Array<i64> = ArrayTrait::new();
    for j in 0_usize..3 {
        let mut m: i64 = *a.at(j);
        for i in 1_usize..2 { let v = *a.at(i * 3 + j); if v > m { m = v; }; };
        for i in 0_usize..7 { let v = *b.at(i * 3 + j); if v > m { m = v; }; };
        dims.append(m + 1);
    };
    // ravel a[i] and b[i]
    let mut flat_b: Array<i64> = ArrayTrait::new();
    for i in 0_usize..7 {
        let mut idx: i64 = 0;
        for j in 0_usize..3 { idx = idx * *dims.at(j) + *b.at(i * 3 + j); };
        flat_b.append(idx);
    };
    let mut count: u32 = 0;
    let mut out: Array<i64> = ArrayTrait::new();
    for i in 0_usize..2 {
        let mut idx: i64 = 0;
        for j in 0_usize..3 { idx = idx * *dims.at(j) + *a.at(i * 3 + j); };
        let mut found: bool = false;
        for kk in 0_usize..7 { if *flat_b.at(kk) == idx { found = true; }; };
        if !found {
            count += 1;
            for j in 0_usize..3 { out.append(*a.at(i * 3 + j)); };
        };
    };
    assert!(count == 1);
    assert!(*out.at(0) == 3);
    assert!(*out.at(1) == 1);
    assert!(*out.at(2) == 2);
    let _ = (na, nb, k);
}
