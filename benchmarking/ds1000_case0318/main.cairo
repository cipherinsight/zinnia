// DS-1000 case 318 — inverse permutation + column gather on Float[8, 6].
// kernel.py:
//   c = empty_like(permutation); c[permutation] = arange(len(permutation))
//   a = a[:, c]
// Sample: permutation = [2, 0, 4, 1, 5, 3] → c = [1, 3, 0, 5, 2, 4].
// a[i][j] = (i+1)*10 + (j+1). Quantisation ×2^32 (Q).

#[executable]
pub fn main() {
    const Q: u128 = 4_294_967_296;
    let mut a: Array<u128> = ArrayTrait::new();
    for i in 0_u32..8 {
        for j in 0_u32..6 {
            let v: u128 = ((i + 1) * 10 + (j + 1)).into();
            a.append(v * Q);
        };
    };
    let mut permutation: Array<u128> = ArrayTrait::new();
    permutation.append(2);
    permutation.append(0);
    permutation.append(4);
    permutation.append(1);
    permutation.append(5);
    permutation.append(3);

    // Build inverse c via mutable scratch.
    let mut c0: u128 = 0;
    let mut c1: u128 = 0;
    let mut c2: u128 = 0;
    let mut c3: u128 = 0;
    let mut c4: u128 = 0;
    let mut c5: u128 = 0;
    let mut k: u32 = 0;
    while k < 6 {
        let p: u128 = *permutation.at(k.try_into().unwrap());
        let kv: u128 = k.into();
        if p == 0 { c0 = kv; }
        if p == 1 { c1 = kv; }
        if p == 2 { c2 = kv; }
        if p == 3 { c3 = kv; }
        if p == 4 { c4 = kv; }
        if p == 5 { c5 = kv; }
        k += 1;
    };
    let mut c: Array<u128> = ArrayTrait::new();
    c.append(c0);
    c.append(c1);
    c.append(c2);
    c.append(c3);
    c.append(c4);
    c.append(c5);

    // Build expected output and assert per-cell.
    let mut i: u32 = 0;
    while i < 8 {
        let mut j: u32 = 0;
        while j < 6 {
            let src_col: u32 = (*c.at(j.try_into().unwrap())).try_into().unwrap();
            let a_idx: u32 = i * 6 + src_col;
            let actual: u128 = *a.at(a_idx.try_into().unwrap());
            let expected_raw: u128 = ((i + 1) * 10 + (src_col + 1)).into();
            let expected: u128 = expected_raw * Q;
            assert!(actual == expected);
            j += 1;
        };
        i += 1;
    };
}
