// DS-1000 case 319 — inverse permutation + row gather on Float[8, 6, 8].
// kernel.py:
//   c = empty_like(permutation); c[permutation] = arange(len(permutation))
//   result = a[c, :, :]
// Sample: permutation = [2,0,4,1,5,3,7,6] → c = [1,3,0,5,2,4,7,6].
// a[i][j][k] = 100*(i+1)+10*(j+1)+(k+1). Quantisation ×2^32 (Q).

#[executable]
pub fn main() {
    const Q: u128 = 4_294_967_296;
    // Flat layout: a[i*48 + j*8 + k].
    let mut a: Array<u128> = ArrayTrait::new();
    for i in 0_u32..8 {
        for j in 0_u32..6 {
            for k in 0_u32..8 {
                let v: u128 = (100 * (i + 1) + 10 * (j + 1) + (k + 1)).into();
                a.append(v * Q);
            };
        };
    };
    let mut permutation: Array<u128> = ArrayTrait::new();
    permutation.append(2);
    permutation.append(0);
    permutation.append(4);
    permutation.append(1);
    permutation.append(5);
    permutation.append(3);
    permutation.append(7);
    permutation.append(6);

    // Build inverse c via mutable scratch (length 8).
    let mut c0: u128 = 0;
    let mut c1: u128 = 0;
    let mut c2: u128 = 0;
    let mut c3: u128 = 0;
    let mut c4: u128 = 0;
    let mut c5: u128 = 0;
    let mut c6: u128 = 0;
    let mut c7: u128 = 0;
    let mut kk: u32 = 0;
    while kk < 8 {
        let p: u128 = *permutation.at(kk.try_into().unwrap());
        let kv: u128 = kk.into();
        if p == 0 { c0 = kv; }
        if p == 1 { c1 = kv; }
        if p == 2 { c2 = kv; }
        if p == 3 { c3 = kv; }
        if p == 4 { c4 = kv; }
        if p == 5 { c5 = kv; }
        if p == 6 { c6 = kv; }
        if p == 7 { c7 = kv; }
        kk += 1;
    };
    let mut c: Array<u128> = ArrayTrait::new();
    c.append(c0);
    c.append(c1);
    c.append(c2);
    c.append(c3);
    c.append(c4);
    c.append(c5);
    c.append(c6);
    c.append(c7);

    // result[i][j][k] = a[c[i]][j][k]; assert per-cell.
    let mut i: u32 = 0;
    while i < 8 {
        let src_i: u32 = (*c.at(i.try_into().unwrap())).try_into().unwrap();
        let mut j: u32 = 0;
        while j < 6 {
            let mut k: u32 = 0;
            while k < 8 {
                let a_idx: u32 = src_i * 48 + j * 8 + k;
                let actual: u128 = *a.at(a_idx.try_into().unwrap());
                let expected_raw: u128 = (100 * (src_i + 1) + 10 * (j + 1) + (k + 1)).into();
                let expected: u128 = expected_raw * Q;
                assert!(actual == expected);
                k += 1;
            };
            j += 1;
        };
        i += 1;
    };
}
