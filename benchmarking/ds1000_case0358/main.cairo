// DS-1000 case 358 — argsort of a.sum(axis=(1,2)), gather b[index, :, :].

#[executable]
pub fn main() {
    const Q: u128 = 4_294_967_296;

    // a[0]=3, a[1]=1, a[2]=2 broadcast across (j,k). Sums: 27, 9, 18 -> argsort [1,2,0]
    let mut a_arr: Array<u128> = ArrayTrait::new();
    let i_vals: Array<u128> = array![3, 1, 2];
    for i in 0_u32..3 {
        for _j in 0_u32..3 {
            for _k in 0_u32..3 {
                a_arr.append((*i_vals.at(i)) * Q);
            };
        };
    };
    let mut b_arr: Array<u128> = ArrayTrait::new();
    for i in 0_u32..3 {
        for j in 0_u32..3 {
            for k in 0_u32..3 {
                let v: u128 = (100 * i + 10 * j + k).into();
                b_arr.append(v);
            };
        };
    };
    let mut claimed: Array<u128> = ArrayTrait::new();
    let perm: Array<u32> = array![1, 2, 0];
    for i in 0_u32..3 {
        for j in 0_u32..3 {
            for k in 0_u32..3 {
                let pi: u32 = *perm.at(i);
                let v: u128 = (100 * pi + 10 * j + k).into();
                claimed.append(v);
            };
        };
    };

    // sums along axis (1, 2)
    let mut sums: Array<u128> = ArrayTrait::new();
    for i in 0_u32..3 {
        let mut s: u128 = 0;
        for j in 0_u32..3 {
            for k in 0_u32..3 {
                s = s + *a_arr.at(i * 9 + j * 3 + k);
            };
        };
        sums.append(s);
    };

    // argsort 3 elements
    let mut i0: u32 = 0;
    let mut i1: u32 = 1;
    let mut i2: u32 = 2;
    let mut s0: u128 = *sums.at(0);
    let mut s1: u128 = *sums.at(1);
    let mut s2: u128 = *sums.at(2);
    if s0 > s1 {
        let tv = s0; s0 = s1; s1 = tv;
        let ti = i0; i0 = i1; i1 = ti;
    }
    if s1 > s2 {
        let tv = s1; s1 = s2; s2 = tv;
        let ti = i1; i1 = i2; i2 = ti;
    }
    if s0 > s1 {
        let tv = s0; s0 = s1; s1 = tv;
        let ti = i0; i0 = i1; i1 = ti;
    }
    let _ = s0; let _ = s1; let _ = s2;
    let idx_perm: Array<u32> = array![i0, i1, i2];

    // result[i][j][k] = b[idx[i]][j][k]
    let mut result: Array<u128> = ArrayTrait::new();
    for i in 0_u32..3 {
        let pi: u32 = *idx_perm.at(i);
        for j in 0_u32..3 {
            for k in 0_u32..3 {
                result.append(*b_arr.at(pi * 9 + j * 3 + k));
            };
        };
    };

    for k in 0_u32..27 {
        assert!(*result.at(k) == *claimed.at(k));
    };
}
