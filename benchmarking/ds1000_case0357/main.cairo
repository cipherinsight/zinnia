// DS-1000 case 357 — argsort axis-0 reversed (descending), gather into b.

#[executable]
pub fn main() {
    const Q: u128 = 4_294_967_296;

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
    // Reversed perm: [0, 2, 1]
    let mut claimed: Array<u128> = ArrayTrait::new();
    let perm: Array<u32> = array![0, 2, 1];
    for i in 0_u32..3 {
        for j in 0_u32..3 {
            for k in 0_u32..3 {
                let pi: u32 = *perm.at(i);
                let v: u128 = (100 * pi + 10 * j + k).into();
                claimed.append(v);
            };
        };
    };

    // ascending argsort, then reverse along axis 0
    let mut jk_perms: Array<u32> = ArrayTrait::new();
    for j in 0_u32..3 {
        for k in 0_u32..3 {
            let v0: u128 = *a_arr.at(0 * 9 + j * 3 + k);
            let v1: u128 = *a_arr.at(1 * 9 + j * 3 + k);
            let v2: u128 = *a_arr.at(2 * 9 + j * 3 + k);
            let mut i0: u32 = 0;
            let mut i1: u32 = 1;
            let mut i2: u32 = 2;
            let mut a0 = v0;
            let mut a1 = v1;
            let mut a2 = v2;
            if a0 > a1 {
                let tv = a0; a0 = a1; a1 = tv;
                let ti = i0; i0 = i1; i1 = ti;
            }
            if a1 > a2 {
                let tv = a1; a1 = a2; a2 = tv;
                let ti = i1; i1 = i2; i2 = ti;
            }
            if a0 > a1 {
                let tv = a0; a0 = a1; a1 = tv;
                let ti = i0; i0 = i1; i1 = ti;
            }
            let _ = a0; let _ = a1; let _ = a2;
            jk_perms.append(i0);
            jk_perms.append(i1);
            jk_perms.append(i2);
        };
    };
    // sort_indices = asort[::-1, :, :] -> i -> 2-i lookup
    let mut sort_indices: Array<u32> = ArrayTrait::new();
    for i in 0_u32..3 {
        for j in 0_u32..3 {
            for k in 0_u32..3 {
                let idx: u32 = *jk_perms.at((j * 3 + k) * 3 + (2 - i));
                sort_indices.append(idx);
            };
        };
    };

    let mut c_arr: Array<u128> = ArrayTrait::new();
    for i in 0_u32..3 {
        for j in 0_u32..3 {
            for k in 0_u32..3 {
                let si: u32 = *sort_indices.at(i * 9 + j * 3 + k);
                c_arr.append(*b_arr.at(si * 9 + j * 3 + k));
            };
        };
    };

    for k in 0_u32..27 {
        assert!(*c_arr.at(k) == *claimed.at(k));
    };
}
