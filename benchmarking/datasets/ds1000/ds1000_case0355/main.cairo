// DS-1000 case 355 — argsort axis-0 on Float[3,3,3], gather into b.
// For each (j, k), argsort a[:, j, k] (3 elements) -> indices; c[i,j,k] = b[idx[i,j,k], j, k].

#[executable]
pub fn main() {
    const Q: u128 = 4_294_967_296;

    // Build a (3x3x3) Q-scaled u128. a[0]=3, a[1]=1, a[2]=2 along i, same on every (j,k).
    let mut a_arr: Array<u128> = ArrayTrait::new();
    let i_vals: Array<u128> = array![3, 1, 2];
    for i in 0_u32..3 {
        for _j in 0_u32..3 {
            for _k in 0_u32..3 {
                a_arr.append((*i_vals.at(i)) * Q);
            };
        };
    };
    // b[i][j][k] = 100*i + 10*j + k
    let mut b_arr: Array<u128> = ArrayTrait::new();
    for i in 0_u32..3 {
        for j in 0_u32..3 {
            for k in 0_u32..3 {
                let v: u128 = (100 * i + 10 * j + k).into();
                b_arr.append(v);
            };
        };
    };
    // Expected with perm = [1,2,0] along i.
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

    // sort_indices[i][j][k]: argsort along axis 0 (3 elements).
    let mut sort_indices: Array<u32> = ArrayTrait::new();
    // First flatten by allocating 27 zeros; we'll set entries in row-major (i, j, k).
    // We compute per-(j,k) the permutation, then push i=0,1,2.
    let mut perm_buf: Array<u32> = ArrayTrait::new();  // 9 entries (j*3+k) each holding 3 elements packed
    // We'll instead compute and fill sort_indices in i-major order using a helper array of (j,k) perms.
    let mut jk_perms: Array<u32> = ArrayTrait::new();   // 9 * 3 = 27 entries, (j*3+k)*3 + i'
    for j in 0_u32..3 {
        for k in 0_u32..3 {
            // Read column [a[0][j][k], a[1][j][k], a[2][j][k]]
            let v0: u128 = *a_arr.at(0 * 9 + j * 3 + k);
            let v1: u128 = *a_arr.at(1 * 9 + j * 3 + k);
            let v2: u128 = *a_arr.at(2 * 9 + j * 3 + k);
            let mut i0: u32 = 0;
            let mut i1: u32 = 1;
            let mut i2: u32 = 2;
            let mut a0 = v0;
            let mut a1 = v1;
            let mut a2 = v2;
            // 3-element sort network.
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
            // suppress unused warnings
            let _ = a0; let _ = a1; let _ = a2;
            jk_perms.append(i0);
            jk_perms.append(i1);
            jk_perms.append(i2);
        };
    };
    // sort_indices in (i, j, k) order
    for i in 0_u32..3 {
        for j in 0_u32..3 {
            for k in 0_u32..3 {
                let idx: u32 = *jk_perms.at((j * 3 + k) * 3 + i);
                sort_indices.append(idx);
            };
        };
    };

    // c[i][j][k] = b[sort_indices[i,j,k]][j][k]
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
