// DS-1000 case 381 — np.argsort(a)[::-1][:len(a)] on Float[100].
// Result is argsort indices reversed (descending order). Quantisation ×2^32.
// Insertion sort over indices; Cairo arrays are append-only so each swap
// rebuilds the index array.

#[executable]
pub fn main() {
    const Q: u128 = 4_294_967_296;
    let n: u32 = 100;

    // a[i] = ((i * 37 + 13) % 1000) + 0.5 * i — deterministic but jumbled.
    let mut a: Array<u128> = ArrayTrait::new();
    let mut i: u32 = 0;
    while i != n {
        let base_int: u128 = ((i * 37 + 13) % 1000).into();
        let base_q: u128 = base_int * Q;
        let i_u128: u128 = i.into();
        let half_q: u128 = (i_u128 * Q) / 2;
        a.append(base_q + half_q);
        i = i + 1;
    };

    // Precomputed expected = argsort(a)[::-1].
    // Computed externally — keep as the claimed witness.
    let expected: Array<u128> = build_expected();

    // Insertion sort on indices.
    let mut idx: Array<u32> = ArrayTrait::new();
    let mut p: u32 = 0;
    while p != n {
        idx.append(p);
        p = p + 1;
    };
    let mut s: u32 = 1;
    while s != n {
        let key_idx: u32 = *idx.at(s);
        let key_val: u128 = *a.at(key_idx);
        let mut j: u32 = s;
        let mut stop: bool = false;
        while !stop {
            if j == 0 {
                stop = true;
            } else if *a.at(*idx.at(j - 1)) <= key_val {
                stop = true;
            } else {
                // shift idx[j-1] → idx[j]
                let mut tmp: Array<u32> = ArrayTrait::new();
                let mut t: u32 = 0;
                while t != n {
                    if t == j {
                        tmp.append(*idx.at(j - 1));
                    } else {
                        tmp.append(*idx.at(t));
                    }
                    t = t + 1;
                };
                idx = tmp;
                j = j - 1;
            }
        };
        let mut tmp2: Array<u32> = ArrayTrait::new();
        let mut t2: u32 = 0;
        while t2 != n {
            if t2 == j {
                tmp2.append(key_idx);
            } else {
                tmp2.append(*idx.at(t2));
            }
            t2 = t2 + 1;
        };
        idx = tmp2;
        s = s + 1;
    };

    // rev[i] = idx[n-1-i]
    let mut rev: Array<u128> = ArrayTrait::new();
    let mut r: u32 = 0;
    while r != n {
        let v: u128 = (*idx.at(n - 1 - r)).into();
        rev.append(v);
        r = r + 1;
    };

    let mut q: u32 = 0;
    while q != n {
        assert!(*rev.at(q) == *expected.at(q));
        q = q + 1;
    };
}

fn build_expected() -> Array<u128> {
    // argsort descending of a[i] = ((i*37+13)%1000) + 0.5*i for i in 0..100.
    // Precomputed externally; embedded here as literals.
    array![
        80, 53, 26, 79, 52, 25, 78, 51, 24, 77, 50, 23, 76, 49, 22, 75, 48, 21, 74, 47,
        20, 73, 46, 19, 99, 72, 45, 98, 18, 71, 44, 97, 17, 70, 43, 96, 16, 69, 42, 95,
        15, 68, 41, 94, 14, 67, 40, 13, 93, 66, 39, 92, 12, 65, 38, 91, 11, 64, 37, 10,
        90, 63, 36, 89, 9, 62, 35, 88, 8, 61, 34, 7, 87, 60, 33, 86, 6, 59, 32, 5,
        85, 58, 31, 84, 4, 57, 30, 83, 3, 56, 29, 82, 2, 55, 28, 81, 1, 54, 27, 0,
    ]
}
