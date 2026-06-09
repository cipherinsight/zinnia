// DS-1000 case 382 — np.argsort(a) on Float[100]. Result is ascending-sorting indices.
// Quantisation ×2^32; insertion sort on an index array (Cairo immutable arrays).

#[executable]
pub fn main() {
    const Q: u128 = 4_294_967_296;
    let n: u32 = 100;

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

    let expected: Array<u128> = build_expected();

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

    let mut q: u32 = 0;
    while q != n {
        let got: u128 = (*idx.at(q)).into();
        assert!(got == *expected.at(q));
        q = q + 1;
    };
}

fn build_expected() -> Array<u128> {
    array![
        0, 27, 54, 1, 81, 28, 55, 2, 82, 29, 56, 3, 83, 30, 57, 4, 84, 31, 58, 85,
        5, 32, 59, 6, 86, 33, 60, 87, 7, 34, 61, 8, 88, 35, 62, 9, 89, 36, 63, 90,
        10, 37, 64, 11, 91, 38, 65, 12, 92, 39, 66, 93, 13, 40, 67, 14, 94, 41, 68, 15,
        95, 42, 69, 16, 96, 43, 70, 17, 97, 44, 71, 18, 98, 45, 72, 99, 19, 46, 73, 20,
        47, 74, 21, 48, 75, 22, 49, 76, 23, 50, 77, 24, 51, 78, 25, 52, 79, 26, 53, 80,
    ]
}
