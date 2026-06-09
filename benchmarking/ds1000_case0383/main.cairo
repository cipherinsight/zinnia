// DS-1000 case 383 — np.argsort(a)[::-1][:N] on Float[100], with N=10.
// Top 10 indices by descending value.

#[executable]
pub fn main() {
    const Q: u128 = 4_294_967_296;
    let n: u32 = 100;
    let big_n: u32 = 10;

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

    let expected: Array<u128> = array![80, 53, 26, 79, 52, 25, 78, 51, 24, 77];

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

    // rev[:N]
    let mut q: u32 = 0;
    while q != big_n {
        let v: u128 = (*idx.at(n - 1 - q)).into();
        assert!(v == *expected.at(q));
        q = q + 1;
    };
}
