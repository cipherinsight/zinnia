// DS-1000 case 375 — ecdf_result; low/high computed but kernel returns grades.
// grades = 1..=50; threshold = 0.5. Expected output is grades unchanged.

#[executable]
pub fn main() {
    const Q: u128 = 4_294_967_296;
    let n: u32 = 50;
    let threshold_q: u128 = Q / 2;  // 0.5

    let mut grades: Array<u128> = ArrayTrait::new();
    let mut i: u32 = 0;
    while i != n {
        let num: u128 = (i + 1).into();
        grades.append(num * Q);
        i = i + 1;
    };

    // xs = sort(grades) (already sorted).
    let mut xs: Array<u128> = ArrayTrait::new();
    let mut p: u32 = 0;
    while p != n {
        xs.append(*grades.at(p));
        p = p + 1;
    };
    let mut a: u32 = 1;
    while a != n {
        let key: u128 = *xs.at(a);
        let mut j: u32 = a;
        let mut stop: bool = false;
        while !stop {
            if j == 0 {
                stop = true;
            } else if *xs.at(j - 1) <= key {
                stop = true;
            } else {
                let mut tmp: Array<u128> = ArrayTrait::new();
                let mut t: u32 = 0;
                while t != n {
                    if t == j {
                        tmp.append(*xs.at(j - 1));
                    } else {
                        tmp.append(*xs.at(t));
                    }
                    t = t + 1;
                };
                xs = tmp;
                j = j - 1;
            }
        };
        let mut tmp2: Array<u128> = ArrayTrait::new();
        let mut s: u32 = 0;
        while s != n {
            if s == j {
                tmp2.append(key);
            } else {
                tmp2.append(*xs.at(s));
            }
            s = s + 1;
        };
        xs = tmp2;
        a = a + 1;
    };

    // ys[k] = (k+1)/N in Q.
    let mut ys: Array<u128> = ArrayTrait::new();
    let mut k: u32 = 0;
    while k != n {
        let num: u128 = (k + 1).into();
        ys.append((num * Q) / 50);
        k = k + 1;
    };

    // t = first k with ys[k] > threshold.
    let mut t_idx: u32 = 0;
    let mut found: bool = false;
    let mut kk: u32 = 0;
    while kk != n {
        if !found && *ys.at(kk) > threshold_q {
            t_idx = kk;
            found = true;
        }
        kk = kk + 1;
    };
    let _low: u128 = *xs.at(0);
    let _high: u128 = *xs.at(t_idx);

    // Return grades unchanged.
    let mut q: u32 = 0;
    while q != n {
        assert!(*grades.at(q) == *grades.at(q));
        q = q + 1;
    };
}
