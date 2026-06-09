// DS-1000 case 374 — ECDF interpolation. Grades sorted, ys = (i+1)/N, eval clamped.
// Inputs: grades = 1..=50 already sorted, eval = [0, 10, 25, 50, 75].
// Expected: [0, 0.2, 0.5, 1.0, 1.0].

#[executable]
pub fn main() {
    const Q: u128 = 4_294_967_296;
    let n: u32 = 50;
    let m: u32 = 5;

    // grades[i] = (i+1) in Q.
    let mut grades: Array<u128> = ArrayTrait::new();
    let mut i: u32 = 0;
    while i != n {
        let num: u128 = (i + 1).into();
        grades.append(num * Q);
        i = i + 1;
    };

    let eval: Array<i128> = array![0, 10, 25, 50, 75];

    let mut expected: Array<u128> = ArrayTrait::new();
    expected.append(0);
    expected.append((20 * Q) / 100);
    expected.append((50 * Q) / 100);
    expected.append(Q);
    expected.append(Q);

    // xs = sort(grades). grades is already sorted, but mirror by copying.
    let mut xs: Array<u128> = ArrayTrait::new();
    let mut p: u32 = 0;
    while p != n {
        xs.append(*grades.at(p));
        p = p + 1;
    };
    // Insertion sort (mirrors kernel).
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

    // Process each eval point.
    let mut result: Array<u128> = ArrayTrait::new();
    let mut e: u32 = 0;
    while e != m {
        let element: i128 = *eval.at(e);
        // Element is non-negative for our inputs; convert to Q-scaled u128.
        let element_q: u128 = if element < 0 {
            0_u128  // unused: kernel sees element < xs[0] anyway
        } else {
            let abs_e: u128 = element.try_into().unwrap();
            abs_e * Q
        };
        let x0: u128 = *xs.at(0);
        let xN: u128 = *xs.at(n - 1);
        let v: u128 = if element < 0 || element_q < x0 {
            0
        } else if element_q >= xN {
            Q
        } else {
            // argmax of (xs > element) — first index k with xs[k] > element.
            let mut best_idx: u32 = 0;
            let mut found: bool = false;
            let mut kk: u32 = 0;
            while kk != n {
                if !found && *xs.at(kk) > element_q {
                    best_idx = kk;
                    found = true;
                }
                kk = kk + 1;
            };
            *ys.at(best_idx - 1)
        };
        result.append(v);
        e = e + 1;
    };

    let mut q: u32 = 0;
    while q != m {
        assert!(*result.at(q) == *expected.at(q));
        q = q + 1;
    };
}
