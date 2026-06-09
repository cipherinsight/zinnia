// DS-1000 case 373 — ecdf_result: sort(grades); ys = arange(1, N+1)/N.
// Returns ys. Quantisation ×2^32; sort still happens to mirror the kernel even
// though only the length affects ys.

#[executable]
pub fn main() {
    const Q: u128 = 4_294_967_296;
    let n: u32 = 50;

    // grades[i] = (50 - i) * 1.5 → strictly decreasing.
    let mut grades: Array<u128> = ArrayTrait::new();
    let mut i: u32 = 0;
    while i != n {
        let g: u128 = ((50 - i) * 15).into();  // 1.5 * 10 = 15, will / 10 next
        grades.append((g * Q) / 10);
        i = i + 1;
    };

    // expected[i] = (i+1)/50 in Q-scale.
    let mut expected: Array<u128> = ArrayTrait::new();
    let mut k: u32 = 0;
    while k != n {
        let num: u128 = (k + 1).into();
        expected.append((num * Q) / 50);
        k = k + 1;
    };

    // Insertion sort grades into xs (values aren't used downstream — only length is —
    // but we run the sort to mirror the kernel's control flow).
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
        // Find insertion point: largest j with xs[j-1] > key.
        let mut stop: bool = false;
        while !stop {
            if j == 0 {
                stop = true;
            } else if *xs.at(j - 1) <= key {
                stop = true;
            } else {
                // shift xs[j-1] → xs[j]; rebuild xs.
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

    // ys[i] = (i+1)/50 in Q.
    let mut ys: Array<u128> = ArrayTrait::new();
    let mut m: u32 = 0;
    while m != n {
        let num: u128 = (m + 1).into();
        ys.append((num * Q) / 50);
        m = m + 1;
    };

    let mut q: u32 = 0;
    while q != n {
        assert!(*ys.at(q) == *expected.at(q));
        q = q + 1;
    };
}
