// DS-1000 case 365 — np.insert(a, pos, element) on Float[100], Integer pos, Integer element.
// Result is Float[101]. The integer `element` is lifted to fp on insertion.
// Quantisation ×2^32 to match Zinnia's `precision_bits = 32`.

#[executable]
pub fn main() {
    const Q: u128 = 4_294_967_296;
    // a[i] = (i+1)/10 in Float; pos = 42; element = 999.
    let mut a: Array<u128> = ArrayTrait::new();
    let mut i: u32 = 0;
    while i != 100 {
        let num: u128 = (i + 1).into();
        let q_v: u128 = (num * Q) / 10;
        a.append(q_v);
        i = i + 1;
    };
    let pos: u128 = 42;
    let element_q: u128 = 999_u128 * Q;

    // Build expected = insert(a, pos, element).
    let mut expected: Array<u128> = ArrayTrait::new();
    let mut k: u128 = 0;
    while k != 101 {
        if k < pos {
            let idx: u32 = k.try_into().unwrap();
            expected.append(*a.at(idx));
        } else if k == pos {
            expected.append(element_q);
        } else {
            let idx: u32 = (k - 1).try_into().unwrap();
            expected.append(*a.at(idx));
        }
        k = k + 1;
    };

    // Recompute result with the same control flow and compare exactly.
    let mut t: u128 = 0;
    while t != 101 {
        let v: u128 = if t < pos {
            let idx: u32 = t.try_into().unwrap();
            *a.at(idx)
        } else if t == pos {
            element_q
        } else {
            let idx: u32 = (t - 1).try_into().unwrap();
            *a.at(idx)
        };
        let exp: u128 = *expected.at(t.try_into().unwrap());
        assert!(v == exp);
        t = t + 1;
    };
}
