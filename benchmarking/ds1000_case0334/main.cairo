// DS-1000 case 334 — Cairo baseline. Elementwise mean of three Float[50].
//
// Driver: a[i] = 0.30 + 0.01·i, b[i] = 0.30 + 0.02·i, c[i] = 0.30 + 0.03·i,
// so mean = 0.30 + 0.02·i. Quantisation ×2^32 to match Zinnia. By symmetry
// of the arithmetic progression (a + c = 2·b), the per-cell truncation
// errors cancel and `sum_q == 3 · mean_q` holds exactly even after the
// individual `/ 100` rescales.

#[executable]
pub fn main() {
    const Q: u128 = 4_294_967_296;
    let mut a: Array<u128> = ArrayTrait::new();
    let mut b: Array<u128> = ArrayTrait::new();
    let mut c: Array<u128> = ArrayTrait::new();
    let mut expected: Array<u128> = ArrayTrait::new();
    for i in 0_u128..50 {
        a.append(((30 + i) * Q) / 100);
        b.append(((30 + 2 * i) * Q) / 100);
        c.append(((30 + 3 * i) * Q) / 100);
        expected.append(((30 + 2 * i) * Q) / 100);
    };

    for i in 0_u32..50 {
        let sum = *a.at(i) + *b.at(i) + *c.at(i);
        assert!(sum == 3 * *expected.at(i));
    };
}
