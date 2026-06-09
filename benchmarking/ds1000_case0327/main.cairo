// DS-1000 case 327 — np.pad on Float[10] with runtime length.
// kernel.py: np.pad(A, (0, length - A.shape[0]), 'constant')
// Driver picks A = [0.1, 0.2, …, 1.0], length = 15 → 5 zero pads.
// Quantisation ×2^32 to match Zinnia. 0.1 = (1 * Q) / 10.

#[executable]
pub fn main() {
    const Q: u128 = 4_294_967_296;
    let length: u32 = 15;
    let n: u32 = 10;
    let mut a: Array<u128> = ArrayTrait::new();
    let mut k: u128 = 1;
    while k <= 10 { a.append((k * Q) / 10); k += 1; };
    let mut out: Array<u128> = ArrayTrait::new();
    for kk in 0_u32..length {
        if kk < n {
            out.append(*a.at(kk.try_into().unwrap()));
        } else {
            out.append(0);
        };
    };
    assert!(*out.at(0) == Q / 10);
    assert!(*out.at(9) == Q);
    assert!(*out.at(10) == 0);
    assert!(*out.at(14) == 0);
}
