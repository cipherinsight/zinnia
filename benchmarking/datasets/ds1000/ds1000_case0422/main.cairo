// DS-1000 case 422 — Cairo baseline.
// Kernel: c = hstack((b[1:], b));  result = np.correlate(a, c, mode='valid')
// len(a)=50, len(b)=50, len(c)=99 → len(result)=50.
// Formula: out[k] = sum_{n=0..49} a[n] * c[n + 49 - k].
// Sample: a[i] = i+1; b = δ_{10}. Expected = [11..50, 1..10].
// Quantisation ×2^32 to match Zinnia. (b is a one-hot, so only one product
// per k contributes; the sum is an integer, exactly representable in Q-scale.)

#[executable]
pub fn main() {
    const Q: u128 = 4_294_967_296;

    let mut a: Array<u128> = ArrayTrait::new();
    for i in 1_u128..51 { a.append(i * Q); };

    let mut b: Array<u128> = ArrayTrait::new();
    for i in 0_u32..50 {
        if i == 10 { b.append(Q); } else { b.append(0); };
    };

    // c = hstack((b[1:], b)).
    let mut c: Array<u128> = ArrayTrait::new();
    for i in 1_u32..50 {
        c.append(*b.at(i.try_into().unwrap()));
    };
    for i in 0_u32..50 {
        c.append(*b.at(i.try_into().unwrap()));
    };

    // Expected: [11..50, 1..10].
    let mut expected: Array<u128> = ArrayTrait::new();
    for k in 11_u128..51 { expected.append(k * Q); };
    for k in 1_u128..11  { expected.append(k * Q); };

    for k in 0_u32..50 {
        let mut s: u128 = 0;
        for n in 0_u32..50 {
            let cidx: u32 = n + 49 - k;
            let an: u128 = *a.at(n.try_into().unwrap());
            let cn: u128 = *c.at(cidx.try_into().unwrap());
            // qmul = (an * cn) / Q.
            s = s + (an * cn) / Q;
        };
        assert!(s == *expected.at(k.try_into().unwrap()));
    };
}
