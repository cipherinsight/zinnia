// DS-1000 case 317 — np.array(a) on Float[10, 20, 5] — identity copy of 1000 cells.
// kernel.py: result = np.array(a) — returns the input as-is.
// Quantisation ×2^32 (Q). Sample: a[i] = (i+1)/1000.

#[executable]
pub fn main() {
    const Q: u128 = 4_294_967_296;
    let mut a: Array<u128> = ArrayTrait::new();
    let mut i: u128 = 0;
    while i < 1000 {
        // sample(i) = (i+1)/1000 → Q-scaled: ((i+1) * Q) / 1000.
        a.append(((i + 1) * Q) / 1000);
        i += 1;
    };

    // Identity copy.
    let mut out: Array<u128> = ArrayTrait::new();
    let mut k: u32 = 0;
    while k < 1000 {
        out.append(*a.at(k.try_into().unwrap()));
        k += 1;
    };

    // Verify against claim (same formula).
    let mut k2: u32 = 0;
    while k2 < 1000 {
        let kk: u128 = k2.into();
        let claim_v: u128 = ((kk + 1) * Q) / 1000;
        assert!(*out.at(k2.try_into().unwrap()) == claim_v);
        k2 += 1;
    };
}
