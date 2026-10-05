// DS-1000 case 300 — np.percentile on Float[20] with integer p.
// kernel.py: result = np.percentile(a, p)
// Sample: a = [1..20] (Q-scaled), p = 25; expected = 5.75 = 23 · Q / 4.
// Quantisation ×2^32 (Q) to match Zinnia's `precision_bits = 32`.

#[executable]
pub fn main() {
    const Q: i128 = 4_294_967_296;
    let mut a: Array<i128> = ArrayTrait::new();
    let mut i: i128 = 0;
    while i < 20 {
        a.append((i + 1) * Q);
        i += 1;
    };
    let p: i128 = 25;
    let claim: i128 = (23 * Q) / 4;  // 5.75 · Q

    // Copy into a sortable array, selection sort ascending.
    let mut s: Array<i128> = ArrayTrait::new();
    let mut k: usize = 0;
    while k < 20 {
        s.append(*a.at(k));
        k += 1;
    };
    let mut taken: Array<bool> = ArrayTrait::new();
    let mut tt: usize = 0;
    while tt < 20 {
        taken.append(false);
        tt += 1;
    };
    let mut sorted: Array<i128> = ArrayTrait::new();
    let mut w: usize = 0;
    while w < 20 {
        let mut mi: usize = 20;
        let mut mv: i128 = 0;
        let mut j: usize = 0;
        while j < 20 {
            if !*taken.at(j) {
                let v = *s.at(j);
                if mi == 20 || v < mv {
                    mi = j;
                    mv = v;
                }
            }
            j += 1;
        };
        sorted.append(mv);
        let mut new_taken: Array<bool> = ArrayTrait::new();
        let mut q: usize = 0;
        while q < 20 {
            if q == mi {
                new_taken.append(true);
            } else {
                new_taken.append(*taken.at(q));
            }
            q += 1;
        };
        taken = new_taken;
        w += 1;
    };

    // rank_q = (n-1) · p · Q / 100. n = 20. rank in Q-scale.
    let n_minus_1: i128 = 19;
    let rank_q: i128 = (n_minus_1 * p * Q) / 100;
    let lo_q: i128 = rank_q / Q;                   // integer floor
    let lo: usize = lo_q.try_into().unwrap();
    let hi: usize = if lo + 1 < 20 { lo + 1 } else { lo };
    let frac_q: i128 = rank_q - lo_q * Q;          // fractional part, Q-scaled
    let diff_q: i128 = *sorted.at(hi) - *sorted.at(lo);
    let result_q: i128 = *sorted.at(lo) + (frac_q * diff_q) / Q;

    assert!(result_q == claim);
}
