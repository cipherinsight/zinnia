// DS-1000 case 0443, Noir baseline.
// Kernel: `A[np.in1d(A, B)]` on A: Integer[20], B: Integer[3] (data-dependent length).
// Output: the count of kept values, then the kept values in order, zero-padded
// to 20 (compaction: noir_hw::compact).
// Membership: A[i] is in B iff (A[i] - B[0]) (A[i] - B[1]) (A[i] - B[2]) = 0
// (a field has no zero divisors): one zero test instead of three.
// @outputs: dynamic int[20x1]

use noir_hw::compact;

fn main(A: [Field; 20], B: [Field; 3]) -> pub [Field; 21] {
    // Input domain: every input element is an int64 / a Q32 code in the window.
    noir_hw::check_inputs(A);
    noir_hw::check_inputs(B);
    let mut keep: [Field; 20] = [0; 20];
    let mut w: [Field; 20] = [0; 20];
    let mut count: Field = 0;
    for i in 0..20 {
        keep[i] = ((A[i] - B[0]) * (A[i] - B[1]) * (A[i] - B[2]) == 0) as Field;
        w[i] = keep[i] * A[i];
        count += keep[i];
    }
    let kept = compact(keep, w);
    let mut out: [Field; 21] = [0; 21];
    out[0] = count;
    for j in 0..20 {
        out[j + 1] = kept[j];
    }
    out
}

