// DS-1000 case 0398, Noir baseline.
// Kernel: B[0] = a*A[0], B[k] = a*A[k] + b*B[k-1] on A: Float[30], a, b: Integer.
// An integer times a Q32 float is exact (no rescale), so the recurrence is
// plain Field arithmetic; nothing is witnessed.
// @outputs: float[30]

global N: u32 = 30;

fn main(A: [Field; 30], a: Field, b: Field) -> pub [Field; 30] {
    // Input domain: every input element is an int64 / a Q32 code in the window.
    noir_hw::check_inputs(A);
    noir_hw::check_input(a);
    noir_hw::check_input(b);
    let mut B = [0; N];
    B[0] = a * A[0];
    for k in 1..N {
        B[k] = a * A[k] + b * B[k - 1];
    }
    B
}
