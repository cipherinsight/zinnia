// DS-1000 case 0399, Noir baseline.
// Kernel: B[0] = a*A[0], B[1] = a*A[1] + b*B[0],
// B[k] = a*A[k] + b*B[k-1] + c*B[k-2] on A: Float[30], a, b, c: Integer.
// An integer times a Q32 float is exact (no rescale), so the recurrence is
// plain Field arithmetic; nothing is witnessed.
// @outputs: float[30]

global N: u32 = 30;

fn main(A: [Field; 30], a: Field, b: Field, c: Field) -> pub [Field; 30] {
    // Input domain: every input element is an int64 / a Q32 code in the window.
    noir_hw::check_inputs(A);
    noir_hw::check_input(a);
    noir_hw::check_input(b);
    noir_hw::check_input(c);
    let mut B = [0; N];
    B[0] = a * A[0];
    B[1] = a * A[1] + b * B[0];
    for k in 2..N {
        B[k] = a * A[k] + b * B[k - 1] + c * B[k - 2];
    }
    B
}
