// pythran_fibo_seq__fibo (Pythran fibo_seq), Noir baseline.
// Kernel: n steps of (a, b) = (a + b, a) from (1, 1), for a private n with
// @requires(0 <= n <= 1024); returns a, the Fibonacci number F(n + 2) (a
// Python int).
// The steps depend on the input only through their number, so the loop is
// (a, b) = M^n (1, 1) with M = [[1, 1], [1, 0]], evaluated by the bits of n:
// M^n is the product of the M^(2^k) (constants) whose bit is set, each
// applied as v + bit * (M^(2^k) v - v), exact in the field.
// A result outside int64 has no proof (NPB_BASELINES.md section 5.7), and
// F(n + 2) < 2^63 exactly for n <= MAX_N = 90, so the statement admits
// 0 <= n <= 90: n is decomposed into N_BITS bits (which also implies the
// input domain, an int64) and compared with MAX_N bit by bit. Every n the
// precondition admits with an int64 result is covered.
// @outputs: int[1]

// The largest n whose result F(n + 2) fits int64 (fixed by the number format,
// not a size), and its bits.
global MAX_N: u32 = 90;
global N_BITS: u32 = 7;

fn main(n: Field) -> pub [Field; 1] {
    let bits: [bool; N_BITS] = n.to_le_bits();
    // n <= MAX_N, from the most significant bit down: `above` is set at the
    // first bit where n has a 1 and MAX_N a 0 while the higher bits agree.
    let mut above: Field = 0;
    let mut agree: Field = 1;
    for t in 0..N_BITS {
        let k = N_BITS - 1 - t;
        let b = bits[k] as Field;
        if (MAX_N >> k) & 1 == 1 {
            agree *= b;
        } else {
            above += agree * b;
            agree *= 1 - b;
        }
    }
    assert(above == 0, "n > MAX_N: the result leaves int64");
    // v = M^n (1, 1), one power of two of M per bit of n.
    let mut v: [Field; 2] = [1, 1];
    let mut m: [[Field; 2]; 2] = [[1, 1], [1, 0]];
    for k in 0..N_BITS {
        let b = bits[k] as Field;
        let mv = [m[0][0] * v[0] + m[0][1] * v[1], m[1][0] * v[0] + m[1][1] * v[1]];
        v = [v[0] + b * (mv[0] - v[0]), v[1] + b * (mv[1] - v[1])];
        m = [
            [m[0][0] * m[0][0] + m[0][1] * m[1][0], m[0][0] * m[0][1] + m[0][1] * m[1][1]],
            [m[1][0] * m[0][0] + m[1][1] * m[1][0], m[1][0] * m[0][1] + m[1][1] * m[1][1]],
        ];
    }
    [v[0]]
}
