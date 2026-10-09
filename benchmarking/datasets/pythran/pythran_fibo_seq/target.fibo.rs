//! pythran_fibo_seq__fibo (Pythran fibo_seq), halo2-base baseline.
//! Kernel: `n` steps of `(a, b) = (a + b, a)` from `(1, 1)`, for a private
//! `n` with `@requires(0 <= n <= 1024)`; returns `a`, the Fibonacci number
//! `F(n + 2)` (a Python int).
//! The steps depend on the input only through their number, so the loop is
//! `(a, b) = M^n (1, 1)` with `M = [[1, 1], [1, 0]]`, evaluated by the bits
//! of `n`: `M^n` is the product of the `M^(2^k)` (constants) whose bit is
//! set, each applied as `v + bit * (M^(2^k) v - v)`, exact in the field.
//! A result outside int64 has no proof (NPB_BASELINES.md section 5.7), and
//! `F(n + 2) < 2^63` exactly for `n <= MAX_N = 90`, so the statement admits
//! `0 <= n <= 90`: `n` is decomposed into `N_BITS` bits (which also implies
//! the input domain, an int64) and checked against `MAX_N`. Every `n` the
//! precondition admits with an int64 result is covered.

use crate::prelude::*;

/// The largest `n` whose result `F(n + 2)` fits int64 (fixed by the number
/// format, not a size).
const MAX_N: u64 = 90;
/// The bits of `MAX_N`.
const N_BITS: usize = 7;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let n = inputs.int("n")?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    // The input, whose domain the bit decomposition below implies (not a
    // hint: the value is the input itself).
    let n = ctx.load_witness(fe(n as i128));
    let bits = q.gate.num_to_bits(ctx, n, N_BITS);
    q.range.check_less_than(ctx, n, Constant(Fr::from(MAX_N + 1)), N_BITS);

    // v = M^n (1, 1), one power of two of M per bit of n.
    let mut v = [Constant(Fr::ONE), Constant(Fr::ONE)];
    for (k, bit) in bits.iter().enumerate() {
        let m = power_of_two_power(k);
        let last = k + 1 == N_BITS;
        let rows = if last { 1 } else { 2 };
        let mut next = v;
        for r in 0..rows {
            // (M^(2^k) - I) v, row r
            let coefficients = [0, 1].map(|c| Constant(fe(m[r][c] as i128 - (r == c) as i128)));
            let step = q.gate.inner_product(ctx, v, coefficients);
            next[r] = Existing(q.gate.mul_add(ctx, *bit, step, v[r]));
        }
        v = next;
    }
    let QuantumCell::Existing(a) = v[0] else { unreachable!("N_BITS > 0") };
    out.int(a);
    Ok(())
}

/// `M^(2^k)` for `M = [[1, 1], [1, 0]]`.
fn power_of_two_power(k: usize) -> [[u128; 2]; 2] {
    let mut m = [[1, 1], [1, 0]];
    for _ in 0..k {
        m = [
            [m[0][0] * m[0][0] + m[0][1] * m[1][0], m[0][0] * m[0][1] + m[0][1] * m[1][1]],
            [m[1][0] * m[0][0] + m[1][1] * m[1][0], m[1][0] * m[0][1] + m[1][1] * m[1][1]],
        ];
    }
    m
}
