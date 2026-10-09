//! pythran_weights__weights (Pythran weights), halo2-base baseline.
//! Kernel: for `input_data: Int[S, L]` and a float `threshold`, row `i`'s
//! weight is `1 / total_i`, where `total_i` counts the rows `r` whose number
//! of positions equal to row `i`'s exceeds `threshold * L`; the weights are
//! stored as float32 (computed in Q32, NPB_BASELINES.md section 5.4: no
//! discrete outcome depends on that rounding).
//! The match count of rows `i` and `r` is symmetric and is `L` for `r = i`,
//! so each unordered pair is counted once and the diagonal's comparison is
//! made once. `count > threshold * L` compares an integer with a float
//! whose Q32 code `L * threshold` can leave the window, so it is one
//! `CMP_BITS`-bit comparison of `count * 2^32 + L * 2^63` with
//! `L * (threshold + 2^63)`, both non-negative. `1 / total` is
//! `floor(2^32 / total)`: `2^32 = q * total + r` with `total - r - 1` and
//! `r` checked to `COUNT_BITS` bits and `q` to 33 bits, which pins `(q, r)`
//! (`|q * total + r| < 2^40`, an integer identity; `r < total`) and leaves
//! `total = 0` without a witness, as NumPy's result is not finite.

use crate::prelude::*;

// Sizes (hw_check.sizes): from fixture.json.
const S: usize = 5; // = input_data_0
const L: usize = 2; // = input_data_1
const CMP_BITS: usize = 66; // = 64 + bits(input_data_1)
const COUNT_BITS: usize = 3; // = bits(input_data_0)

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let data = inputs.ints("input_data", &[S, L])?;
    let threshold = inputs.float("threshold")?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let data = q.load_ints(ctx, &data);
    let threshold = q.load_float(ctx, threshold);
    let half_window = 1i128 << 63;
    // L * (threshold + 2^63): the right side of every comparison.
    let bar = q.gate.mul_add(ctx, threshold, Constant(fe(L as i128)), Constant(fe(L as i128 * half_window)));
    // count > threshold * L, for a count in [0, L]
    let exceeds = |ctx: &mut Ctx, count: QuantumCell<Fr>| {
        let left = q.gate.mul_add(ctx, count, Constant(fe(SCALE)), Constant(fe(L as i128 * half_window)));
        q.range.is_less_than(ctx, bar, left, CMP_BITS)
    };
    let self_match = exceeds(ctx, Constant(fe(L as i128)));
    let mut over = vec![vec![self_match; S]; S];
    for i in 0..S {
        for r in i + 1..S {
            let equal: Vec<Cell> = (0..L).map(|c| q.eq(ctx, data[i * L + c], data[r * L + c])).collect();
            let count = q.sum(ctx, equal);
            let bit = exceeds(ctx, Existing(count));
            over[i][r] = bit;
            over[r][i] = bit;
        }
    }
    let weights: Vec<Cell> = over
        .iter()
        .map(|row| {
            let total = q.sum(ctx, row.iter().copied());
            reciprocal_count(&q, ctx, total)
        })
        .collect();
    out.floats(&weights);
    Ok(())
}

/// `1 / total` in Q32, `floor(2^32 / total)`, for a `total` in `[0, S]`
/// (pinned as the header says).
fn reciprocal_count(q: &Q32, ctx: &mut Ctx, total: Cell) -> Cell {
    let t = crate::q32::signed(total.value()) as u64;
    let (quotient, remainder) = if t == 0 { (0, 0) } else { ((SCALE as u64) / t, (SCALE as u64) % t) };
    // Pinned by the recomposition and the three range checks below.
    let quotient = ctx.load_witness(Fr::from(quotient));
    let remainder = ctx.load_witness(Fr::from(remainder));
    let recomposed = q.gate.mul_add(ctx, quotient, total, remainder);
    q.gate.assert_is_const(ctx, &recomposed, &fe(SCALE));
    q.range.range_check(ctx, quotient, 33);
    q.range.range_check(ctx, remainder, COUNT_BITS);
    let gap = q.gate.sub(ctx, total, remainder);
    let gap = q.gate.sub(ctx, gap, Constant(Fr::ONE));
    q.range.range_check(ctx, gap, COUNT_BITS);
    quotient
}
