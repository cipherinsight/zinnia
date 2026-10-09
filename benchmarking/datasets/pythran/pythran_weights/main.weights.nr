// pythran_weights__weights (Pythran weights), Noir baseline.
// Kernel: for `input_data: Int[S, L]` and a float `threshold`, row i's weight
// is 1 / total_i, where total_i counts the rows r whose number of positions
// equal to row i's exceeds threshold * L; the weights are stored as float32
// (computed in Q32, NPB_BASELINES.md section 5.4: no discrete outcome depends
// on that rounding).
// The match count of rows i and r is symmetric and is L for r = i, so each
// unordered pair is counted once and the diagonal's comparison is made once.
// Compare count*SCALE with threshold*L using the frozen Noir 64-bit gap
// comparator. 1 / total is floor(2^32 / total):
// 2^32 = q * total + r with r and total - r - 1 checked to COUNT_BITS bits and
// q to 42 bits, which pins (q, r) (an integer identity far below p; r < total)
// and leaves total = 0 without a witness, as NumPy's result is not finite.
// COUNT_BITS and the 42 bits are multiples of 14: no range table
// beyond the 64-bit checks' (NOIR_BASELINES.md).
// @outputs: float[S]

use noir_hw::{check_input, check_inputs, canonical_gt, SCALE};

// Sizes (hw_check.sizes): from fixture.json.
global S: u32 = 5; // = input_data_0
global L: u32 = 2; // = input_data_1
global COUNT_BITS: u32 = 14; // = 14 * cdiv(bits(input_data_0), 14)

unconstrained fn divmod_hint(total: Field) -> (Field, Field) {
    let t = total as u64;
    if t == 0 {
        (0, 0)
    } else {
        ((4294967296 / t) as Field, (4294967296 % t) as Field)
    }
}

// floor(2^32 / total) for a total in [0, S].
fn reciprocal_count(total: Field) -> Field {
    // Safety: pinned by the recomposition and the range checks.
    let (quotient, remainder) = unsafe { divmod_hint(total) };
    assert(quotient * total + remainder == SCALE, "reciprocal: recomposition");
    quotient.assert_max_bit_size::<42>();
    remainder.assert_max_bit_size::<COUNT_BITS>();
    (total - remainder - 1).assert_max_bit_size::<COUNT_BITS>();
    quotient
}

fn main(input_data: [[Field; L]; S], threshold: Field) -> pub [Field; S] {
    // Input domain: every input element is an int64 or a Q32 code in the window.
    for i in 0..S {
        check_inputs(input_data[i]);
    }
    check_input(threshold);
    let bar = (L as Field) * threshold;
    let self_match = canonical_gt((L as Field) * SCALE, bar) as Field;
    let mut totals: [Field; S] = [self_match; S];
    for i in 0..S {
        for r in i + 1..S {
            let mut count: Field = 0;
            for c in 0..L {
                count += (input_data[i][c] == input_data[r][c]) as Field;
            }
            let over = canonical_gt(count * SCALE, bar) as Field;
            totals[i] += over;
            totals[r] += over;
        }
    }
    let mut weights: [Field; S] = [0; S];
    for i in 0..S {
        weights[i] = reciprocal_count(totals[i]);
    }
    weights
}
