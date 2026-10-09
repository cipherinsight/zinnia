// pythran_weights__weights (Pythran weights), Noir baseline.
// Kernel: for `input_data: Int[S, L]` and a float `threshold`, row i's weight
// is 1 / total_i, where total_i counts the rows r whose number of positions
// equal to row i's exceeds threshold * L; the weights are stored as float32
// (computed in Q32, NPB_BASELINES.md section 5.4: no discrete outcome depends
// on that rounding).
// The match count of rows i and r is symmetric and is L for r = i, so each
// unordered pair is counted once and the diagonal's comparison is made once.
// count > threshold * L compares an integer with a float whose Q32 code
// L * threshold can leave the window: with d = (count * 2^32 + L * 2^63) -
// L * (threshold + 2^63), both terms in [0, 2^CMP_BITS) (threshold's window
// check), the hinted bit `over` is pinned by d - 1 (over) or -d (not over)
// lying in [0, 2^CMP_BITS). 1 / total is floor(2^32 / total):
// 2^32 = q * total + r with r and total - r - 1 checked to COUNT_BITS bits and
// q to 42 bits, which pins (q, r) (an integer identity far below p; r < total)
// and leaves total = 0 without a witness, as NumPy's result is not finite.
// CMP_BITS, COUNT_BITS and the 42 bits are multiples of 14: no range table
// beyond the 64-bit checks' (NOIR_BASELINES.md).
// @outputs: float[S]

use noir_hw::{check_input, check_inputs, OFFSET, SCALE};

// Sizes (hw_check.sizes): from fixture.json.
global S: u32 = 5; // = input_data_0
global L: u32 = 2; // = input_data_1
global CMP_BITS: u32 = 70; // = 14 * cdiv(64 + bits(input_data_1), 14)
global COUNT_BITS: u32 = 14; // = 14 * cdiv(bits(input_data_0), 14)

unconstrained fn is_positive_hint(d: Field) -> bool {
    // d is a small signed integer: positive iff below p / 2 and nonzero.
    (d != 0) & d.lt(OFFSET * OFFSET)
}

// d > 0 for an integer d in (-2^CMP_BITS, 2^CMP_BITS).
fn is_positive(d: Field) -> bool {
    // Safety: pinned by the range check (d - 1 or -d in [0, 2^CMP_BITS)).
    let positive = unsafe { is_positive_hint(d) };
    let checked = if positive { d - 1 } else { -d };
    checked.assert_max_bit_size::<CMP_BITS>();
    positive
}

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
    // L * (threshold + 2^63) and L * 2^63
    let bar = (L as Field) * (threshold + OFFSET);
    let base = (L as Field) * OFFSET;
    let self_match = is_positive((L as Field) * SCALE + base - bar) as Field;
    let mut totals: [Field; S] = [self_match; S];
    for i in 0..S {
        for r in i + 1..S {
            let mut count: Field = 0;
            for c in 0..L {
                count += (input_data[i][c] == input_data[r][c]) as Field;
            }
            let over = is_positive(count * SCALE + base - bar) as Field;
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
