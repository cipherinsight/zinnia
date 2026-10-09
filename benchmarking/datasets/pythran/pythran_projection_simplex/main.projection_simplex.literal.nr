// pythran_projection_simplex__projection_simplex: literal translation of kernel.py (LITERAL_BASELINES.md).
// Stable bubble sort; each mask call independently uses prefix counts and pairwise equality tests.
// Nonempty checks preserve IndexError for [-1] on an empty selection.
// Comparison domain follows optimized canonical gaps, including values beyond signed64.
// @outputs: float[N]
use noir_hw::{
    canonical_div_nonzero, check_input, check_inputs, divmod_pos, gt, index_pos, SCALE,
};
global N: u32 = 10; // = v_0
global INDEX_BITS: u32 = 8; // = 8 + 20 * min(1, (v_0 - 1) // 128)
fn main(v: [Field; N], z: Field) -> pub [Field; N] {
    check_inputs(v);
    check_input(z);
    let mut sorted = v;
    for pass in 0..N - 1 {
        for j in 0..N - 1 - pass {
            let swap = gt(sorted[j], sorted[j + 1]);
            let a = sorted[j];
            let b = sorted[j + 1];
            sorted[j] = if swap { b } else { a };
            sorted[j + 1] = if swap { a } else { b };
        }
    }
    let mut u = [0; N];
    for i in 0..N {
        u[i] = sorted[N - 1 - i];
    }
    let mut cumsum = [0; N];
    let mut total: Field = 0;
    for i in 0..N {
        total += u[i];
        cumsum[i] = total;
    }
    let mut cssv = [0; N];
    for i in 0..N {
        cssv[i] = cumsum[i] - z * SCALE;
    }
    let mut ind = [0; N];
    for i in 0..N {
        ind[i] = (i + 1) as Field;
    }
    let mut quotients = [0; N];
    for i in 0..N {
        quotients[i] = divmod_pos(cssv[i], ind[i]).0;
    }
    let mut gaps = [0; N];
    for i in 0..N {
        gaps[i] = u[i] - quotients[i];
    }
    let mut cond = [false; N];
    for i in 0..N {
        cond[i] = noir_hw::canonical_gt(gaps[i], 0);
    }
    let (count, selected) = compact_literal(ind, cond);
    assert(count != 0);
    let rho = selected[index_pos::<INDEX_BITS>(count - 1, N)];
    let (count, selected) = compact_literal(cssv, cond);
    assert(count != 0);
    let numerator = selected[index_pos::<INDEX_BITS>(count - 1, N)];
    let theta = canonical_div_nonzero(numerator, rho * SCALE);
    let mut differences = [0; N];
    for i in 0..N {
        differences[i] = v[i] - theta;
    }
    let mut w = [0; N];
    for i in 0..N {
        w[i] = if noir_hw::canonical_gt(differences[i], 0) { differences[i] } else { 0 };
    }
    w
}
fn compact_literal(xs: [Field; N], keep: [bool; N]) -> (Field, [Field; N]) {
    let mut count: Field = 0;
    let mut positions = [0; N];
    for i in 0..N {
        positions[i] = count;
        count += keep[i] as Field;
    }
    let mut result = [0; N];
    for j in 0..N {
        for i in 0..N {
            result[j] += ((keep[i] & (positions[i] == j as Field)) as Field) * xs[i];
        }
    }
    (count, result)
}
