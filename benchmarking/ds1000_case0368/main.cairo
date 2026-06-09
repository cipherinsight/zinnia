// DS-1000 case 368 — np.isclose(a, a[0], atol=0, rtol=1e-5).all() on Integer[3, 3].
// Mirrors target.rs: float-cast each integer, compute rtol*|a[0][j]| per column,
// then |a[i][j] - a[0][j]| <= rtol*|a[0][j]| across all 9 cells, AND-reduced.
// Quantisation ×2^32 = Q to match Zinnia's `precision_bits = 32`.
//
// Use i128 for diffs (signed), then take abs as u128 for the comparison against
// rtol_abs_b. rtol_q = floor(1e-5 * Q) = 42_949 (truncated; matches Zinnia).

#[executable]
pub fn main() {
    const Q: u128 = 4_294_967_296;
    let rtol_q: u128 = 42_949;  // floor(1e-5 * Q)

    // a = [[10, 20, 30], [10, 20, 30], [10, 21, 30]] → not all close.
    let a: Array<i128> = array![10, 20, 30, 10, 20, 30, 10, 21, 30];
    let expected: bool = false;

    // rtol_abs_b[j] = rtol * |a[0][j]| (fp).
    let mut rtol_abs_b: Array<u128> = ArrayTrait::new();
    let mut j: u32 = 0;
    while j != 3 {
        let b: i128 = *a.at(j);
        let abs_b: u128 = if b < 0 { (-b).try_into().unwrap() } else { b.try_into().unwrap() };
        let b_q: u128 = abs_b * Q;
        let prod: u128 = (b_q * rtol_q) / Q;
        rtol_abs_b.append(prod);
        j = j + 1;
    };

    let mut all_close: bool = true;
    let mut i: u32 = 0;
    while i != 3 {
        let mut jj: u32 = 0;
        while jj != 3 {
            let v: i128 = *a.at(i * 3 + jj);
            let b: i128 = *a.at(jj);
            let diff: i128 = v - b;
            let abs_d_int: u128 = if diff < 0 { (-diff).try_into().unwrap() } else { diff.try_into().unwrap() };
            let abs_d_q: u128 = abs_d_int * Q;
            let leq: bool = abs_d_q <= *rtol_abs_b.at(jj);
            all_close = all_close && leq;
            jj = jj + 1;
        };
        i = i + 1;
    };

    assert!(all_close == expected);
}
