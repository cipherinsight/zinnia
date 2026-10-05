// DS-1000 case 0322 -- Noir baseline.
// np.argwhere(a == np.min(a)) on Integer[2,2].
// Mirrors Zinnia's lowering ALGORITHM and contract: min via qlt_signed
// select chain (re-projecting intermediates), keep indicators, then a
// count-addressed scatter of keep-gated (row, col) coordinates into a
// packed 4x2 payload, bound as [len, n, payload...] (numpy argwhere
// result, zero-padded envelope).

use noir_fpc::{qlt_signed, to_signed};

fn main(a: [Field; 4]) -> pub [Field; 10] {
    // Running min: select chain, projecting each intermediate.
    let (s0, m0) = to_signed(a[0]);
    let (s1, m1) = to_signed(a[1]);
    let lt1: Field = if qlt_signed(s0, m0, s1, m1) { 1 } else { 0 };
    let min1 = lt1 * a[0] + (1 - lt1) * a[1];
    let (sm1, mm1) = to_signed(min1);
    let (s2, m2) = to_signed(a[2]);
    let lt2: Field = if qlt_signed(sm1, mm1, s2, m2) { 1 } else { 0 };
    let min2 = lt2 * min1 + (1 - lt2) * a[2];
    let (sm2, mm2) = to_signed(min2);
    let (s3, m3) = to_signed(a[3]);
    let lt3: Field = if qlt_signed(sm2, mm2, s3, m3) { 1 } else { 0 };
    let minv = lt3 * min2 + (1 - lt3) * a[3];
    // Keep indicators.
    let mut keep: [Field; 4] = [0; 4];
    for k in 0..4 {
        keep[k] = if a[k] == minv { 1 } else { 0 };
    }
    // Count-addressed scatter: cell k writes keep-gated (row, col) at
    // flat positions (2*cnt, 2*cnt + 1), merged by address equality
    // over the feasible slot range; cnt advances by keep.
    let mut payload: [Field; 8] = [0; 8];
    let mut cnt: Field = 0;
    for k in 0..4 {
        let row: Field = (k / 2) as Field;
        let col: Field = (k % 2) as Field;
        let wr = keep[k] * row;
        let wc = keep[k] * col;
        let base = cnt * 2;
        for s in 0..8 {
            if s < 2 * k + 1 {
                let hit: Field = if base == s as Field { 1 } else { 0 };
                payload[s] = payload[s] + hit * (wr - payload[s]);
            }
        }
        for s in 0..8 {
            if (0 < s) & (s < 2 * k + 2) {
                let hit: Field = if base + 1 == s as Field { 1 } else { 0 };
                payload[s] = payload[s] + hit * (wc - payload[s]);
            }
        }
        if k == 0 {
            payload[0] = wr;
            payload[1] = wc;
        }
        cnt = cnt + keep[k];
    }
    [cnt * 2, cnt,
     payload[0], payload[1], payload[2], payload[3],
     payload[4], payload[5], payload[6], payload[7]]
}
