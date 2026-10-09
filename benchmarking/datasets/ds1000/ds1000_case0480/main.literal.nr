// DS-1000 case 0480: literal translation of kernel.py (LITERAL_BASELINES.md).
// argmax scans left to right with strict improvement; both indexed rechecks are computed.
// @outputs: int[1]
fn main(x: [Field; 8], y: [Field; 8], a: Field, b: Field) -> pub Field {
    noir_hw::check_inputs(x); noir_hw::check_inputs(y);
    noir_hw::check_input(a); noir_hw::check_input(b);
    let mut xm = [false; 8]; let mut ym = [false; 8];
    for i in 0..8 { xm[i] = x[i] == a; }
    for i in 0..8 { ym[i] = y[i] == b; }
    let mut hits = [false; 8];
    for i in 0..8 { hits[i] = xm[i] & ym[i]; }
    let mut best = hits[0] as Field; let mut result: Field = 0;
    for i in 1..8 {
        let improve = noir_hw::gt(hits[i] as Field, best);
        best = if improve { hits[i] as Field } else { best };
        result = if improve { i as Field } else { result };
    }
    let nx = x[noir_hw::index_pos::<8>(result, 8)] != a;
    let ny = y[noir_hw::index_pos::<8>(result, 8)] != b;
    result = if nx | ny { -1 } else { result };
    result
}
