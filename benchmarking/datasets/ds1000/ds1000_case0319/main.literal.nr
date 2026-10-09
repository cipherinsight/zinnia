// DS-1000 case 0319: literal translation of kernel.py (LITERAL_BASELINES.md).
// Enforce permutation precondition; build inverse by private-index scatter, then standard indexed reads.
// empty_like starts with zeros: the permutation guarantees every slot is overwritten.
// @outputs: float[384]
fn main(a: [[Field; 48]; 8], permutation: [Field; 8]) -> pub [Field; 384] {
    for i in 0..8 { noir_hw::check_inputs(a[i]); }
    noir_hw::check_inputs(permutation);
    for j in 0..8 {
        let mut count: Field = 0;
        for k in 0..8 { count += (permutation[k] == j as Field) as Field; }
        assert(count == 1);
    }
    let mut c = [0; 8];
    let mut arange = [0; 8];
    for i in 0..8 { arange[i] = i as Field; }
    for i in 0..8 {
        for k in 0..8 {
            c[k] = if permutation[i] == k as Field { arange[i] } else { c[k] };
        }
    }
    let mut result = [0; 384];
    for i in 0..8 {
        let plane = a[noir_hw::index_pos::<8>(c[i], 8)];
        for j in 0..48 { result[i*48+j] = plane[j]; }
    }
    result
}
