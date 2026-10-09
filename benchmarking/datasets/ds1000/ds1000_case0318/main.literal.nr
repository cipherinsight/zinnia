// DS-1000 case 0318: literal translation of kernel.py (LITERAL_BASELINES.md).
// Enforce permutation precondition; build inverse by private-index scatter, then standard indexed reads.
// empty_like starts with zeros: the permutation guarantees every slot is overwritten.
// @outputs: float[48]
fn main(a: [Field; 48], permutation: [Field; 6]) -> pub [Field; 48] {
    noir_hw::check_inputs(a);
    noir_hw::check_inputs(permutation);
    for j in 0..6 {
        let mut count: Field = 0;
        for k in 0..6 { count += (permutation[k] == j as Field) as Field; }
        assert(count == 1);
    }
    let mut c = [0; 6];
    let mut arange = [0; 6];
    for i in 0..6 { arange[i] = i as Field; }
    for i in 0..6 {
        for k in 0..6 {
            c[k] = if permutation[i] == k as Field { arange[i] } else { c[k] };
        }
    }
    let mut result = [0; 48];
    for i in 0..8 {
        let mut row = [0; 6];
        for k in 0..6 { row[k] = a[i*6+k]; }
        for j in 0..6 { result[i*6+j] = row[noir_hw::index_pos::<8>(c[j], 6)]; }
    }
    result
}
