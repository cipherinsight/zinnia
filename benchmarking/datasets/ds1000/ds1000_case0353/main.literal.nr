// DS-1000 case 0353, literal translation of kernel.py (LITERAL_BASELINES.md).
// Independent axis scans, elementwise maximum, row-major ravel, equality membership.
// Masking uses running counts and an equality test per input/output row pair.
// @outputs: dynamic int[2x3]
fn ravel<let N: u32>(rows: [Field; N], dims: [Field; 3]) -> [Field; N/3] {
    let mut size: Field = 1;
    for d in dims { size *= d; size.assert_max_bit_size::<63>(); }
    let mut strides: [Field; 3] = [1; 3];
    for i in 0..2 { let k = 1-i; strides[k] = strides[k+1]*dims[k+1]; }
    let mut result = [0; N/3];
    for r in 0..N/3 {
        let mut flat: Field = 0;
        for k in 0..3 {
            rows[r*3+k].assert_max_bit_size::<63>();
            assert(noir_hw::lt(rows[r*3+k], dims[k]));
            flat += rows[r*3+k]*strides[k];
        }
        result[r] = flat;
    }
    result
}
fn main(A: [Field; 6], B: [Field; 21]) -> pub [Field; 7] {
    noir_hw::check_inputs(A); noir_hw::check_inputs(B);
    let mut bmax = [0; 3];
    for c in 0..3 { let mut acc = B[c]; for r in 1..7 { acc = noir_hw::max(acc, B[r*3+c]); } bmax[c] = acc; }
    let mut amax = [0; 3];
    for c in 0..3 { let mut acc = A[c]; for r in 1..2 { acc = noir_hw::max(acc, A[r*3+c]); } amax[c] = acc; }
    let mut maxima = [0; 3];
    for c in 0..3 { maxima[c] = noir_hw::max(bmax[c], amax[c]); }
    let mut dims = [0; 3];
    for c in 0..3 { dims[c] = maxima[c]+1; }
    let flat_a = ravel(A, dims);
    let flat_b = ravel(B, dims);
    let mut member = [false; 2];
    for i in 0..2 { let mut found = false; for j in 0..7 { found = found | (flat_a[i] == flat_b[j]); } member[i] = found; }
    let mut keep = [false; 2];
    for i in 0..2 { keep[i] = !member[i]; }
    let mut count: Field = 0; let mut positions = [0; 2];
    for i in 0..2 { positions[i] = count; count += keep[i] as Field; }
    let mut result = [0; 7]; result[0] = count;
    for j in 0..2 {
        let mut hits = [false; 2];
        for i in 0..2 { hits[i] = keep[i] & (positions[i] == j as Field); }
        for c in 0..3 {
            let mut value: Field = 0;
            for i in 0..2 { value += (hits[i] as Field)*A[i*3+c]; }
            result[1+j*3+c] = value;
        }
    }
    result
}
