// DS-1000 case 0485: literal translation of kernel.py (LITERAL_BASELINES.md).
// tensordot uses its defining sum in index order with one final Q32 rescale per output (L5).
// @outputs: float[90]
fn main(A: [Field; 90], B: [Field; 9]) -> pub [Field; 90] {
    noir_hw::check_inputs(A); noir_hw::check_inputs(B);
    let mut result = [0; 90];
    for i in 0..5 { for j in 0..6 { for k in 0..3 {
        let mut raw: Field = 0;
        for p in 0..3 { raw += A[(i*6+j)*3+p] * B[p*3+k]; }
        result[(i*6+j)*3+k] = noir_hw::rescale(raw);
    } } }
    result
}
