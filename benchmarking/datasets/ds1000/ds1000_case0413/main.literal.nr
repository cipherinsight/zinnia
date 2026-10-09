// DS-1000 case 0413: literal translation of kernel.py (LITERAL_BASELINES.md).
// Complex values are interleaved (real, imaginary) Q32 components, each checked.
// Exact nonzero imaginary test; length counts complex items, padded unit is two.
// Prefix-count equality scan computes every input/output pair.
// @outputs: dynamic float[10x2]
fn main(x: [Field; 20]) -> pub [Field; 21] {
    noir_hw::check_inputs(x);
    let mut imag = [0; 10];
    for i in 0..10 { imag[i] = x[2 * i + 1]; }
    let mut keep = [false; 10];
    for i in 0..10 { keep[i] = imag[i] != 0; }
    let mut prefix: [Field; 11] = [0; 11];
    for i in 0..10 { prefix[i + 1] = prefix[i] + keep[i] as Field; }
    let mut result = [0; 21];
    result[0] = prefix[10];
    for j in 0..10 {
        let mut real: Field = 0;
        let mut imag: Field = 0;
        for i in 0..10 {
            let active = keep[i] & (prefix[i] == j as Field);
            real += (active as Field) * x[2 * i];
            imag += (active as Field) * x[2 * i + 1];
        }
        result[2 * j + 1] = real;
        result[2 * j + 2] = imag;
    }
    result
}
