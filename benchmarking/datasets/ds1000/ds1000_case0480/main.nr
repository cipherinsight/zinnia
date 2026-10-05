// DS-1000 case 0480, Noir baseline.
// Kernel: `result = ((x == a) & (y == b)).argmax()` on x, y: Integer[8], then
// `result = -1` if `x[result] != a or y[result] != b`.
// This is the first index i with x[i] == a and y[i] == b, or -1 if there is
// none: when a match exists, argmax returns the first one and the re-check
// passes; when none does, argmax returns 0 and the re-check at index 0 fails.
// So the circuit does not gather x[result], y[result]: scanning from the last
// index down to the first, starting at -1, gives the same result for every
// input.
// @outputs: int[1]

fn main(x: [Field; 8], y: [Field; 8], a: Field, b: Field) -> pub Field {
    let mut result: Field = -1;
    for k in 0..8 {
        let i = 7 - k;
        if (x[i] == a) & (y[i] == b) {
            result = i as Field;
        }
    }
    result
}
