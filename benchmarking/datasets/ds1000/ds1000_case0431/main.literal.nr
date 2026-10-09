// DS-1000 case 0431: literal translation of kernel.py (LITERAL_BASELINES.md).
// Each mean/std occurrence is independent; std uses mean, full deviations, raw squares, one rescale of their sum, mean and sqrt (optimized arithmetic).
// @outputs: bool[8]
global N: u32 = 8;
fn mean(a: [Field; N]) -> Field {
    let mut total: Field = 0;
    for i in 0..N { total += a[i]; }
    noir_hw::floor_div_const::<N>(total)
}
fn std(a: [Field; N]) -> Field {
    let avg = mean(a);
    let mut dev = [0; N];
    for i in 0..N { dev[i] = a[i] - avg; }
    let mut squares = [0; N];
    for i in 0..N { squares[i] = dev[i] * dev[i]; }
    let mut total: Field = 0;
    for i in 0..N { total += squares[i]; }
    noir_hw::sqrt(noir_hw::floor_div_const::<N>(noir_hw::rescale(total)))
}
fn main(a: [Field; N]) -> pub [bool; 8] {
    noir_hw::check_inputs(a);
    let low = mean(a) - 2 * std(a);
    let high = mean(a) + 2 * std(a);
    let mut above = [false; N];
    for i in 0..N { above[i] = noir_hw::gt(a[i], low); }
    let mut below = [false; N];
    for i in 0..N { below[i] = noir_hw::lt(a[i], high); }
    let mut inside = [false; N];
    for i in 0..N { inside[i] = above[i] & below[i]; }
    let mut result = [false; N];
    for i in 0..N { result[i] = !inside[i]; }
    result
}
