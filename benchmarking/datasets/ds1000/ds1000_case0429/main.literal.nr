// DS-1000 case 0429: literal translation of kernel.py (LITERAL_BASELINES.md).
// Each mean/std occurrence is independent; std uses mean, full deviations, rounded squares, mean and sqrt.
// @outputs: float[2]
global N: u32 = 30;
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
    for i in 0..N { squares[i] = noir_hw::mul(dev[i], dev[i]); }
    let mut total: Field = 0;
    for i in 0..N { total += squares[i]; }
    noir_hw::sqrt(noir_hw::floor_div_const::<N>(total))
}
fn main(a: [Field; N]) -> pub [Field; 2] {
    noir_hw::check_inputs(a);
    let low = mean(a) - 2 * std(a);
    let high = mean(a) + 2 * std(a);
    [low, high]
}
