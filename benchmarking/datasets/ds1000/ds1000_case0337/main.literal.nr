// DS-1000 case 0337: literal translation of kernel.py (LITERAL_BASELINES.md).
// fliplr and diag are views, materialized as wire aliases; domain-check only the selected elements (L7).
// @outputs: float[8]
fn main(a: [Field; 80]) -> pub [Field; 8] {
    let mut flipped = [[0; 10]; 8];
    for i in 0..8 { for j in 0..10 { flipped[i][j] = a[i * 10 + 9 - j]; } }
    let mut anti = [0; 8];
    for i in 0..8 { anti[i] = noir_hw::checked(flipped[i][i]); }
    anti
}
