// DS-1000 case 0326: literal translation of kernel.py (LITERAL_BASELINES.md).
// No shared arcsin entry exists: local half-angle arcsin(v)=2 atan(v/(1+sqrt(1-v*v))).
// Use the baseline's degree-9 atan coefficients and exact raw integer root as the
// arithmetic contract inside arcsin, through shared gadgets only.
// Materialize radians before the separate degrees conversion by Q32 180/pi.
// @outputs: float[1]
global ATAN_COEFS: [Field; 10] = [
    -7750000, 47082270, -134277928, 249279572, -361114211,
    470446531, -612650751, 858930835, -1431654192, 4294967292,
];
fn arcsin(value: Field) -> Field {
    let shifted = value + noir_hw::SCALE;
    shifted.assert_max_bit_size::<64>();
    (2 * noir_hw::SCALE - shifted).assert_max_bit_size::<64>();
    let root = noir_hw::isqrt(noir_hw::SCALE * noir_hw::SCALE - value * value);
    let denominator = noir_hw::SCALE + root;
    let t = noir_hw::div(value, denominator);
    let t2 = noir_hw::mul(t, t);
    let polynomial = noir_hw::poly(t2, ATAN_COEFS);
    let angle = noir_hw::mul(t, polynomial);
    2 * angle
}
fn main(value: Field) -> pub Field {
    noir_hw::check_input(value);
    let radians = arcsin(value);
    noir_hw::mul(radians, 246083499208)
}
