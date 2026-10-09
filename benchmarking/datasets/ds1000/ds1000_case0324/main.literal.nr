// DS-1000 case 0324: literal translation of kernel.py (LITERAL_BASELINES.md).
// deg2rad on integers is exact multiplication by Q32 pi/180; shared trigonometric gadgets.
// @outputs: float[1]
fn main(degree: Field) -> pub Field {
    noir_hw::check_input(degree);
    let deg = noir_hw::cos(degree * 74961321);
    deg
}
