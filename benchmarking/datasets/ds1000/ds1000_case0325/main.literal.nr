// DS-1000 case 0325: literal translation of kernel.py (LITERAL_BASELINES.md).
// deg2rad on integers is exact multiplication by Q32 pi/180; shared trigonometric gadgets.
// @outputs: int[1]
fn main(number: Field) -> pub Field {
    noir_hw::check_input(number);
    let deg = noir_hw::sin(number * 74961321);
    let rad = noir_hw::sin(noir_hw::from_int(number));
    noir_hw::gt(rad, deg) as Field
}
