// DS-1000 case 0420: literal translation of kernel.py (LITERAL_BASELINES.md).
// Both where branches computed eagerly; powers computed independently; integer literals multiply Q32 exactly.
// L4: select zero for the polynomial operand when either clamp applies;
// otherwise unused powers can overflow Q32 for valid inputs (e.g. x=10000).
// @outputs: float[1]
fn main(x: Field, x_min: Field, x_max: Field) -> pub Field {
    noir_hw::check_input(x);
    noir_hw::check_input(x_min);
    noir_hw::check_input(x_max);
    let below = noir_hw::lt(x, noir_hw::from_int(x_min));
    let above = noir_hw::gt(x, noir_hw::from_int(x_max));
    let t = if below | above { 0 } else { x };
    let square = noir_hw::mul(t, t);
    let first = 3 * square;
    // Separate power call: recompute the square for x**3.
    let cube = noir_hw::mul(t, noir_hw::mul(t, t));
    let second = 2 * cube;
    let smooth = first - second;
    let inner = if above { noir_hw::from_int(x_max) } else { smooth };
    if below { noir_hw::from_int(x_min) } else { inner }
}
