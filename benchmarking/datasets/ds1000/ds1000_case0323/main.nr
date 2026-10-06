// DS-1000 case 0323, Noir baseline.
// Kernel: `np.sin(np.deg2rad(degree))` for an Integer degree. deg2rad of an
// integer is the integer times the Q32 constant pi/180 (exact, no rescale).
// @outputs: float[1]

global DEG_TO_RAD_Q: Field = 74961321; // round(pi / 180 * 2^32)

fn main(degree: Field) -> pub Field {
    // Input domain: every input element is an int64 / a Q32 code in the window.
    noir_hw::check_input(degree);
    noir_hw::sin(degree * DEG_TO_RAD_Q)
}
