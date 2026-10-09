// DS-1000 case 0409, Noir LITERAL baseline (LITERAL_BASELINES.md).
// Literal translation of kernel.py:
//   x_new = np.array(x)
//   y_new = np.array(y)
//   z = x_new + y_new
// np.array(...) on an array is a copy (no-op on the values); + is elementwise
// addition, one gadget call per element (L5). No computational library call, no control
// flow and no repeated subexpression here, so there is no room for a
// literal/optimized difference: this file is the same as the optimized
// baseline.
// @outputs: float[48]

fn main(x: [Field; 48], y: [Field; 48]) -> pub [Field; 48] {
    noir_hw::check_inputs(x);
    noir_hw::check_inputs(y);
    let x_new = x; // np.array(x)
    let y_new = y; // np.array(y)
    let mut z = [0; 48];
    for i in 0..48 {
        z[i] = x_new[i] + y_new[i];
    }
    z
}
