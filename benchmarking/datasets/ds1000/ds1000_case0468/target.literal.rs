//! DS-1000 case 0468: literal translation of kernel.py (LITERAL_BASELINES.md).
//! Clipped 3x3 windows on a 4x4 integer input, column-first window order.
//! The helper preserves source statement and loop order; all slice bounds are shape constants.
use crate::prelude::*;

fn window(arr: &[Cell], shape: (usize, usize)) -> Vec<Cell> {
    let mut ans = Vec::new();
    // floor(shape / 2).astype(int): compile-time, positive shape arithmetic.
    let r_win = shape.0 / 2;
    let c_win = shape.1 / 2;
    let (x, y) = (4usize, 4usize);
    for j in 0..y {
        let ymin = j.saturating_sub(c_win);
        let ymax = y.min(j + c_win + 1);
        for i in 0..x {
            let xmin = i.saturating_sub(r_win);
            let xmax = x.min(i + r_win + 1);
            // ans.append(arr[xmin:xmax, ymin:ymax]), flattened row-major.
            for r in xmin..xmax {
                for c in ymin..ymax { ans.push(arr[r * y + c]); }
            }
        }
    }
    ans
}

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let a = inputs.ints("a", &[4, 4])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);
    let a = q.load_ints(ctx, &a);
    let size = (3, 3);
    let result = window(&a, size);
    out.ints(&result);
    Ok(())
}
