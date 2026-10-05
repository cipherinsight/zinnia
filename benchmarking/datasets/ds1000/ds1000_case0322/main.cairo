//! DS-1000 case 0322: the (row, column) pairs where `a` attains its minimum,
//! in row-major order.
//!
//!     def kernel(a: Integer[Array, "2 2"]):
//!         result = np.argwhere(a == np.min(a))
//!         return result
//!
//! The output has shape `(count, 2)`; its length implies the shape.
// output: [i64]
const ROWS: u32 = 2;
const COLS: u32 = 2;

#[executable]
fn main(a: Array<i64>) -> Array<i64> {
    assert!(a.len() == ROWS * COLS, "a has the wrong length");
    let mut lowest = *a.at(0);
    for v in a.span() {
        if *v < lowest {
            lowest = *v;
        }
    }
    let mut positions = array![];
    for i in 0..ROWS {
        for j in 0..COLS {
            if *a.at(i * COLS + j) == lowest {
                positions.append(i.into());
                positions.append(j.into());
            }
        }
    }
    positions
}
