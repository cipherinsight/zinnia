//! DS-1000 case 0467: clipped 3-by-3 windows, row-first traversal.
//! Each list element is a row-major window; its shape is fixed by its position.
// output: [[i64]]

#[executable]
fn main(a: Array<i64>) -> Array<Array<i64>> {
    assert!(a.len() == 16, "a has the wrong shape");
    // floor(shape[axis] / 2), for the kernel's constant shape (3, 3).
    let r_win: u32 = 3 / 2;
    let c_win: u32 = 3 / 2;
    let mut ans = array![];
    for i in 0_u32..4 {
        let xmin = if i < r_win { 0 } else { i - r_win };
        let xmax = if i + r_win + 1 > 4 { 4 } else { i + r_win + 1 };
        for j in 0_u32..4 {
            let ymin = if j < c_win { 0 } else { j - c_win };
            let ymax = if j + c_win + 1 > 4 { 4 } else { j + c_win + 1 };
            let mut window = array![];
            for row in xmin..xmax {
                for col in ymin..ymax {
                    window.append(*a.at(row * 4 + col));
                }
            }
            ans.append(window);
        }
    }
    ans
}
