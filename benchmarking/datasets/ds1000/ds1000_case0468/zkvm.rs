//! DS-1000 case 0468: clipped 3-by-3 windows, column-first traversal.
//! Each list element is a row-major window; its shape is fixed by its position.
use crate::prelude::*;

pub type Output = Vec<Vec<i64>>;

pub fn main(input: &mut impl Input) -> Output {
    let a = input.read_matrix::<i64, 4, 4>();
    // floor(shape[axis] / 2), for the kernel's constant shape (3, 3).
    let r_win = 3usize / 2;
    let c_win = 3usize / 2;
    let mut ans = Vec::with_capacity(16);
    for j in 0usize..4 {
        let ymin = j.saturating_sub(c_win);
        let ymax = (j + c_win + 1).min(4);
        for i in 0usize..4 {
            let xmin = i.saturating_sub(r_win);
            let xmax = (i + r_win + 1).min(4);
            let mut window = Vec::with_capacity((xmax - xmin) * (ymax - ymin));
            for row in &a[xmin..xmax] {
                window.extend_from_slice(&row[ymin..ymax]);
            }
            ans.push(window);
        }
    }
    ans
}
