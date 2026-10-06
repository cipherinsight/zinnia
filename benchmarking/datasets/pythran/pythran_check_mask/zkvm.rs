//! Pythran check_mask: marks the rows of `db` whose target bit is set and
//! whose remaining bits cover `mask`.
//!
//! ```python
//! def check_mask(db: Boolean[Array, '3 4'], out: Boolean[Array, '3'], mask: Tuple[Integer, Integer, Integer]=(1, 0, 1)):
//!     for idx, line in enumerate(db):
//!         target, vector = (line[0], line[1:])
//!         if (mask == np.bitwise_and(mask, vector)).all():
//!             if target == 1:
//!                 out[idx] = 1
//!     return out
//! ```
//!
//! The tuple `mask` arrives as its three items, one parameter each.
use crate::prelude::*;

pub mod check_mask {
    use super::*;

    pub type Output = Array1<bool>;

    pub fn main(args: &mut Args<impl Input>) -> Output {
        let db: Array2<bool> = args.array2();
        let mut out: Array1<bool> = args.array1();
        let mask: [i64; 3] = [args.scalar(), args.scalar(), args.scalar()];
        for (idx, line) in db.rows().into_iter().enumerate() {
            let (target, vector) = (line[0], line.slice(s![1..]));
            assert_eq!(vector.len(), mask.len(), "operands could not be broadcast together");
            // mask == mask & vector, elementwise (a bool is 0 or 1).
            let covered = mask.iter().zip(vector).all(|(&m, &v)| (m & v as i64) == m);
            if covered && target {
                out[idx] = true;
            }
        }
        out
    }
}
