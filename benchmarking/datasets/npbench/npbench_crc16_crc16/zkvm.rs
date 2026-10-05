//! NPBench crc16: CRC-16-CCITT of a byte sequence, bit by bit.
//!
//! ```python
//! def crc16(data: Integer[Array, '1600'], poly: Integer=33800):
//!     crc = 65535
//!     for b in data:
//!         cur_byte = 255 & b
//!         for _ in range(0, 8):
//!             if crc & 1 ^ cur_byte & 1:
//!                 crc = crc >> 1 ^ poly
//!             else:
//!                 crc >>= 1
//!             cur_byte >>= 1
//!     crc = ~crc & 65535
//!     crc = crc << 8 | crc >> 8 & 255
//!     return crc & 65535
//! ```
use crate::prelude::*;

pub mod crc16 {
    use super::*;

    pub type Output = i64;

    pub fn main(args: &mut Args<impl Input>) -> Output {
        let data: Array1<i64> = args.array1();
        let poly: i64 = args.scalar();
        // Python binds `>>` before `&`, `&` before `^`, and `^` before `|`.
        let mut crc: i64 = 65535;
        for &b in &data {
            let mut cur_byte = 255 & b;
            for _ in 0..8 {
                if (crc & 1) ^ (cur_byte & 1) != 0 {
                    crc = (crc >> 1) ^ poly;
                } else {
                    crc >>= 1;
                }
                cur_byte >>= 1;
            }
        }
        crc = !crc & 65535;
        crc = (crc << 8) | ((crc >> 8) & 255);
        crc & 65535
    }
}
