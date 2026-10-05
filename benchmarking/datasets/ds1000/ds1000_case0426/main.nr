// DS-1000 case 0426, Noir baseline.
// Kernel: ((a[:, None] & (1 << np.arange(m))[::-1]) > 0).astype(int) with
// m = 15 on a: Integer[20]: the low 15 bits of each value, most significant
// first (Integer[20, 15]).
// a & 2^k > 0 is bit k of a's two's complement, i.e. bit k of a mod 2^15, also
// for negative a; those are the low bits of a + 2^63. to_le_bytes on a + 2^63
// proves it lies in [0, 2^64) (a in the window) and gives its bytes; the two
// low bytes are split into bits.
// @outputs: int[300]

use noir_hw::OFFSET;

global M: u32 = 15;

fn main(a: [Field; 20]) -> pub [Field; 300] {
    let mut result: [Field; 300] = [0; 300];
    for i in 0..20 {
        let bytes: [u8; 8] = (a[i] + OFFSET).to_le_bytes();
        let low = bytes[0] as Field + 256 * (bytes[1] as Field);
        let bits: [bool; 16] = low.to_le_bits();
        for j in 0..M {
            result[i * M + j] = bits[M - 1 - j] as Field;
        }
    }
    result
}
