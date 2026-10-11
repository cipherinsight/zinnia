//! CRC-16-CCITT, matching zkvm.rs (including signed poly bit operations).
// output: i64
use crate::io::{ArgsTrait};
use crate::np::twos_complement;

#[executable]
fn main(inputs: Array<Array<i64>>) -> Array<i64> {
    let mut args = ArgsTrait::new(inputs);
    let data: Array<i64> = args.array1();
    let poly: i64 = args.scalar();
    let poly = twos_complement(poly);
    let mut crc: u64 = 65535;
    for b in data {
        let mut byte = twos_complement(b) & 255;
        for _ in 0..8_u32 {
            // Arithmetic right shift of the signed Rust i64 bit pattern.
            let shifted = crc / 2 + (crc & 0x8000000000000000);
            crc = if (crc & 1) ^ (byte & 1) != 0 { shifted ^ poly } else { shifted };
            byte /= 2;
        };
    };
    crc = (crc ^ 0xffffffffffffffff) & 65535;
    crc = ((crc * 256) | ((crc / 256) & 255)) & 65535;
    let mut out = args.output();
    out.append(crc.try_into().unwrap());
    out
}
