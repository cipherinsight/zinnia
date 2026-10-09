// npbench_crc16_crc16__crc16: literal translation of kernel.py (LITERAL_BASELINES.md).
// Bit arrays implement signed shifts, XOR and masks directly; standard bit decomposition only.
// @outputs: int[1]

// Sizes (hw_check.sizes): from fixture.json.
global L: u32 = 16; // = data_0
global WORD: u32 = 64;
global SIGN: u32 = WORD - 1;
global OUT_BITS: u32 = 16;
global OFFSET: Field = 9223372036854775808;

fn twos_complement_bits(x: Field) -> [Field; WORD] {
    let offset: [bool; WORD] = (x + OFFSET).to_le_bits();
    let mut bits = [0; WORD];
    for i in 0..WORD { bits[i] = offset[i] as Field; }
    bits[SIGN] = 1-bits[SIGN];
    bits
}
fn low_byte_bits(b: Field) -> [Field; 8] {
    let bits = twos_complement_bits(b);
    let mut low = [0; 8];
    for i in 0..8 { low[i] = bits[i]; }
    low
}

fn xor(a: Field, b: Field) -> Field {
    a + b - 2 * a * b
}

fn main(data: [Field; L], poly: Field) -> pub Field {
    let p = twos_complement_bits(poly);
    // crc = 65535: the low 16 bits set, the rest clear.
    let mut crc: [Field; WORD] = [0; WORD];
    for i in 0..OUT_BITS {
        crc[i] = 1;
    }
    for b in data {
        // cur_byte = 255 & b.
        let mut cur_byte = low_byte_bits(b);
        for _ in 0..8 {
            // if crc & 1 ^ cur_byte & 1: crc = crc >> 1 ^ poly else: crc >>= 1
            let feedback = xor(crc[0], cur_byte[0]);
            let old = crc;
            for i in 0..WORD {
                let index = if i < SIGN { i + 1 } else { SIGN };
                let shifted = old[index];
                let xored = xor(shifted, p[i]);
                crc[i] = if feedback == 1 { xored } else { shifted };
            }
            // cur_byte >>= 1
            for i in 0..7 {
                cur_byte[i] = cur_byte[i + 1];
            }
            cur_byte[7] = 0;
        }
    }
    // crc = ~crc & 65535
    let mut masked = [0; WORD];
    for i in 0..WORD {
        let complemented = 1-crc[i];
        masked[i] = if i < OUT_BITS { complemented } else { 0 };
    }
    // crc = crc << 8 | crc >> 8 & 255
    let mut combined = [0; WORD];
    for i in 0..WORD {
        let left = if i >= 8 { masked[i-8] } else { 0 };
        let right = if i < 8 { masked[i+8] } else { 0 };
        combined[i] = left + right - left*right;
    }
    // return crc & 65535
    let mut result: Field = 0;
    for i in 0..OUT_BITS { result += combined[i] * (1 << i) as Field; }
    result
}
