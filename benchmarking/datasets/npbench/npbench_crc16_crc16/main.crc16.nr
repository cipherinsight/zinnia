// Source CRC recurrence over 64 Boolean bits, not GF2 parity history.
// Arithmetic right shift repeats bit63. Final complement/mask/byte swap.
// @outputs: int[1]
// Sizes (hw_check.sizes): from fixture.json.
global L: u32 = 16; // = data_0
global WORD: u32 = 64;
global SIGN: u32 = WORD - 1;
global OUT_BITS: u32 = 16;
global OFFSET: Field = 9223372036854775808;
global HIGH_OFFSET: Field = 36028797018963968;
unconstrained fn offset_bits<let N: u32>(x: Field) -> [Field; N] {
    let v = (x + OFFSET) as u64;
    let mut bits: [Field; N] = [0; N];
    for j in 0..N {
        bits[j] = ((v >> (j as u64)) & 1) as Field;
    }
    bits
}

unconstrained fn high_hint(b: Field) -> Field {
    (((b + OFFSET) as u64) >> 8) as Field - HIGH_OFFSET
}

// The eight low bits of an int64 b (255 & b, two's complement), pinned by
// b = sum_j 2^j d_j + 256 h with each d_j a bit and h + 2^55 in [0, 2^56): the
// right side lies in [-2^63, 2^63) (b's domain), and the decomposition of a
// value there is unique.
fn low_byte_bits(b: Field) -> [Field; 8] {
    // Safety: pinned by the bit checks, the range check and the recomposition.
    let bits = unsafe { offset_bits::<8>(b) };
    // Safety: as above.
    let high = unsafe { high_hint(b) };
    let mut low: Field = 0;
    for j in 0..8 {
        assert(bits[j] * (bits[j] - 1) == 0, "low_byte_bits: not a bit");
        low += bits[j] * (1 << j) as Field;
    }
    (high + HIGH_OFFSET).assert_max_bit_size::<56>();
    assert(b == 256 * high + low, "low_byte_bits: recomposition");
    bits
}

// The 64 two's-complement bits of an int64 x (low first), pinned by
// x = sum_{k < 63} 2^k p_k - 2^63 p_63 with every p_k a bit (unique; the right
// side lies in [-2^63, 2^63), x's domain).
fn twos_complement_bits(x: Field) -> [Field; WORD] {
    // Safety: pinned by the bit checks and the recomposition.
    let offset = unsafe { offset_bits::<WORD>(x) };
    let mut bits = offset;
    bits[SIGN] = 1 - offset[SIGN];
    let mut sum: Field = 0;
    for k in 0..WORD {
        assert(bits[k] * (bits[k] - 1) == 0, "twos_complement_bits: not a bit");
        sum += offset[k] * 2.pow_32(k as Field);
    }
    assert(sum == x + OFFSET, "twos_complement_bits: recomposition");
    bits
}

fn xor(a: Field, b: Field) -> Field { a + b - 2 * a * b }
fn main(data: [Field; L], poly: Field) -> pub Field {
    let p = twos_complement_bits(poly);
    let mut crc: [Field; WORD] = [0; WORD];
    for i in 0..OUT_BITS { crc[i] = 1; }
    for b in data {
        let mut cur_byte = low_byte_bits(b);
        for _ in 0..8 {
            let feedback = xor(crc[0], cur_byte[0]);
            let old = crc;
            for i in 0..WORD {
                let index = if i < SIGN { i + 1 } else { SIGN };
                let shifted = old[index];
                let xored = xor(shifted, p[i]);
                crc[i] = if feedback == 1 { xored } else { shifted };
            }
            for i in 0..7 { cur_byte[i] = cur_byte[i + 1]; }
            cur_byte[7] = 0;
        }
    }
    let mut result: Field = 0;
    for i in 0..OUT_BITS {
        result += (1 - crc[i]) * (1 << ((i + 8) % OUT_BITS)) as Field;
    }
    result
}
