// npbench_crc16_crc16__crc16 (NPBench crc16), Noir baseline.
// Kernel: CRC-16-CCITT of the low bytes of data: Int[L] with a private int64
// poly, bit by bit: crc starts at 65535; for each of the STEPS = 8 L data bits
// d_t (low bit first), c_t = (crc & 1) ^ d_t, then crc = (crc >> 1) ^ poly if
// c_t else crc >> 1; the result is ~crc & 65535 with its two bytes swapped.
// crc is an int64 throughout (>> is arithmetic, so its bit 63 repeats), and
// the steps are linear over GF(2) given the c_t: bit i of crc after step t is
// bit min(63, i + t) of 65535 XOR the bits c_u & p[min(63, i + t - 1 - u)] for
// u < t, p the bits of poly. An XOR of bits is the parity of their integer
// sum, so each c_t is the parity of d_t + bit_t(65535) + sum_u c_u p[t - 1 - u]
// (the terms with index 63 summed as p[63] times a running count of the c_u),
// and the 16 output bits likewise. Each parity is a hinted bit pinned by
// sum = 2 h + bit with h range-checked. This costs one product per
// (step, earlier step) pair, about 7,000, against two gates per bit and step
// when crc is carried as 64 bits.
// The decompositions of the inputs into bits are their domain checks: a data
// element is 256 h + (its low byte) with h in [-2^55, 2^55), and poly is the
// two's-complement sum of 64 bits, both in [-2^63, 2^63).
// @outputs: int[1]

// Sizes (hw_check.sizes): from fixture.json.
global L: u32 = 16; // = data_0
global STEPS: u32 = 8 * L;

// The initial crc, from the kernel's text.
global INIT: u64 = 65535;
// The bits of an int64, and the index of its sign bit.
global WORD: u32 = 64;
global SIGN: u32 = WORD - 1;
// The output keeps the low 16 bits.
global OUT_BITS: u32 = 16;
global OFFSET: Field = 9223372036854775808; // 2^63
global HIGH_OFFSET: Field = 36028797018963968; // 2^55
// The parity sums are below 2^15; their halves are checked at the 14-bit limb
// width of the 56-bit check, so no other range table is built.
global HALF_BITS: u32 = 14;

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

unconstrained fn half_hint(sum: Field) -> Field {
    ((sum as u64) / 2) as Field
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

// The parity of a sum of at most STEPS + 2 bits: sum = 2 h + bit with bit a
// bit and h a HALF_BITS-bit value (the sum is a small non-negative integer,
// so (h, bit) is its unique decomposition).
fn parity(sum: Field) -> Field {
    // Safety: pinned by the bit check and the range check.
    let half = unsafe { half_hint(sum) };
    let bit = sum - 2 * half;
    assert(bit * (bit - 1) == 0, "parity: not a bit");
    half.assert_max_bit_size::<HALF_BITS>();
    bit
}

fn init_bit(i: u32) -> Field {
    let index = if i < SIGN { i } else { SIGN };
    ((INIT >> (index as u64)) & 1) as Field
}

// The parity of: bit i + t of INIT, `extra`, and c_u p[i + t - 1 - u] for
// u < t (index capped at 63), i.e. bit i of crc after t steps (XOR `extra`).
// count[x] is c[0] + ... + c[x - 1].
fn crc_bit(c: [Field; STEPS], count: [Field; STEPS + 1], p: [Field; WORD], i: u32, t: u32, extra: Field) -> Field {
    // Below index 63: u from i + t - 63 on.
    let first = if i + t > SIGN { i + t - SIGN } else { 0 };
    let mut sum = p[SIGN] * count[first] + init_bit(i + t) + extra;
    for u in first..t {
        sum += c[u] * p[i + t - 1 - u];
    }
    parity(sum)
}

fn main(data: [Field; L], poly: Field) -> pub Field {
    let mut d: [Field; STEPS] = [0; STEPS];
    for e in 0..L {
        let bits = low_byte_bits(data[e]);
        for j in 0..8 {
            d[8 * e + j] = bits[j];
        }
    }
    let p = twos_complement_bits(poly);
    let mut c: [Field; STEPS] = [0; STEPS];
    let mut count: [Field; STEPS + 1] = [0; STEPS + 1];
    for t in 0..STEPS {
        c[t] = crc_bit(c, count, p, 0, t, d[t]);
        count[t + 1] = count[t] + c[t];
    }
    // ~crc & 65535 with its bytes swapped: output bit k is NOT crc bit
    // (k + 8) % 16.
    let mut result: Field = 0;
    for i in 0..OUT_BITS {
        let bit = 1 - crc_bit(c, count, p, i, STEPS, 0);
        result += bit * (1 << ((i + OUT_BITS / 2) % OUT_BITS)) as Field;
    }
    result
}
