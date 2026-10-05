// DS-1000 case 426 — bit decomposition matrix.
// kernel.py: ((a[:,None] & (1 << np.arange(m))[::-1]) > 0).astype(int)
// a = [0..20), m = 5 → 20×5 bit matrix (MSB-first).

#[executable]
pub fn main() {
    let m: u32 = 5;
    let mut a: Array<u64> = ArrayTrait::new();
    let mut i: u64 = 0;
    while i < 20 { a.append(i); i += 1; };
    for i in 0_usize..20 {
        let v: u64 = *a.at(i);
        for j in 0_u32..m {
            let shift: u32 = m - 1 - j;
            let mask: u64 = 1_u64 * pow2_u64(shift);
            let bit: u64 = if (v & mask) > 0 { 1 } else { 0 };
            // No claim comparison; this mirrors the kernel's compute path.
            let _ = bit;
        };
    };
}

fn pow2_u64(k: u32) -> u64 {
    let mut r: u64 = 1;
    let mut i: u32 = 0;
    while i < k { r = r * 2; i += 1; };
    r
}
