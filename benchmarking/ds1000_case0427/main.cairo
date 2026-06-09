// DS-1000 case 427 — XOR-reduce then bit decomposition.
// kernel.py: res starts at 0; res ^= a[i] for each i; output is m bits of res MSB-first.
// a = [1..21], m = 5. XOR(1..20) = 20 = 0b10100. Expected bits: [1,0,1,0,0].

fn pow2_u64(k: u32) -> u64 {
    let mut r: u64 = 1;
    let mut i: u32 = 0;
    while i < k { r = r * 2; i += 1; };
    r
}

#[executable]
pub fn main() {
    let m: u32 = 5;
    let mut a: Array<u64> = ArrayTrait::new();
    let mut i: u64 = 1;
    while i <= 20 { a.append(i); i += 1; };
    let mut res: u64 = 0;
    for k in 0_usize..20 { res = res ^ *a.at(k); };
    let mut j: u32 = 0;
    while j < m {
        let shift: u32 = m - 1 - j;
        let mask: u64 = pow2_u64(shift);
        let _bit: u64 = if (res & mask) > 0 { 1 } else { 0 };
        j += 1;
    };
    assert!(res == 20);
}
