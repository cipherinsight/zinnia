// DS-1000 case 0328 — Risc0 guest program.

use risc0_zkvm::guest::env;

fn main() {
    const N: usize = 5;
    let length: u32 = env::read();
    let mut a: Vec<i64> = Vec::with_capacity(N);
    for _ in 0..N { a.push(env::read()); }
    let mut out: Vec<i64> = Vec::new();
    if (length as usize) > N {
        for i in 0..length as usize { out.push(if i < N { a[i] } else { 0 }); }
    } else {
        for i in 0..N { out.push(if i < length as usize { a[i] } else { 0 }); }
    }
    for v in &out {
        let claim: i64 = env::read();
        assert_eq!(*v, claim);
    }
}
