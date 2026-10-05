// DS-1000 case 0299 — SP1 guest program.
// Kernel: temp = (a - a.min()).ravel(); b = zeros((200, max(temp)+1)); b[i, temp[i]]=1.

#![no_main]
sp1_zkvm::entrypoint!(main);

pub fn main() {
    const R: usize = 10;
    const C: usize = 20;
    const N: usize = R * C;
    let cols: u32 = sp1_zkvm::io::read::<u32>();
    let mut flat: Vec<i64> = Vec::with_capacity(N);
    for _ in 0..N { flat.push(sp1_zkvm::io::read::<i64>()); }
    let m = *flat.iter().min().unwrap();
    let temp: Vec<i64> = flat.iter().map(|v| v - m).collect();
    let mut b = vec![vec![0u64; cols as usize]; N];
    for i in 0..N { b[i][temp[i] as usize] = 1; }
    for row in &b {
        for v in row {
            let claim: u64 = sp1_zkvm::io::read::<u64>();
            assert_eq!(*v, claim);
        }
    }
}
