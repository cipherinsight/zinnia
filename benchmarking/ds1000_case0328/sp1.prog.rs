// DS-1000 case 0328 — SP1 guest program.
// Kernel: if length > A.shape[0]: pad zeros to length;
//         else: A.copy() with A[length:] zeroed.
// A: Integer[5], length: Integer. We pick length=8 (pad branch).

#![no_main]
sp1_zkvm::entrypoint!(main);

pub fn main() {
    const N: usize = 5;
    let length: u32 = sp1_zkvm::io::read::<u32>();
    let mut a: Vec<i64> = Vec::with_capacity(N);
    for _ in 0..N { a.push(sp1_zkvm::io::read::<i64>()); }
    let out_len = if (length as usize) > N { length as usize } else { N };
    let mut out: Vec<i64> = Vec::with_capacity(out_len);
    if (length as usize) > N {
        for i in 0..length as usize {
            out.push(if i < N { a[i] } else { 0 });
        }
    } else {
        for i in 0..N {
            out.push(if i < length as usize { a[i] } else { 0 });
        }
    }
    for v in &out {
        let claim: i64 = sp1_zkvm::io::read::<i64>();
        assert_eq!(*v, claim);
    }
}
