// DS-1000 case 344 — SP1 guest program.
// Kernel: `np.array(list(mystr), dtype=int)` on Integer scalar → echoes the input.
// We mirror Zinnia's IR which treats this as a passthrough of the scalar.

#![no_main]
sp1_zkvm::entrypoint!(main);

pub fn main() {
    let mystr: i64 = sp1_zkvm::io::read::<i64>();
    let claimed: i64 = sp1_zkvm::io::read::<i64>();

    let result: i64 = mystr;

    assert_eq!(result, claimed);
}
