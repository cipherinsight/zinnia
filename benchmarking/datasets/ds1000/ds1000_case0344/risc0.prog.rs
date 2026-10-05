// DS-1000 case 344 — Risc0 guest program.

use risc0_zkvm::guest::env;

fn main() {
    let mystr: i64 = env::read();
    let claimed: i64 = env::read();

    let result: i64 = mystr;

    assert_eq!(result, claimed);
}
