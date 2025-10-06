// Risc 0 program code
use risc0_zkvm::guest::env;

fn main() {
    // Read input array a[4]
    let mut a: [i32; 4] = [0; 4];
    for i in 0..4 {
        a[i] = env::read();
    }

    // Read number and result
    let number: i32 = env::read();
    let is_contained: i32 = env::read();

    // Compute found flag
    let mut found: i32 = 0;
    for i in 0..4 {
        if a[i] == number {
            found = 1;
        }
    }

    assert_eq!(is_contained, found);
}
