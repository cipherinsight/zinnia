// Risc 0 program code
use risc0_zkvm::guest::env;

fn main() {
    // Read arrays
    let mut x: [i32; 9] = [0; 9];
    let mut y: [i32; 9] = [0; 9];
    for i in 0..9 {
        x[i] = env::read();
    }
    for i in 0..9 {
        y[i] = env::read();
    }

    let a: i32 = env::read();
    let b: i32 = env::read();
    let result: i32 = env::read();

    // Compute expected index
    let mut found_index: i32 = -1;
    for i in 0..9 {
        if x[i] == a && y[i] == b && found_index == -1 {
            found_index = i as i32;
        }
    }

    assert_eq!(result, found_index);
}
