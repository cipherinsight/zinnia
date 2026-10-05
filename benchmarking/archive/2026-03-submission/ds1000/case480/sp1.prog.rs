// SP1 program code
#![no_main]
sp1_zkvm::entrypoint!(main);

pub fn main() {
    // Read arrays
    let mut x: [i32; 9] = [0; 9];
    let mut y: [i32; 9] = [0; 9];
    for i in 0..9 {
        x[i] = sp1_zkvm::io::read::<i32>();
    }
    for i in 0..9 {
        y[i] = sp1_zkvm::io::read::<i32>();
    }

    let a: i32 = sp1_zkvm::io::read::<i32>();
    let b: i32 = sp1_zkvm::io::read::<i32>();
    let result: i32 = sp1_zkvm::io::read::<i32>();

    let mut found_index: i32 = -1;
    for i in 0..9 {
        if x[i] == a && y[i] == b && found_index == -1 {
            found_index = i as i32;
        }
    }

    assert_eq!(result, found_index);
}
