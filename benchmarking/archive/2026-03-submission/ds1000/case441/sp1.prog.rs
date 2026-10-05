// SP1 program code
#![no_main]
sp1_zkvm::entrypoint!(main);

pub fn main() {
    // Read inputs
    let mut a: [i32; 4] = [0; 4];
    for i in 0..4 {
        a[i] = sp1_zkvm::io::read::<i32>();
    }
    let number: i32 = sp1_zkvm::io::read::<i32>();
    let is_contained: i32 = sp1_zkvm::io::read::<i32>();

    let mut found: i32 = 0;
    for i in 0..4 {
        if a[i] == number {
            found = 1;
        }
    }

    assert_eq!(is_contained, found);
}
