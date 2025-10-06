// SP1 program code
#![no_main]
sp1_zkvm::entrypoint!(main);

pub fn main() {
    let mut a: [i32; 3] = [0; 3];
    let mut b: [i32; 3] = [0; 3];
    let mut c: [i32; 3] = [0; 3];
    let mut result: [i32; 3] = [0; 3];

    for i in 0..3 {
        a[i] = sp1_zkvm::io::read::<i32>();
    }
    for i in 0..3 {
        b[i] = sp1_zkvm::io::read::<i32>();
    }
    for i in 0..3 {
        c[i] = sp1_zkvm::io::read::<i32>();
    }
    for i in 0..3 {
        result[i] = sp1_zkvm::io::read::<i32>();
    }

    let mut expected: [i32; 3] = [0; 3];
    for i in 0..3 {
        let ab_max = if a[i] > b[i] { a[i] } else { b[i] };
        expected[i] = if ab_max > c[i] { ab_max } else { c[i] };
    }

    for i in 0..3 {
        assert_eq!(result[i], expected[i]);
    }
}
