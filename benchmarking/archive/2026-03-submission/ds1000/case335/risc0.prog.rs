// Risc 0 program code
use risc0_zkvm::guest::env;

fn main() {
    // read a, b, c (3-element arrays)
    let mut a: Vec<i32> = Vec::new();
    let mut b: Vec<i32> = Vec::new();
    let mut c: Vec<i32> = Vec::new();
    for _ in 0..3 { a.push(env::read()); }
    for _ in 0..3 { b.push(env::read()); }
    for _ in 0..3 { c.push(env::read()); }

    // read result (3-element array)
    let mut result: Vec<i32> = Vec::new();
    for _ in 0..3 { result.push(env::read()); }

    // compute elementwise max across a,b,c
    let mut computed: Vec<i32> = Vec::new();
    for i in 0..3 {
        let mut max_val = a[i as usize];
        if b[i as usize] > max_val {
            max_val = b[i as usize];
        }
        if c[i as usize] > max_val {
            max_val = c[i as usize];
        }
        computed.push(max_val);
    }

    // compare with result
    for i in 0..3 {
        assert_eq!(result[i as usize], computed[i as usize]);
    }

    // env::commit(&output);
}
