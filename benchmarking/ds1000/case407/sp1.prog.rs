// SP1 program code
#![no_main]
sp1_zkvm::entrypoint!(main);

pub fn main() {
    // read a
    let mut a: Vec<i32> = Vec::new();
    for _ in 0..10 {
        a.push(sp1_zkvm::io::read::<i32>());
    }

    // read accmap
    let mut accmap: Vec<i32> = Vec::new();
    for _ in 0..10 {
        accmap.push(sp1_zkvm::io::read::<i32>());
    }

    // read result
    let mut result: Vec<i32> = Vec::new();
    for _ in 0..3 {
        result.push(sp1_zkvm::io::read::<i32>());
    }

    // compute expected sums
    let mut sum0: i32 = 0;
    let mut sum1: i32 = 0;
    let mut sum2: i32 = 0;

    for i in 0..10 {
        if accmap[i as usize] == 0 {
            sum0 += a[i as usize];
        }
        if accmap[i as usize] == 1 {
            sum1 += a[i as usize];
        }
        if accmap[i as usize] == 2 {
            sum2 += a[i as usize];
        }
    }

    let expected: Vec<i32> = vec![sum0, sum1, sum2];

    for j in 0..3 {
        assert_eq!(result[j as usize], expected[j as usize]);
    }

    // sp1_zkvm::io::commit_slice(&output);
}
