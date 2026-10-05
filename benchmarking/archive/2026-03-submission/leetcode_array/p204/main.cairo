#[executable]
pub fn main() {
    let n = 1000_u32;
    let result = 168_u32;

    if n == 0_u32 {
        assert!(result == 0_u32);
    }
    if n == 1_u32 {
        assert!(result == 0_u32);
    }

    if !(n == 0_u32) && !(n == 1_u32) {
        let mut count = 0_u32;
        for i in 2_u32..n {
            let mut is_prime = true;
            for j in 2_u32..i {
                if i % j == 0_u32 {
                    is_prime = false;
                }
            }
            if is_prime {
                count += 1_u32;
            }
        }
        assert!(count == result);
    }
}