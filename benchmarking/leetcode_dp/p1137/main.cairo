#[executable]
pub fn main() {
    let n: u128 = 100_u128;
    let sol: u128 = 98079530178586034536500564_u128;

    assert!(n >= 0_u128);
    assert!(n < 101_u128);

    if (n == 0_u128) {
        assert!(sol == 0_u128);
    } else if (n == 1_u128) {
        assert!(sol == 1_u128);
    } else if (n == 2_u128) {
        assert!(sol == 1_u128);
    } else {
        let mut a: u128 = 0_u128;
        let mut b: u128 = 1_u128;
        let mut c: u128 = 1_u128;

        for i in 3_u128..101_u128 {
            let t: u128 = a + b + c;
            a = b;
            b = c;
            c = t;

            if i == n {
                assert!(sol == c);
            }
        }
    }
}