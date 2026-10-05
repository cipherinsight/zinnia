fn main(n: Field, sol: Field) {
    assert(!n.lt(0),  "n must be >= 0");
    assert(n.lt(101), "n must be < 101");

    if (n == 0) {
        assert(sol == 0, "T0 = 0");
    } else if (n == 1) {
        assert(sol == 1, "T1 = 1");
    } else if (n == 2) {
        assert(sol == 1, "T2 = 1");
    } else {
        let mut a: Field = 0;
        let mut b: Field = 1;
        let mut c: Field = 1;

        for i in 3..101 {
            let t = a + b + c;
            a = b;
            b = c;
            c = t;

            if Field::from(i as u128) == n {
                assert(sol == c, "T_n mismatch");
            }
        }
    }
}
