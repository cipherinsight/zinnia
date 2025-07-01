fn main(n: Field, result: Field) {
    assert(!n.lt(0), "n must be >= 0");
    assert(n.lt(101), "n must be <= 100");

    if n == 0 {
        assert(result == 0, "result must be 0 when n = 0");
    }
    if n == 1 {
        assert(result == 0, "result must be 0 when n = 1");
    }

    if !(n == 0) & !(n == 1) {
        let mut is_prime: [Field; 101] = [1; 101];
        is_prime[0] = 0;
        is_prime[1] = 0;

        let mut count: Field = 0;

        for i in 2..101 {
            if is_prime[i] == 1 {
                count = count + 1;
                for j in 2..101 {
                    if j % i == 0 {
                        is_prime[j] = 0;
                    }
                }
            }

            if Field::from(i as u128) == n {
                assert(count == result, "At i = i, expected count result, got count");
            }
        }
    }
}