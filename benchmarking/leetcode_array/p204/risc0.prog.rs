use risc0_zkvm::guest::env;

fn main() {
    // TODO: Implement your guest code here
    // read the input
    let n: u32 = env::read();
    let result: u32 = env::read();

    assert!(n <= 1000, "n must be between 0 and 1000 inclusive");

    // For n equal to 0 or 1, the expected result is 0.
    if n == 0 || n == 1 {
        assert_eq!(result, 0, "For n = 0 or 1, result must be 0");
    } else {
        // Initialize a vector with 1001 elements set to 1.
        let mut is_prime = vec![1; 1001];
        let mut number_of_primes = 0;

        // Iterate from 2 to 1000 inclusive.
        for i in 2..=1000 {
            if is_prime[i] == 1 {
                number_of_primes += 1;
                // Mark all multiples of i as non-prime.
                for j in (i..=1000).step_by(i) {
                    is_prime[j] = 0;
                }
            }
            // When i equals n, verify that the number of primes found equals the result.
            if (i as u32) == n {
                assert_eq!(
                    number_of_primes, result,
                    "At i = {}, expected number of primes to be {}",
                    i, result
                );
            }
        }
    }

    // write public output to the journal
    // env::commit(&input);
}
