use risc0_zkvm::guest::env;

fn main() {
    // TODO: Implement your guest code here
    // read the input
    let n: u32 = env::read();
    let sol: u128 = env::read();
    assert!(n != 0 || sol == 0, "For n = 0, sol must be 0");
    assert!(n != 1 || sol == 1, "For n = 1, sol must be 1");
    assert!(n != 2 || sol == 1, "For n = 2, sol must be 1");

    let (mut a, mut b, mut c) = (0, 1, 1);

    for i in 3..=100 {
        let next = a + b + c;
        a = b;
        b = c;
        c = next;

        assert!(
            n != i || sol == c,
            "For n = {}, expected sol = {}, but got {}",
            i,
            c,
            sol
        );
    }

    // write public output to the journal
    // env::commit(&input);
}
