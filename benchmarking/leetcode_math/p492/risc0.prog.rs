use risc0_zkvm::guest::env;

fn main() {
    // TODO: Implement your guest code here
    // read the input
    let area = env::read();
    let expected_l = env::read();
    let expected_w = env::read();

    let mut w = area;

    for i in 1..=1000 {
        if area % i == 0 {
            w = i;
        }
        if i * i >= area {
            break;
        }
    }

    let mut answer_l = area / w;
    let mut answer_w = w;

    if answer_w > answer_l {
        std::mem::swap(&mut answer_l, &mut answer_w);
    }

    assert!(
        answer_l == expected_l && answer_w == expected_w,
        "Expected dimensions ({}, {}), but got ({}, {})",
        expected_l,
        expected_w,
        answer_l,
        answer_w
    );

    // write public output to the journal
    // env::commit(&input);
}
