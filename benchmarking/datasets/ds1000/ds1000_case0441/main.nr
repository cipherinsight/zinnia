// DS-1000 case 0441, Noir baseline.
// Kernel: `number in a` on a: Integer[4], number: Integer.
// `number in a` is `(a == number).any()`: the product of the differences
// number - a[k] is zero exactly when one of them is (a field has no zero
// divisors), so one zero test replaces four equality tests and their `or`.
// @outputs: bool[1]

fn main(a: [Field; 4], number: Field) -> pub bool {
    // Input domain: every input element is an int64 / a Q32 code in the window.
    noir_hw::check_inputs(a);
    noir_hw::check_input(number);
    let mut product: Field = 1;
    for k in 0..4 {
        product *= number - a[k];
    }
    product == 0
}
