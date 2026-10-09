// pythran_multitype__times (Pythran multitype), Noir baseline.
// Kernel: `n * m` on two private integers, under the preconditions
// 0 <= n <= 1024 and 0 <= m <= 1024 (`@requires`, asserted).
// The precondition checks imply the int64 domain, so they replace the window
// check of either input (CONVENTIONS 1.6); the product is exact.
// @outputs: int[1]

// The bound of both preconditions (`@requires`), from the kernel's text.
global BOUND: Field = 1024;

// 0 <= x <= BOUND: x and BOUND - x are both 11-bit values (BOUND < 2^11), so
// neither wrapped around the field.
fn assert_in_bound(x: Field) {
    x.assert_max_bit_size::<11>();
    (BOUND - x).assert_max_bit_size::<11>();
}

fn main(n: Field, m: Field) -> pub Field {
    assert_in_bound(n);
    assert_in_bound(m);
    n * m
}
