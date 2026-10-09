// pythran_multitype__times: literal translation of kernel.py (LITERAL_BASELINES.md).
// Exact integer product. Halo2 standard input loaders replace forbidden local witnesses; source preconditions are asserted.
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
    assert_in_bound(n); assert_in_bound(m);
    n * m
}
