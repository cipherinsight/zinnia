// pythran_fibo_seq__fibo: literal translation of kernel.py (LITERAL_BASELINES.md).
// Private range(n) is bounded by the explicit 1024 precondition with selected state. Output uses field arithmetic without an int64 guard, matching the optimized statement.
// @outputs: int[1]

fn main(n: Field) -> pub [Field; 1] {
    noir_hw::check_input(n);
    assert(noir_hw::le(0, n));
    assert(noir_hw::le(n, 1024));
    let mut a: Field = 1;
    let mut b: Field = 1;
    for step in 0..1024 {
        let active = noir_hw::lt(step as Field, n);
        let next = a + b;
        let na = if active { next } else { a };
        b = if active { a } else { b };
        a = na;
    }
    [a]
}
