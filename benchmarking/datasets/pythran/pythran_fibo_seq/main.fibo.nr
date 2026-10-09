// Sequential Fibonacci recurrence, preserving kernel.py's private n loop.
// No matrix powering, derived n<=90 shortcut, or extra state range guards.
// The intended int64-result comparison covers n<=90; larger n is field arithmetic.
// @outputs: int[1]
use noir_hw::window;
global MAX_STEPS: u32 = 1024;
fn main(n: Field) -> pub [Field; 1] {
    let nw = window(n);
    assert(nw >= window(0));
    assert(nw <= window(MAX_STEPS as Field));
    let mut a: Field = 1;
    let mut b: Field = 1;
    for step in 0..MAX_STEPS {
        let active = nw > window(step as Field);
        let next = a + b;
        let next_a = if active { next } else { a };
        b = if active { a } else { b };
        a = next_a;
    }
    [a]
}
