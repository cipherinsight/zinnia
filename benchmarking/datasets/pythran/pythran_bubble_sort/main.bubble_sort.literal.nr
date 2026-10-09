// pythran_bubble_sort__bubble_sort: literal translation of kernel.py (LITERAL_BASELINES.md).
// Source bubble loops, fresh bounds per comparison. swap_test=True is unconditional, so break depends only on loop shape.
// @outputs: int[N]

// Sizes (hw_check.sizes): from fixture.json.
global N: u32 = 11; // = list0_0
fn main(list0: [Field; N]) -> pub [Field; N] {
    noir_hw::check_inputs(list0);
    let mut list1 = list0;
    for i in 0..N - 1 {
        // swap_test starts False, becomes True unconditionally on each j;
        // N-i-1 >= 1, so the final break is compile-time unreachable.
        for j in 0..N - i - 1 {
            let swap = noir_hw::gt(list1[j], list1[j + 1]);
            let left = list1[j];
            let right = list1[j + 1];
            list1[j] = if swap { right } else { left };
            list1[j + 1] = if swap { left } else { right };
        }
    }
    list1
}
