// DS-1000 case 0441: literal translation of kernel.py (LITERAL_BASELINES.md).
// NumPy array membership is elementwise equality followed by an index-order any reduction.
// @outputs: bool[1]
fn main(a: [Field; 4], number: Field) -> pub bool {
    noir_hw::check_inputs(a);
    noir_hw::check_input(number);
    let mut equal = [false; 4];
    for i in 0..4 { equal[i] = number == a[i]; }
    let mut contained = false;
    for i in 0..4 { contained = contained | equal[i]; }
    contained
}
