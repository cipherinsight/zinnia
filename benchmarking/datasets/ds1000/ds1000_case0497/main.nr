// DS-1000 case 0497, Noir baseline.
// Kernel: `np.pad(a, ((0, 93 - 42), (0, 13 - 10)), 'constant',
// constant_values=element)` on Float[42, 10] and an Integer element
// (`shape = (93, 13)` is a constant): each row of a followed by 3 copies of
// float(element), then 51 rows of it. The padding depends on the private
// element, so those values are public copies of the one converted value.
// Precondition: |element| < 2^31, so that float(element) fits Q32.
// @outputs: float[1209]

use noir_hw::from_int;

fn main(a: [Field; 420], element: Field) -> pub [Field; 1209] {
    let fill = from_int(element);
    let mut result = [fill; 1209];
    for i in 0..42 {
        for j in 0..10 {
            result[i * 13 + j] = a[i * 10 + j];
        }
    }
    result
}
