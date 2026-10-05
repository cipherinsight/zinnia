// DS-1000 case 400 — `np.array([])`. Empty output, no IR work. Trivial.

#[executable]
pub fn main() {
    let _result: Array<u128> = ArrayTrait::new();
}
