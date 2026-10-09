// pythran_insertion_sort__insertion_sort: literal translation of kernel.py (LITERAL_BASELINES.md).
// While bounded by i; private j retained, active gates every effect. Reads use indexing; writes scan equality tests. At j=0 the inactive -1 read is valid.
// @outputs: float[N]
global N: u32 = 10; // = list2_0
global INDEX_BITS: u32 = 8; // = 8 + 20 * min(1, (list2_0 - 1) // 128)

fn main(list2: [Field; N]) -> pub [Field; N] {
    noir_hw::check_inputs(list2);
    let mut values = list2;
    for i in 1..N {
        let save = values[i];
        let mut j = i as Field;
        let mut active = true;
        for _step in 0..i {
            let positive = noir_hw::gt(j, 0);
            let compared = values[noir_hw::index_pos::<INDEX_BITS>(j - 1, N)];
            let larger = noir_hw::gt(compared, save);
            active = active & positive & larger;
            let shifted = values[noir_hw::index_pos::<INDEX_BITS>(j - 1, N)];
            for k in 0..N {
                values[k] = if active & (j == k as Field) {
                    shifted
                } else {
                    values[k]
                };
            }
            j = if active { j - 1 } else { j };
        }
        for k in 0..N {
            values[k] = if j == k as Field { save } else { values[k] };
        }
    }
    values
}
