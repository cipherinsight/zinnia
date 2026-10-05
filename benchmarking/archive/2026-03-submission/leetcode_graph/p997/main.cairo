#[executable]
pub fn main() {
    let judge_id = 10_u32;
    let trust_graph = array![array![1_u32, 1_u32, 0_u32, 0_u32, 1_u32, 1_u32, 0_u32, 1_u32, 0_u32, 1_u32], array![1_u32, 1_u32, 1_u32, 0_u32, 0_u32, 1_u32, 0_u32, 1_u32, 0_u32, 1_u32], array![0_u32, 1_u32, 1_u32, 1_u32, 1_u32, 1_u32, 1_u32, 0_u32, 0_u32, 1_u32], array![0_u32, 0_u32, 1_u32, 1_u32, 1_u32, 0_u32, 1_u32, 1_u32, 0_u32, 1_u32], array![0_u32, 0_u32, 0_u32, 1_u32, 1_u32, 1_u32, 0_u32, 0_u32, 1_u32, 1_u32], array![1_u32, 0_u32, 0_u32, 0_u32, 1_u32, 1_u32, 1_u32, 1_u32, 1_u32, 1_u32], array![0_u32, 1_u32, 0_u32, 0_u32, 0_u32, 1_u32, 1_u32, 1_u32, 0_u32, 1_u32], array![0_u32, 0_u32, 0_u32, 1_u32, 1_u32, 0_u32, 1_u32, 1_u32, 1_u32, 1_u32], array![0_u32, 0_u32, 0_u32, 0_u32, 0_u32, 0_u32, 0_u32, 1_u32, 1_u32, 1_u32], array![0_u32, 0_u32, 0_u32, 0_u32, 0_u32, 0_u32, 0_u32, 0_u32, 0_u32, 1_u32]];

    assert!(judge_id >= 1_u32);
    assert!(judge_id < 11);

    let judge_idx: u32 = judge_id - 1_u32;

    for i in 0..10_u32 {
        for j in 0..10_u32 {
            let fi = i;
            let fj = j;

            if (fi == judge_idx) && (fj != judge_idx) {
                assert!(*trust_graph.at(i).at(j) == 0_u32);
            }

            if (fj == judge_idx) && (fi != judge_idx) {
                assert!(*trust_graph.at(i).at(j) == 1_u32);
            }
        }
    }
}