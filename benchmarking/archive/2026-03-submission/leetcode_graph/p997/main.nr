fn main(trust_graph: [[Field; 10]; 10], judge_id: Field) {
    assert(!judge_id.lt(1), "judge_id must be >= 1");
    assert(judge_id.lt(11), "judge_id must be <= 10");

    let judge_idx = judge_id - 1;

    for i in 0..10 {
        for j in 0..10 {
            let fi = Field::from(i as u128);
            let fj = Field::from(j as u128);

            if (fi == judge_idx) & (fj != judge_idx) {
                assert(trust_graph[i][j] == 0, "trust_graph must be 0 when i is judge");
            }

            if (fj == judge_idx) & (fi != judge_idx) {
                assert(trust_graph[i][j] == 1, "trust_graph must be 1 when j is judge");
            }
        }
    }
}
