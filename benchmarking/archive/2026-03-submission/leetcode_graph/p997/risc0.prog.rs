use risc0_zkvm::guest::env;

fn main() {
    // TODO: Implement your guest code here
    // read the input
    let mut trust_graph: Vec<Vec<u32>> = Vec::new();
    for i in 0..10 {
        let mut tmp: Vec<u32> = Vec::new();
        for j in 0..10 {
            tmp.push(env::read());
        }
        trust_graph.push(tmp);
    }
    let judge_id: u32 = env::read();
    assert_eq!(trust_graph.len(), 10, "Trust graph must have 10 rows");
    for row in trust_graph.iter() {
        assert_eq!(
            row.len(),
            10,
            "Each row in the trust graph must have 10 columns"
        );
    }

    let judge_index = judge_id - 1;

    for i in 0..10 {
        for j in 0..10 {
            if i == judge_index && i != j {
                assert_eq!(
                    trust_graph[i as usize][j as usize], 0,
                    "Judge (index {}) should not trust anyone, but trusts person {}",
                    i, j
                );
            }
            if j == judge_index && i != j {
                assert_eq!(
                    trust_graph[i as usize][j as usize], 1,
                    "Everyone should trust the judge (index {}), but person {} does not",
                    j, i
                );
            }
        }
    }

    // write public output to the journal
    // env::commit(&input);
}
