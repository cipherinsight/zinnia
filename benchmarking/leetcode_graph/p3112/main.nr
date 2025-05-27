fn main(graph: [[Field; 10]; 10], disappear: [Field; 10], answers: [Field; 10]) {
    let mut g: [[Field; 10]; 10] = graph;
    let zero: Field = 0;  // no minus 1 in Noir so we use 0

    for k in 0..10 {
        for i in 0..10 {
            for j in 0..10 {
                if (g[i][k] != zero) & (g[k][j] != zero) {
                    let sum = g[i][k] + g[k][j];
                    let old = g[i][j];
                    let new = if old.lt(sum) { old } else { sum };
                    g[i][j] = new;
                }
            }
        }
    }

    for i in 0..10 {
        let gi = g[0][i];
        let di = disappear[i];
        if (gi != zero) & !di.lt(gi) {
            assert(answers[i] == gi, "answers[i]");
        } else {
            assert(answers[i] == zero, "answers[i] must be 0");
        }
    }
}
