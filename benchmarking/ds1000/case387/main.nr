fn main(a: [[Field; 4]; 4], result: [[[Field; 2]; 2]; 4]) {
    for k in 0..4 {
        let i_0: u64 = k / 2;
        let j_0: u64 = k % 2;
        for u in 0..2 {
            for v in 0..2 {
                let t1: u64 = i_0 * 2 + u;
                let t2: u64 = j_0 * 2 + v;
                let orig = a[t1][t2];
                let out  = result[k][u][v];
                assert(orig == out, "Mismatch at block {k}, pos ({u},{v})");
            }
        }
    }
}
