fn main(bank: [[Field; 5]; 5], expected: Field) {
    for i in 0..5 {
        for j in 0..5 {
            assert((bank[i][j] == 0) | (bank[i][j] == 1), "bank must be 0 or 1");
        }
    }

    let mut res: Field = 0;
    for si in 0..5 {
        for sj in 0..5 {
            for ti in 0..5 {
                for tj in 0..5 {
                    if si < ti {
                        if (bank[si][sj] == 1) & (bank[ti][tj] == 1) {
                            let mut blocked: bool = false;
                            for k in (si + 1)..ti {
                                let mut row_block: bool = false;
                                for j in sj..tj {
                                    if bank[k][j] == 1 {
                                        row_block = true;
                                    }
                                }
                                if row_block {
                                    blocked = true;
                                }
                            }
                            if !blocked {
                                res = res + 1;
                            }
                        }
                    }
                }
            }
        }
    }

    assert(res == expected, "Expected paths, got");
}