#[executable]
pub fn main() {
    let bank = array![array![0_u32, 1_u32, 0_u32, 0_u32, 0_u32], array![0_u32, 0_u32, 0_u32, 0_u32, 0_u32], array![1_u32, 0_u32, 1_u32, 0_u32, 0_u32], array![0_u32, 0_u32, 0_u32, 0_u32, 0_u32], array![0_u32, 1_u32, 0_u32, 1_u32, 0_u32]];
    let expected: u32 = 7_u32;

    for i in 0..5_u32 {
        for j in 0..5_u32 {
            assert!((*bank.at(i).at(j) == 0_u32) || (*bank.at(i).at(j) == 1_u32));
        }
    }

    let mut res = 0_u32;
    for si in 0..5_u32 {
        for sj in 0..5_u32 {
            for ti in 0..5_u32 {
                for tj in 0..5_u32 {
                    if si < ti {
                        if (*bank.at(si).at(sj) == 1_u32) && (*bank.at(ti).at(tj) == 1_u32) {
                            let mut blocked: bool = false;
                            for k in (si + 1_u32)..ti {
                                let mut row_block: bool = false;
                                for j in sj..tj {
                                    if *bank.at(k).at(j) == 1_u32 {
                                        row_block = true;
                                    }
                                }
                                if row_block {
                                    blocked = true;
                                }
                            }
                            if !blocked {
                                res = res + 1_u32;
                            }
                        }
                    }
                }
            }
        }
    }

    assert!(res == expected);
}
