#[executable]
pub fn main() {
    let answers = array![1_u32, 1_u32, 1_u32, 1_u32, 1_u32, 1_u32, 1_u32, 1_u32, 0_u32, 0_u32];
    let disappear = array![1_u32, 2_u32, 3_u32, 4_u32, 5_u32, 6_u32, 7_u32, 8_u32, 1_u32, 10_u32];
    let mut g = array![array![1_u32, 1_u32, 1_u32, 1_u32, 1_u32, 1_u32, 1_u32, 1_u32, 2_u32, 0_u32],  array![1_u32, 1_u32, 1_u32, 0_u32, 1_u32, 1_u32, 0_u32, 1_u32, 1_u32, 0_u32],  array![1_u32, 6_u32, 1_u32, 0_u32, 0_u32, 1_u32, 0_u32, 1_u32, 1_u32, 0_u32],  array![0_u32, 1_u32, 1_u32, 0_u32, 0_u32, 4_u32, 0_u32, 1_u32, 1_u32, 0_u32],  array![0_u32, 6_u32, 6_u32, 0_u32, 0_u32, 1_u32, 0_u32, 8_u32, 8_u32, 0_u32],  array![0_u32, 1_u32, 1_u32, 5_u32, 1_u32, 0_u32, 8_u32, 1_u32, 3_u32, 0_u32],  array![1_u32, 1_u32, 6_u32, 3_u32, 4_u32, 0_u32, 8_u32, 1_u32, 3_u32, 0_u32],  array![9_u32, 1_u32, 1_u32, 6_u32, 9_u32, 9_u32, 1_u32, 1_u32, 1_u32, 0_u32],  array![9_u32, 1_u32, 1_u32, 9_u32, 6_u32, 9_u32, 9_u32, 1_u32, 8_u32, 0_u32],  array![0_u32, 0_u32, 0_u32, 0_u32, 0_u32, 0_u32, 0_u32, 0_u32, 0_u32, 0_u32]];


    for k in 0..10_u32 {
        for i in 0..10_u32 {
            for j in 0..10_u32 {
                if (*g.at(i).at(k) != 0_u32) && (*g.at(k).at(j) != 0_u32) {
                    let sum = *g.at(i).at(k) + *g.at(k).at(j);
                    let old = *g.at(i).at(j);
                    let mut new: u32 = 0;
                    if old < sum {
                        new = old;
                    } else {
                        new = sum;
                    }
                    for m in 0..10_u32 {
                        if m == i {
                            let mut tmp_arr = g.pop_front().unwrap();
                            for t in 0..10_u32 {
                                if t == j {
                                    tmp_arr.pop_front().unwrap();
                                    tmp_arr.append(new);
                                } else {
                                    let tmp_val = tmp_arr.pop_front().unwrap();
                                    tmp_arr.append(tmp_val);
                                }
                            }
                            g.append(tmp_arr);
                        } else {
                            let mut tmp_arr = g.pop_front().unwrap();
                            g.append(tmp_arr);
                        }
                    }
                }
            }
        }
    }

    for i in 0..10_u32 {
        let gi = *g.at(0).at(i);
        let di = *disappear.at(i);
        if (gi != 0_u32) && di >= gi {
            assert!(*answers.at(i) == gi);
        } else {
            assert!(*answers.at(i) == 0_u32);
        }
    }

}