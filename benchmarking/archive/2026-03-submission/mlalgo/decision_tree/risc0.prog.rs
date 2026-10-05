// Risc 0 program code
use risc0_zkvm::guest::env;

fn main() {
    // read training_x (10x2)
    let mut training_x: Vec<[f64; 2]> = Vec::new();
    for _ in 0..10 {
        let x0: f64 = env::read();
        let x1: f64 = env::read();
        training_x.push([x0, x1]);
    }

    // read training_y (10)
    let mut training_y: Vec<i32> = Vec::new();
    for _ in 0..10 {
        training_y.push(env::read());
    }

    // read testing_x (2x2)
    let mut testing_x: Vec<[f64; 2]> = Vec::new();
    for _ in 0..2 {
        let x0: f64 = env::read();
        let x1: f64 = env::read();
        testing_x.push([x0, x1]);
    }

    // read testing_y (2)
    let mut testing_y: Vec<i32> = Vec::new();
    for _ in 0..2 {
        testing_y.push(env::read());
    }

    let thr: [f64; 3] = [0.5, 1.0, 1.5];

    let mut best_err: i32 = 999;
    let mut best_feat_r = 0;
    let mut best_thr_r = 0;
    let mut best_feat_l = 0;
    let mut best_thr_l = 0;
    let mut best_feat_rr = 0;
    let mut best_thr_rr = 0;

    // exhaustive search (static loops)
    for fr in 0..2 {
        for tr in 0..3 {
            for fl in 0..2 {
                for tl in 0..3 {
                    for fr2 in 0..2 {
                        for tr2 in 0..3 {
                            let mut err = 0;
                            for i in 0..10 {
                                let go_right = if training_x[i][fr as usize] >= thr[tr as usize] { 1 } else { 0 };
                                let mut pred = 0;
                                if go_right == 0 {
                                    pred = if training_x[i][fl as usize] >= thr[tl as usize] { 1 } else { 0 };
                                } else {
                                    pred = if training_x[i][fr2 as usize] >= thr[tr2 as usize] { 1 } else { 0 };
                                }
                                if pred != training_y[i] {
                                    err += 1;
                                }
                            }
                            if err < best_err {
                                best_err = err;
                                best_feat_r = fr;
                                best_thr_r = tr;
                                best_feat_l = fl;
                                best_thr_l = tl;
                                best_feat_rr = fr2;
                                best_thr_rr = tr2;
                            } else if err == best_err {
                                if (fr < best_feat_r)
                                    || (fr == best_feat_r && (tr < best_thr_r
                                        || (tr == best_thr_r
                                            && (fl < best_feat_l
                                                || (fl == best_feat_l
                                                    && (tl < best_thr_l
                                                        || (tl == best_thr_l
                                                            && (fr2 < best_feat_rr
                                                                || (fr2 == best_feat_rr && tr2 < best_thr_rr)))))))))
                                {
                                    best_feat_r = fr;
                                    best_thr_r = tr;
                                    best_feat_l = fl;
                                    best_thr_l = tl;
                                    best_feat_rr = fr2;
                                    best_thr_rr = tr2;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // evaluate test set
    let mut test_errors = 0;
    for i in 0..2 {
        let go_right = if testing_x[i][best_feat_r as usize] >= thr[best_thr_r as usize] { 1 } else { 0 };
        let mut pred = 0;
        if go_right == 0 {
            pred = if testing_x[i][best_feat_l as usize] >= thr[best_thr_l as usize] { 1 } else { 0 };
        } else {
            pred = if testing_x[i][best_feat_rr as usize] >= thr[best_thr_rr as usize] { 1 } else { 0 };
        }
        if pred != testing_y[i] {
            test_errors += 1;
        }
    }

    assert!(test_errors <= 1);
}
