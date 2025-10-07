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
    let mut training_y: Vec<f64> = Vec::new();
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
    let mut testing_y: Vec<f64> = Vec::new();
    for _ in 0..2 {
        testing_y.push(env::read());
    }

    let n_train = 10.0_f64;
    let alpha = 1.0_f64;
    let n_classes = 2.0_f64;

    let mut count_c0 = 0.0;
    let mut count_c1 = 0.0;
    let mut count1_c0_f0 = 0.0;
    let mut count1_c0_f1 = 0.0;
    let mut count1_c1_f0 = 0.0;
    let mut count1_c1_f1 = 0.0;

    // count stats
    for i in 0..10 {
        let yi = training_y[i];
        let x0 = training_x[i][0];
        let x1 = training_x[i][1];

        if yi == 0.0 {
            count_c0 += 1.0;
            if x0 >= 0.5 {
                count1_c0_f0 += 1.0;
            }
            if x1 >= 0.5 {
                count1_c0_f1 += 1.0;
            }
        } else {
            count_c1 += 1.0;
            if x0 >= 0.5 {
                count1_c1_f0 += 1.0;
            }
            if x1 >= 0.5 {
                count1_c1_f1 += 1.0;
            }
        }
    }

    // priors
    let prior0 = (count_c0 + alpha) / (n_train + n_classes * alpha);
    let prior1 = (count_c1 + alpha) / (n_train + n_classes * alpha);

    let denom0 = count_c0 + 2.0 * alpha;
    let denom1 = count_c1 + 2.0 * alpha;

    let theta0_f0 = (count1_c0_f0 + alpha) / denom0;
    let theta0_f1 = (count1_c0_f1 + alpha) / denom0;
    let theta1_f0 = (count1_c1_f0 + alpha) / denom1;
    let theta1_f1 = (count1_c1_f1 + alpha) / denom1;

    // prediction
    let mut preds = [0.0_f64, 0.0_f64];
    for i in 0..2 {
        let x0 = testing_x[i][0];
        let x1 = testing_x[i][1];

        let t0_f0 = if x0 >= 0.5 { theta0_f0 } else { 1.0 - theta0_f0 };
        let t0_f1 = if x1 >= 0.5 { theta0_f1 } else { 1.0 - theta0_f1 };
        let score0 = prior0 * t0_f0 * t0_f1;

        let t1_f0 = if x0 >= 0.5 { theta1_f0 } else { 1.0 - theta1_f0 };
        let t1_f1 = if x1 >= 0.5 { theta1_f1 } else { 1.0 - theta1_f1 };
        let score1 = prior1 * t1_f0 * t1_f1;

        let pred = if score1 >= score0 { 1.0 } else { 0.0 };
        preds[i] = pred;
    }

    assert_eq!(preds[0], testing_y[0]);
    assert_eq!(preds[1], testing_y[1]);

    // env::commit(&output);
}
