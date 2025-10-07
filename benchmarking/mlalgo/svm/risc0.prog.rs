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

    // initialize
    let mut w = [0.0_f64, 0.0_f64];
    let mut b = 0.0_f64;
    let m = 10.0_f64;
    let lr = 0.05_f64;

    // fixed 100 iterations
    for _ in 0..100 {
        let mut gw0 = 0.0_f64;
        let mut gw1 = 0.0_f64;
        let mut gb = 0.0_f64;

        for i in 0..10 {
            let score = training_x[i][0] * w[0] + training_x[i][1] * w[1] + b;
            let margin = training_y[i] * score;
            if margin < 1.0 {
                gw0 += -training_y[i] * training_x[i][0];
                gw1 += -training_y[i] * training_x[i][1];
                gb += -training_y[i];
            }
        }

        gw0 = gw0 / m + w[0];
        gw1 = gw1 / m + w[1];
        gb = gb / m;

        w[0] = w[0] - lr * gw0;
        w[1] = w[1] - lr * gw1;
        b = b - lr * gb;
    }

    // evaluation
    for i in 0..2 {
        let pred = testing_x[i][0] * w[0] + testing_x[i][1] * w[1] + b;
        assert!(testing_y[i] * pred > 0.1);
    }

    // env::commit(&output);
}
