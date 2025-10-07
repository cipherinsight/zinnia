// SP1 program code
#![no_main]
sp1_zkvm::entrypoint!(main);

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

pub fn main() {
    // read training_x (10x2)
    let mut training_x: Vec<[f64; 2]> = Vec::new();
    for _ in 0..10 {
        let x0 = sp1_zkvm::io::read::<f64>();
        let x1 = sp1_zkvm::io::read::<f64>();
        training_x.push([x0, x1]);
    }

    // read training_y (10)
    let mut training_y: Vec<f64> = Vec::new();
    for _ in 0..10 {
        training_y.push(sp1_zkvm::io::read::<f64>());
    }

    // read testing_x (2x2)
    let mut testing_x: Vec<[f64; 2]> = Vec::new();
    for _ in 0..2 {
        let x0 = sp1_zkvm::io::read::<f64>();
        let x1 = sp1_zkvm::io::read::<f64>();
        testing_x.push([x0, x1]);
    }

    // read testing_y (2)
    let mut testing_y: Vec<f64> = Vec::new();
    for _ in 0..2 {
        testing_y.push(sp1_zkvm::io::read::<f64>());
    }

    // initialize weights and bias
    let mut weights = [0.0_f64, 0.0_f64];
    let mut bias = 0.0_f64;
    let m = 10.0_f64;

    // gradient descent loop
    for _ in 0..100 {
        let mut z: Vec<f64> = Vec::new();
        for i in 0..10 {
            z.push(training_x[i][0] * weights[0] + training_x[i][1] * weights[1] + bias);
        }

        let mut preds: Vec<f64> = Vec::new();
        for i in 0..10 {
            preds.push(sigmoid(z[i]));
        }

        let mut errors: Vec<f64> = Vec::new();
        for i in 0..10 {
            errors.push(preds[i] - training_y[i]);
        }

        let mut dw0 = 0.0_f64;
        let mut dw1 = 0.0_f64;
        let mut db = 0.0_f64;
        for i in 0..10 {
            dw0 += training_x[i][0] * errors[i];
            dw1 += training_x[i][1] * errors[i];
            db += errors[i];
        }
        dw0 /= m;
        dw1 /= m;
        db /= m;

        weights[0] -= 0.2 * dw0;
        weights[1] -= 0.2 * dw1;
        bias -= 0.2 * db;
    }

    // test evaluation
    let mut mismatches = 0;
    for i in 0..2 {
        let z_test = testing_x[i][0] * weights[0] + testing_x[i][1] * weights[1] + bias;
        let prob = sigmoid(z_test);
        let pred = if prob >= 0.5 { 1.0 } else { 0.0 };
        if (pred - testing_y[i]).abs() > 1e-9 {
            mismatches += 1;
        }
    }

    assert_eq!(mismatches, 0);

    // sp1_zkvm::io::commit_slice(&output);
}
