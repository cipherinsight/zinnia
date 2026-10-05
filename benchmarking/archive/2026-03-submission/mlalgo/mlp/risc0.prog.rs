// Risc 0 program code
use risc0_zkvm::guest::env;

fn main() {
    // read training_x (10x2)
    let mut training_x: Vec<[f64; 2]> = Vec::new();
    for _ in 0..10 {
        let a = env::read::<f64>();
        let b = env::read::<f64>();
        training_x.push([a, b]);
    }

    // read training_y (10)
    let mut training_y: Vec<f64> = Vec::new();
    for _ in 0..10 {
        training_y.push(env::read());
    }

    // read testing_x (2x2)
    let mut testing_x: Vec<[f64; 2]> = Vec::new();
    for _ in 0..2 {
        let a = env::read::<f64>();
        let b = env::read::<f64>();
        testing_x.push([a, b]);
    }

    // read testing_y (2)
    let mut testing_y: Vec<f64> = Vec::new();
    for _ in 0..2 {
        testing_y.push(env::read());
    }

    let train_m = 10usize;
    let test_m = 2usize;
    let hidden_dim = 3usize;
    let lr = 0.02_f64;

    // init params
    let mut W1 = vec![
        vec![0.02, -0.01, 0.016],
        vec![0.014, 0.004, -0.006]
    ];
    let mut b1 = vec![0.0, 0.0, 0.0];
    let mut W2 = vec![0.05, -0.03, 0.02];
    let mut b2 = 0.0_f64;

    // training loop
    for _ in 0..10 {
        let mut H = vec![vec![0.0; hidden_dim]; train_m];
        let mut A = vec![vec![0.0; hidden_dim]; train_m];
        let mut preds = vec![0.0; train_m];

        for i in 0..train_m {
            for k in 0..hidden_dim {
                let mut h = 0.0;
                h += training_x[i][0] * W1[0][k];
                h += training_x[i][1] * W1[1][k];
                h += b1[k];
                H[i][k] = h;
                A[i][k] = h * h;
            }
            let mut out = 0.0;
            for k in 0..hidden_dim {
                out += A[i][k] * W2[k];
            }
            out += b2;
            preds[i] = out;
        }

        let mut errors = vec![0.0; train_m];
        for i in 0..train_m {
            errors[i] = preds[i] - training_y[i];
        }

        let inv_m2 = 2.0 / (train_m as f64);

        let mut dW2 = vec![0.0; hidden_dim];
        let mut db2 = 0.0;
        for i in 0..train_m {
            let e = errors[i];
            db2 += e;
            for k in 0..hidden_dim {
                dW2[k] += e * A[i][k];
            }
        }
        for k in 0..hidden_dim {
            dW2[k] *= inv_m2;
        }
        db2 *= inv_m2;

        let mut dW1 = vec![vec![0.0; hidden_dim]; 2];
        let mut db1g = vec![0.0; hidden_dim];
        for i in 0..train_m {
            let e = errors[i];
            for k in 0..hidden_dim {
                let dh = (4.0 / (train_m as f64)) * e * W2[k] * H[i][k];
                dW1[0][k] += dh * training_x[i][0];
                dW1[1][k] += dh * training_x[i][1];
                db1g[k] += dh;
            }
        }

        for k in 0..hidden_dim {
            W2[k] -= lr * dW2[k];
            b1[k] -= lr * db1g[k];
            W1[0][k] -= lr * dW1[0][k];
            W1[1][k] -= lr * dW1[1][k];
        }
        b2 -= lr * db2;
    }

    // test forward
    let mut test_preds = vec![0.0; test_m];
    for i in 0..test_m {
        let h0 = testing_x[i][0] * W1[0][0] + testing_x[i][1] * W1[1][0] + b1[0];
        let h1 = testing_x[i][0] * W1[0][1] + testing_x[i][1] * W1[1][1] + b1[1];
        let h2 = testing_x[i][0] * W1[0][2] + testing_x[i][1] * W1[1][2] + b1[2];
        let a0 = h0 * h0;
        let a1 = h1 * h1;
        let a2 = h2 * h2;
        let yhat = a0 * W2[0] + a1 * W2[1] + a2 * W2[2] + b2;
        test_preds[i] = yhat;
    }

    let mut se_sum = 0.0;
    for i in 0..test_m {
        let diff = test_preds[i] - testing_y[i];
        se_sum += diff * diff;
    }
    let test_mse = se_sum / (test_m as f64);
    assert!(test_mse <= 50.0);

    // env::commit(&output);
}
