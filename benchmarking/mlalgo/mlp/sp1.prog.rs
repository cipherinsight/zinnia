// SP1 program code
#![no_main]
sp1_zkvm::entrypoint!(main);

pub fn main() {
    let mut training_x: [[f64; 2]; 10] = [[0.0; 2]; 10];
    let mut training_y: [f64; 10] = [0.0; 10];
    let mut testing_x: [[f64; 2]; 2] = [[0.0; 2]; 2];
    let mut testing_y: [f64; 2] = [0.0; 2];

    for i in 0..10 {
        for j in 0..2 {
            training_x[i][j] = sp1_zkvm::io::read::<f64>();
        }
    }
    for i in 0..10 {
        training_y[i] = sp1_zkvm::io::read::<f64>();
    }
    for i in 0..2 {
        for j in 0..2 {
            testing_x[i][j] = sp1_zkvm::io::read::<f64>();
        }
    }
    for i in 0..2 {
        testing_y[i] = sp1_zkvm::io::read::<f64>();
    }

    let mut W1 = [[0.02, -0.01, 0.016],
                  [0.014, 0.004, -0.006]];
    let mut b1 = [0.0, 0.0, 0.0];
    let mut W2 = [0.05, -0.03, 0.02];
    let mut b2 = 0.0;
    let lr = 0.02;
    let steps = 50usize;

    for _step in 0..steps {
        let mut H = [[0.0; 3]; 10];
        let mut A = [[0.0; 3]; 10];
        let mut preds = [0.0; 10];

        for i in 0..10 {
            for k in 0..3 {
                let h = training_x[i][0]*W1[0][k] + training_x[i][1]*W1[1][k] + b1[k];
                H[i][k] = h;
                A[i][k] = h*h;
            }
            preds[i] = A[i][0]*W2[0] + A[i][1]*W2[1] + A[i][2]*W2[2] + b2;
        }

        let mut errors = [0.0; 10];
        for i in 0..10 {
            errors[i] = preds[i] - training_y[i];
        }

        let inv_m2 = 2.0 / 10.0;
        let mut dW2 = [0.0; 3];
        let mut db2 = 0.0;
        for i in 0..10 {
            let e = errors[i];
            db2 += e;
            for k in 0..3 {
                dW2[k] += e*A[i][k];
            }
        }
        for k in 0..3 {
            dW2[k] *= inv_m2;
        }
        db2 *= inv_m2;

        let mut dW1 = [[0.0; 3]; 2];
        let mut db1g = [0.0; 3];
        for i in 0..10 {
            let e = errors[i];
            for k in 0..3 {
                let dh = (4.0/10.0)*e*W2[k]*H[i][k];
                dW1[0][k] += dh*training_x[i][0];
                dW1[1][k] += dh*training_x[i][1];
                db1g[k] += dh;
            }
        }

        for k in 0..3 {
            W2[k] -= lr*dW2[k];
            b1[k] -= lr*db1g[k];
            W1[0][k] -= lr*dW1[0][k];
            W1[1][k] -= lr*dW1[1][k];
        }
        b2 -= lr*db2;
    }

    let mut test_preds = [0.0; 2];
    for i in 0..2 {
        let h0 = testing_x[i][0]*W1[0][0] + testing_x[i][1]*W1[1][0] + b1[0];
        let h1 = testing_x[i][0]*W1[0][1] + testing_x[i][1]*W1[1][1] + b1[1];
        let h2 = testing_x[i][0]*W1[0][2] + testing_x[i][1]*W1[1][2] + b1[2];
        let a0 = h0*h0;
        let a1 = h1*h1;
        let a2 = h2*h2;
        test_preds[i] = a0*W2[0] + a1*W2[1] + a2*W2[2] + b2;
    }

    let mut se_sum = 0.0;
    for i in 0..2 {
        let diff = test_preds[i] - testing_y[i];
        se_sum += diff*diff;
    }
    let test_mse = se_sum / 2.0;

    assert!(test_mse <= 50.0);
}
