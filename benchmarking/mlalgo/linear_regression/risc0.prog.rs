use risc0_zkvm::guest::env;

fn main() {
    // TODO: Implement your guest code here
    // read the input
    let mut training_x: Vec<Vec<f64>> = Vec::new();
    for i in 0..10 {
        let mut tmp: Vec<f64> = Vec::new();
        for j in 0..2 {
            tmp.push(env::read());
        }
        training_x.push(tmp);
    }
    let mut training_y: Vec<f64> = Vec::new();
    for j in 0..10 {
        training_y.push(env::read());
    }
    let mut testing_x: Vec<Vec<f64>> = Vec::new();
    for i in 0..2 {
        let mut tmp: Vec<f64> = Vec::new();
        for j in 0..2 {
            tmp.push(env::read());
        }
        testing_x.push(tmp);
    }
    let mut testing_y: Vec<f64> = Vec::new();
    for j in 0..2 {
        testing_y.push(env::read());
    }

    let mut weights = [0.0; 2];
    let mut bias = 0.0;
    let m = training_y.len() as f64;
    let learning_rate = 0.02;

    // Gradient descent loop
    for _ in 0..100 {
        let mut predictions = vec![0.0; training_y.len()];
        let mut errors = vec![0.0; training_y.len()];

        // Compute predictions and errors
        for (i, x) in training_x.iter().enumerate() {
            predictions[i] = x[0] * weights[0] + x[1] * weights[1] + bias;
            errors[i] = predictions[i] - training_y[i];
        }

        // Compute gradients
        let mut dw = [0.0; 2];
        let mut db = 0.0;

        for (i, x) in training_x.iter().enumerate() {
            dw[0] += x[0] * errors[i];
            dw[1] += x[1] * errors[i];
            db += errors[i];
        }

        dw[0] /= m;
        dw[1] /= m;
        db /= m;

        // Update parameters
        weights[0] -= learning_rate * dw[0];
        weights[1] -= learning_rate * dw[1];
        bias -= learning_rate * db;
    }

    // Evaluate model
    let mut test_error = 0.0;
    for (i, x) in testing_x.iter().enumerate() {
        let prediction = x[0] * weights[0] + x[1] * weights[1] + bias;
        let error = prediction - testing_y[i];
        test_error += error * error;
    }
    test_error /= testing_y.len() as f64;

    println!("{}", test_error);
    assert!(test_error <= 1.0, "Test error too high: {}", test_error);

    // write public output to the journal
    // env::commit(&input);
}
