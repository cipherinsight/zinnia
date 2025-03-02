//! A simple program that takes a number `n` as input, and writes the `n-1`th and `n`th fibonacci
//! number as an output.

// These two lines are necessary for the program to properly compile.
//
// Under the hood, we wrap your main function with some extra code so that it behaves properly
// inside the zkVM.
#![no_main]
sp1_zkvm::entrypoint!(main);

use alloy_sol_types::SolType;
use fibonacci_lib::PublicValuesStruct;

pub fn main() {
    // Read an input to the program.
    //
    // Behind the scenes, this compiles down to a custom system call which handles reading inputs
    // from the prover.
    let mut training_x: Vec<Vec<f64>> = Vec::new();
    for i in 0..10 {
        let mut tmp: Vec<f64> = Vec::new();
        for j in 0..2 {
            tmp.push(sp1_zkvm::io::read::<f64>());
        }
        training_x.push(tmp);
    }
    let mut training_y: Vec<f64> = Vec::new();
    for j in 0..10 {
        training_y.push(sp1_zkvm::io::read::<f64>());
    }
    let mut testing_x: Vec<Vec<f64>> = Vec::new();
    for i in 0..2 {
        let mut tmp: Vec<f64> = Vec::new();
        for j in 0..2 {
            tmp.push(sp1_zkvm::io::read::<f64>());
        }
        testing_x.push(tmp);
    }
    let mut testing_y: Vec<f64> = Vec::new();
    for j in 0..2 {
        testing_y.push(sp1_zkvm::io::read::<f64>());
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

    // Encode the public values of the program.
    // let bytes = PublicValuesStruct::abi_encode(&PublicValuesStruct { n, a, b });

    // Commit to the public values of the program. The final proof will have a commitment to all the
    // bytes that were committed to.
    // sp1_zkvm::io::commit_slice(&bytes);
}
