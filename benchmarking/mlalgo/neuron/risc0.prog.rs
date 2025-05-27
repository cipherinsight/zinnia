use risc0_zkvm::guest::env;

fn main() {
    // TODO: Implement your guest code here
    // read the input
    let mut training_data: Vec<Vec<f64>> = Vec::new();
    for i in 0..10 {
        let mut tmp: Vec<f64> = Vec::new();
        for j in 0..2 {
            tmp.push(env::read());
        }
        training_data.push(tmp);
    }
    let mut training_labels: Vec<i32> = Vec::new();
    for j in 0..10 {
        training_labels.push(env::read());
    }
    let mut initial_weights: Vec<f64> = Vec::new();
    for j in 0..2 {
        initial_weights.push(env::read());
    }
    let mut testing_data: Vec<Vec<f64>> = Vec::new();
    for i in 0..2 {
        let mut tmp: Vec<f64> = Vec::new();
        for j in 0..2 {
            tmp.push(env::read());
        }
        testing_data.push(tmp);
    }
    let mut testing_labels: Vec<i32> = Vec::new();
    for j in 0..2 {
        testing_labels.push(env::read());
    }

    let n = training_data.len();

    let mut weights = [initial_weights[0], initial_weights[1]];
    // Perceptron training loop
    for _ in 0..50 {
        for i in 0..n {
            let activation = training_data[i][0] * weights[0] + training_data[i][1] * weights[1];
            let prediction = if activation >= 0.0 { 1 } else { -1 };
            if prediction != training_labels[i] {
                if training_labels[i] == 1 {
                    weights[0] += training_data[i][0];
                    weights[1] += training_data[i][1];
                } else {
                    weights[0] -= training_data[i][0];
                    weights[1] -= training_data[i][1];
                }
            }
        }
    }

    let m = testing_data.len();

    // Test the trained model
    for i in 0..m {
        let activation = testing_data[i][0] * weights[0] + testing_data[i][1] * weights[1];
        let prediction = if activation >= 0.0 { 1 } else { -1 };
        assert!(
            testing_labels[i] == (if prediction >= 0 { 1 } else { -1 }),
            "Mismatch in prediction at index {}: expected {}, but got {}",
            i,
            testing_labels[i],
            prediction
        );
    }

    // write public output to the journal
    // env::commit(&input);
}
