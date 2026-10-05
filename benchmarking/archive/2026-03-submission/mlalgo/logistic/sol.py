import json

from zinnia import *


@zk_circuit
def verify_solution(
        training_x: NDArray[float, 10, 2],
        training_y: NDArray[float, 10],
        testing_x: NDArray[float, 2, 2],
        testing_y: NDArray[float, 2]
):
    # Logistic Regression (binary) with fixed hyperparams and iteration count
    # Shapes are all static to satisfy circuit constraints.

    # Initialize
    weights = np.zeros((2,))   # training_x has 2 features
    bias = 0.0
    m = 10.0                   # len(training_y)

    # Gradient descent (fixed steps)
    for _ in range(100):
        # Linear scores
        z = np.dot(training_x, weights) + bias
        # Sigmoid
        preds = 1.0 / (1.0 + np.exp(-z))

        # Errors (pred - y) for BCE gradient
        errors = preds - training_y

        # Gradients
        dw = (1.0 / m) * np.dot(training_x.T, errors)
        db = (1.0 / m) * np.sum(errors)

        # Update
        weights = weights - 0.2 * dw
        bias = bias - 0.2 * db

    # Evaluate on testing set (2 examples)
    z_test = np.dot(testing_x, weights) + bias
    probs = 1.0 / (1.0 + np.exp(-z_test))

    # Turn probs into {0.0, 1.0} with a static loop
    mismatches = 0
    for i in range(2):
        pred = 1.0 if probs[i] >= 0.5 else 0.0
        if pred != testing_y[i]:
            mismatches += 1

    # Require perfect accuracy on this simple, linearly separable instance
    assert mismatches == 0


if __name__ == '__main__':
    # A tiny, linearly separable dataset (5 negatives, 5 positives)
    training_x = [
        [-2.0, -1.0],
        [-1.5, -1.3],
        [-1.2, -1.8],
        [-2.1, -2.2],
        [-1.7, -1.4],
        [ 2.0,  1.0],
        [ 1.5,  1.6],
        [ 1.2,  1.9],
        [ 2.1,  2.2],
        [ 1.7,  1.3],
    ]
    training_y = [
        0.0, 0.0, 0.0, 0.0, 0.0,
        1.0, 1.0, 1.0, 1.0, 1.0
    ]

    testing_x = [
        [-1.8, -1.7],  # should be 0
        [ 1.8,  1.6],  # should be 1
    ]
    testing_y = [0.0, 1.0]

    assert verify_solution(training_x, training_y, testing_x, testing_y)

    # Parse inputs
    program = ZKCircuit.from_method(verify_solution).compile()
    parsed_inputs = program.argparse(training_x, training_y, testing_x, testing_y)
    json_dict = {}
    for entry in parsed_inputs.entries:
        json_dict[entry.get_key()] = entry.get_value()
    with open('./sol.py.in', 'w') as f:
        json.dump(json_dict, f, indent=2)
