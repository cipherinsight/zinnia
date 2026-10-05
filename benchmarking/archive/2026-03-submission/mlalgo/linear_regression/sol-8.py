import json

from zinnia import *


@zk_circuit
def verify_solution(
        training_x: NDArray[float, 80, 2],
        training_y: NDArray[float, 80],
        testing_x: NDArray[float, 2, 2],
        testing_y: NDArray[float, 2]
):
    weights = np.zeros((training_x.shape[1], ))
    bias = 0
    m = float(len(training_y))

    # Gradient descent loop
    for _ in range(100):
        predictions = np.dot(training_x, weights) + bias
        errors = predictions - training_y

        # Compute gradients
        dw = (1.0 / m) * np.dot(training_x.T, errors)
        db = (1.0 / m) * np.sum(errors)

        # Update parameters
        weights -= 0.02 * dw
        bias -= 0.02 * db

    # Evaluate model
    test_predictions = np.dot(testing_x, weights) + bias
    test_error = np.sum((test_predictions - testing_y) * (test_predictions - testing_y)) / len(testing_y)

    print(test_error)
    assert test_error <= 1.0
