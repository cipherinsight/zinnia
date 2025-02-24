from zinnia import *


@zk_circuit
def train_linear_regression(
        training_x: NDArray[float, 10, 2],
        training_y: NDArray[int, 10],
        testing_x: NDArray[float, 2, 2],
        testing_y: NDArray[int, 2]
):
    weights = np.zeros((training_x.shape[1], ))
    bias = 0
    m = len(training_y)

    # Gradient descent loop
    for _ in range(200):
        predictions = np.dot(training_x, weights) + bias
        errors = predictions - training_y

        # Compute gradients
        dw = (1 / m) * np.dot(training_x.T, errors)
        db = (1 / m) * np.sum(errors)

        # Update parameters
        weights -= 0.02 * dw
        bias -= 0.02 * db

    # Evaluate model
    test_predictions = np.dot(testing_x, weights) + bias
    test_error = np.sum((test_predictions - testing_y) ** 2) / len(testing_y)

    assert test_error <= 0.1


assert train_linear_regression(
    np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11]]),
    np.array([3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
    np.array([[11, 12], [12, 13]]),
    np.array([13, 14])
)