import json

from zinnia import *


@zk_circuit
def verify_solution(
        training_data: NDArray[float, 40, 2],
        training_labels: NDArray[int, 40],
        initial_weights: NDArray[float, 2],
        testing_data: NDArray[float, 2, 2],
        testing_labels: NDArray[int, 2],
):
    n, d = training_data.shape
    weights = initial_weights
    for _ in range(50):
        for i in range(n):
            activation = np.dot(weights, training_data[i])
            prediction = 1 if activation >= 0 else -1
            if prediction != training_labels[i]:
                weights += training_data[i] if training_labels[i] == 1 else -training_data[i]
    m = testing_data.shape[0]
    for i in range(m):
        activation = np.dot(weights, testing_data[i])
        prediction = 1 if activation >= 0 else -1
        assert testing_labels[i] == 1 if prediction >= 0 else testing_labels[i] == -1