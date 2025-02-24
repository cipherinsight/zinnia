from zinnia import *


@zk_circuit
def train_neuron(
        training_data: NDArray[float, 10, 2],
        training_labels: NDArray[int, 10],
        initial_weights: NDArray[float, 2],
        testing_data: NDArray[float, 2, 2],
        testing_labels: NDArray[int, 2],
):
    n, d = training_data.shape
    weights = initial_weights
    for _ in range(100):
        for i in range(n):
            activation = np.dot(weights, training_data[i])
            prediction = 1 if activation > 0 else -1
            if prediction != training_labels[i]:
                weights += training_labels[i] * training_data[i]
    correct = 0
    m = testing_data.shape[0]
    for i in range(m):
        activation = np.dot(weights, testing_data[i])
        prediction = 1 if activation > 0 else -1
        if prediction == testing_labels[i]:
            correct += 1
    assert correct == m


assert train_neuron(
    np.array([[1, 1], [1, 2], [2, 1], [-2, 2], [3, -3], [-3, -4], [-4, 3], [4, -4], [-5, -5], [-5, 6]]),
    np.array([1, 1, 1, -1, -1, -1, -1, -1, -1, -1]),
    np.array([0.0, 0.0]),
    np.array([[5, 6], [-1, -2]]),
    np.array([1, -1]),
)