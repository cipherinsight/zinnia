import json

from zinnia import *


@zk_circuit
def verify_solution(
        training_x: NDArray[float, 10, 2],
        training_y: NDArray[float, 10],
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


# assert verify_solution(
#     np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11]]),
#     np.array([3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
#     np.array([[11, 12], [12, 13]]),
#     np.array([13, 14])
# )

# entries = ZKCircuit.from_method(verify_solution).argparse(
#     np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11]]),
#     np.array([3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
#     np.array([[11, 12], [12, 13]]),
#     np.array([13, 14])
# ).entries
#
# json_dict = {}
# for entry in entries:
#     json_dict[entry.get_key()] = entry.get_value()
# print(json.dumps(json_dict, indent=2))
#
# json_dict = {}
# json_dict["training_x"] = [float(x) for x in np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11]]).flatten().tolist()]
# json_dict["training_y"] = [float(x) for x in np.array([3, 4, 5, 6, 7, 8, 9, 10, 11, 12]).flatten().tolist()]
# json_dict["testing_x"] = [float(x) for x in np.array([[11, 12], [12, 13]]).flatten().tolist()]
# json_dict["testing_y"] = [float(x) for x in np.array([13, 14]).flatten().tolist()]
# print(json.dumps(json_dict, indent=2))

# with open("compiled.rs", "w") as f:
#     f.write(ZKCircuit.from_method(verify_solution).compile().source)

# 1628463 cells rust version
# 1299675 cells python version