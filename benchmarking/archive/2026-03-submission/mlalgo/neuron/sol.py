import json

from zinnia import *


@zk_circuit
def verify_solution(
        training_data: NDArray[float, 10, 2],
        training_labels: NDArray[int, 10],
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


# assert verify_solution(
#     np.array([[1, 1], [1, 2], [2, 1], [-2, 2], [3, -3], [-3, -4], [-4, 3], [4, -4], [-5, -5], [-5, 6]]),
#     np.array([1, 1, 1, -1, -1, -1, -1, -1, -1, -1]),
#     np.array([0.0, 0.0]),
#     np.array([[5, 6], [-1, -2]]),
#     np.array([1, -1]),
# )
#
# entries = ZKCircuit.from_method(verify_solution).argparse(
#     np.array([[1, 1], [1, 2], [2, 1], [-2, 2], [3, -3], [-3, -4], [-4, 3], [4, -4], [-5, -5], [-5, 6]]),
#     np.array([1, 1, 1, -1, -1, -1, -1, -1, -1, -1]),
#     np.array([0.0, 0.0]),
#     np.array([[5, 6], [-1, -2]]),
#     np.array([1, -1]),
# ).entries
#
# json_dict = {}
# for entry in entries:
#     json_dict[entry.get_key()] = entry.get_value()
# print(json.dumps(json_dict, indent=2))
#
# json_dict = {}
# json_dict["training_data"] = [float(x) for x in np.array([[1, 1], [1, 2], [2, 1], [-2, 2], [3, -3], [-3, -4], [-4, 3], [4, -4], [-5, -5], [-5, 6]]).flatten().tolist()]
# json_dict["training_labels"] = [int(x) for x in [1, 1, 1, -1, -1, -1, -1, -1, -1, -1]]
# json_dict["testing_data"] = [float(x) for x in np.array([[5, 6], [-1, -2]]).flatten().tolist()]
# json_dict["testing_labels"] = [int(x) for x in [1, -1]]
# json_dict["initial_weights"] = [float(x) for x in [0.0, 0.0]]
# print(json.dumps(json_dict, indent=2))

# with open("compiled.rs", "w") as f:
#     f.write(ZKCircuit.from_method(verify_solution).compile().source)

# rust version 380900 advice cells
# python version 381143 advice cells