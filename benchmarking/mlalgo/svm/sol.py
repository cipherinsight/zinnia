import json

from zinnia import *


@zk_circuit
def verify_solution(
        training_x: NDArray[float, 10, 2],
        training_y: NDArray[float, 10],
        testing_x: NDArray[float, 2, 2],
        testing_y: NDArray[float, 2]
):
    # Minimal linear SVM (primal) with hinge loss and L2 regularization
    # Objective (scaled): 0.5*||w||^2 + (1/m) * sum_i max(0, 1 - y_i (w·x_i + b))
    # Subgradient updates with fixed iterations (static loop bound)

    w = np.zeros((2,), dtype=float)
    b = 0.0
    m = 10.0
    lr = 0.05

    for _ in range(100):  # fixed, statically-known iteration count
        # accumulate subgradients
        gw0 = 0.0
        gw1 = 0.0
        gb = 0.0
        for i in range(10):
            score = training_x[i, 0] * w[0] + training_x[i, 1] * w[1] + b
            margin = training_y[i] * score
            if margin < 1.0:
                # subgradient of hinge term: -y_i * x_i (and -y_i for bias)
                gw0 -= training_y[i] * training_x[i, 0]
                gw1 -= training_y[i] * training_x[i, 1]
                gb -= training_y[i]
        # add gradient of 0.5*||w||^2, and average hinge part by m
        gw0 = gw0 / m + w[0]
        gw1 = gw1 / m + w[1]
        gb = gb / m

        # update
        w[0] -= lr * gw0
        w[1] -= lr * gw1
        b -= lr * gb

    # Evaluate on testing set: require correct sign on both examples with a small margin
    for i in range(2):
        pred = testing_x[i, 0] * w[0] + testing_x[i, 1] * w[1] + b
        assert testing_y[i] * pred > 0.1  # positive margin ⇒ correct classification


if __name__ == '__main__':
    # Linearly separable toy dataset (y in {-1, +1})
    training_x = [
        [0.0, 0.0],
        [0.2, 0.1],
        [0.3, -0.2],
        [-0.2, 0.1],
        [0.1, -0.1],
        [2.0, 2.0],
        [2.1, 1.9],
        [1.8, 2.2],
        [2.2, 2.1],
        [1.9, 1.8],
    ]
    training_y = [-1.0, -1.0, -1.0, -1.0, -1.0, +1.0, +1.0, +1.0, +1.0, +1.0]

    testing_x = [
        [0.1, 0.0],   # should be -1
        [2.05, 2.0],  # should be +1
    ]
    testing_y = [-1.0, +1.0]

    assert verify_solution(training_x, training_y, testing_x, testing_y)

    # Parse inputs
    program = ZKCircuit.from_method(verify_solution).compile()
    parsed_inputs = program.argparse(training_x, training_y, testing_x, testing_y)
    json_dict = {}
    for entry in parsed_inputs.entries:
        json_dict[entry.get_key()] = entry.get_value()
    with open('./sol.py.in', 'w') as f:
        json.dump(json_dict, f, indent=2)
