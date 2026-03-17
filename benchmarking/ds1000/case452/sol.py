import json

from zinnia import *


@zk_circuit
def verify_solution(X: DynamicNDArray[int, 20, 2], result: DynamicNDArray[float, 20, 2]):
    l1_0 = np.stack((X[0:4], -X[0:4]), axis=0).max(axis=0).reshape((1, 4)).sum(axis=1)
    l1_1 = np.stack((X[4:8], -X[4:8]), axis=0).max(axis=0).reshape((1, 4)).sum(axis=1)
    l1_2 = np.stack((X[8:12], -X[8:12]), axis=0).max(axis=0).reshape((1, 4)).sum(axis=1)
    l1_3 = np.stack((X[12:16], -X[12:16]), axis=0).max(axis=0).reshape((1, 4)).sum(axis=1)
    l1_4 = np.stack((X[16:20], -X[16:20]), axis=0).max(axis=0).reshape((1, 4)).sum(axis=1)
    l1 = np.concatenate((l1_0, l1_1, l1_2, l1_3, l1_4), axis=0)
    expected = np.concatenate((
        X[0:4] / l1[0],
        X[4:8] / l1[1],
        X[8:12] / l1[2],
        X[12:16] / l1[3],
        X[16:20] / l1[4],
    ), axis=0)
    assert result == expected


if __name__ == '__main__':
    X = [
        [1, 2, 3, 6],
        [4, 5, 6, 5],
        [1, 2, 5, 5],
        [4, 5, 10, 25],
        [5, 2, 10, 25]
    ]

    import numpy as np
    l1 = np.abs(X).sum(axis=1)
    result = (np.array(X) / l1.reshape(-1, 1)).tolist()

    assert verify_solution(X, result)

    # Parse inputs
    program = ZKCircuit.from_method(verify_solution).compile()
    parsed_inputs = program.argparse(X, result)
    json_dict = {}
    for entry in parsed_inputs.entries:
        json_dict[entry.get_key()] = entry.get_value()
    with open('./sol.py.in', 'w') as f:
        json.dump(json_dict, f, indent=2)
