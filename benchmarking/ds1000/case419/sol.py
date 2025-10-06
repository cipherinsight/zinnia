import json

from zinnia import *


@zk_circuit
def verify_solution(data: NDArray[float, 2, 5], result: NDArray[float, 2, 1]):
    # data =
    # [[4, 2, 5, 6, 7],
    #  [5, 4, 3, 5, 7]]
    # bin_size = 3
    # Reverse each row, take bins of 3, compute mean, then reverse result
    # Steps:
    #   new_data = [[7,6,5,2,4],
    #               [7,5,3,4,5]]
    #   trimmed = new_data[:, :3] -> [[7,6,5],[7,5,3]]
    #   reshaped = (2,1,3)
    #   mean = [[6],[5]]
    #   reverse result → same shape [[6],[5]]

    bin_size = 3
    new_data = data[:, ::-1]
    trimmed = new_data[:, :(5 // bin_size) * bin_size]
    reshaped = trimmed.reshape((data.shape[0], 1, bin_size))
    bin_data_mean = np.mean(reshaped, axis=-1)[:, ::-1]
    expected = bin_data_mean
    assert result == expected


if __name__ == '__main__':
    data = [
        [4, 2, 5, 6, 7],
        [5, 4, 3, 5, 7]
    ]
    result = [
        [6.0],
        [5.0]
    ]

    assert verify_solution(data, result)

    # Parse inputs
    program = ZKCircuit.from_method(verify_solution).compile()
    parsed_inputs = program.argparse(data, result)
    json_dict = {}
    for entry in parsed_inputs.entries:
        json_dict[entry.get_key()] = entry.get_value()
    with open('./sol.py.in', 'w') as f:
        json.dump(json_dict, f, indent=2)
