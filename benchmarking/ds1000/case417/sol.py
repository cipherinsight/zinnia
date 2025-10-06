import json

from zinnia import *


@zk_circuit
def verify_solution(data: NDArray[float, 10], result: NDArray[float, 3]):
    # data = [4, 2, 5, 6, 7, 5, 4, 3, 5, 7]
    # bin_size = 3
    # Reverse the array → [7,5,3,4,5,7,6,5,2,4]
    # Trim to multiple of 3 → first 9 elements: [7,5,3,4,5,7,6,5,2]
    # reshape to (3,3): [[7,5,3],[4,5,7],[6,5,2]]
    # mean along last axis → [5,5.33,4.33]

    bin_size = 3
    new_data = data[::-1]
    trimmed = new_data[:(10 // bin_size) * bin_size]
    reshaped = trimmed.reshape((3, bin_size))
    bin_data_mean = np.mean(reshaped, axis=1)
    expected = bin_data_mean
    assert result == expected


if __name__ == '__main__':
    data = [4, 2, 5, 6, 7, 5, 4, 3, 5, 7]
    result = [5.0, 5.3333333333, 4.3333333333]

    assert verify_solution(data, result)

    # Parse inputs
    program = ZKCircuit.from_method(verify_solution).compile()
    parsed_inputs = program.argparse(data, result)
    json_dict = {}
    for entry in parsed_inputs.entries:
        json_dict[entry.get_key()] = entry.get_value()
    with open('./sol.py.in', 'w') as f:
        json.dump(json_dict, f, indent=2)
