import json

from zinnia import *


@zk_circuit
def verify_solution(data: NDArray[int, 10], result: NDArray[int, 3]):
    # data = [4, 2, 5, 6, 7, 5, 4, 3, 5, 7]
    # bin_size = 3
    # Drop last element to make length multiple of 3 → data[:9]
    # reshape to (3, 3): [[4,2,5], [6,7,5], [4,3,5]]
    # max along axis=1 → [5,7,5]

    bin_size = 3
    trimmed = data[:(10 // bin_size) * bin_size]
    reshaped = trimmed.reshape((3, bin_size))
    bin_data_max = reshaped.max(axis=1)
    expected = bin_data_max
    assert result == expected


if __name__ == '__main__':
    data = [4, 2, 5, 6, 7, 5, 4, 3, 5, 7]
    result = [5, 7, 5]

    assert verify_solution(data, result)

    # Parse inputs
    program = ZKCircuit.from_method(verify_solution).compile()
    parsed_inputs = program.argparse(data, result)
    json_dict = {}
    for entry in parsed_inputs.entries:
        json_dict[entry.get_key()] = entry.get_value()
    with open('./sol.py.in', 'w') as f:
        json.dump(json_dict, f, indent=2)
