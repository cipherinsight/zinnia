import json

from zinnia import *


@zk_circuit
def verify_solution(data: NDArray[float, 10], result: NDArray[float, 3]):
    # data = [4, 2, 5, 6, 7, 5, 4, 3, 5, 7]
    # bin_size = 3
    # Drop last element to make length multiple of 3 → data[:9]
    # reshape to (3, 3): [[4,2,5], [6,7,5], [4,3,5]]
    # mean along axis=1 → [3.67, 6, 4]

    bin_size = 3
    trimmed = data[:(10 // bin_size) * bin_size]
    reshaped = trimmed.reshape((3, bin_size))
    bin_data_mean = np.mean(reshaped, axis=1)
    assert result == bin_data_mean


if __name__ == '__main__':
    data = [4, 2, 5, 6, 7, 5, 4, 3, 5, 7]
    result = [3.6666666666666667, 6, 4]

    assert verify_solution(data, result)

    # Parse inputs
    program = ZKCircuit.from_method(verify_solution).compile()
    parsed_inputs = program.argparse(data, result)
    json_dict = {}
    for entry in parsed_inputs.entries:
        json_dict[entry.get_key()] = entry.get_value()
    with open('./sol.py.in', 'w') as f:
        json.dump(json_dict, f, indent=2)
