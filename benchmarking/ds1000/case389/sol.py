import json

from zinnia import *


@zk_circuit
def verify_solution(a: NDArray[int, 4, 2, 3], result: NDArray[int, 4, 6]):
    # a shape: (n=4, nrows=2, ncols=3)
    # Goal: tile the 4 small (2x3) blocks into a (h=4, w=6) array in row-major block order.

    # Equivalent to:
    # n, nrows, ncols = a.shape
    # out = a.reshape(h//nrows, -1, nrows, ncols).swapaxes(1, 2).reshape(h, w)
    # Since swapaxes isn't available, use transpose((0, 2, 1, 3)).

    nrows = 2
    ncols = 3
    h = 4
    w = 6

    step1 = a.reshape((h // nrows, 2, nrows, ncols))   # (2, 2, 2, 3)
    step2 = step1.transpose((0, 2, 1, 3))               # swapaxes(1,2) -> (0,2,1,3)
    computed = step2.reshape((h, w))                     # (4, 6)

    assert result == computed


if __name__ == '__main__':
    a = [
        [[0, 1, 2],
         [6, 7, 8]],

        [[3, 4, 5],
         [9, 10, 11]],

        [[12, 13, 14],
         [18, 19, 20]],

        [[15, 16, 17],
         [21, 22, 23]],
    ]
    result = [
        [0, 1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10, 11],
        [12, 13, 14, 15, 16, 17],
        [18, 19, 20, 21, 22, 23],
    ]
    assert verify_solution(a, result)

    # Parse inputs
    program = ZKCircuit.from_method(verify_solution).compile()
    parsed_inputs = program.argparse(a, result)
    json_dict = {}
    for entry in parsed_inputs.entries:
        json_dict[entry.get_key()] = entry.get_value()
    with open('./sol.py.in', 'w') as f:
        json.dump(json_dict, f, indent=2)
