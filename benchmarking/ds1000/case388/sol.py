import json

from zinnia import *


@zk_circuit
def verify_solution(a: NDArray[int, 4, 5], result: NDArray[int, 4, 2, 2]):
    # a =
    # [[ 1,  5,  9, 13, 17],
    #  [ 2,  6, 10, 14, 18],
    #  [ 3,  7, 11, 15, 19],
    #  [ 4,  8, 12, 16, 20]]
    #
    # Patch size = 2. If shape not divisible, drop the extra row/column.
    # We drop the last column to make width divisible by 2.
    #
    # Expected result =
    # [
    #   [[ 1,  5],
    #    [ 2,  6]],
    #   [[ 9, 13],
    #    [10, 14]],
    #   [[ 3,  7],
    #    [ 4,  8]],
    #   [[11, 15],
    #    [12, 16]],
    # ]
    #
    # NumPy reference (without swapaxes):
    # x = a[:(a.shape[0]//2)*2, :(a.shape[1]//2)*2]
    # out = x.reshape(x.shape[0]//2, 2, x.shape[1]//2, 2).transpose((0, 2, 1, 3)).reshape((-1, 2, 2))

    patch = 2
    # 1) Trim to multiples of patch size
    rows = (a.shape[0] // patch) * patch  # 4
    cols = (a.shape[1] // patch) * patch  # 4
    x = a[:rows, :cols]                   # (4, 4)

    # 2) Blockify -> (rows/2, 2, cols/2, 2) == (2, 2, 2, 2)
    blk = x.reshape((rows // patch, patch, cols // patch, patch))

    # 3) Reorder so that we iterate column blocks inside each row block: (0, 2, 1, 3)
    perm = blk.transpose((0, 2, 1, 3))

    # 4) Flatten blocks -> (num_blocks, 2, 2) == (4, 2, 2)
    computed = perm.reshape(( (rows // patch) * (cols // patch), patch, patch ))

    assert result == computed


if __name__ == '__main__':
    a = [
        [1, 5, 9, 13, 17],
        [2, 6, 10, 14, 18],
        [3, 7, 11, 15, 19],
        [4, 8, 12, 16, 20],
    ]
    result = [
        [[1, 5], [2, 6]],
        [[9, 13], [10, 14]],
        [[3, 7], [4, 8]],
        [[11, 15], [12, 16]],
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
