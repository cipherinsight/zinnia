# No. 360
# Problem:
#
# >>> arr = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
# >>> arr
# array([[ 1,  2,  3,  4],
#        [ 5,  6,  7,  8],
#        [ 9, 10, 11, 12]])
# I am deleting the 3rd row
# array([[ 1,  2,  3,  4],
#        [ 5,  6,  7,  8]])
# Are there any good way ?  Please consider this to be a novice question.
#
#
# A:
# <code>
import json

import numpy as np
# a = np.arange(12).reshape(3, 4)
# </code>
# a = ... # put solution in this variable
# BEGIN SOLUTION
# <code>
#
# ------------------------------------------------------------
# b = np.delete(a, 2, axis = 0)

import numpy as np

from zinnia import *

@zk_circuit
def verify_solution(data: NDArray[int, 3, 32], result: NDArray[int, 2, 32]):
    assert data[:2] == result


# assert verify_solution(a, b)

# Parse inputs
# program = ZKCircuit.from_method(verify_solution).compile()
# parsed_inputs = program.argparse(a, b)
# json_dict = {}
# for entry in parsed_inputs.entries:
#     json_dict[entry.get_key()] = entry.get_value()
# print(json.dumps(json_dict, indent=2))
# print(ZKCircuit.from_method(verify_solution).compile().source)
