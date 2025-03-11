# No. 309
# Problem:
# How can I get get the position (indices) of the largest value in a multi-dimensional NumPy array `a`?
# Note that I want to get the raveled index of it, in C order.
# A:
# <code>
import json

import numpy as np

a = np.array([[10,50,30],[60,20,40]])
# </code>
# result = ... # put solution in this variable
# BEGIN SOLUTION
# <code>
#
# ------------------------------------------------------------
result = a.argmax()


from zinnia import *

@zk_circuit
def verify_solution(a: NDArray[int, 2, 3], result: int):
    assert a.argmax() == result


# assert verify_solution(a, result)

# Parse inputs
# program = ZKCircuit.from_method(verify_solution).compile()
# parsed_inputs = program.argparse(a, result)
# json_dict = {}
# for entry in parsed_inputs.entries:
#     json_dict[entry.get_key()] = entry.get_value()
# print(json.dumps(json_dict, indent=2))
