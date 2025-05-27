# No. 459
# Problem:
# I want to be able to calculate the mean of A:
#  import numpy as np
#  A = ['33.33', '33.33', '33.33', '33.37']
#  NA = np.asarray(A)
#  AVG = np.mean(NA, axis=0)
#  print AVG
# This does not work, unless converted to:
# A = [33.33, 33.33, 33.33, 33.37]
# Is it possible to compute AVG WITHOUT loops?
# A:
# <code>
# import numpy as np
# A = ['33.33', '33.33', '33.33', '33.37']
# NA = np.asarray(A)
# </code>
# AVG = ... # put solution in this variable
# BEGIN SOLUTION
# <code>
#
# ------------------------------------------------------------
# AVG = np.mean(NA.astype(float), axis = 0)
import json

import numpy as np
from zinnia import *

# A = [33.33, 33.33, 33.33, 33.37]
# AVG = np.mean(np.asarray(A).astype(float), axis = 0)


@zk_circuit
def verify_solution(A: NDArray[float, 4], AVG: float):
    # print(np.mean(np.asarray(A).astype(float), axis = 0))
    # print(AVG)
    assert AVG == (np.sum(np.asarray(A).astype(float), axis = 0) / len(A))


# assert verify_solution(A, AVG)

# Parse inputs
# program = ZKCircuit.from_method(verify_solution).compile()
# parsed_inputs = program.argparse(A, AVG)
# json_dict = {}
# for entry in parsed_inputs.entries:
#     json_dict[entry.get_key()] = entry.get_value()
# print(json.dumps(json_dict, indent=2))
