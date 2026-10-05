# OUT OF SCOPE - not part of any measurement. The DS-1000 program is out of scope
# (imports: import copy; non_numeric_data: test input array_of_arrays: object array (e.g. arrays of different shapes)).
# This file is the earlier generator output, which typed that input as a number;
# it is kept only for reference. See datasets/case_sets/ds1000_perf.json.

from zinnia import *
import numpy as np

# DS-1000 NumPy acceptance-study kernel.
# The body below is generated from the original DS-1000 reference solution.
# Type annotations/preconditions are intentionally not inferred here.

@zk_circuit
def kernel(pairs: Integer[Array, "3 2"], array_of_arrays: Integer[Array, "3"]):
    import copy
    result = copy.deepcopy(array_of_arrays)
    return result
