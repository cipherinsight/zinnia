# OUT OF SCOPE - not part of any measurement. The DS-1000 program is out of scope
# (non_numeric_data: test input mystr: test input is a str).
# This file is the earlier generator output, which typed that input as a number;
# it is kept only for reference. See datasets/case_sets/ds1000_perf.json.

from zinnia import *
import numpy as np

# DS-1000 NumPy acceptance-study kernel.
# The body below is generated from the original DS-1000 reference solution.
# Type annotations/preconditions are intentionally not inferred here.

@zk_circuit
def kernel(mystr: Integer):
    result = np.array(list(mystr), dtype = int)
    return result
