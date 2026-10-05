from zinnia import *
import numpy as np

# Calibration program (not a DS-1000 case): returns its private input as the
# public output. Its proving and verification times are each system's fixed
# cost: process start-up, key loading, minimum trace size, recursion and
# wrapping steps, output commitment.

@zk_circuit
def kernel(x: Integer):
    return x
