import json

import numpy as np
from zinnia import *

@zk_circuit
def verify_solution(A: NDArray[float, 64], AVG: float):
    assert AVG == (np.sum(np.asarray(A).astype(float), axis = 0) / len(A))
