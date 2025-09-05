import json

import numpy as np

from zinnia import *


@zk_circuit
def verify_solution(a: NDArray[float, 4, 4], power: float, desired_result: NDArray[float, 4, 4]):
    assert a ** power == desired_result
