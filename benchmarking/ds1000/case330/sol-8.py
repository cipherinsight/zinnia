import json

import numpy as np

from zinnia import *


@zk_circuit
def verify_solution(a: NDArray[float, 16, 2], power: float, desired_result: NDArray[float, 16, 2]):
    assert a ** power == desired_result

