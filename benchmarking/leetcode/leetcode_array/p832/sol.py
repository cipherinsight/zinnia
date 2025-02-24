# 832. Flipping an Image
# Easy
# Topics
# Companies
#
# Given an n x n binary matrix image, flip the image horizontally, then invert it, and return the resulting image.
#
# To flip an image horizontally means that each row of the image is reversed.
#
#     For example, flipping [1,1,0] horizontally results in [0,1,1].
#
# To invert an image means that each 0 is replaced by 1, and each 1 is replaced by 0.
#
#     For example, inverting [0,1,1] results in [1,0,0].
import random

from zinnia import *


@zk_circuit
def verify_solution(
    image: Public[NDArray[int, 10, 10]],
    result: Public[NDArray[int, 10, 10]]
):
    shape = image.shape
    assert image == 0 or image == 1
    assert result == 0 or result == 1
    for i in range(shape[0]):
        for j in range(shape[1]):
            assert result[i][j] == 1 - image[i][shape[1] - 1 - j]


ZKCircuit.from_method(verify_solution).compile()