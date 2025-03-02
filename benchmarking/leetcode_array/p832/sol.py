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
import json
import random

from zinnia import *


@zk_circuit
def verify_solution(
    image: NDArray[int, 10, 10],
    result: NDArray[int, 10, 10]
):
    shape = image.shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            assert image[i][j] == 1 or image[i][j] == 0
            assert result[i][j] == 1 or result[i][j] == 0
    for i in range(shape[0]):
        for j in range(shape[1]):
            assert result[i][j] == 1 - image[i][shape[1] - 1 - j]


def generate_solution(
    image: NDArray[int, 10, 10]
):
    shape = image.shape
    assert np.logical_or(image == 0, image == 1).all()
    result = np.zeros_like(image)
    for i in range(shape[0]):
        for j in range(shape[1]):
            result[i][j] = 1 - image[i][shape[1] - 1 - j]
    return result


# circuit = ZKCircuit.from_method(verify_solution)
# print(circuit.compile().source)
# np.random.seed(0)
# image = np.random.randint(0, 2, (10, 10))
# result = generate_solution(image)
# json_dict = {}
# for entry in circuit.argparse(image, result).entries:
#     json_dict[entry.get_key()] = entry.value
# print(json.dumps(json_dict))
# json_dict = {}
# json_dict['image'] = list(int(x) for x in image.astype(int).flatten())
# json_dict['result'] = list(int(x) for x in result.astype(int).flatten())
# print(json.dumps(json_dict))
