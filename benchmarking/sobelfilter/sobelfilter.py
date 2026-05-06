# Source: Pythran tests/cases/sobelfilter.py
# Original #pythran export: sobelFilter(int list, int, int)
from zinnia import *


@zk_circuit
def sobelFilter(original_image: NDArray[Integer, 64], cols: int, rows: int):
    edge_image = list(range(len(original_image)))
    for i in range(rows):
        edge_image[i * cols] = 255
        edge_image[((i + 1) * cols) - 1] = 255

    for i in range(1, cols - 1):
        edge_image[i] = 255
        edge_image[i + ((rows - 1) * cols)] = 255

    for iy in range(1, rows - 1):
        for ix in range(1, cols - 1):
            sum_x = 0
            sum_y = 0
            sum = 0
            sum_x += original_image[ix - 1 + (iy - 1) * cols] * -1
            sum_x += original_image[ix + (iy - 1) * cols] * -2
            sum_x += original_image[ix + 1 + (iy - 1) * cols] * -1
            sum_x += original_image[ix - 1 + (iy + 1) * cols] * 1
            sum_x += original_image[ix + (iy + 1) * cols] * 2
            sum_x += original_image[ix + 1 + (iy + 1) * cols] * 1
            sum_x = min(255, max(0, sum_x))
            sum_y += original_image[ix - 1 + (iy - 1) * cols] * 1
            sum_y += original_image[ix + 1 + (iy - 1) * cols] * -1
            sum_y += original_image[ix - 1 + (iy) * cols] * 2
            sum_y += original_image[ix + 1 + (iy) * cols] * -2
            sum_y += original_image[ix - 1 + (iy + 1) * cols] * 1
            sum_y += original_image[ix + 1 + (iy + 1) * cols] * -1
            sum_y = min(255, max(0, sum_y))

            sum = abs(sum_x) + abs(sum_y)

            edge_image[ix + iy * cols] = 255 - (255 & sum)
    _zinnia_result = edge_image
