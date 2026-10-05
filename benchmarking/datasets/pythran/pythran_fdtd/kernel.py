# pythran program fdtd.py (serge-sans-paille/pythran pythran/tests/cases).
# Function bodies are the upstream text. Entry-function annotations are derived from upstream
# types by benchmarking/datasets/tools/acceptance_case_set.py; see datasets/case_sets/acceptance.json.

from zinnia import *
import numpy as np

@zk_circuit
def fdtd(input_grid: Float[Array, '1000 1000'], steps: Integer):
    grid = input_grid.copy()
    old_grid = np.zeros_like(input_grid)
    previous_grid = np.zeros_like(input_grid)
    l_x = grid.shape[0]
    l_y = grid.shape[1]
    for i in range(steps):
        np.copyto(previous_grid, old_grid)
        np.copyto(old_grid, grid)
        for x in range(l_x):
            for y in range(l_y):
                grid[x, y] = 0.0
                if 0 < x + 1 < l_x:
                    grid[x, y] += old_grid[x + 1, y]
                if 0 < x - 1 < l_x:
                    grid[x, y] += old_grid[x - 1, y]
                if 0 < y + 1 < l_y:
                    grid[x, y] += old_grid[x, y + 1]
                if 0 < y - 1 < l_y:
                    grid[x, y] += old_grid[x, y - 1]
                grid[x, y] /= 2.0
                grid[x, y] -= previous_grid[x, y]
    return grid
