from zinnia import *
# Zinnia benchmark contract draft: generated for review.
# Keep kernel body faithful; refine TODO contracts before final benchmark runs.
# Categories: in-place-mutation, loop-bounded
# Expected outcome: compile/prove candidate
# Precondition: Loop trip counts must be finite and tied to scalar bounds or array dimensions.
# Precondition: Use writable fixed/bounded array shapes and record an aliasing policy for mutated inputs.
import numpy as np

@zk_circuit
def fdtd(input_grid: Float[Array, 'N<=1024 M<=1024'], steps: Integer):
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
