1. Calibrate uses vectorized scaling followed by inclusive clipping to the requested range.
2. Background subtracts the per-column minimum using broadcasting.
3. Energy masks readings at or below each channel floor, squares the remaining values, and sums per observation.
4. Distances use pairwise broadcasting across the two point collections and sum squared coordinate differences.
5. Pooling adds the four aligned stride-two slices for each 2x2 image block.
6. Smoothing copies each round's grid, updates only interior cells from the old grid, and repeats for the configured three rounds.
