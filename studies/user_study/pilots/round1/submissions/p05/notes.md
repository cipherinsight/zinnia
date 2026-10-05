# Notes

1. Calibration uses elementwise affine conversion followed by inclusive clipping.
2. Background correction broadcasts each column's minimum across all samples.
3. Thresholded energy masks each value independently before summing squared values by row.
4. Pairwise differences are broadcast across point pairs, then squared and reduced by coordinate.
5. Pooling reshapes the two spatial dimensions into adjacent pairs and sums the pair axes.
6. Smoothing copies the current grid for each round, replaces only interior cells from the prior grid, and preserves edges.
