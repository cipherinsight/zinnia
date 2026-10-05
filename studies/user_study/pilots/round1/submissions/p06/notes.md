# Notes

1. Calibration uses elementwise affine adjustment followed by inclusive clipping.
2. Background correction subtracts each column's minimum through broadcasting.
3. Threshold energy masks readings with a strict per-channel comparison before reducing squared values by row.
4. Pairwise distances use broadcast coordinate differences, square them, then sum coordinates.
5. Pooling reshapes each image into 2-by-2 blocks and sums the two block axes.
6. Smoothing creates a new grid for each round, leaving copied edge values unchanged while updating interiors from the prior grid.
