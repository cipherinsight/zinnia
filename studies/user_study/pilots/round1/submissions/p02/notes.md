# Implementation notes

- `calibrate.py`: Apply the affine calibration to the whole vector, then clip both inclusive limits.
- `background.py`: Take column minima along the sample axis and subtract them by broadcasting.
- `threshold_energy.py`: Select qualifying readings elementwise, square them, and sum each row.
- `distances.py`: Broadcast the two point collections into all pairs and sum squared coordinate differences.
- `pool.py`: Add the four stride-two views that cover each non-overlapping block.
- `smooth.py`: Copy the grid each round, update only the interior from the previous grid, and retain the copied edges.
