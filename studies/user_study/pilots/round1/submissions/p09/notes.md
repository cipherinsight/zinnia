calibrate.py: Applies the affine calibration and clips every reading to the inclusive output range.

background.py: Finds each channel's column minimum and subtracts it with broadcasting.

threshold_energy.py: Masks readings at or below their per-channel floor, squares the remaining readings, and sums by observation.

distances.py: Uses broadcasting to form all point differences, then sums squared coordinates.

pool.py: Takes four strided 2D slices for each non-overlapping block and adds them while preserving batch and channel axes.

smooth.py: Copies the current grid on each round so edges remain fixed and all interior updates use the round's starting grid.
