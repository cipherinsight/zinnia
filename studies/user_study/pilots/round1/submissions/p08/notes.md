`calibrate.py` applies the affine calibration and clips each reading to the required interval.

`background.py` finds each channel minimum and broadcasts it across the sample rows before subtraction.

`threshold_energy.py` masks readings at or below each channel floor, squares the remaining values, and sums per observation.

`distances.py` uses broadcasting to form every pairwise coordinate difference, then sums squared differences.

`pool.py` selects the four parity-offset views of each 2x2 block and adds them while preserving batch and channel axes.

`smooth.py` copies the grid at each round and computes all interior updates from that round's unchanged source grid.
