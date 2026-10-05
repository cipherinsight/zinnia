# Implementation notes

- `calibrate.py`: Scale and offset each reading, then use `np.clip` for the inclusive bounds.
- `background.py`: Take one minimum per column and broadcast its subtraction over all samples.
- `threshold_energy.py`: Select readings strictly above their channel thresholds, square them, and sum by observation.
- `distances.py`: Broadcast the two point collections into pairwise coordinate differences, then sum their squares over coordinates.
- `pool.py`: Add the four strided views corresponding to each position in every non-overlapping block.
- `smooth.py`: Copy the grid each round so all interior updates read the same prior grid; copied edge cells stay fixed.
