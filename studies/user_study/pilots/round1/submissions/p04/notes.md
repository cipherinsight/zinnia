# Implementation choices

- `calibrate.py`: Apply the affine calibration to the whole vector, then clip to the inclusive bounds.
- `background.py`: Take one minimum per column and broadcast it across the samples for subtraction.
- `threshold_energy.py`: Select qualifying squared readings per channel, then sum each row.
- `distances.py`: Broadcast point pairs to form coordinate differences and sum their squares along the coordinate axis.
- `pool.py`: Add the four strided views corresponding to each position in a 2 by 2 block.
- `smooth.py`: Copy the complete grid each round, replacing only interior cells using values from the preceding round.
