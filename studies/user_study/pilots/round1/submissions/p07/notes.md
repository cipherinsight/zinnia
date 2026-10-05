# Notes

1. `calibrate.py` uses elementwise arithmetic followed by inclusive lower and upper clipping.
2. `background.py` finds each column minimum and broadcasts it across the sample rows.
3. `threshold_energy.py` zeros readings at or below their channel threshold before summing squares by row.
4. `distances.py` broadcasts the two point collections to form coordinate differences for every pair.
5. `pool.py` reshapes the spatial dimensions into pairs and sums the two block axes.
6. `smooth.py` copies each round's starting grid, then replaces only interior cells using integer floor division.
