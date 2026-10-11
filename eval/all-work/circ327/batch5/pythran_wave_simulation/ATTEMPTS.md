# pythran_wave_simulation

Category: `not_expressible: dynamic shape`

PARTICLE_COUNT has no declared upper bound. The fixed two-coordinate return does not bound intermediate particle arrays. sqrt/division are additional blockers.

as_written: not expressible

kernel.py:41: masspoints = np.empty((2, count, 2), np.float64). PARTICLE_COUNT has no declared upper bound. The fixed two-coordinate return does not bound intermediate particle arrays. sqrt/division are additional blockers.

rewrite: not expressible

PARTICLE_COUNT has no declared upper bound. The fixed two-coordinate return does not bound intermediate particle arrays. sqrt/division are additional blockers.

