# pythran_calculate_u

Category: `not_expressible: dynamic loop`

dt != 0 gives neither a positive-step guarantee nor a bound on (t_stop-t)/dt; do not infer a cap from fixture values. Fixed-point division is an additional blocker.

as_written: not expressible

kernel.py:12: while t <= t_stop:. dt != 0 gives neither a positive-step guarantee nor a bound on (t_stop-t)/dt; do not infer a cap from fixture values. Fixed-point division is an additional blocker.

rewrite: not expressible

dt != 0 gives neither a positive-step guarantee nor a bound on (t_stop-t)/dt; do not infer a cap from fixture values. Fixed-point division is an additional blocker.

