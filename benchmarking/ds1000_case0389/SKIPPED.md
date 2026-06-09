# DS-1000 case 0389 -- handwritten Noir skipped (specialization-required)

The kernel takes a runtime scalar parameter whose value determines the
output shape (n_unique for 0389=0298, h/w for 0389=0389). Noir requires
compile-time array sizes for `pub` returns and compile-time for-loop
bounds. A faithful runtime-parameter implementation would also have to
change the Python-side INPUT_BUILDERS to provide max-sized claimed
arrays, which crosses out of the Noir-baseline scope.

Same category as 0388 / 0390 (patch reshape on Float[100,200]):
naturally inexpressible in hand-written Noir without specialising on the
trace-time parameter value, so the original baseline asserted
`param == sample_value`. Carrying that cheat into the Zinnia-vs-Noir
comparison was unfair; the case is dropped from the sweep instead.
`main.nr.skipped_specialization` preserves the original for the record.

Zinnia handles the kernel as-written (its IR produces the per-cell mux
mask for the runtime parameter; the output is trace-pinned to the
sample-value-derived shape).
