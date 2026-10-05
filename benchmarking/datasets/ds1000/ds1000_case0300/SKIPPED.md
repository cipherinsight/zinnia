Skipped: np.percentile requires a runtime floor of (n-1)*p/100 to pick the
two array elements to interpolate between. Noir has no native floor and no
range_check primitive comparable to halo2's; expressing the lo/hi bracket
needs an unconstrained witness plus a sort + indicator gather. Achievable
but would need ~80 lines and a multi-pass debug. Deferred for coverage budget.
