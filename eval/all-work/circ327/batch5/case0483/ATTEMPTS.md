# case0483

Category: `not_expressible: dynamic shape`

degree is unconstrained; even with 100 samples, NumPy can return degree+1 coefficients for higher-degree underdetermined fits. Padding has no domain-implied capacity.

as_written: not expressible

kernel.py:11: result = np.polyfit(x, y, degree). degree is unconstrained; even with 100 samples, NumPy can return degree+1 coefficients for higher-degree underdetermined fits. Padding has no domain-implied capacity.

rewrite: not expressible

degree is unconstrained; even with 100 samples, NumPy can return degree+1 coefficients for higher-degree underdetermined fits. Padding has no domain-implied capacity.

