# case0407

Category: `not_expressible: dynamic shape`

The 100 map entries have no declared upper bound; bincount length is max(adjusted accmap)+1, not 100.

as_written: not expressible

kernel.py:14: result = np.bincount(accmap, weights = a). The 100 map entries have no declared upper bound; bincount length is max(adjusted accmap)+1, not 100.

rewrite: not expressible

The 100 map entries have no declared upper bound; bincount length is max(adjusted accmap)+1, not 100.

