# case0299

Category: `not_expressible: dynamic shape`

The 200 input integers have no declared value-range bound; max(a)-min(a)+1 is not bounded by their count.

as_written: not expressible

kernel.py:12: b = np.zeros((a.size, temp.max()+1)). The 200 input integers have no declared value-range bound; max(a)-min(a)+1 is not bounded by their count.

rewrite: not expressible

The 200 input integers have no declared value-range bound; max(a)-min(a)+1 is not bounded by their count.

