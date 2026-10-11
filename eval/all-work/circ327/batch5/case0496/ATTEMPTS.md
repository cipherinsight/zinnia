# case0496

Category: `not_expressible: dynamic shape`

The two requested shape values have no declared upper bounds; input shape 42x10 supplies only lower bounds for valid padding.

as_written: not expressible

kernel.py:11: result = np.pad(a, ((0, shape[0]-a.shape[0]), (0, shape[1]-a.shape[1])), 'constant'). The two requested shape values have no declared upper bounds; input shape 42x10 supplies only lower bounds for valid padding.

rewrite: not expressible

The two requested shape values have no declared upper bounds; input shape 42x10 supplies only lower bounds for valid padding.

