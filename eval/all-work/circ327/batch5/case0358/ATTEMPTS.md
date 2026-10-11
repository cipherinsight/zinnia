# case0358

Category: `needs_gadget: sort`

Sorting is required to preserve the data-dependent reordering of b; the brief prohibits writing a sorting gadget.

as_written: needs gadget

kernel.py:11: index = np.argsort(a.sum(axis = (1, 2))). Sorting is required to preserve the data-dependent reordering of b; the brief prohibits writing a sorting gadget.

rewrite: needs gadget

Sorting is required to preserve the data-dependent reordering of b; the brief prohibits writing a sorting gadget.

