# case0373

Category: `accepted_with_rewrite`

Remove unused sorting; ECDF ordinate output depends only on length 50.

as_written: needs gadget

needs gadget: sort

rewrite: accepted

Compilation accepted.

- Remove unused sorted values: ecdf_result returns only arange(1,51)/50; len(sort(grades)) is always 50. Constant output is encoded at scale 2^32.
