# case0458

Category: `needs_gadget: sqrt`

Euclidean norms require a square root; replacing them by squared distances changes the output.

as_written: needs gadget

kernel.py:11: result = np.triu(np.linalg.norm(a - a[:, None], axis = -1)). Euclidean norms require a square root; replacing them by squared distances changes the output.

rewrite: needs gadget

Euclidean norms require a square root; replacing them by squared distances changes the output.

