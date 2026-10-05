def kernel(x):
    R = 3
    current = x.copy()
    for _ in range(R):
        updated = current.copy()
        updated[1:-1, 1:-1] = (
            4 * current[1:-1, 1:-1]
            + current[:-2, 1:-1]
            + current[2:, 1:-1]
            + current[1:-1, :-2]
            + current[1:-1, 2:]
        ) // 8
        current = updated
    return current
