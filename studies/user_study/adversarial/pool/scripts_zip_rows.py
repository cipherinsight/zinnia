def kernel(x):
    batches = []
    for batch in x:
        rows = []
        for top, bottom in zip(batch[::2], batch[1::2]):
            row = top + bottom
            rows.append(row[::2] + row[1::2])
        batches.append(rows)
    return np.array(batches)
