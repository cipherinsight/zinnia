def kernel(x):
    batches = []
    for batch in x:
        rows = []
        for i in range(0, batch.shape[0], 2):
            row = []
            for j in range(0, batch.shape[1], 2):
                row.append(batch[i:i+2,j:j+2,:].sum(axis=(0,1)))
            rows.append(row)
        batches.append(rows)
    return np.array(batches)
