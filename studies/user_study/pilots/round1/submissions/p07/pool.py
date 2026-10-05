def kernel(x):
    batch, height, width, channels = x.shape
    blocks = x.reshape(batch, height // 2, 2, width // 2, 2, channels)
    return np.sum(blocks, axis=(2, 4))
